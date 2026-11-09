# Copyright (C) 2026 Shichao Wu
# This program is free software under the GNU General Public License,
# version 3 or (at your option) any later version.
"""Mode FFT correlations with an actual finite observation-time window.

Yearly response harmonics are evaluated on each source mode's own clock.
Correlations use one inverse FFT per mode and polarization. The complete
cross-mode Gram is computed from windowed time-domain templates; no mode
powers are summed incoherently. Gram anchors sum the response harmonics in frequency before their IFFT,
so memory does not grow with the number of response harmonics.
"""
import numpy

from pycbc.filter.subsample_arrival_time import correlation_at
from pycbc.tdi.ssb_yearly_harmonics import ORBITAL_ANGULAR_FREQUENCY


class WindowedModeFFT:
    """Prepare one intrinsic/sky template for coherent arrival-time search.

    grid provides size, duration, delta_t, frequencies and harmonic_shift;
    duration must be harmonic_shift orbital periods. Source arrays have
    shape (K,N), with FD phases referred to the grid start. response(r)
    returns the (2,C,N) array for one yearly order. data is the unitary FFT
    of observed time data, shape (C,N). It must already be zero outside the
    observation interval in time. Each real noise component has variance 1
    before that window. Frequency weights/whitening belong in the source.

    observation=(first,last) is a half-open pair of time SAMPLE indices.
    delay_bounds are inclusive integer sample offsets. The crop spans the
    union of translated observation intervals and cannot exceed one period.
    Gram workspaces use complex64; Grams accumulate in complex128.
    Accuracy of source/response interpolation belongs to the caller.
    """
    def __init__(self, grid, waveforms, times, response, orders, data,
                 observation, delay_bounds, xp=numpy, dtype=numpy.complex64,
                 max_cache_bytes=4_000_000_000):
        self.grid, self.xp, self.dtype = grid, xp, numpy.dtype(dtype)
        self.orders = numpy.asarray(orders, dtype=int)
        self.size, self.duration = int(grid.size), float(grid.duration)
        self.dt = float(grid.delta_t)
        first, last = map(int, observation)
        jlo, jhi = map(int, delay_bounds)
        if not 0 <= first < last <= self.size or jlo > jhi:
            raise ValueError('invalid observation or delay bounds')
        self.first, self.last = first, last
        self.crop_first, self.crop_last = first-jhi, last-jlo
        self.crop_size = self.crop_last-self.crop_first
        if self.crop_size > self.size:
            raise ValueError('observation plus delay span exceeds the FFT period')
        data = xp.asarray(data, dtype=self.dtype)
        b = xp.asarray(waveforms, dtype=self.dtype)
        t = xp.asarray(times, dtype=xp.float64)
        if b.ndim != 2 or b.shape[-1] != self.size or t.shape != b.shape:
            raise ValueError('source arrays must have shape (K,N)')
        if data.ndim != 2 or data.shape[-1] != self.size:
            raise ValueError('data must have shape (C,N)')
        self.nmode, self.nchannel = b.shape[0], data.shape[0]
        self.bounds = jlo, jhi
        self.cache_bytes = b.nbytes+t.nbytes
        self.max_workspace_bytes = int(max_cache_bytes)
        self.b, self.t, self.response = b, t, response
        self.spectra = xp.zeros((self.nmode, 2, self.size), dtype=self.dtype)
        for r, order in enumerate(self.orders):
            rho = xp.asarray(response(r), dtype=self.dtype)
            if rho.shape != (2, self.nchannel, self.size):
                raise ValueError('response callback must return (2,C,N)')
            for k in range(self.nmode):
                phase = (b[k]*xp.exp(1j*int(order)*ORBITAL_ANGULAR_FREQUENCY*t[k])).astype(self.dtype)
                h = rho*phase[None, None]
                product = (h.conj()*data[None]).sum(axis=1)
                self.spectra[k] += xp.roll(product, -int(order)*grid.harmonic_shift, axis=-1)
                del h, product, phase
            del rho

    def correlations(self, sample_offsets):
        """Complex z (K,2,L) on FFT delays, including the common carrier."""
        xp = self.xp
        indices = numpy.asarray(sample_offsets, dtype=int)
        if indices.ndim != 1 or not len(indices):
            raise ValueError('sample_offsets must be a nonempty vector')
        output = xp.empty((self.nmode, 2, len(indices)), dtype=xp.complex128)
        selected = xp.asarray(indices % self.size)
        for k in range(self.nmode):
            transformed = xp.fft.ifft(self.spectra[k], axis=-1)*self.size
            output[k] = transformed[:, selected]
        carrier = xp.exp(2j*numpy.pi*float(self.grid.frequencies[0])*
                         xp.asarray(indices*self.dt))
        return output*carrier[None, None]

    def fractional_correlations(self, delays, chunk=2):
        """Evaluate the same correlation Fourier polynomial off its grid.

        Nonzero source spectra must have guard bins wider than the largest
        yearly harmonic shift; otherwise rolled terms alias off-grid phases.
        """
        xp = self.xp
        values = correlation_at(self.spectra, delays, self.duration, chunk=chunk, xp=xp)
        carrier = xp.exp(2j*numpy.pi*float(self.grid.frequencies[0])*xp.asarray(delays))
        return values*carrier[None, None]

    def gram(self, sample_offsets):
        """Full G from exact windowed templates, with harmonics summed first.

        One IFFT per mode/polarization/channel/anchor is needed, rather than
        one per yearly harmonic. Anchor blocks bound workspace memory.
        """
        xp = self.xp
        indices = numpy.asarray(sample_offsets, dtype=int)
        if numpy.any(indices < self.bounds[0]) or numpy.any(indices > self.bounds[1]):
            raise ValueError('Gram delay lies outside the prepared bounds')
        out = xp.empty((self.nmode, 2, self.nmode, 2, len(indices)), dtype=xp.complex128)
        one = self.nmode*2*self.nchannel*self.size*self.dtype.itemsize
        if one > self.max_workspace_bytes:
            raise MemoryError('one Gram anchor exceeds the workspace limit')
        chunk = max(1, min(len(indices), self.max_workspace_bytes//one))
        observation = xp.arange(self.first, self.last)
        root = self.dtype.type(numpy.sqrt(self.size))
        for first in range(0, len(indices), chunk):
            lags = indices[first:first+chunk]
            spectra = xp.zeros((len(lags), self.nmode, 2, self.nchannel, self.size), dtype=self.dtype)
            for r, order in enumerate(self.orders):
                rho = xp.asarray(self.response(r), dtype=self.dtype)
                for k in range(self.nmode):
                    phase = (self.b[k]*xp.exp(1j*int(order)*ORBITAL_ANGULAR_FREQUENCY*self.t[k])).astype(self.dtype)
                    h = rho*phase[None, None]
                    for j, lag in enumerate(lags):
                        coefficient = self.dtype.type(numpy.exp(1j*int(order)*ORBITAL_ANGULAR_FREQUENCY*lag*self.dt))
                        spectra[j, k] += h*coefficient
                    del h, phase
                del rho
            for j, lag in enumerate(lags):
                h = xp.empty((self.nmode, 2, self.nchannel, self.last-self.first), dtype=self.dtype)
                take = (observation-int(lag)) % self.size
                for k in range(self.nmode):
                    u = xp.fft.ifft(spectra[j, k], axis=-1)*root
                    h[k] = u[..., take]
                    del u
                matrix = h.reshape(self.nmode*2, -1)
                block = 16384
                if xp is not numpy and matrix.shape[1] > 4*block:
                    # A single 8-by-millions skinny GEMM underutilizes a GPU.
                    # Sum complete cross-Grams from parallel time chunks;
                    # no statistic or mode powers are summed incoherently.
                    pad = (-matrix.shape[1]) % block
                    padded = xp.pad(matrix, ((0, 0), (0, pad))) if pad else matrix
                    parts = xp.ascontiguousarray(padded.reshape(
                        self.nmode*2, -1, block).transpose(1, 0, 2)).astype(xp.complex128)
                    g = (parts.conj() @ parts.swapaxes(1, 2)).sum(axis=0)
                    del padded, parts
                else:
                    matrix = matrix.astype(xp.complex128)
                    g = matrix.conj() @ matrix.T
                out[..., first+j] = g.reshape(self.nmode, 2, self.nmode, 2)
                del h, matrix, g
            del spectra
        return out

    def interpolated_gram(self, delays, anchors, matrices):
        """Convex linear interpolation of complete anchor Grams.

        This preserves PSD but is an approximation between anchors. Check
        its statistic error against gram() at withheld integer delays.
        """
        xp = self.xp
        delays, anchors = numpy.asarray(delays), numpy.asarray(anchors)
        if len(anchors) == 1:
            return xp.broadcast_to(matrices[..., :1], matrices.shape[:-1]+(len(delays),))
        if numpy.any(delays < anchors[0]) or numpy.any(delays > anchors[-1]):
            raise ValueError('Gram interpolation would extrapolate')
        at = numpy.clip(numpy.searchsorted(anchors, delays, side='right')-1, 0, len(anchors)-2)
        weight = (delays-anchors[at])/(anchors[at+1]-anchors[at])
        at, weight = xp.asarray(at), xp.asarray(weight)
        return matrices[..., at]*(1-weight)+matrices[..., at+1]*weight
