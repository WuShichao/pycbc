# Copyright (C) 2026  Shichao Wu
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""A template family whose parameters enter through linear exponents.

Around a reference source, the SSB-frame template of basis ``a`` (plus or
cross) and channel ``c`` at shape coordinates ``xi`` is

    y_ac(f; Delta) = b(f) sum_k rho_k,ac(f) exp(i k Omega (t(f) + Delta))
                     exp(-2 pi i (f - f0) Delta),
    b = b0 exp(sum_i xi_i D_i + 1/2 sum_ij xi_i xi_j D_ij),
    t = t0 + sum_i xi_i T_i + 1/2 sum_ij xi_i xi_j T_ij,
    rho_k = (1/N_T) sum_j R0(t_j) exp(delta . L(t_j)) exp(-i k Omega t_j),

with ``xi_i`` the intrinsic coordinates, ``delta`` the two sky offsets and
``Delta`` the arrival time. ``b0`` is the whitened source and ``t0`` its
stationary time, ``D_i`` and ``T_i`` their derivatives in ``xi_i`` and
``D_ij``, ``T_ij`` the second derivatives (zero for a first-order family),
``R0``
the yearly response samples of
:func:`pycbc.tdi.ssb_yearly_harmonics.yearly_response_samples` on a coarse
frequency grid and ``L`` their sky log-derivatives. The correlation with
the data at every arrival time on the grid costs one FFT per basis.

Arrays live on numpy or cupy, as :func:`pycbc.scheme.array_module` picks
when the surrogate is made.
"""

import numpy

from pycbc.scheme import array_module
from pycbc.tdi.ssb_yearly_harmonics import ORBITAL_ANGULAR_FREQUENCY


class LinearExponentSurrogate(object):
    """The linear-exponent template family of one reference source.

    Parameters
    ----------
    grid : pycbc.tdi.ssb_yearly_harmonics.SSBGrid
        The SSB time grid; its frequencies are the fine grid.
    coarse_frequencies : array_like
        Uniformly spaced frequencies of the response samples.
    response_samples : array_like, shape ``(N_T, 2, nchannels, ncoarse)``
        ``R0`` at ``N_T`` times over one orbit.
    sky_derivatives : array_like, shape ``(2,) + response_samples.shape``
        ``L``, the log-derivatives of ``R0`` in the two sky offsets.
    source : array_like, complex, shape ``(N,)``
        ``b0`` on the fine grid, whitened.
    stationary_time : array_like, shape ``(N,)``
        ``t0`` on the fine grid, s.
    log_derivatives : array_like, complex, shape ``(nintrinsic, N)``
        ``D_i``.
    time_derivatives : array_like, shape ``(nintrinsic, N)``
        ``T_i``.
    second_log_derivatives, second_time_derivatives : array_like, optional
        ``D_ij`` and ``T_ij`` for ``i <= j`` in row-major order, shape
        ``(nintrinsic (nintrinsic + 1) / 2, N)``. Along a nearly degenerate
        combination of the intrinsic parameters the first-order family
        decorrelates from the source within a small fraction of a shape
        unit; the second-order terms carry it much further.
    kmax : int
        Highest yearly harmonic kept.
    dtype : numpy dtype
        Complex type of the arrays, e.g. ``numpy.complex64`` on a GPU.
    """
    def __init__(self, grid, coarse_frequencies, response_samples,
                 sky_derivatives, source, stationary_time, log_derivatives,
                 time_derivatives, kmax=4, dtype=numpy.complex128,
                 second_log_derivatives=None, second_time_derivatives=None):
        xp = self.xp = array_module()
        self.grid = grid
        self.size = grid.size
        self.shift = grid.harmonic_shift
        self.orders = numpy.arange(-kmax, kmax + 1)
        self.samples = int(numpy.shape(response_samples)[0])
        if 2 * kmax + 1 > self.samples:
            raise ValueError("too few response samples for kmax")
        real = numpy.float32 if dtype == numpy.complex64 else numpy.float64
        self.dtype, self.real = dtype, real
        self.R0 = xp.asarray(response_samples, dtype=dtype)
        self.L = xp.asarray(sky_derivatives, dtype=dtype)
        self.b0 = xp.asarray(source, dtype=dtype)
        self.t0 = xp.asarray(stationary_time, dtype=numpy.float64)
        self.D = xp.asarray(log_derivatives, dtype=dtype)
        self.T = xp.asarray(time_derivatives, dtype=numpy.float64)
        self.nintrinsic = n = self.D.shape[0]
        self.pairs = [(i, j) for i in range(n) for j in range(i, n)]
        if second_log_derivatives is None:
            self.D2 = self.T2 = None
            self._D, self._T = self.D, self.T
        else:
            self.D2 = xp.asarray(second_log_derivatives, dtype=dtype)
            self.T2 = xp.asarray(second_time_derivatives, dtype=numpy.float64)
            if self.D2.shape[0] != len(self.pairs) or \
                    self.T2.shape[0] != len(self.pairs):
                raise ValueError("need one second derivative per pair i <= j")
            # the second-order terms enter like further coordinates, with
            # weights xi_i xi_j (halved on the diagonal)
            self._D = xp.concatenate([self.D, self.D2])
            self._T = xp.concatenate([self.T, self.T2])
        coarse = numpy.asarray(coarse_frequencies, dtype=float)
        x = (grid.frequencies - coarse[0]) / (coarse[1] - coarse[0])
        i = numpy.clip(numpy.floor(x).astype(int), 0, len(coarse) - 2)
        self._index = xp.asarray(i)
        self._weight = xp.asarray(numpy.clip(x - i, 0, 1), dtype=real)
        self.coarse_frequencies = coarse

    def _weights(self, xi):
        """The coefficient of each row of the stacked derivatives."""
        xi = numpy.asarray(xi, dtype=float)
        if self.D2 is None:
            return xi
        return numpy.concatenate([xi, [(0.5 if i == j else 1.0) * xi[i]
                                       * xi[j] for i, j in self.pairs]])

    def _gradient(self, xi, i):
        """``d/dxi_i`` of the exponent and of ``t`` at ``xi``."""
        dlog, dt = self.D[i], self.T[i]
        if self.D2 is not None:
            for p, (a, c) in enumerate(self.pairs):
                if i in (a, c):
                    other = c if a == i else a
                    dlog = dlog + float(xi[other]) * self.D2[p]
                    dt = dt + float(xi[other]) * self.T2[p]
        return dlog, dt

    def fine(self, values):
        """Coarse-grid ``values`` (last axis) on the fine grid."""
        i, w = self._index, self._weight
        return values[..., i] * (1 - w) + values[..., i + 1] * w

    def pieces(self, xi):
        """``b``, ``t`` and ``rho_k`` (shape ``(2 kmax + 1, 2, nchannels,
        ncoarse)``) at shape coordinates ``xi``: the intrinsic ones, then
        the two sky offsets."""
        xi = numpy.asarray(xi, dtype=float)
        if xi.shape != (self.nintrinsic + 2,):
            raise ValueError("xi must have {} entries".format(
                self.nintrinsic + 2))
        b, t = self._source(xi[:self.nintrinsic])
        return b, t, self._response(xi[-2:])

    def _source(self, intrinsic):
        """``b`` and ``t`` at the intrinsic shape coordinates."""
        xp = self.xp
        weights = self._weights(intrinsic)
        if xp is not numpy:
            return self._pieces_kernel(xp.asarray(weights))
        b = self.b0 * xp.exp(xp.tensordot(weights.astype(self.dtype),
                                          self._D, axes=1))
        t = self.t0 + xp.tensordot(weights, self._T, axes=1)
        return b, t

    def _response(self, sky_offsets):
        """``rho_k`` at the two sky offsets."""
        xp = self.xp
        sky = xp.exp(float(sky_offsets[0]) * self.L[0]
                     + float(sky_offsets[1]) * self.L[1])
        rho = xp.fft.fft(self.R0 * sky, axis=0) / self.samples
        return rho[xp.asarray(self.orders % self.samples)]

    def responses(self, sky_offsets):
        """``rho_k`` at each row of sky offsets ``(M, 2)``, shape ``(M, 2
        kmax + 1, 2, nchannels, ncoarse)``."""
        xp = self.xp
        d = xp.asarray(numpy.atleast_2d(sky_offsets), dtype=self.real)
        expo = (d[:, 0, None, None, None, None] * self.L[0][None]
                + d[:, 1, None, None, None, None] * self.L[1][None])
        rho = xp.fft.fft(self.R0[None] * xp.exp(expo), axis=1) \
            / self.samples
        return rho[:, xp.asarray(self.orders % self.samples)]

    _PIECES = r"""
    #include <cupy/complex.cuh>
    typedef complex<{real}> cplx;
    extern "C" __global__ void pieces(
        const cplx* b0, const cplx* D, const double* t0, const double* T,
        const double* xi, const long long n, const int nint, cplx* b,
        double* t) {
      long long f = (long long)blockDim.x * blockIdx.x + threadIdx.x;
      if (f >= n) return;
      cplx e(0, 0);
      double tt = t0[f];
      for (int i = 0; i < nint; i++) {
        e += ({real})xi[i] * D[i * n + f];
        tt += xi[i] * T[i * n + f];
      }
      b[f] = b0[f] * exp(e);
      t[f] = tt;
    }
    """

    def _pieces_kernel(self, intrinsic, threads=256):
        """``b`` and ``t`` in one pass on cupy."""
        xp = self.xp
        if not hasattr(self, '_pieces'):
            real = 'float' if self.dtype == numpy.complex64 else 'double'
            self._pieces = xp.RawModule(code=self._PIECES.replace(
                '{real}', real)).get_function('pieces')
            self._D = xp.ascontiguousarray(self._D)
            self._T = xp.ascontiguousarray(self._T)
        b = xp.empty(self.size, dtype=self.dtype)
        t = xp.empty(self.size, dtype=xp.float64)
        self._pieces(((self.size + threads - 1) // threads,), (threads,), (
            self.b0, self._D, self.t0, self._T,
            xp.ascontiguousarray(intrinsic, dtype=xp.float64),
            numpy.int64(self.size), numpy.int32(len(intrinsic)), b, t))
        return b, t

    def _phase(self, k, b, t, delta=0.0):
        omega = k * ORBITAL_ANGULAR_FREQUENCY
        return (self.xp.exp(1j * omega * (t + delta)) * b).astype(self.dtype)

    def correlate(self, data, b, t, rho):
        """``z_a(Delta) = sum_c,f conj(y_ac(f; Delta)) X_c(f)`` at every
        arrival time ``Delta = j delta_t`` of the grid; ``data`` is the
        SSB-time data spectrum, shape ``(nchannels, N)``. Returns shape
        ``(2, N)``."""
        spectrum = self.correlation_spectrum(data, b, t, rho)
        return self.xp.fft.ifft(spectrum, axis=-1) * self.size

    def correlation_spectrum(self, data, b, t, rho):
        """The spectrum ``a_m`` of the correlation, ``z(Delta) = sum_m a_m
        exp(2 pi i m Delta / T)`` with ``T`` the grid's duration: evaluated
        at any ``Delta``, see
        :mod:`pycbc.filter.subsample_arrival_time`."""
        xp = self.xp
        data = data.astype(self.dtype, copy=False)
        if xp is not numpy and self.dtype == numpy.complex64:
            return self._correlation_spectrum_kernel(data, b, t, rho)
        acc = xp.zeros((2, self.size), dtype=self.dtype)
        # exp(i k Omega t) b by recurrence in k, one exponential in all
        step = xp.exp(1j * ORBITAL_ANGULAR_FREQUENCY * t).astype(self.dtype)
        phase = (b * step ** int(self.orders[0])).astype(self.dtype)
        for i, k in enumerate(self.orders):
            if i:
                phase = phase * step
            # both bases and every channel of one harmonic at once
            weighted = xp.conj(self.fine(rho[i].astype(self.dtype,
                                                       copy=False)) * phase)
            prod = (weighted * data[None]).sum(axis=1)
            # the k-th harmonic is a shift of k bins per orbital period
            acc += xp.roll(prod, -int(k) * self.shift, axis=-1)
        return acc

    _KERNEL = r"""
    #include <cupy/complex.cuh>
    extern "C" __global__ void corr(
        const complex<float>* data, const complex<float>* b,
        const double* t, const int* index, const float* weight,
        const complex<float>* rho, complex<float>* out, const int n,
        const int nch, const int ncoarse, const int nk, const int kmin,
        const int shift, const double turns_per_second) {
      // out[a, f] = sum_k sum_c conj(y_kac(f + k shift)) X_c(f + k shift)
      int f = blockDim.x * blockIdx.x + threadIdx.x;
      if (f >= n) return;
      complex<float> acc0(0, 0), acc1(0, 0);
      for (int q = 0; q < nk; q++) {
        int k = kmin + q;
        int j = f + k * shift;
        j = j < 0 ? j + n : (j >= n ? j - n : j);
        // Omega t in turns, reduced in double, then k turns in single
        double turns = t[j] * turns_per_second;
        float c, sn;
        sincospif(2.0f * k * (float)(turns - floor(turns)), &sn, &c);
        complex<float> ph = b[j] * complex<float>(c, sn);
        int i = index[j];
        float w = weight[j];
        for (int ch = 0; ch < nch; ch++) {
          const complex<float>* r0 = rho + ((q * 2 + 0) * nch + ch) * ncoarse;
          const complex<float>* r1 = rho + ((q * 2 + 1) * nch + ch) * ncoarse;
          complex<float> f0 = r0[i] * (1.0f - w) + r0[i + 1] * w;
          complex<float> f1 = r1[i] * (1.0f - w) + r1[i + 1] * w;
          complex<float> x = data[ch * n + j];
          acc0 += conj(f0 * ph) * x;
          acc1 += conj(f1 * ph) * x;
        }
      }
      out[f] = acc0;
      out[n + f] = acc1;
    }
    """

    def _correlation_spectrum_kernel(self, data, b, t, rho):
        """correlation_spectrum in one pass of a fused kernel, single
        precision on cupy."""
        xp = self.xp
        if not hasattr(self, '_corr'):
            self._corr = xp.RawKernel(self._KERNEL, 'corr')
            self._index32 = self._index.astype(xp.int32)
            self._weight32 = self._weight.astype(xp.float32)
        out = xp.empty((2, self.size), dtype=xp.complex64)
        nch = data.shape[0]
        if abs(int(self.orders[0])) * self.shift >= self.size:
            raise ValueError("the harmonics shift beyond the grid")
        threads = 256
        self._corr(((self.size + threads - 1) // threads,), (threads,), (
            xp.ascontiguousarray(data), xp.ascontiguousarray(
                b, dtype=xp.complex64), xp.ascontiguousarray(
                t, dtype=xp.float64), self._index32, self._weight32,
            xp.ascontiguousarray(rho, dtype=xp.complex64), out,
            numpy.int32(self.size), numpy.int32(nch),
            numpy.int32(rho.shape[-1]), numpy.int32(len(self.orders)),
            numpy.int32(self.orders[0]), numpy.int32(self.shift),
            numpy.float64(ORBITAL_ANGULAR_FREQUENCY / (2 * numpy.pi))))
        return out

    def template(self, b, t, rho, delta):
        """The template ``y_ac(f; Delta)`` on the fine grid, shape
        ``(2, nchannels, N)``."""
        xp = self.xp
        nch = rho.shape[2]
        y = xp.zeros((2, nch, self.size), dtype=self.dtype)
        for i, k in enumerate(self.orders):
            phase = self._phase(k, b, t, delta)
            y += self.fine(rho[i]) * phase
        shift = xp.asarray(self.grid.frequencies - self.grid.first_frequency)
        return y * xp.exp(-2j * numpy.pi * shift * delta).astype(self.dtype)

    def tangents(self, xi=None, delta=0.0):
        """The template and its derivatives in every shape coordinate at
        ``xi`` (default the reference) and arrival time ``delta``: shapes
        ``(2, nchannels, N)`` and ``(nintrinsic + 2, 2, nchannels, N)``."""
        xp = self.xp
        xi = numpy.zeros(self.nintrinsic + 2) if xi is None else xi
        b, t, rho = self.pieces(xi)
        sky = xp.exp(float(xi[-2]) * self.L[0] + float(xi[-1]) * self.L[1])
        y = self.template(b, t, rho, delta)
        shift = xp.exp(-2j * numpy.pi * xp.asarray(
            self.grid.frequencies - self.grid.first_frequency)
            * delta).astype(self.dtype)
        # filled in place: a list stacked at the end would hold every
        # derivative twice
        out = xp.empty((self.nintrinsic + 2,) + y.shape, dtype=y.dtype)
        for i in range(self.nintrinsic):
            d = out[i]
            d[...] = 0
            dlog, dt = self._gradient(xi, i)
            for j, k in enumerate(self.orders):
                factor = dlog + 1j * k * ORBITAL_ANGULAR_FREQUENCY * dt
                d += self.fine(rho[j]) * (self._phase(k, b, t, delta)
                                          * factor.astype(self.dtype))
            d *= shift
        for axis in range(2):
            drho = xp.fft.fft(self.R0 * sky * self.L[axis], axis=0) \
                / self.samples
            drho = drho[xp.asarray(self.orders % self.samples)]
            out[self.nintrinsic + axis] = self.template(b, t, drho, delta)
        return y, out


class DailyGram(object):
    """The Gram matrix ``G_ab(Delta)`` of a surrogate's bases over the data
    window, on a daily grid of arrival times.

    In the stationary-phase picture frequency ``f`` sits at SSB time
    ``t(f) + Delta``, so binning the template power by day gives

        G_ab(Delta) = sum_m sum_tau Q_ab,m(tau) W_m(tau + Delta),
        Q_ab,m(tau) = sum_{t(f) in day tau} |b(f)|^2
                      sum_{k' - k = m} conj(rho_k,a) . rho_k',b (fbar_tau),
        W_m(t) = W(t) exp(i m Omega t),

    with ``W`` the data window and the response read at each day's mean
    frequency. ``G`` is in the units of the correlation: with ``W = 1`` it
    is ``sum_f conj(y_a) y_b``.

    Parameters
    ----------
    surrogate : LinearExponentSurrogate
        The family; its reference stationary time sizes the grid.
    day : float
        The bin width, s.
    window : callable, optional
        ``W(t)`` on SSB times; default one inside the grid's data span.
    """
    def __init__(self, surrogate, day=86400.0, window=None):
        xp = self.xp = surrogate.xp
        self.surrogate = surrogate
        grid = surrogate.grid
        self.day = float(day)
        t0 = surrogate.t0
        t_min, t_max = float(xp.min(t0)), float(xp.max(t0))
        self.t_lo = min(t_min, grid.start) - 2 * self.day
        if window is None:
            def window(t):
                return ((t >= grid.start) & (t < grid.stop)).astype(float)
        span = int(numpy.ceil((max(t_max, grid.stop) - self.t_lo) / self.day))
        days = self.t_lo + (numpy.arange(span) + 0.5) * self.day
        need = span + int(numpy.sum(window(days) > 0)) + 2
        self.size = 1 << int(numpy.ceil(numpy.log2(need)))
        days = self.t_lo + (numpy.arange(self.size) + 0.5) * self.day
        w = window(days)
        kmax = int(surrogate.orders[-1])
        self.harmonics = numpy.arange(-2 * kmax, 2 * kmax + 1)
        omega = ORBITAL_ANGULAR_FREQUENCY
        self._Wm = xp.asarray(numpy.array([
            numpy.fft.fft(w * numpy.exp(1j * m * omega * days))
            for m in self.harmonics]))
        nk = len(surrogate.orders)
        mmap = numpy.zeros((len(self.harmonics), nk * nk))
        for k in range(nk):
            for l in range(nk):
                mmap[l - k + 2 * kmax, k * nk + l] = 1.0
        self._mmap = xp.asarray(mmap.astype(complex))
        lag = numpy.arange(grid.size)
        delta = numpy.where(lag < grid.size // 2, lag, lag - grid.size) \
            * grid.delta_t
        x = numpy.mod(delta, self.day * self.size) / self.day - 0.5
        self._lag_j = xp.asarray(numpy.floor(x).astype(int) % self.size)
        self._lag_fr = xp.asarray(x - numpy.floor(x))
        self._freqs = xp.asarray(grid.frequencies)

    def __call__(self, b, t, rho):
        """``(g00, g11, g01)`` on the daily grid for one template."""
        xp, s = self.xp, self.surrogate
        power = xp.abs(b.astype(xp.complex128)) ** 2
        tau = xp.floor((t - self.t_lo) / self.day).astype(xp.int64)
        ok = (tau >= 0) & (tau < self.size)
        tau = xp.where(ok, tau, 0)
        power = xp.where(ok, power, 0)
        pw = xp.bincount(tau, weights=power, minlength=self.size)
        fbar = xp.bincount(tau, weights=power * self._freqs,
                           minlength=self.size) / xp.maximum(pw, 1e-300)
        coarse = s.coarse_frequencies
        x = (fbar - coarse[0]) / (coarse[1] - coarse[0])
        i = xp.clip(xp.floor(x).astype(xp.int64), 0, len(coarse) - 2)
        w = xp.clip(x - i, 0, 1)
        r = rho.astype(xp.complex128)
        rk = r[..., i] * (1 - w) + r[..., i + 1] * w
        nk = len(s.orders)
        pk = xp.einsum('kacd,lbcd->klabd', xp.conj(rk), rk).reshape(
            nk * nk, 2, 2, self.size)
        q = xp.tensordot(self._mmap, pk, axes=([1], [0])) * pw
        # sum_tau Q(tau) W_m(tau + lag), Q complex: conjugate it first
        g = xp.fft.ifft(xp.conj(xp.fft.fft(xp.conj(q), axis=-1))
                        * self._Wm[:, None, None, :], axis=-1).sum(axis=0)
        return g[0, 0].real, g[1, 1].real, g[0, 1]

    def at_lags(self, gram):
        """The daily ``(g00, g11, g01)`` interpolated to every arrival time
        of the surrogate's grid, in its FFT order."""
        j, fr = self._lag_j, self._lag_fr
        j1 = (j + 1) % self.size
        return tuple(g[j] * (1 - fr) + g[j1] * fr for g in gram)

    def at_delays(self, gram, delays):
        """The daily ``(g00, g11, g01)`` interpolated to arbitrary arrival
        times ``delays``, s."""
        xp = self.xp
        x = xp.mod(xp.asarray(delays, dtype=xp.float64),
                   self.day * self.size) / self.day - 0.5
        j = xp.floor(x).astype(xp.int64) % self.size
        fr = x - xp.floor(x)
        j1 = (j + 1) % self.size
        return tuple(g[j] * (1 - fr) + g[j1] * fr for g in gram)


def lal_plus_source(params, frequencies, epoch):
    """The plus polarisation at inclination 0, unit distance and zero phase
    on ``frequencies``, referenced to ``epoch``, and its stationary time.

    The clock is the sparse-TDI source's: frequency ``params['f_lower']``
    is reached at ``params['tdi_t_start']``.
    """
    from pycbc.tdi.sources import LALFDSource
    from pycbc.waveform import get_fd_waveform_sequence
    approximant = params.get('approximant', 'IMRPhenomD')
    if approximant == 'TDISparse':
        approximant = params.get('tdi_approximant', 'IMRPhenomD')
    shape = {k: params[k] for k in ('mass1', 'mass2', 'spin1z', 'spin2z')}
    frequencies = numpy.asarray(frequencies, dtype=float)
    hp, _ = get_fd_waveform_sequence(
        approximant=approximant, distance=1.0, inclination=0.0,
        coa_phase=0.0, sample_points=numpy.concatenate(
            [[params['f_lower']], frequencies]), **shape)
    hp = numpy.asarray(hp)[1:]
    source = LALFDSource(f_lower=params['f_lower'],
                         f_upper=params['tdi_f_upper'], distance=1.0,
                         inclination=0.0, coa_phase=0.0,
                         t_start=params.get('tdi_t_start', 0.0), **shape)
    first = source.stationary_times[0]
    time = numpy.interp(frequencies, source.sample_frequencies,
                        source.stationary_times - first + source.t_start)
    clock = 2 * numpy.pi * frequencies * (first - source.t_start)
    return hp * numpy.exp(1j * clock + 2j * numpy.pi * frequencies
                          * epoch), time


def surrogate_from_source(grid, whitening, orbit, terms, params, intrinsic,
                          sky_scales, coarse_frequencies, to_source=None,
                          source=lal_plus_source, step=0.5, kmax=4,
                          samples=32, dtype=numpy.complex128, order=1,
                          directions=None, derivatives='stencil'):
    """Build the linear-exponent family around a reference source.

    Parameters
    ----------
    grid : SSBGrid
        The SSB time grid.
    whitening : array_like
        ``sqrt(4 delta_f / S_n)`` on the grid's frequencies.
    orbit, terms
        The orbit and TDI channel terms of the response.
    params : dict
        The reference point, in the coordinates named by ``intrinsic`` plus
        the source parameters ``to_source`` needs, including
        ``eclipticlongitude`` and ``eclipticlatitude``.
    intrinsic : dict
        Intrinsic coordinate names and their shape units, in order.
    sky_scales : tuple
        Shape units of the longitude and latitude offsets, rad.
    coarse_frequencies : array_like
        Uniform frequencies of the response samples.
    to_source : callable, optional
        Maps a point to the parameters of ``source``; default the identity.
    source : callable
        ``source(params, frequencies, epoch) -> (h, t)``.
    step : float
        Central-difference step of the derivative columns, shape units;
        halved for a coordinate until its displaced sources are valid.
    kmax, samples : int
        Yearly harmonics kept and response samples per orbit.
    dtype : numpy dtype
        Complex type of the family's arrays.
    order : {1, 2}
        Order of the expansion in the intrinsic coordinates; 2 adds the
        second derivatives from the same stencil plus the four corners of
        each pair.
    directions : array_like, optional
        Columns ``v_k`` in shape units along which the stencil steps,
        default the coordinate axes; the derivatives are transformed back
        to the coordinates. Along a nearly degenerate combination the
        curvature is what is left after large axis terms cancel, so their
        stencil errors do not; stepping along the metric's eigenvectors
        measures that curvature directly.
    derivatives : {'stencil', 'taylorf2'}
        How the derivatives are taken: central differences of the source,
        or exactly, by Taylor-mode arithmetic through the TaylorF2 phase
        pinned at ``f_lower`` (see :mod:`pycbc.waveform.taylorf2_jets`),
        for the intrinsic coordinates mchirp, q, spin1z and spin2z. Along a
        weakly measured coordinate the stencil's error is of the order of
        the step squared times large higher derivatives; the second option
        has none.
    """
    from pycbc.tdi.ssb_yearly_harmonics import (sky_log_derivatives,
                                                yearly_response_samples)
    to_source = (lambda point: dict(point)) if to_source is None \
        else to_source
    f = grid.frequencies
    whitening = numpy.asarray(whitening, dtype=float)

    def evaluate(point):
        h, t = source(to_source(point), f, grid.start)
        return h * whitening, t

    b0, t0 = evaluate(params)
    names = list(intrinsic)
    ok0 = numpy.abs(b0) > 0

    def log_ratio(b):
        """ln(b / b0), the phase unwrapped along frequency."""
        ratio = numpy.where(ok0 & (numpy.abs(b) > 0),
                            b / numpy.where(ok0, b0, 1), 1.0)
        return numpy.log(numpy.abs(ratio)) + 1j * numpy.unwrap(
            numpy.angle(ratio))

    if derivatives == 'taylorf2':
        from pycbc.waveform.taylorf2_jets import taylorf2_derivatives
        order_names = ('mchirp', 'q', 'spin1z', 'spin2z')
        if sorted(names) != sorted(order_names):
            raise ValueError("TaylorF2 derivatives need the intrinsic "
                             "coordinates " + ', '.join(order_names))
        out = taylorf2_derivatives(params, intrinsic, f,
                                   params['f_lower'], order=order)
        # from the fixed order to the family's
        perm = [order_names.index(n) for n in names]
        log_derivatives = list(out[0][perm])
        time_derivatives = list(out[1][perm])
        second = None
        if order == 2:
            fixed = [(i, j) for i in range(4) for j in range(i, 4)]
            rows = []
            for i in range(4):
                for j in range(i, 4):
                    a_, b_ = sorted((perm[i], perm[j]))
                    rows.append(fixed.index((a_, b_)))
            second = (out[2][rows], out[3][rows])
    elif derivatives == 'stencil':
        nint = len(names)
        V = numpy.eye(nint) if directions is None else numpy.asarray(
            directions, dtype=float)
        if V.shape != (nint, nint):
            raise ValueError("need one direction per intrinsic coordinate")

        def displaced(offsets):
            """ln(b / b0) and t - t0 at the reference plus sum_k u_k v_k."""
            point = dict(params)
            for k, u in offsets.items():
                for name, v in zip(names, V[:, k]):
                    point[name] = point[name] + u * v * intrinsic[name]
            b, t = evaluate(point)
            return log_ratio(b), t - t0

        # per direction, halve the step until the displaced sources are
        # valid: a weakly measured coordinate has a shape unit beyond its
        # physical range
        steps, axis = [], []
        for k in range(nint):
            h = step
            for _ in range(60):
                try:
                    axis.append([displaced({k: sign * h}) for sign in (1, -1)])
                    break
                except (RuntimeError, ValueError):
                    h *= 0.5
            else:
                raise ValueError("no valid derivative step along direction "
                                 "{}".format(k))
            steps.append(h)
        # derivatives in u, then in xi = V u: gradient W^T g, Hessian W^T H W
        # with W = V^-1
        W = numpy.linalg.inv(V)
        grad_log = [(axis[k][0][0] - axis[k][1][0]) / (2 * steps[k])
                    for k in range(nint)]
        grad_t = [(axis[k][0][1] - axis[k][1][1]) / (2 * steps[k])
                  for k in range(nint)]
        log_derivatives = [sum(W[k, i] * grad_log[k] for k in range(nint))
                           for i in range(nint)]
        time_derivatives = [sum(W[k, i] * grad_t[k] for k in range(nint))
                            for i in range(nint)]
        second = None
        if order == 2:
            hess_log, hess_t = {}, {}
            for k in range(nint):
                (lp, tp), (lm, tm) = axis[k]
                hess_log[k, k] = (lp + lm) / steps[k] ** 2
                hess_t[k, k] = (tp + tm) / steps[k] ** 2
                for m in range(k + 1, nint):
                    corners = {(sa, sb): displaced({k: sa * steps[k],
                                                    m: sb * steps[m]})
                               for sa in (1, -1) for sb in (1, -1)}
                    norm = 4 * steps[k] * steps[m]
                    for out, part in ((hess_log, 0), (hess_t, 1)):
                        out[k, m] = out[m, k] = (
                            corners[1, 1][part] - corners[1, -1][part]
                            - corners[-1, 1][part]
                            + corners[-1, -1][part]) / norm
            second_log, second_time = [], []
            for i in range(nint):
                for j in range(i, nint):
                    second_log.append(sum(W[k, i] * W[m, j] * hess_log[k, m]
                                          for k in range(nint)
                                          for m in range(nint)))
                    second_time.append(sum(W[k, i] * W[m, j] * hess_t[k, m]
                                           for k in range(nint)
                                           for m in range(nint)))
            second = (numpy.array(second_log), numpy.array(second_time))
        elif order != 1:
            raise ValueError("order must be 1 or 2")
    else:
        raise ValueError("derivatives must be 'stencil' or 'taylorf2'")
    lon, lat = params['eclipticlongitude'], params['eclipticlatitude']
    response = yearly_response_samples(orbit, terms, coarse_frequencies, lon,
                                       lat, lon, lat, samples=samples)
    sky = sky_log_derivatives(orbit, terms, coarse_frequencies, lon, lat,
                              scales=sky_scales, samples=samples)
    return LinearExponentSurrogate(
        grid, coarse_frequencies, response, sky, b0, t0,
        numpy.array(log_derivatives), numpy.array(time_derivatives),
        kmax=kmax, dtype=dtype,
        second_log_derivatives=None if second is None else second[0],
        second_time_derivatives=None if second is None else second[1])
