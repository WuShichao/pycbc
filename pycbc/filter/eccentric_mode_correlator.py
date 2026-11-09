# Copyright (C) 2026 Shichao Wu
# This program is free software under the GNU General Public License,
# version 3 or (at your option) any later version.
"""Coherent eccentric-mode correlations with separate clocks and time windows.

This direct, streamed implementation is a reference for optimized search
engines. It evaluates a supplied finite yearly-response expansion on each
mode's own clock; it does not replace the sum by a single clock or drop
cross-mode Gram terms. No native waveform generator is called here.
"""
import numpy

from pycbc.tdi.ssb_yearly_harmonics import ORBITAL_ANGULAR_FREQUENCY


class EccentricModeCorrelator:
    """A supplied eccentric template over arrival-time offsets.

    ``waveforms`` and ``times`` have shape (K,F); ``waveforms`` are the
    complex, whitened source modes at zero mean anomaly and delay.
    ``response`` has shape (R,2,C,F), with yearly integer ``orders`` of
    length R, two polarization bases and C data channels. All modes must
    share that angular pattern, as the (2,2,n) family does. The response
    is reconstructed at times[k,f] + delay. Any frequency weights must
    already be included in waveforms and the data's noise convention.

    ``time_window`` optionally masks each mode by its own shifted SPA
    time in [start,stop). This is an SPA observation mask, not an exact
    time-domain truncation. Accuracy of the supplied response harmonics,
    source interpolation and frequency quadrature belongs to the caller.
    ``xp`` selects NumPy or CuPy. No custom CUDA kernels are required.
    """
    def __init__(self, frequencies, waveforms, times, response, orders,
                 time_window=None, xp=numpy):
        self.xp = xp
        self.f = xp.asarray(frequencies, dtype=xp.float64)
        self.b = xp.asarray(waveforms, dtype=xp.complex128)
        self.t = xp.asarray(times, dtype=xp.float64)
        self.rho = xp.asarray(response, dtype=xp.complex128)
        raw = numpy.asarray(orders)
        if (self.f.ndim != 1 or not self.f.size or
                not bool(xp.all(xp.isfinite(self.f))) or
                bool(xp.any(self.f <= 0)) or bool(xp.any(xp.diff(self.f) <= 0))):
            raise ValueError('frequencies must be positive, finite and increasing')
        if (self.b.ndim != 2 or not self.b.shape[0] or
                self.b.shape[1] != self.f.size or self.t.shape != self.b.shape):
            raise ValueError('waveforms and times must have shape (K,F), K > 0')
        if (self.rho.ndim != 4 or self.rho.shape[1] != 2 or
                not self.rho.shape[2] or self.rho.shape[-1] != self.f.size):
            raise ValueError('response must have shape (R,2,C,F)')
        if (raw.shape != (self.rho.shape[0],) or not raw.size or
                not numpy.all(numpy.isfinite(raw)) or
                not numpy.all(raw == numpy.floor(raw)) or
                len(numpy.unique(raw)) != len(raw)):
            raise ValueError('orders must be distinct finite integers')
        self.orders = xp.asarray(raw, dtype=xp.float64)
        for a in (self.b, self.t, self.rho):
            if not bool(xp.all(xp.isfinite(a))):
                raise ValueError('source and response arrays must be finite')
        self.time_window = None
        if time_window is not None:
            start, stop = map(float, time_window)
            if not numpy.isfinite(start + stop) or start >= stop:
                raise ValueError('time_window must be finite and increasing')
            self.time_window = (start, stop)

    def iter_statistics(self, data, delays, lag_chunk=4, frequency_chunk=16384):
        """Yield (lag slice,z,G) with shapes (K,2,L) and (K,2,K,2,L).

        Each yielded block includes the entire frequency band coherently.
        Frequency chunks only bound temporary memory; their complex
        correlations and full Gram matrices are summed before any statistic.
        Noise variance is not rescaled here. Delays need not form a grid.
        """
        xp = self.xp
        data = xp.asarray(data, dtype=xp.complex128)
        delays = xp.asarray(delays, dtype=xp.float64)
        if data.shape != (self.rho.shape[2], self.f.size):
            raise ValueError('data must have shape (C,F) matching the response')
        if (delays.ndim != 1 or not delays.size or
                not bool(xp.all(xp.isfinite(delays))) or
                not bool(xp.all(xp.isfinite(data)))):
            raise ValueError('data and a nonempty delay vector must be finite')
        if any(isinstance(n, bool) or not isinstance(n, (int, numpy.integer))
               or n <= 0 for n in (lag_chunk, frequency_chunk)):
            raise ValueError('chunk sizes must be positive integers')
        k = self.b.shape[0]
        for first in range(0, len(delays), lag_chunk):
            last = min(first + lag_chunk, len(delays))
            delta = delays[first:last]
            z = xp.zeros((k, 2, len(delta)), dtype=xp.complex128)
            gram = xp.zeros((k, 2, k, 2, len(delta)), dtype=xp.complex128)
            for start in range(0, len(self.f), frequency_chunk):
                stop = min(start + frequency_chunk, len(self.f))
                sl = slice(start, stop)
                time = self.t[None, :, sl] + delta[:, None, None]
                orbit = xp.exp(1j * ORBITAL_ANGULAR_FREQUENCY *
                               time[:, :, None] * self.orders[None, None, :, None])
                response = xp.einsum('lkrf,racf->lkacf', orbit, self.rho[..., sl])
                phase = xp.exp(-2j * numpy.pi * delta[:, None] * self.f[None, sl])
                wave = self.b[None, :, sl] * phase[:, None]
                if self.time_window is not None:
                    a, b = self.time_window
                    wave = wave * ((time >= a) & (time < b))
                templates = response * wave[:, :, None, None]
                z += xp.einsum('lkacf,cf->kal', templates.conj(), data[:, sl])
                gram += xp.einsum('lkacf,lmbcf->kambl', templates.conj(), templates)
            yield slice(first, last), z, gram

    def at(self, data, delays, max_output_bytes=256 * 1024**2, **kwargs):
        """Collect streamed correlations and Grams for a finite delay vector.

        Use iter_statistics directly when retaining all delays would exceed
        max_output_bytes. Temporary response/template arrays are additional.
        """
        k, n = self.b.shape[0], len(delays)
        need = (2*k + 4*k*k) * n * numpy.dtype(complex).itemsize
        if need > max_output_bytes:
            raise MemoryError('collected mode statistics exceed the output memory limit')
        z = self.xp.empty((k, 2, n), dtype=self.xp.complex128)
        gram = self.xp.empty((k, 2, k, 2, n), dtype=self.xp.complex128)
        for sl, values, matrix in self.iter_statistics(data, delays, **kwargs):
            z[..., sl], gram[..., sl] = values, matrix
        return z, gram
