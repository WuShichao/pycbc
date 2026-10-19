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
    b = b0 exp(sum_i xi_i D_i),   t = t0 + sum_i xi_i T_i,
    rho_k = (1/N_T) sum_j R0(t_j) exp(delta . L(t_j)) exp(-i k Omega t_j),

with ``xi_i`` the intrinsic coordinates, ``delta`` the two sky offsets and
``Delta`` the arrival time. ``b0`` is the whitened source and ``t0`` its
stationary time, ``D_i`` and ``T_i`` their derivatives in ``xi_i``, ``R0``
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
    kmax : int
        Highest yearly harmonic kept.
    dtype : numpy dtype
        Complex type of the arrays, e.g. ``numpy.complex64`` on a GPU.
    """
    def __init__(self, grid, coarse_frequencies, response_samples,
                 sky_derivatives, source, stationary_time, log_derivatives,
                 time_derivatives, kmax=4, dtype=numpy.complex128):
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
        self.nintrinsic = self.D.shape[0]
        coarse = numpy.asarray(coarse_frequencies, dtype=float)
        x = (grid.frequencies - coarse[0]) / (coarse[1] - coarse[0])
        i = numpy.clip(numpy.floor(x).astype(int), 0, len(coarse) - 2)
        self._index = xp.asarray(i)
        self._weight = xp.asarray(numpy.clip(x - i, 0, 1), dtype=real)
        self.coarse_frequencies = coarse

    def fine(self, values):
        """Coarse-grid ``values`` (last axis) on the fine grid."""
        i, w = self._index, self._weight
        return values[..., i] * (1 - w) + values[..., i + 1] * w

    def pieces(self, xi):
        """``b``, ``t`` and ``rho_k`` (shape ``(2 kmax + 1, 2, nchannels,
        ncoarse)``) at shape coordinates ``xi``: the intrinsic ones, then
        the two sky offsets."""
        xp = self.xp
        xi = numpy.asarray(xi, dtype=float)
        if xi.shape != (self.nintrinsic + 2,):
            raise ValueError("xi must have {} entries".format(
                self.nintrinsic + 2))
        intrinsic = xp.asarray(xi[:self.nintrinsic])
        b = self.b0 * xp.exp(xp.tensordot(intrinsic.astype(self.dtype),
                                          self.D, axes=1))
        t = self.t0 + xp.tensordot(intrinsic, self.T, axes=1)
        sky = xp.exp(float(xi[-2]) * self.L[0] + float(xi[-1]) * self.L[1])
        rho = xp.fft.fft(self.R0 * sky, axis=0) / self.samples
        return b, t, rho[xp.asarray(self.orders % self.samples)]

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
        acc = xp.zeros((2, self.size), dtype=self.dtype)
        for i, k in enumerate(self.orders):
            phase = self._phase(k, b, t)
            for a in range(2):
                prod = xp.zeros(self.size, dtype=self.dtype)
                for c in range(data.shape[0]):
                    prod += xp.conj(self.fine(rho[i, a, c]) * phase) * data[c]
                # the k-th harmonic is a shift of k bins per orbital period
                acc[a] += xp.roll(prod, -int(k) * self.shift)
        return acc

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
        out = []
        for i in range(self.nintrinsic):
            d = xp.zeros_like(y)
            for j, k in enumerate(self.orders):
                factor = self.D[i] + 1j * k * ORBITAL_ANGULAR_FREQUENCY \
                    * self.T[i]
                d += self.fine(rho[j]) * (self._phase(k, b, t, delta)
                                          * factor.astype(self.dtype))
            out.append(d * shift)
        for axis in range(2):
            drho = xp.fft.fft(self.R0 * sky * self.L[axis], axis=0) \
                / self.samples
            drho = drho[xp.asarray(self.orders % self.samples)]
            out.append(self.template(b, t, drho, delta))
        return y, xp.stack(out)


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
                          samples=32, dtype=numpy.complex128):
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
    log_derivatives, time_derivatives = [], []
    for name, scale in intrinsic.items():
        # halve the step until both displaced sources are valid: a weakly
        # measured coordinate has a shape unit beyond its physical range
        h = step
        for _ in range(60):
            try:
                moved = []
                for sign in (1, -1):
                    point = dict(params)
                    point[name] = point[name] + sign * h * scale
                    moved.append(evaluate(point))
                break
            except (RuntimeError, ValueError):
                h *= 0.5
        else:
            raise ValueError("no valid derivative step for " + name)
        (bp, tp), (bm, tm) = moved
        ok = (numpy.abs(bp) > 0) & (numpy.abs(bm) > 0)
        ratio = numpy.where(ok, bp / numpy.where(ok, bm, 1), 1.0)
        log_derivatives.append((numpy.log(numpy.abs(ratio)) + 1j
                                * numpy.unwrap(numpy.angle(ratio)))
                               / (2 * h))
        time_derivatives.append((tp - tm) / (2 * h))
    lon, lat = params['eclipticlongitude'], params['eclipticlatitude']
    response = yearly_response_samples(orbit, terms, coarse_frequencies, lon,
                                       lat, lon, lat, samples=samples)
    sky = sky_log_derivatives(orbit, terms, coarse_frequencies, lon, lat,
                              scales=sky_scales, samples=samples)
    return LinearExponentSurrogate(
        grid, coarse_frequencies, response, sky, b0, t0,
        numpy.array(log_derivatives), numpy.array(time_derivatives),
        kmax=kmax, dtype=dtype)
