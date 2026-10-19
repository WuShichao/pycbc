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
        return xp.fft.ifft(acc, axis=-1) * self.size

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
