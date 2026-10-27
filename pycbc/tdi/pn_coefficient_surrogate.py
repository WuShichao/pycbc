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
"""A template family exact in the TaylorF2 phase of its intrinsic
parameters.

The TaylorF2 phase is linear in its coefficients over fixed frequency
functions,

    Psi(f; theta) = sum_n A_n(theta) f^(n/3) + B_n(theta) f^(n/3) ln f,

(:func:`pycbc.waveform.taylorf2_jets.taylorf2_basis_coefficients`), so
about a reference source ``b0`` the family

    b(f; theta) = b0(f) (Mc / Mc0)^(5/6) exp(i dphase(f)),
    t(f; theta) = t0(f) + dt(f),

with ``dphase`` and ``dt`` the changes of the phase and time pinned at the
lower frequency, follows the source over the whole parameter range, not a
cell: what is left is the reference waveform's departure from TaylorF2 in
its parameter dependence (deep in the inspiral, as for stellar-mass
binaries in the LISA band, at the level of 1e-6 rad) and the Newtonian
amplitude. The response and the sky offsets are those of
:class:`pycbc.tdi.linear_exponent_surrogate.LinearExponentSurrogate`.
"""

import numpy

from pycbc.tdi.linear_exponent_surrogate import (LinearExponentSurrogate,
                                                 _host)
from pycbc.waveform.taylorf2_jets import (basis_phase_and_time,
                                          taylorf2_basis_coefficients)

NAMES = ('mchirp', 'q', 'spin1z', 'spin2z')


class PNCoefficientSurrogate(LinearExponentSurrogate):
    """The family of a :class:`LinearExponentSurrogate` built about a
    reference source, with its intrinsic part replaced by the exact TaylorF2
    phase.

    Parameters
    ----------
    family : LinearExponentSurrogate
        The family about the reference; its first derivatives serve
        :meth:`tangents`.
    reference : dict
        The reference's ``mchirp``, ``q`` (``m1 / m2``), ``spin1z`` and
        ``spin2z``.
    scales : dict
        Their shape units: ``theta = reference + scale * xi``.
    intrinsic : list of str
        The family's intrinsic coordinates, in its order: these four.
    f_lower : float
        The frequency at which the source's phase and time are fixed, Hz.
    """
    def __init__(self, family, reference, scales, intrinsic, f_lower):
        self.__dict__.update(family.__dict__)
        if sorted(intrinsic) != sorted(NAMES):
            raise ValueError("the intrinsic coordinates must be "
                             + ', '.join(NAMES))
        self.order = [list(intrinsic).index(n) for n in NAMES]
        self.theta0 = numpy.array([reference[n] for n in NAMES], dtype=float)
        self.units = numpy.array([scales[n] for n in NAMES], dtype=float)
        self.f_lower = float(f_lower)
        self._A0, self._B0, self._la0 = taylorf2_basis_coefficients(
            *self.theta0)
        xp = self.xp
        f = numpy.asarray(self.grid.frequencies, dtype=float)
        self._f = xp.asarray(f)
        self._cbrt = xp.asarray(numpy.cbrt(f))
        self._logf = xp.asarray(numpy.log(f))

    def _source(self, intrinsic):
        theta = self.theta0 + self.units * numpy.asarray(
            intrinsic, dtype=float)[self.order]
        A, B, la = taylorf2_basis_coefficients(*theta)
        return self.source_from_coefficients(A - self._A0, B - self._B0,
                                             la - self._la0)

    def source_from_coefficients(self, dA, dB, dla=0.0):
        """``b`` and ``t`` of the template whose basis coefficients differ
        from the reference's by ``dA`` and ``dB`` (eight each) and whose log
        amplitude by ``dla``: any point of the coefficient space, physical
        or not, as a template bank laid out there needs."""
        dA, dB = numpy.asarray(dA, dtype=float), numpy.asarray(dB, dtype=float)
        dla = float(dla)
        xp = self.xp
        if xp is numpy:
            phase, dt = basis_phase_and_time(dA, dB, self.grid.frequencies,
                                             self.f_lower)
            b = self.b0 * numpy.exp(dla + 1j * phase).astype(self.dtype)
            return b, self.t0 + dt
        return self._source_kernel(dA, dB, dla)

    _PN_KERNEL = r"""
    #include <cupy/complex.cuh>
    typedef complex<{real}> cplx;
    extern "C" __global__ void pn_source(
        const cplx* b0, const double* t0, const double* f,
        const double* cbrt_f, const double* log_f, const double* coef,
        const double c0, const double c1, const double f_lower,
        const double amplitude, const long long n, cplx* b, double* t) {
      // coef: the differences dA_0..7, dB_0..7 of the basis coefficients;
      // c0, c1 the pinning value and slope of the phase at f_lower
      long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
      if (i >= n) return;
      double x = cbrt_f[i], lf = log_f[i], fi = f[i];
      double xi = 1.0 / x;
      double p = xi * xi * xi * xi * xi;           // f^(-5/3)
      double psi = 0, dpsi = 0;
      for (int k = 0; k < 8; k++) {
        double m = (k - 5) / 3.0;
        psi += (coef[k] + coef[8 + k] * lf) * p;
        dpsi += (coef[k] * m + coef[8 + k] * (m * lf + 1)) * p;
        p *= x;
      }
      dpsi /= fi;
      const double two_pi = 6.283185307179586;
      double phase = -(psi - c0 - c1 * (fi - f_lower));
      phase -= two_pi * floor(phase / two_pi);
      float sn, cs;
      sincosf((float)phase, &sn, &cs);
      b[i] = b0[i] * cplx(amplitude * cs, amplitude * sn);
      t[i] = t0[i] + (dpsi - c1) / two_pi;
    }
    """

    def _source_kernel(self, dA, dB, dla, threads=256):
        xp = self.xp
        if not hasattr(self, '_pn_kernel'):
            real = 'float' if self.dtype == numpy.complex64 else 'double'
            self._pn_kernel = xp.RawModule(code=self._PN_KERNEL.replace(
                '{real}', real)).get_function('pn_source')
        # the pinning constants from the same sums at f_lower
        fl = self.f_lower
        lx, m = numpy.log(fl), (numpy.arange(8) - 5) / 3.0
        p = fl ** m
        c0 = float(numpy.sum((dA + dB * lx) * p))
        c1 = float(numpy.sum((dA * m + dB * (m * lx + 1)) * p) / fl)
        b = xp.empty(self.size, dtype=self.dtype)
        t = xp.empty(self.size, dtype=xp.float64)
        self._pn_kernel(((self.size + threads - 1) // threads,), (threads,), (
            self.b0, self.t0, self._f, self._cbrt, self._logf,
            xp.asarray(numpy.concatenate([dA, dB])), numpy.float64(c0),
            numpy.float64(c1), numpy.float64(fl),
            numpy.float64(numpy.exp(dla)), numpy.int64(self.size), b, t))
        return b, t

    _DAYS_KERNEL = r"""
    #include <cupy/complex.cuh>
    typedef complex<{real}> cplx;
    extern "C" __global__ void pn_days(
        const cplx* b0, const double* t0, const double* f,
        const double* cbrt_f, const double* log_f, const double* coef,
        const double* c1, const double* power, const double t_lo,
        const double per_day, const long long n, const long long size,
        const int chunk, double* pw, double* fw) {
      // the day sums of DailyGram's ``days`` kernel for template
      // blockIdx.y of a batch, its stationary time from the basis
      // coefficients' differences coef[16 y ...] as in pn_source
      const unsigned all = 0xffffffffu;
      const double two_pi = 6.283185307179586;
      long long y = blockIdx.y;
      coef += 16 * y;
      pw += y * size;
      fw += y * size;
      int lane = threadIdx.x & 31;
      long long warp = ((long long)blockDim.x * blockIdx.x + threadIdx.x)
          >> 5;
      long long start = warp * 32 * chunk;
      long long stop = min(start + 32LL * chunk, n), current = -1;
      double p = 0, m = 0;
      for (long long base = start; base < stop; base += 32) {
        long long j = base + lane, tau = -1;
        double v = 0, vf = 0;
        if (j < stop) {
          double x = cbrt_f[j], lf = log_f[j], fi = f[j];
          double xi = 1.0 / x;
          double q = xi * xi * xi * xi * xi;
          double dpsi = 0;
          for (int k = 0; k < 8; k++) {
            double e = (k - 5) / 3.0;
            dpsi += (coef[k] * e + coef[8 + k] * (e * lf + 1)) * q;
            q *= x;
          }
          double t = t0[j] + (dpsi / fi - c1[y]) / two_pi;
          tau = (long long)floor((t - t_lo) * per_day);
          v = power[y] * ((double)b0[j].real() * b0[j].real()
                          + (double)b0[j].imag() * b0[j].imag());
          vf = v * fi;
        }
        long long first = __shfl_sync(all, tau, 0);
        if (__all_sync(all, tau == first || j >= stop)) {
          for (int h = 16; h > 0; h >>= 1) {
            v += __shfl_down_sync(all, v, h);
            vf += __shfl_down_sync(all, vf, h);
          }
          if (lane == 0) {
            if (first != current) {
              if (current >= 0 && current < size) {
                atomicAdd(pw + current, p);
                atomicAdd(fw + current, m);
              }
              current = first; p = 0; m = 0;
            }
            p += v;
            m += vf;
          }
        } else if (j < stop && tau >= 0 && tau < size) {
          atomicAdd(pw + tau, v);
          atomicAdd(fw + tau, vf);
        }
      }
      if (lane == 0 && current >= 0 && current < size) {
        atomicAdd(pw + current, p);
        atomicAdd(fw + current, m);
      }
    }
    """

    _EDGES_KERNEL = r"""
    __device__ double pn_time(const double* t0, const double* f,
                              const double* cbrt_f, const double* log_f,
                              const double* coef, double c1, long long j) {
      const double two_pi = 6.283185307179586;
      double x = cbrt_f[j], lf = log_f[j];
      double xi = 1.0 / x;
      double q = xi * xi * xi * xi * xi;
      double dpsi = 0;
      for (int k = 0; k < 8; k++) {
        double e = (k - 5) / 3.0;
        dpsi += (coef[k] * e + coef[8 + k] * (e * lf + 1)) * q;
        q *= x;
      }
      return t0[j] + (dpsi / f[j] - c1) / two_pi;
    }

    extern "C" __global__ void pn_day_edges(
        const double* t0, const double* f, const double* cbrt_f,
        const double* log_f, const double* coef, const double* c1,
        const double t_lo, const double day, const long long n,
        const long long size, long long* edges) {
      // edges[y, tau]: the first bin whose stationary time is at least the
      // start of day tau, by bisection, for template blockIdx.y
      long long tau = (long long)blockDim.x * blockIdx.x + threadIdx.x;
      long long y = blockIdx.y;
      if (tau > size) return;
      const double* c = coef + 16 * y;
      double edge = t_lo + tau * day;
      long long lo = 0, hi = n;
      while (lo < hi) {
        long long mid = (lo + hi) >> 1;
        if (pn_time(t0, f, cbrt_f, log_f, c, c1[y], mid) < edge) lo = mid + 1;
        else hi = mid;
      }
      edges[y * (size + 1) + tau] = lo;
    }
    """

    def _cumulative(self):
        """The reference's power and power times frequency, summed over the
        bins below each bin."""
        if not hasattr(self, '_cum'):
            xp = self.xp
            p = xp.abs(self.b0.astype(xp.complex128)) ** 2
            zero = xp.zeros(1)
            self._cum = (xp.concatenate([zero, xp.cumsum(p)]),
                         xp.concatenate([zero, xp.cumsum(p * self._f)]))
        return self._cum

    def _monotone(self, coef, c1, t_begin, t_end, samples=8192):
        """Whether each template's stationary times, on an even subset of
        the bins, lie below ``t_begin``, then rise through ``[t_begin,
        t_end)``, then stay at or above ``t_end``: the day edges inside are
        then those a bisection finds."""
        xp = self.xp
        if not hasattr(self, '_dpsi_basis'):
            idx = numpy.unique(numpy.linspace(0, self.size - 1, samples
                                              ).astype(int))
            f = numpy.asarray(self.grid.frequencies, dtype=float)[idx]
            lf = numpy.log(f)
            e = (numpy.arange(8) - 5) / 3.0
            p = f[None] ** e[:, None]
            basis = numpy.concatenate([e[:, None] * p,
                                       (e[:, None] * lf + 1) * p]) / f
            self._dpsi_basis = (xp.asarray(idx), xp.asarray(basis))
        idx, basis = self._dpsi_basis
        t = self.t0[idx][None] + (xp.asarray(coef) @ basis
                                  - xp.asarray(c1)[:, None]) / (2 * numpy.pi)
        side = (t >= t_begin).astype(xp.int8) + (t >= t_end)
        inside = (side[:, 1:] == 1) & (side[:, :-1] == 1)
        ok = xp.all(xp.diff(side, axis=1) >= 0, axis=1)
        ok &= xp.all((xp.diff(t, axis=1) >= 0) | ~inside, axis=1)
        return _host(ok)

    def day_sums(self, thetas, daily_gram, reach=31 * 86400.0, chunk=16,
                 threads=256):
        """Each template's day sums ``pw``, ``fw`` ``(M, days)`` of
        ``daily_gram``'s grid, for parameter rows ``thetas`` ``(M, 4)``
        (mchirp, q, spin1z, spin2z), on cupy: the stationary time in closed
        form, no waveform stored.

        Only the days a Gram matrix at arrival times within ``reach`` (s)
        of zero reads are kept, those within ``reach`` of the data window;
        the rest are zero."""
        xp = self.xp
        thetas = numpy.atleast_2d(numpy.asarray(thetas, dtype=float))
        coef, c1, power = [], [], []
        fl = self.f_lower
        lx, e = numpy.log(fl), (numpy.arange(8) - 5) / 3.0
        p = fl ** e
        for theta in thetas:
            A, B, la = taylorf2_basis_coefficients(*theta)
            dA, dB = A - self._A0, B - self._B0
            coef.append(numpy.concatenate([dA, dB]))
            c1.append(numpy.sum((dA * e + dB * (e * lx + 1)) * p) / fl)
            power.append(numpy.exp(2 * (la - self._la0)))
        coef, c1 = numpy.array(coef), numpy.array(c1)
        power = xp.asarray(numpy.array(power))
        M, size = len(thetas), daily_gram.size
        if not hasattr(self, '_edges_kernel'):
            self._edges_kernel = xp.RawModule(
                code=self._EDGES_KERNEL).get_function('pn_day_edges')
        # a chirp's days are runs of bins: each day's sums are differences
        # of the cumulative sums at its first bin
        edges = xp.empty((M, size + 1), dtype=xp.int64)
        self._edges_kernel(
            ((size + threads) // threads, M), (threads,), (
                self.t0, self._f, self._cbrt, self._logf, xp.asarray(coef),
                xp.asarray(c1), numpy.float64(daily_gram.t_lo),
                numpy.float64(daily_gram.day), numpy.int64(self.size),
                numpy.int64(size), edges))
        P, Q = self._cumulative()
        pw = (P[edges[:, 1:]] - P[edges[:, :-1]]) * power[:, None]
        fw = (Q[edges[:, 1:]] - Q[edges[:, :-1]]) * power[:, None]
        # the days that matter: within reach of the window
        live = numpy.nonzero(_host(daily_gram._w) > 0)[0]
        pad = int(numpy.ceil(reach / daily_gram.day)) + 1
        first = max(0, live[0] - pad)
        stop = min(size, live[-1] + pad + 1)
        pw[:, :first] = 0
        pw[:, stop:] = 0
        fw[:, :first] = 0
        fw[:, stop:] = 0
        rising = self._monotone(coef, c1,
                                daily_gram.t_lo + first * daily_gram.day,
                                daily_gram.t_lo + stop * daily_gram.day)
        if not rising.all():
            # bins out of day order: sum them one by one
            rows = numpy.nonzero(~rising)[0]
            if not hasattr(self, '_days_kernel'):
                real = 'float' if self.dtype == numpy.complex64 \
                    else 'double'
                self._days_kernel = xp.RawModule(
                    code=self._DAYS_KERNEL.replace('{real}', real)
                ).get_function('pn_days')
            pr = xp.zeros((len(rows), size), dtype=xp.float64)
            fr = xp.zeros((len(rows), size), dtype=xp.float64)
            warps = (self.size + 32 * chunk - 1) // (32 * chunk)
            self._days_kernel(
                ((32 * warps + threads - 1) // threads, len(rows)),
                (threads,), (
                    self.b0, self.t0, self._f, self._cbrt, self._logf,
                    xp.asarray(coef[rows]), xp.asarray(c1[rows]),
                    xp.ascontiguousarray(power[xp.asarray(rows)]),
                    numpy.float64(daily_gram.t_lo),
                    numpy.float64(1 / daily_gram.day),
                    numpy.int64(self.size), numpy.int64(size),
                    numpy.int32(chunk), pr, fr))
            pw[xp.asarray(rows)] = pr
            fw[xp.asarray(rows)] = fr
        return pw, fw
