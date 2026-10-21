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
"""Correlations of many templates at once by one non-uniform FFT.

For :class:`pycbc.tdi.pn_coefficient_surrogate.PNCoefficientSurrogate` the
correlation of basis ``a`` with the SSB-time data at arrival time ``Delta``,

    z_a(theta, Delta) = sum_g conj(y_a(g; theta, Delta)) X(g),

depends on the template only through the changes ``dC`` of its TaylorF2
basis coefficients, which enter linearly in the exponent: with the basis
functions ``psi_m(f)`` (``f^(n/3)``, ``f^(n/3) ln f``),

    z_a = A sum_k sum_g S_ka(g) exp(i [dC . (psi(g) - k Omega psi'(g) / 2 pi)
                                     + Delta' (2 pi (f_g - f_l) - k Omega)]),
    S_ka(g) = conj(b0(g)) exp(-i k Omega t0(g)) sum_c conj(rho_kac(g)) X_c(g),

the yearly harmonic ``k`` entering through the arrival time ``t = Psi' / 2
pi``, also linear in ``dC``. The phase and time pinned at ``f_l`` shift the
arrival time ``Delta'`` and give each template a constant phase. Over a region
of templates the ``dC`` span, beyond the constant and linear parts in ``f``
that phase and time absorb, about three directions: on the two leading ones
and arrival time the region is one type-1 transform onto a uniform grid
(:class:`PhaseGrid`), the third entering through its Taylor series, and any
template of the region, or any node of the grid, is read off it, at a cost
that does not grow with their number. What a template keeps outside the
directions, and what its sky's response keeps outside the region's sky
basis (:class:`SkyBasis`), are measured for each template.
"""

import numpy

from pycbc.tdi.ssb_yearly_harmonics import ORBITAL_ANGULAR_FREQUENCY
from pycbc.waveform.taylorf2_jets import taylorf2_basis_coefficients

_EXPONENTS = (numpy.arange(8) - 5) / 3.0


def _basis(f):
    """``psi_m(f)`` and ``psi_m'(f)``, shape ``(16, ...)``: the ``A`` terms
    ``f^(n/3)`` then the ``B`` terms ``f^(n/3) ln f``."""
    f = numpy.asarray(f, dtype=float)
    lf = numpy.log(f)
    psi, dpsi = [], []
    for m in _EXPONENTS:
        p = f ** m
        psi.append(p)
        dpsi.append(m * p / f)
    for m in _EXPONENTS:
        p = f ** m
        psi.append(p * lf)
        dpsi.append(p * (m * lf + 1) / f)
    return numpy.array(psi), numpy.array(dpsi)


def source_power_weights(family):
    """The source's power each of the response's coarse frequencies
    carries: each fine bin's power shared between the two coarse
    frequencies it interpolates, by its interpolation weights. As the
    weights sum to one, a fine bin's response error squared is at most the
    weighted sum of its coarse frequencies' (by convexity), so the error of
    the coarse harmonics in this weighting bounds the template's."""
    power = numpy.abs(_host(family.b0)) ** 2
    i = _host(family._index)
    w = _host(family._weight).astype(float)
    n = len(family.coarse_frequencies)
    return (numpy.bincount(i, power * (1 - w), minlength=n)
            + numpy.bincount(i + 1, power * w, minlength=n))


class SkyBasis(object):
    """An orthonormal basis of the response harmonics over a region of the
    sky: the leading singular vectors of the harmonics at ``points`` (sky
    offsets ``(M, 2)``), weighted by the source's power at each coarse
    frequency, as many as bring every point's relative error below a third
    of ``tolerance``. The response of any point is then its projection,
    whose error :meth:`project` gives exactly.

    It behaves as the list of basis vectors, each of the harmonics' shape
    ``(nk, 2, nchannels, ncoarse)``."""
    def __init__(self, family, points, tolerance=1e-4, weight=None,
                 chunk=64, max_points=300, max_rank=64):
        xp = self.xp = family.xp
        self.family = family
        points = numpy.atleast_2d(numpy.asarray(points, dtype=float))
        if len(points) > max_points:
            pick = numpy.random.default_rng(0).choice(len(points),
                                                      max_points,
                                                      replace=False)
            points = points[numpy.sort(pick)]
        if weight is None:
            weight = source_power_weights(family)
        self._root = xp.sqrt(xp.asarray(weight))
        A = self._rows(points, chunk)                  # (M, D) weighted
        K = A @ xp.conj(A.T)
        values, vectors = xp.linalg.eigh(K)
        order = xp.argsort(values)[::-1]
        values, vectors = values[order], vectors[:, order]
        keep = int(min(max_rank, xp.sum(values > values[0] * 1e-14)))
        # the points' coordinates on the leading vectors, A V / sqrt(l)
        C = vectors[:, :keep] * xp.sqrt(values[:keep])[None]
        norm2 = xp.sum(xp.abs(A) ** 2, axis=1)
        left = norm2[:, None] - xp.cumsum(xp.abs(C) ** 2, axis=1)
        err = _host(xp.sqrt(xp.maximum(left, 0) / norm2[:, None]))
        # a third of the tolerance at the training points, so that points
        # between them meet it too
        good = numpy.nonzero(err.max(axis=0) <= tolerance / 3)[0]
        rank = int(good[0]) + 1 if len(good) else keep
        self.rank = rank
        self.tolerance = float(tolerance)
        self._U = (xp.conj(vectors[:, :rank].T) @ A) / xp.sqrt(
            values[:rank])[:, None]                    # (rank, D)
        del A
        nk = len(family.orders)
        self._shape = (nk,) + tuple(family.R0.shape[1:])
        # coarse frequencies no fine bin uses have no weight: zero there
        inverse = xp.where(self._root > 0, 1 / xp.maximum(self._root,
                                                          1e-300), 0)
        self._vectors = [(u.reshape(self._shape) * inverse).astype(
            family.R0.dtype) for u in self._U]

    def _rows(self, points, chunk=64):
        xp = self.xp
        rows = []
        for i in range(0, len(points), chunk):
            rho = self.family.responses(points[i:i + chunk]).astype(
                xp.complex128)
            rows.append((rho * self._root).reshape(len(rho), -1))
        return xp.concatenate(rows)

    def __len__(self):
        return self.rank

    def __iter__(self):
        return iter(self._vectors)

    def __getitem__(self, i):
        return self._vectors[i]

    def project(self, points, chunk=64):
        """The coordinates ``(M, rank)`` of each point's response on the
        basis and their exact relative errors ``(M,)``."""
        xp = self.xp
        coords, errors = [], []
        for i in range(0, len(points), chunk):
            A = self._rows(points[i:i + chunk])
            C = A @ xp.conj(self._U.T)
            # the residual itself: its norm from the norms' difference
            # would lose it below 1e-8 of the response
            res = A - C @ self._U
            coords.append(_host(C))
            errors.append(_host(xp.sqrt(xp.sum(xp.abs(res) ** 2, axis=1)
                                        / xp.sum(xp.abs(A) ** 2, axis=1))))
        return numpy.concatenate(coords), numpy.concatenate(errors)


class PhaseNUFFTCorrelation(object):
    """Correlations of a batch of templates of a PN-coefficient family with
    the data, at many arrival times, by one type-3 NUFFT.

    Parameters
    ----------
    family : pycbc.tdi.pn_coefficient_surrogate.PNCoefficientSurrogate
        The family; its sky offsets are fixed at ``sky``.
    data : array, shape ``(nchannels, N)``
        The SSB-time data spectrum.
    sky : tuple
        The two sky offsets of the strengths :meth:`direct` uses, shape
        units.
    eps : float
        Requested relative accuracy of the transform.
    sky_points : array, shape ``(M, 2)``, optional
        Sky offsets spanning a region: the response harmonics are
        represented on the leading singular vectors of their values at
        these points (see :class:`SkyBasis`), each a set of node
        strengths, and :meth:`sky_error` gives any point's exact error. By
        default the response at ``sky`` alone.
    sky_tolerance : float
        The largest relative error of the sky basis at ``sky_points``.
    """
    def __init__(self, family, data, sky=(0.0, 0.0), eps=1e-6,
                 sky_points=None, sky_tolerance=1e-4):
        self.family = family
        xp = self.xp = family.xp
        self.eps = float(eps)
        grid = family.grid
        f = numpy.asarray(grid.frequencies, dtype=float)
        self.f_lower = family.f_lower
        self.orders = family.orders
        self.sky = numpy.asarray(sky, dtype=float)
        self._data = xp.asarray(data)
        self._power = source_power_weights(family)
        if sky_points is None:
            sky_points = self.sky[None]
        self.sky_basis = SkyBasis(family, sky_points, sky_tolerance,
                                  self._power)
        self._rho = list(self.sky_basis)
        self.sky_sets = len(self._rho)
        # (harmonic, basis, bin) at the sky offsets ``sky``
        self._rho_at = family._response(self.sky)
        self.strengths = xp.stack([self.strength(i, -1) for i in range(
            len(self.orders))])
        self.f = f
        self._logf = xp.asarray(numpy.log(f))
        # weights of the batch geometry: the template power at each
        # frequency, on a coarse grid
        step = max(1, len(f) // 4000)
        coarse = slice(0, len(f), step)
        power = numpy.abs(family.b0[coarse].get() if hasattr(
            family.b0, 'get') else family.b0[coarse]) ** 2
        self._fc = f[coarse]
        self._wc = power / power.sum()
        self._psic, _ = _basis(self._fc)
        self._A0 = family._A0
        self._B0 = family._B0
        self._la0 = family._la0

    def strength(self, harmonic, vector):
        """Node strengths ``(2, N)`` of harmonic index ``harmonic`` for sky
        basis vector ``vector``; ``-1`` for the response at ``sky``."""
        if not hasattr(self, '_rho'):
            return self.strengths[harmonic]
        xp, fam = self.xp, self.family
        k = self.orders[harmonic]
        phase = xp.exp(-1j * k * ORBITAL_ANGULAR_FREQUENCY
                       * xp.asarray(fam.t0))
        rho = (self._rho_at if vector < 0
               else self._rho[vector])[harmonic]
        weighted = xp.conj(fam.fine(rho)) * self._data[None]
        return (xp.conj(xp.asarray(fam.b0)) * phase
                * weighted.sum(axis=1)).astype(xp.complex64)

    def sky_weights(self, offsets):
        """The weights of the strength sets for each template's sky offsets,
        shape ``(P, M)``: the conjugates of its response's coordinates on
        the sky basis (the strengths carry the conjugate response)."""
        offsets = numpy.atleast_2d(numpy.asarray(offsets, dtype=float))
        if getattr(self, 'sky_basis', None) is None:
            return numpy.ones((1, len(offsets)))
        c, _ = self.sky_basis.project(offsets)
        return numpy.conj(c).T

    def sky_error(self, offsets):
        """Each template's exact relative error, in the template's norm, of
        the response harmonics its sky weights give (see
        :class:`SkyBasis`)."""
        offsets = numpy.atleast_2d(numpy.asarray(offsets, dtype=float))
        if getattr(self, 'sky_basis', None) is None:
            return numpy.zeros(len(offsets))
        return self.sky_basis.project(offsets)[1]

    @classmethod
    def from_arrays(cls, strengths, frequencies, f_lower, orders, A0, B0,
                    la0, weight_frequencies, weights, xp=numpy, eps=1e-6):
        """An instance from its arrays: the node strengths ``(harmonic,
        basis, bin)``, the grid's frequencies, the reference's basis
        coefficients and the coarse weights of the batch geometry."""
        self = cls.__new__(cls)
        self.xp, self.eps = xp, float(eps)
        self.strengths = xp.asarray(strengths, dtype=_complex(xp))
        self.f = numpy.asarray(frequencies, dtype=float)
        self.f_lower = float(f_lower)
        self.orders = numpy.asarray(orders)
        self._logf = xp.asarray(numpy.log(self.f))
        self._fc = numpy.asarray(weight_frequencies, dtype=float)
        self._wc = numpy.asarray(weights, dtype=float)
        self._psic, _ = _basis(self._fc)
        self._A0, self._B0, self._la0 = A0, B0, la0
        self.sky_sets = 1
        self.sky = numpy.zeros(2)
        self.sky_basis = None
        return self

    def direct(self, thetas, delays):
        """The correlations of :meth:`correlate_grid` term by term, without
        a transform or a projection on directions: for tests."""
        xp = self.xp
        thetas = numpy.atleast_2d(numpy.asarray(thetas, dtype=float))
        dC, amp = self._coefficients(thetas)
        psi_l, dpsi_l = _basis([self.f_lower])
        out = xp.zeros((2, len(thetas), len(delays)), dtype=complex)
        df = xp.asarray(2 * numpy.pi * (self.f - self.f_lower))
        for j in range(len(thetas)):
            pin0, pin1 = dC[j] @ psi_l[:, 0], dC[j] @ dpsi_l[:, 0]
            phase, slope = self._projected_basis(dC[j][:, None])
            phase, slope = phase[0], slope[0]
            for i, k in enumerate(self.orders):
                w = k * ORBITAL_ANGULAR_FREQUENCY
                arg = phase - w / (2 * numpy.pi) * slope
                for n, delay in enumerate(delays):
                    total = arg + (delay - pin1 / (2 * numpy.pi)) * (df - w)
                    out[:, j, n] += (self.strengths[i] * xp.exp(1j * total)
                                     ).sum(axis=-1)
            out[:, j] *= amp[j] * numpy.exp(-1j * pin0)
        return out

    def _projected_basis(self, D):
        """``D^T psi(f)`` and ``D^T psi'(f)`` on the grid, shape ``(ndir,
        N)``, a basis function at a time."""
        xp = self.xp
        lf = self._logf
        inv_f = xp.exp(-lf)
        x = xp.zeros((D.shape[1], len(lf)))
        dx = xp.zeros_like(x)
        for m in range(16):
            n = _EXPONENTS[m % 8]
            if not numpy.any(D[m]):
                continue
            p = xp.exp(n * lf)
            if m < 8:
                psi, dpsi = p, n * p * inv_f
            else:
                psi, dpsi = p * lf, p * (n * lf + 1) * inv_f
            w = xp.asarray(D[m])[:, None]
            x += w * psi[None]
            dx += w * dpsi[None]
        return x, dx

    def _coefficients(self, thetas):
        """``dC`` (M, 16) and the amplitude ratios."""
        dC, amp = [], []
        for theta in thetas:
            A, B, la = taylorf2_basis_coefficients(*theta)
            dC.append(numpy.concatenate([A - self._A0, B - self._B0]))
            amp.append(numpy.exp(la - self._la0))
        return numpy.array(dC), numpy.array(amp)

    def frame(self, thetas, nall):
        """The phase frame of a region given by sample points ``thetas``:
        its leading ``nall`` directions and what :meth:`project` needs.

        The weighted phases of the points, less their constant and linear
        parts, are decomposed by singular values on a coarse grid; each
        direction is a combination of the points' own coefficient changes,
        so no ill-conditioned inverse of the basis enters the nodes."""
        dC, _ = self._coefficients(numpy.atleast_2d(thetas))
        fc, w, psi = self._fc, self._wc, self._psic
        sw = numpy.sqrt(w)
        X = numpy.vstack([numpy.ones_like(fc), fc - self.f_lower]).T
        coef = numpy.linalg.lstsq(X * sw[:, None], (psi * sw).T,
                                  rcond=None)[0]
        weighted = (psi - (X @ coef).T) * sw              # (16, nc)
        # the singular values of dC @ weighted through the QR factors of the
        # basis: only (M, 16) and (nc, 16) matrices, whatever M
        Q, R = numpy.linalg.qr(weighted.T)                # (nc, 16), (16, 16)
        U, sig, vs = numpy.linalg.svd(dC @ R.T, full_matrices=False)
        vt = vs @ Q.T
        nall = int(min(nall, numpy.sum(sig > sig[0] * 1e-12)))
        directions = dC.T @ U[:, :nall] / sig[:nall]       # (16, nall)
        # dC @ weighted @ vt.T is dC @ R.T @ vs.T: (M, 16) products only
        return dict(coef=coef, weighted=weighted, vt=vt[:nall],
                    root=R.T, vs=vs[:nall], directions=directions,
                    line_dir=directions.T @ coef.T)

    def project(self, thetas, frame):
        """Each template's phase coordinates ``s`` (M, nall), the arrival-time
        shift and the constant factor, in ``frame``."""
        thetas = numpy.atleast_2d(numpy.asarray(thetas, dtype=float))
        dC, amp = self._coefficients(thetas)
        s = (dC @ frame['root']) @ frame['vs'].T
        remainder = dC - s @ frame['directions'].T
        line = remainder @ frame['coef'].T
        psi_l, dpsi_l = _basis([self.f_lower])
        pin0, pin1 = dC @ psi_l[:, 0], dC @ dpsi_l[:, 0]
        line_dir = frame['line_dir']
        shift = (line[:, 1] + s @ line_dir[:, 1] - pin1) / (2 * numpy.pi)
        const = numpy.exp(1j * (line[:, 0] + s @ line_dir[:, 0] - pin0)) \
            * amp
        return s, shift, const

    def phase_residual(self, thetas, frame):
        """The root-mean-square, weighted by the template power, of the
        phase each template keeps outside ``frame``'s directions, less its
        constant and linear parts, rad: by Cauchy-Schwarz it bounds the
        power-weighted mean phase error the frame makes."""
        thetas = numpy.atleast_2d(numpy.asarray(thetas, dtype=float))
        dC, _ = self._coefficients(thetas)
        y = dC @ frame['root']
        r = y - (y @ frame['vs'].T) @ frame['vs']
        return numpy.sqrt(numpy.sum(r ** 2, axis=1))

    def time_offsets(self, thetas):
        """Each template's arrival-time offset from the reference, the
        power-weighted mean over frequency of ``t_theta(f) - t0(f)``, s."""
        thetas = numpy.atleast_2d(numpy.asarray(thetas, dtype=float))
        dC, _ = self._coefficients(thetas)
        _, dpsi = _basis(self._fc)
        _, dpsi_l = _basis([self.f_lower])
        dt = (dC @ dpsi - (dC @ dpsi_l[:, 0])[:, None]) / (2 * numpy.pi)
        return dt @ self._wc

    def nodes(self, frame):
        """The transform's node coordinates in ``frame``, over all
        harmonics: one row per direction, then the arrival-time row."""
        xp = self.xp
        L = xp.asarray(frame['line_dir'])
        df_lower = xp.asarray(self.f - self.f_lower)
        x_phase, x_slope = self._projected_basis(frame['directions'])
        x_phase = x_phase - L[:, :1] - L[:, 1:] * df_lower[None]
        x_slope = x_slope - L[:, 1:]
        return x_phase, x_slope, 2 * numpy.pi * df_lower

    def node_row(self, parts, d, harmonic):
        """Row ``d`` of the nodes of one harmonic from :meth:`nodes`'
        parts: a phase direction, or the arrival time for ``d = -1``."""
        x_phase, x_slope, omega = parts
        w = harmonic * ORBITAL_ANGULAR_FREQUENCY
        if d < 0:
            return omega - w
        return x_phase[d] - w / (2 * numpy.pi) * x_slope[d]


def _complex(xp):
    """Single precision on a GPU, double on the CPU."""
    return numpy.complex128 if xp is numpy else xp.complex64


def _real(xp):
    return numpy.float64 if xp is numpy else xp.float32


def _host(array):
    return array.get() if hasattr(array, 'get') else numpy.asarray(array)
