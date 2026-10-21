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

import math

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

    def correlate_grid(self, thetas, delays, **kwargs):
        """``z`` of shape ``(2, M, L)`` for parameter rows ``thetas`` ``(M,
        4)`` (mchirp, q, spin1z, spin2z) and arrival times ``delays``
        ``(L,)``, s, through a :class:`PhaseGrid` over the batch itself: one
        type-1 transform whose cost does not grow with the number of
        templates, then interpolation to each."""
        delays = numpy.asarray(delays, dtype=float)
        grid = PhaseGrid(self, thetas, (delays.min(), delays.max()),
                         **kwargs)
        return grid.evaluate(thetas, delays)

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


class PhaseGrid(object):
    """The correlations of a PN-coefficient family over a region, on a
    uniform grid of two phase coordinates and arrival time, from one type-1
    transform per Taylor order and basis; any template of the region is
    then an interpolation.

    The correlation is band-limited in each coordinate by the extent ``X``
    of the nodes; the grid steps are ``pi / (oversample X)`` and the
    Kaiser-windowed sinc interpolation uses ``half_width`` points either
    side. The third phase direction enters through its Taylor series about
    the region's middle, with as many orders as bring the largest omitted
    one below ``taylor_tolerance``.

    Parameters
    ----------
    correlator : PhaseNUFFTCorrelation
        The family's data-side strengths and basis.
    region : array, shape ``(M, 4)``
        Points (mchirp, q, spin1z, spin2z) spanning the region.
    delay_range : tuple
        The arrival times to cover, s.
    time_range : tuple, optional
        The grid times ``delay + shift`` to cover, s, in place of those
        ``delay_range`` gives every template of the region.
    phase_tolerance : float
        The largest power-weighted rms phase, rad, a template may keep
        outside the grid's directions (see
        :meth:`PhaseNUFFTCorrelation.phase_residual`); :meth:`inside` is
        false beyond it.
    """
    def __init__(self, correlator, region, delay_range, ndir=2,
                 taylor_tolerance=1e-4, oversample=(4.0, 4.0, 2.0),
                 half_width=(6, 6, 12), beta=(14.0, 14.0, 18.0),
                 upsampfac=2.0, nufft_options=None, margin=0.05,
                 memory_limit=2.2e9, phase_tolerance=1e-4,
                 time_range=None):
        import logging
        import time
        xp = self.xp = correlator.xp
        cplx, real = _complex(xp), _real(xp)
        self.correlator = correlator
        region = numpy.atleast_2d(numpy.asarray(region, dtype=float))
        ndir = max(1, min(ndir, len(region)))
        frame = correlator.frame(region, ndir + 1)
        self.frame = frame
        ndir = min(ndir, frame['directions'].shape[1])
        self.ndir = ndir
        self.phase_tolerance = float(phase_tolerance)
        self.phase_residual = float(correlator.phase_residual(
            region, frame).max())
        s, shift, _ = correlator.project(region, frame)
        parts = correlator.nodes(frame)
        orders = correlator.orders
        rows = list(range(ndir)) + [-1]

        def extent(d):
            lo_, hi_ = numpy.inf, -numpy.inf
            for k in orders:
                r = correlator.node_row(parts, d, k)
                lo_, hi_ = min(lo_, float(r.min())), max(hi_, float(r.max()))
            return lo_, hi_
        # Taylor variable: the third direction about the region's middle
        self.terms, self.middle, self.s3_half = 1, 0.0, numpy.inf
        third = s.shape[1] > ndir
        if third:
            self.middle = 0.5 * (s[:, ndir].max() + s[:, ndir].min())
            self.s3_half = numpy.abs(s[:, ndir] - self.middle).max() * (
                1 + margin)
            x3lo, x3hi = extent(ndir)
            reach = max(abs(x3lo), abs(x3hi)) * float(
                numpy.abs(s[:, ndir] - self.middle).max()) * (1 + margin)
            # the templates' spacing along it, as along the grid's axes
            self.s3_step = numpy.pi / (oversample[0] * max(
                0.5 * (x3hi - x3lo), 1e-300))
            term = 1.0
            while term > taylor_tolerance:
                term *= reach / self.terms
                self.terms += 1
        if time_range is None:
            time_range = (delay_range[0] + shift.min(),
                          delay_range[1] + shift.max())
        lo = [s[:, d].min() for d in range(ndir)] + [time_range[0]]
        hi = [s[:, d].max() for d in range(ndir)] + [time_range[1]]
        self.half_width, self.beta = half_width, beta
        self.centre, self.step, self.start, self.size = [], [], [], []
        for d, row in enumerate(rows):
            xlo, xhi = extent(row)
            xc = 0.5 * (xlo + xhi)
            X = max(0.5 * (xhi - xlo), 1e-300)
            h = numpy.pi / (oversample[d] * X)
            pad = margin * (hi[d] - lo[d])
            n = int(numpy.ceil((hi[d] - lo[d] + 2 * pad) / h)) \
                + 2 * half_width[d] + 2
            n += n % 2
            mid = 0.5 * (hi[d] + lo[d])
            self.centre.append(xc)
            self.step.append(h)
            self.start.append(mid - (n // 2) * h)
            self.size.append(n)
        npoly = correlator.sky_sets
        item = numpy.dtype(cplx).itemsize
        need = 2 * self.terms * npoly * numpy.prod(self.size) * item * (
            1 + upsampfac ** len(self.size) / (2 * self.terms * npoly))
        logging.info("PhaseGrid: grid %s, %d Taylor terms, about %.2f GB",
                     self.size, self.terms, need / 1e9)
        if xp is not numpy:
            # what the device can hold now, with room for the transform
            xp.get_default_memory_pool().free_all_blocks()
            free = xp.cuda.runtime.memGetInfo()[0]
            memory_limit = min(memory_limit, free - 4e8)
        if need > memory_limit:
            err = MemoryError(
                "the region needs a grid of {:.2f} GB ({} Taylor terms, "
                "grid {}); split it into smaller regions".format(
                    need / 1e9, self.terms, self.size))
            err.need = need
            raise err
        # the grid first, while the device memory is free: an array
        # allocated after the transforms' work space may land in host-shared
        # memory, where every read crosses the bus
        shape = tuple(self.size) if len(self.size) == 3 else (
            self.size[0], 1, self.size[1])
        self.grid = xp.empty((self.terms * npoly * 2,) + shape, dtype=cplx)
        # scaled nodes and strengths, a harmonic at a time, in single
        # precision: F(mid + m h) = sum c exp(i mid (x - xc)) exp(i m h (x -
        # xc)), times exp(i (mid + m h) xc), applied in evaluate
        N = len(correlator.f)
        total = N * len(orders)
        scaled = [xp.empty(total, dtype=real) for _ in rows]
        ix3 = xp.empty(total, dtype=cplx) if third else None
        centring = []
        for i, k in enumerate(orders):
            cut = slice(i * N, (i + 1) * N)
            phase = 0.0
            for d, row in enumerate(rows):
                x = correlator.node_row(parts, row, k)
                scaled[d][cut] = ((x - self.centre[d]) * self.step[d])
                phase = phase + 0.5 * (hi[d] + lo[d]) * (x - self.centre[d])
            if third:
                x3 = correlator.node_row(parts, ndir, k)
                phase = phase + self.middle * x3
                ix3[cut] = 1j * x3
            centring.append(xp.exp(1j * phase).astype(cplx))
        parts = None
        if xp is numpy:
            import finufft as lib
            options = dict(nufft_options or {})
        else:
            import cufinufft as lib
            options = dict(gpu_method=1, gpu_sort=0, gpu_maxbatchsize=1)
            options.update(nufft_options or {})
        blocked = _BlockedNodes(scaled, len(orders), N, self.size,
                                correlator.eps, xp)
        del scaled
        blocked.plan(lib, options, cplx, upsampfac, correlator.eps)
        self.blocks = blocked.width
        logging.info("PhaseGrid: %d bins a block, moments to order %d",
                     blocked.width, blocked.order)
        tick = time.time()
        base = xp.empty((2, total), dtype=cplx)
        for p in range(npoly):
            for i in range(len(orders)):
                cut = slice(i * N, (i + 1) * N)
                base[:, cut] = correlator.strength(i, p) * centring[i][None]
            current = base
            for q in range(self.terms):
                if q:
                    current = current * ix3[None] / q
                for a in range(2):
                    self.grid[(q * npoly + p) * 2 + a] = blocked.transform(
                        current[a]).reshape(shape)
        del current, base, ix3, centring, blocked
        if xp is not numpy:
            xp.cuda.Stream.null.synchronize()
            xp.get_default_memory_pool().free_all_blocks()
        grids = self.grid
        logging.info("PhaseGrid %s, %d vectors, %.2f s", self.size,
                     len(grids), time.time() - tick)

    def grid_twof(self, gram, noise_variance=1.0, s3_values=(0.0,),
                  sky=None, chunk=None):
        """The F-statistic ``2F = z^H G^-1 z`` at every node of the grid:
        each node of the two phase coordinates is a template, for each of
        ``s3_values`` of the third (offsets from the region's middle) and
        each sky position, at every arrival-time sample of the grid, with no
        interpolation.

        The constant phase and the amplitude a template carries cancel in
        ``2F``, the latter against the Gram matrix's ``A^2``; ``gram(T,
        sky)`` gives the reference's Gram matrix ``(g00, g11, g01)`` at grid
        times ``T`` and sky offsets ``sky``, which stand for the arrival
        times since the Gram matrix changes over a day far less than a
        template's time offset matters.

        Every (sky, third coordinate) pair weighs the grid's vectors
        (Taylor orders and sky basis vectors) differently; one matrix product
        per block of nodes gives them all.

        Parameters
        ----------
        sky : array, shape ``(S, 2)``
            Sky offsets; the correlator's ``sky`` when not given.

        Returns the largest 2F over arrival time, shape ``(S,
        len(s3_values), n1, n2)``, the arrival-time grid index where it is
        reached, and the nodes' phase coordinates ``(s1, s2)``."""
        xp = self.xp
        c = self.correlator
        npoly = c.sky_sets
        shape = self.grid.shape[1:]
        n1, n2, n3 = shape
        sky = numpy.atleast_2d(c.sky if sky is None else sky)
        T = self.start[-1] + numpy.arange(n3) * self.step[-1]
        G = []
        for d in sky:
            g00, g11, g01 = (xp.asarray(v) / noise_variance
                             for v in gram(T, d))
            det = xp.maximum(g00 * g11 - xp.abs(g01) ** 2, 1e-300)
            G.append((g00 / det, g11 / det, g01 / det))
        mono = c.sky_weights(sky)                          # (npoly, S)
        s3 = numpy.asarray(s3_values, dtype=float)
        taylor = s3[None, :] ** numpy.arange(self.terms)[:, None]
        # (S, n_s3, terms, npoly), the vectors' order in the grid
        weight = (mono.T[:, None, None, :] * taylor.T[None, :, :, None])
        combos = len(sky) * len(s3)
        weight = xp.asarray(weight.reshape(combos, -1) / noise_variance,
                            dtype=self.grid.dtype)
        vectors = self.grid.reshape(self.terms * npoly, 2, n1, n2 * n3)
        if chunk is None:
            # the products' output, two bases of every combination, within
            # about as much memory as a vector of the grid
            chunk = max(1, n1 // (2 * combos))
        best = xp.empty((combos, n1, n2))
        where = xp.empty((combos, n1, n2), dtype=xp.int64)
        for r0 in range(0, n1, chunk):
            rows = slice(r0, min(r0 + chunk, n1))
            nr = rows.stop - rows.start
            sub = vectors[:, :, rows].reshape(len(weight[0]), -1)
            z = (weight @ sub).reshape(combos, 2, nr, n2, n3)
            for i in range(combos):
                h00, h11, h01 = G[i // len(s3)]
                z0 = z[i, 0].astype(xp.complex128)
                z1 = z[i, 1].astype(xp.complex128)
                twof = (h11 * xp.abs(z0) ** 2 + h00 * xp.abs(z1) ** 2
                        - 2 * xp.real(xp.conj(z0) * h01 * z1))
                best[i, rows] = twof.max(axis=-1)
                where[i, rows] = twof.argmax(axis=-1)
            del z
        nodes = [self.start[d] + numpy.arange(shape[d]) * self.step[d]
                 for d in range(self.ndir)]
        out = (len(sky), len(s3), n1, n2)
        return best.reshape(out), where.reshape(out), nodes

    def time_limits(self):
        """The grid times, s, interpolation can reach."""
        hw = self.half_width[-1]
        return (self.start[-1] + hw * self.step[-1],
                self.start[-1] + (self.size[-1] - hw - 1) * self.step[-1])

    def inside(self, thetas, delay_range=None):
        """Whether each template's interpolation stays on the grid and its
        third coordinate within the Taylor series' reach, for arrival times
        in ``delay_range`` (not checked when not given)."""
        s, shift, _ = self.correlator.project(thetas, self.frame)
        ok = numpy.ones(len(shift), dtype=bool)
        hw = self.half_width
        for d in range(self.ndir):
            lo = self.start[d] + hw[d] * self.step[d]
            hi = self.start[d] + (self.size[d] - hw[d] - 1) * self.step[d]
            ok &= (s[:, d] >= lo) & (s[:, d] <= hi)
        if delay_range is not None:
            lo, hi = self.time_limits()
            ok &= (delay_range[0] + shift >= lo) & (
                delay_range[1] + shift <= hi)
        if self.terms > 1:
            ok &= numpy.abs(s[:, self.ndir] - self.middle) <= self.s3_half
        ok &= self.correlator.phase_residual(thetas, self.frame) \
            <= self.phase_tolerance
        return ok

    def evaluate(self, thetas, delays, sky=None, chunk=1024):
        """``z`` of shape ``(2, M, L)`` for templates ``thetas`` of the
        region at arrival times ``delays``, s: ``(L,)`` for all, or ``(M,
        L)``, each template its own; ``sky``, ``(M, 2)``, their sky offsets,
        by default the correlator's ``sky``.

        The vectors of the grid (Taylor orders, sky basis vectors, bases) are
        combined with each template's weights while interpolating along the
        phase coordinates, so the arrival-time interpolation works on the
        two bases only."""
        xp = self.xp
        s, shift, const = self.correlator.project(thetas, self.frame)
        delays = numpy.asarray(delays, dtype=float)
        if delays.ndim == 1:
            delays = numpy.broadcast_to(delays, (len(shift), len(delays)))
        nd = self.ndir
        hw, beta = self.half_width, self.beta
        M, nl = delays.shape
        npoly = self.correlator.sky_sets
        weight = numpy.ones((M, self.terms, npoly), dtype=complex)
        if self.terms > 1:
            s3 = s[:, nd] - self.middle
            weight = weight * (s3[:, None] ** numpy.arange(self.terms)
                               )[:, :, None]
        if npoly > 1 or getattr(self.correlator, 'sky_basis',
                                None) is not None:
            if sky is None:
                sky = numpy.broadcast_to(self.correlator.sky, (M, 2))
            weight = weight * self.correlator.sky_weights(sky).T[:, None, :]
        weight = weight.reshape(M, -1)                    # (M, terms npoly)
        out = xp.empty((2, M, nl), dtype=self.grid.dtype)
        for j0 in range(0, M, chunk):
            js = numpy.arange(j0, min(j0 + chunk, M))
            taps = []
            for d in range(nd):
                u = (s[js, d] - self.start[d]) / self.step[d]
                i0 = numpy.floor(u).astype(int) - hw[d] + 1
                offs = u[:, None] - (i0[:, None] + numpy.arange(2 * hw[d]))
                taps.append((i0, _kaiser_taps(offs, hw[d], beta[d])))
            if nd == 1:
                taps.append((numpy.zeros(len(js), int),
                             numpy.ones((len(js), 1))))
            (i1, w1), (i2, w2) = taps
            H = _phase_combine(self.grid, i1, w1, i2, w2, weight[js], xp)
            T = delays[js] + shift[js][:, None]
            vals = _time_interpolate(H, (T - self.start[-1]) / self.step[-1],
                                     hw[-1], beta[-1], xp)
            cphase = sum(s[js, d] * self.centre[d] for d in range(nd))
            cphase = cphase[:, None] + T * self.centre[-1]
            out[:, js] = vals * xp.asarray(numpy.exp(1j * cphase),
                                           dtype=self.grid.dtype)[None]
        return out * xp.asarray(const)[None, :, None]


class _BlockedNodes(object):
    """The type-1 transform of the PhaseGrid's nodes, ``sum_j c_j exp(i m .
    y_j)`` on the grid's modes ``m``, with the nodes of runs of ``width``
    consecutive frequency bins of each harmonic gathered at the run's
    middle node ``y_b``: with ``delta_j = y_j - y_b``,

        exp(i m . delta_j) = sum_alpha (i m)^alpha delta_j^alpha / alpha!,

    so the transform is ``sum_alpha m^alpha i^|alpha| / alpha!`` times the
    transform of the moments ``sum_{j in b} c_j delta_j^alpha`` at the run
    middles. With ``e = sum_d (n_d / 2) |delta_jd|``, the largest over the
    bins, the terms past order ``R`` change each bin's term by at most
    ``e^(R+1) exp(e) / (R+1)!`` of ``|c_j|``, kept below the transform's
    ``eps``; the run width is the one of least cost, one bin (the plain
    transform) when no run pays."""
    #: the cost of a moment's pass over the bins, in kernel points spread
    moment_cost = 1.0

    def __init__(self, scaled, nk, N, size, eps, xp, max_order=8,
                 widths=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)):
        self.xp, self.nk, self.N = xp, nk, N
        self.size = list(size)
        D = len(scaled)
        half = [n // 2 for n in size]
        kernel = (int(numpy.ceil(-numpy.log10(eps))) + 1) ** D
        total = nk * N
        best = (total * kernel, 1, 0)
        for width in widths[1:]:
            if width > N:
                break
            e = 0
            for d in range(D):
                y = self._runs(scaled[d], width)
                e = e + half[d] * xp.abs(y - y[:, width // 2][:, None])
            e = float(e.max())
            order = None
            for R in range(1, max_order + 1):
                if e ** (R + 1) * numpy.exp(e) / math.factorial(R + 1) \
                        <= eps:
                    order = R
                    break
            if order is None:
                continue
            terms = math.comb(order + D, D)
            cost = terms * (total / width * kernel
                            + self.moment_cost * total)
            if cost < best[0]:
                best = (cost, width, order)
        _, self.width, self.order = best
        if self.width == 1:
            self.nodes = scaled
            self.alphas = [(0,) * D]
            return
        runs = [self._runs(y, self.width) for y in scaled]
        self.nodes = [xp.ascontiguousarray(y[:, self.width // 2])
                      for y in runs]
        self.deltas = [y - y[:, self.width // 2][:, None] for y in runs]
        self.alphas = _nested(self.order, D)

    def _runs(self, values, width, zero=False):
        """``values`` of the harmonics' bins as runs ``(runs, width)``,
        each harmonic's last run padded with its last value, or with zeros
        (strengths)."""
        xp = self.xp
        v = values.reshape(self.nk, self.N)
        extra = (-self.N) % width
        if extra:
            pad = (xp.zeros((self.nk, extra), dtype=v.dtype) if zero
                   else xp.repeat(v[:, -1:], extra, axis=1))
            v = xp.concatenate([v, pad], axis=1)
        return v.reshape(-1, width)

    def plan(self, lib, options, cplx, upsampfac, eps, memory=2.5e8):
        """The transform's plan, for as many moments at a time as fit in
        ``memory`` bytes of output."""
        cells = int(numpy.prod(self.size))
        self.batch = int(max(1, min(len(self.alphas),
                                    memory // (8 * cells))))
        self.cplx = cplx
        self._plan = lib.Plan(1, tuple(self.size), n_trans=self.batch,
                              eps=eps, isign=1,
                              dtype=numpy.dtype(cplx).name,
                              upsampfac=upsampfac, **options)
        self._plan.setpts(*self.nodes)
        self._modes = [self.xp.arange(n, dtype=numpy.float64) - n // 2
                       for n in self.size]

    def _mode_power(self, alpha):
        out = None
        for d, a in enumerate(alpha):
            if a == 0:
                continue
            shape = [1] * len(self.size)
            shape[d] = self.size[d]
            v = (self._modes[d] ** a).reshape(shape)
            out = v if out is None else out * v
        return out

    def transform(self, strengths):
        """The transform of one vector of strengths, shape the grid's."""
        xp = self.xp
        if self.width == 1:
            if self.batch == 1:
                return self._plan.execute(xp.ascontiguousarray(
                    strengths, dtype=self.cplx))
            stack = xp.zeros((self.batch, len(strengths)), dtype=self.cplx)
            stack[0] = strengths
            return self._plan.execute(stack)[0]
        c = self._runs(xp.asarray(strengths, dtype=self.cplx), self.width,
                       zero=True)
        moments = self._moments(c)
        total = None
        stack = xp.zeros((self.batch, len(c)), dtype=self.cplx)
        for g in range(0, len(self.alphas), self.batch):
            group = self.alphas[g:g + self.batch]
            for i, alpha in enumerate(group):
                stack[i] = next(moments)
            out = self._plan.execute(stack)
            for i, alpha in enumerate(group):
                factor = 1j ** sum(alpha) / numpy.prod(
                    [math.factorial(a) for a in alpha])
                power = self._mode_power(alpha)
                term = out[i] * (factor if power is None
                                 else factor * power)
                total = term if total is None else total + term
        return total

    def _moments(self, c):
        """The moments ``sum_{j in b} c_j delta_j^alpha`` in the order of
        :attr:`alphas`, each power one product from the last."""
        def walk(w, d, left):
            if d == len(self.deltas) - 1:
                for _ in range(left + 1):
                    yield w.sum(axis=1)
                    w = w * self.deltas[d]
                return
            for a in range(left + 1):
                yield from walk(w, d + 1, left - a)
                w = w * self.deltas[d]
        return walk(c, 0, self.order)


def _nested(order, D):
    """The multi-indices of ``D`` non-negative integers summing to at most
    ``order``, the first varying slowest."""
    if D == 1:
        return [(a,) for a in range(order + 1)]
    return [(a,) + rest for a in range(order + 1)
            for rest in _nested(order - a, D - 1)]


def _kaiser_taps(offsets, half_width, beta):
    """Kaiser-windowed cardinal sine at ``offsets`` (in grid steps)."""
    from scipy.special import i0
    u = numpy.clip(offsets / half_width, -1, 1)
    window = i0(beta * numpy.sqrt(1 - u * u)) / i0(beta)
    return numpy.sinc(offsets) * window


_TIME_INTERPOLATE = r"""
#include <cupy/complex.cuh>
__device__ float bessel_i0(float x) {
  // series of I0, enough for the window's arguments
  float term = 1.0f, sum = 1.0f, y = 0.25f * x * x;
  for (int k = 1; k < 40; k++) {
    term *= y / ((float)k * k);
    sum += term;
    if (term < 1e-8f * sum) break;
  }
  return sum;
}

extern "C" __global__ void time_interpolate(
    const complex<float>* H, const double* u, const int V, const int J,
    const int L, const int n3, const int hw, const float beta,
    const float norm, complex<float>* out) {
  // out[v, j, l] = sum_c K(u[j, l] - i) H[v, j, i], i within hw of u
  long long k = (long long)blockDim.x * blockIdx.x + threadIdx.x;
  if (k >= (long long)V * J * L) return;
  int l = k % L;
  int j = (k / L) % J;
  int v = k / ((long long)L * J);
  double x = u[(long long)j * L + l];
  int first = (int)floor(x) - hw + 1;
  complex<float> acc(0, 0);
  for (int c = 0; c < 2 * hw; c++) {
    int i = first + c;
    if (i < 0 || i >= n3) continue;
    float d = (float)(x - i);
    float w = d / hw;
    if (fabsf(w) >= 1.0f) continue;
    float sinc = fabsf(d) < 1e-6f ? 1.0f : sinpif(d) / (3.14159265f * d);
    float window = bessel_i0(beta * sqrtf(1.0f - w * w)) / norm;
    acc += (sinc * window) * H[((long long)v * J + j) * n3 + i];
  }
  out[k] = acc;
}
"""


def _time_interpolate(H, u, half_width, beta, xp, threads=256):
    if xp is numpy:
        return _time_interpolate_cpu(H, u, half_width, beta)
    return _time_interpolate_gpu(H, u, half_width, beta, xp, threads)


def _time_interpolate_cpu(H, u, half_width, beta):
    V, J, n3 = H.shape
    first = numpy.floor(u).astype(int) - half_width + 1        # (J, L)
    idx = first[..., None] + numpy.arange(2 * half_width)      # (J, L, C)
    w = _kaiser_taps(u[..., None] - idx, half_width, beta)
    valid = (idx >= 0) & (idx < n3)
    w = numpy.where(valid, w, 0.0)
    idx = numpy.clip(idx, 0, n3 - 1)
    rows = numpy.arange(J)[:, None, None]
    return (H[:, rows, idx] * w[None]).sum(axis=-1)


def _time_interpolate_gpu(H, u, half_width, beta, xp, threads=256):
    """Kaiser-sinc interpolation of ``H`` ``(V, J, n3)`` at fractional
    indices ``u`` ``(J, L)``: ``(V, J, L)``."""
    from scipy.special import i0
    kernel = _kernel(xp, _TIME_INTERPOLATE, 'time_interpolate')
    rdt = numpy.float32
    V, J, n3 = H.shape
    L = u.shape[1]
    out = xp.empty((V, J, L), dtype=H.dtype)
    total = V * J * L
    kernel(((total + threads - 1) // threads,), (threads,), (
        xp.ascontiguousarray(H), xp.asarray(u, dtype=xp.float64),
        numpy.int32(V), numpy.int32(J), numpy.int32(L), numpy.int32(n3),
        numpy.int32(half_width), rdt(beta), rdt(i0(beta)), out))
    return out


_KERNELS = {}


def _kernel(xp, code, name):
    """A compiled kernel, compiled once per process."""
    if name not in _KERNELS:
        _KERNELS[name] = xp.RawKernel(code, name)
    return _KERNELS[name]


_PHASE_COMBINE = r"""
#include <cupy/complex.cuh>
extern "C" __global__ void phase_combine(
    const complex<float>* grid, const int* i1, const float* w1,
    const int* i2, const float* w2, const complex<float>* weight,
    const int V,
    const int J, const int n1, const int n2, const int n3, const int A,
    const int B, complex<float>* out) {
  // out[c, j, t] = sum_{v = c mod 2} weight[j, v / 2]
  //                sum_a,b w1[j, a] w2[j, b] grid[v, i1[j] + a, i2[j] + b, t]
  long long k = (long long)blockDim.x * blockIdx.x + threadIdx.x;
  if (k >= 2LL * J * n3) return;
  int t = k % n3;
  int j = (k / n3) % J;
  int c = k / ((long long)n3 * J);
  complex<float> total(0, 0);
  for (int v = c; v < V; v += 2) {
    complex<float> wv = weight[(long long)j * (V / 2) + v / 2];
    complex<float> acc(0, 0);
    for (int a = 0; a < A; a++) {
      int x = i1[j] + a;
      if (x < 0 || x >= n1) continue;
      complex<float> row(0, 0);
      for (int b = 0; b < B; b++) {
        int y = i2[j] + b;
        if (y < 0 || y >= n2) continue;
        row += w2[j * B + b]
            * grid[(((long long)v * n1 + x) * n2 + y) * n3 + t];
      }
      acc += w1[j * A + a] * row;
    }
    total += wv * acc;
  }
  out[k] = total;
}
"""


def _phase_combine(grid, i1, w1, i2, w2, weight, xp, threads=256):
    if xp is numpy:
        return _phase_combine_cpu(grid, i1, w1, i2, w2, weight)
    return _phase_combine_gpu(grid, i1, w1, i2, w2, weight, xp, threads)


def _phase_combine_cpu(grid, i1, w1, i2, w2, weight):
    V, n1, n2, n3 = grid.shape
    J, A = numpy.shape(w1)
    B = numpy.shape(w2)[1]
    out = numpy.zeros((2, J, n3), dtype=grid.dtype)
    for j in range(J):
        a = i1[j] + numpy.arange(A)
        b = i2[j] + numpy.arange(B)
        wa = numpy.where((a >= 0) & (a < n1), w1[j], 0.0)
        wb = numpy.where((b >= 0) & (b < n2), w2[j], 0.0)
        a, b = numpy.clip(a, 0, n1 - 1), numpy.clip(b, 0, n2 - 1)
        local = grid[:, a[:, None], b[None, :], :]             # (V, A, B, n3)
        H = numpy.einsum('vabt,a,b->vt', local, wa, wb)
        w = numpy.repeat(weight[j], 2)                         # (V,)
        out[0, j] = (w[0::2, None] * H[0::2]).sum(axis=0)
        out[1, j] = (w[1::2, None] * H[1::2]).sum(axis=0)
    return out


def _phase_combine_gpu(grid, i1, w1, i2, w2, weight, xp, threads=256):
    """Interpolate ``grid`` ``(V, n1, n2, n3)`` along its phase axes for
    each template and combine its vectors (pairs of bases) with the
    template's complex ``weight`` ``(J, V / 2)``: ``(2, J, n3)``."""
    kernel = _kernel(xp, _PHASE_COMBINE, 'phase_combine')
    cplx, rdt = xp.complex64, xp.float32
    V, n1, n2, n3 = grid.shape
    J, A = numpy.shape(w1)
    B = numpy.shape(w2)[1]
    out = xp.empty((2, J, n3), dtype=cplx)
    total = 2 * J * n3
    kernel(((total + threads - 1) // threads,), (threads,), (
        xp.ascontiguousarray(grid, dtype=cplx),
        xp.asarray(i1, dtype=xp.int32), xp.asarray(w1, dtype=rdt),
        xp.asarray(i2, dtype=xp.int32), xp.asarray(w2, dtype=rdt),
        xp.asarray(weight, dtype=cplx), numpy.int32(V),
        numpy.int32(J), numpy.int32(n1), numpy.int32(n2), numpy.int32(n3),
        numpy.int32(A), numpy.int32(B), out))
    return out


def _complex(xp):
    """Single precision on a GPU, double on the CPU."""
    return numpy.complex128 if xp is numpy else xp.complex64


def _real(xp):
    return numpy.float64 if xp is numpy else xp.float32


def phase_grid_search(correlator, region, delay_range, gram,
                      noise_variance=1.0, sky=None, top=20, depth=0,
                      max_depth=8, **grid_options):
    """The F-statistic over a region of templates and sky positions, on
    NUFFT grids: each grid gives ``2F`` at every node, third phase
    coordinate and sky position, and arrival-time sample (see
    :meth:`PhaseGrid.grid_twof`). While one grid cannot hold the region, it
    is halved along its leading phase coordinate, or the arrival times are
    once the region is down to a few points; so is a region whose points
    keep more phase outside the grid's directions than the grid's
    ``phase_tolerance``.

    The third phase coordinate is searched at the grid's spacing of the
    first two, ``pi / (oversample X)`` with ``X`` the half-extent of its
    node coordinates.

    Parameters
    ----------
    correlator : PhaseNUFFTCorrelation
        The family's strengths, on a sky basis spanning the positions
        searched (``sky_points``).
    region : array, shape ``(M, 4)``
        Points (mchirp, q, spin1z, spin2z) spanning the region.
    delay_range : tuple
        Arrival times searched, s.
    gram : callable
        ``gram(T, sky)``, the reference's Gram matrix at grid times, see
        :meth:`PhaseGrid.grid_twof`.
    sky : array, shape ``(S, 2)``
        Sky offsets searched; the correlator's ``sky`` when not given.
    top : int
        Candidates kept.

    Returns
    -------
    dict
        ``candidates``, the ``top`` largest ``2F`` values with their phase
        coordinates, third coordinate, sky offsets, arrival time and the
        region point nearest each, and ``templates``, the number of
        templates (nodes, third values and sky positions) searched.
    """
    import logging
    region = numpy.atleast_2d(region)
    sky = numpy.atleast_2d(correlator.sky if sky is None else sky)
    try:
        grid = PhaseGrid(correlator, region, delay_range, **grid_options)
        if grid.phase_residual > grid.phase_tolerance:
            del grid
            if len(region) < 4:
                raise ValueError("the region's phase does not fit the "
                                 "grid's directions")
            raise MemoryError("the region's phase leaves the grid's "
                              "directions")
    except MemoryError:
        if depth >= max_depth:
            raise
        if len(region) >= 4:
            frame = correlator.frame(region, 1)
            s, _, _ = correlator.project(region, frame)
            half = numpy.median(s[:, 0])
            parts = [(region[s[:, 0] <= half], delay_range),
                     (region[s[:, 0] > half], delay_range)]
        else:
            mid = 0.5 * (delay_range[0] + delay_range[1])
            parts = [(region, (delay_range[0], mid)),
                     (region, (mid, delay_range[1]))]
        found, count = [], 0
        for part, delays in parts:
            out = phase_grid_search(correlator, part, delays, gram,
                                    noise_variance, sky, top, depth + 1,
                                    max_depth, **grid_options)
            found.extend(out['candidates'])
            count += out['templates']
        found.sort(key=lambda c: -c['twof'])
        return dict(candidates=found[:top], templates=count)
    if grid.terms > 1 and numpy.isfinite(grid.s3_half):
        n = int(numpy.ceil(2 * grid.s3_half / grid.s3_step)) + 1
        s3_values = numpy.linspace(-1, 1, n) * grid.s3_half
    else:
        s3_values = numpy.zeros(1)
    best, where, axes = grid.grid_twof(gram, noise_variance, s3_values, sky)
    best, where = _host(best), _host(where)
    flat = numpy.argsort(best, axis=None)[::-1][:top]
    found = []
    s_region, _, _ = correlator.project(region, grid.frame)
    for k in flat:
        i0, i3, i1, i2 = numpy.unravel_index(k, best.shape)
        coords = [axes[0][i1]] + ([axes[1][i2]] if grid.ndir > 1 else [])
        dist = numpy.sum((s_region[:, :len(coords)] - coords) ** 2, axis=1)
        found.append(dict(
            twof=float(best[i0, i3, i1, i2]), s=coords,
            s3=float(s3_values[i3] + grid.middle), sky=sky[i0],
            time=float(grid.start[-1] + where[i0, i3, i1, i2]
                       * grid.step[-1]),
            nearest=region[numpy.argmin(dist)]))
    count = int(numpy.prod(best.shape))
    logging.info("phase grid search: grid %s, %d templates, 2F max %.1f",
                 grid.size, count, best.max())
    del grid
    return dict(candidates=found, templates=count)


def _host(array):
    return array.get() if hasattr(array, 'get') else numpy.asarray(array)
