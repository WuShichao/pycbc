# Copyright (C) 2026 Shichao Wu
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
"""The main row of a precessing source's modes, as a phase correction of an
aligned-spin cell's harmonics, for any waveform model.

A precessing model twists co-precessing modes up with Euler angles: the
co-precessing harmonic ``(l, m, n)`` contributes to the inertial index ``m'``
through ``D^l_{m'm}(alpha, beta, gamma)(t)``. Its main row ``m' = m`` is, for
small opening angles, the aligned harmonic times ``exp(-i dpsi_k(f))`` (the
amplitude ratio stays within a few 1e-3 of one): ``dpsi_k`` holds the slow
rotation phase ``2 int (1 - cos beta) d alpha`` at the harmonic's stationary
time and the co-precessing dynamics' own change.

:class:`PrecessionMainRow` stores ``dpsi_k`` as a few shapes per harmonic on a
cell's frequency grid, with coefficients that are a quadratic emulator in the
intrinsic parameters ``lam = (Mc, e, chi, q, chi1_perp, chi2_perp, dphi)``
(spins at the cell's ``f_ref``; ``dphi`` the in-plane spins' azimuth
difference; a common rotation of both is the observer's azimuth and leaves the
main row unchanged). Parts a PE absorbs are not represented: a phase common to
all harmonics, a common arrival time and ``(n - m)`` times a mean-anomaly
offset. The shapes are pinned at ``f_ref`` (value and slope zero on the
``(2, 2, 2)`` harmonic) through the common phase and time, so the cell's
clock convention holds unchanged.

:func:`build_precession_main_row` fits it from any ``generator(lam, inplane)``
returning ``{key: (amplitude (2, N), psi (N,), present (N,))}`` with the main
row in ``amplitude[0]`` (``pycbc.waveform.eccentric_mode_family.efpe_modes``
with ``hlm=True`` for pyEFPEHM).
"""
from itertools import combinations_with_replacement

import numpy

__all__ = ['PrecessionMainRow', 'build_precession_main_row', 'features',
           'isotropic_log_prior', 'isotropic_log_prior_sqrt']


def features(lam):
    """The emulator's design matrix of rows ``lam`` (M, 7): one, the eight
    base variables and their products (45 columns)."""
    lam = numpy.atleast_2d(numpy.asarray(lam, dtype=float))
    mc, e, chi, q, s1, s2, dphi = lam.T
    base = numpy.column_stack([numpy.log(mc / 30.), numpy.log(q / 1.5),
                               (e - .05) / .005, chi / .1, s1, s2,
                               s1 * s2 * numpy.cos(dphi),
                               s1 * s2 * numpy.sin(dphi)])
    cols = [numpy.ones(len(lam))] + list(base.T)
    cols += [base[:, i] * base[:, j] for i, j in
             combinations_with_replacement(range(base.shape[1]), 2)]
    return numpy.column_stack(cols)


class PrecessionMainRow(object):
    """The main-row phase corrections ``dpsi_k(f; lam)`` of the harmonics
    ``keys`` on the grid ``f``: ``shapes[key]`` (nshape, len(f)), fitted on
    the observation window and held at their edge values beyond, and the
    emulator matrix ``emulator``
    (45, nshape). ``dpsi_k = coefficients(lam) @ shapes[key]``.

    The validation numbers of the build (held-out error, rad) are in
    ``diagnostics``."""

    def __init__(self, keys, f, shapes, emulator, f_ref, diagnostics=None):
        self.keys = [tuple(int(v) for v in k) for k in keys]
        self.f = numpy.asarray(f, dtype=float)
        self.shapes = {tuple(int(v) for v in k): numpy.asarray(v, dtype=float)
                       for k, v in shapes.items()}
        self.emulator = numpy.asarray(emulator, dtype=float)
        self.f_ref = float(f_ref)
        self.nshape = self.emulator.shape[1]
        self.diagnostics = dict(diagnostics or {})

    def coefficients(self, lam):
        """The shape coefficients (M, nshape) of rows ``lam`` (M, 7)."""
        return features(lam) @ self.emulator

    def phase(self, key, lam, frequencies=None):
        """``dpsi_key`` (M, len) at rows ``lam``, on ``frequencies`` (by
        default the grid ``f``; linear interpolation, the edge values
        outside)."""
        key = tuple(int(v) for v in key)
        c = self.coefficients(lam)
        if key not in self.shapes:
            n = len(self.f) if frequencies is None else len(frequencies)
            return numpy.zeros((len(c), n))
        shapes = self.shapes[key]
        if frequencies is not None:
            frequencies = numpy.asarray(frequencies, dtype=float)
            shapes = numpy.array([numpy.interp(frequencies, self.f, s)
                                  for s in shapes])
        return c @ shapes


def _pin(shapes_by_key, f, f_ref, key222=(2, 2, 2)):
    """Each shape less the common phase and arrival time that zero its value
    and slope on the (2, 2, 2) harmonic at f_ref."""
    out = {}
    s222 = shapes_by_key[key222]
    for k, S in shapes_by_key.items():
        out[k] = numpy.array(S, dtype=float)
    for i in range(s222.shape[0]):
        v = numpy.interp(f_ref, f, s222[i])
        # the slope at f_ref: central differences interpolated there
        slope = numpy.interp(f_ref, f[1:-1], (s222[i, 2:] - s222[i, :-2])
                             / (f[2:] - f[:-2]))
        for k, S in out.items():
            S[i] = S[i] - v - slope * (f - f_ref)
    return out


def build_precession_main_row(generator, draws, f, f_ref, weight, keys=None,
                              nshape=2, train_fraction=0.75, pool=None,
                              smooth_degree=8):
    """Fit :class:`PrecessionMainRow` from ``generator`` at ``draws`` (M, 7).

    ``generator(lam, inplane)`` returns the modes of the source ``lam`` with
    its in-plane spins (``inplane=True``) or without (the aligned source of
    the same masses, eccentricity and aligned spins). ``weight[key]`` (len(f))
    is the SNR^2 density that weights the fit (zero outside the observation).
    ``keys`` default to every (2, 2, n) the generator returns for all draws.
    ``pool``: an optional map for the generator calls."""
    f = numpy.asarray(f, dtype=float)
    draws = numpy.atleast_2d(numpy.asarray(draws, dtype=float))
    mapper = map if pool is None else pool.map
    pairs = list(mapper(_draw_pair, [(generator, lam) for lam in draws]))
    ok = [i for i, p in enumerate(pairs) if p is not None]
    if keys is None:
        keys = sorted(set.intersection(*[set(pairs[i]) for i in ok]))
        keys = [k for k in keys if tuple(k[:2]) == (2, 2)]
    keys = [tuple(int(v) for v in k) for k in keys]
    blocks, ws = [], []
    for k in keys:
        have = numpy.logical_and.reduce([pairs[i][k][1] for i in ok])
        w = numpy.where(have, weight[k], 0.)
        blocks.append(numpy.nonzero(w > 0)[0])
        ws.append(w[w > 0])
    wall = numpy.concatenate(ws)
    wall = wall / wall.sum()
    sw = numpy.sqrt(wall)
    edges = numpy.cumsum([0] + [len(b) for b in blocks])
    Y = numpy.array([numpy.concatenate([pairs[i][k][0][b] for k, b in
                                        zip(keys, blocks)]) for i in ok])
    V = numpy.column_stack([numpy.ones(len(wall)),
                            numpy.concatenate([2 * numpy.pi * (f[b] - f_ref)
                                               for b in blocks]),
                            numpy.concatenate([numpy.full(len(b), k[2] - k[1],
                                                          dtype=float)
                                               for k, b in zip(keys, blocks)])])
    # each harmonic's phase is defined mod 2 pi: per draw, the integer shifts
    # (2 pi times -2..2 per block) that leave the least weighted residual
    # after the absorbed parts. The residual is quadratic in the shifts:
    # with P the weighted projector off the absorbed span and e_j a block's
    # indicator, |P (y - 2 pi sum s_j e_j)|^2 = |Py|^2 - 4 pi s.g + 4 pi^2
    # s.M.s, g_j = <Py, P e_j>, M_jk = <P e_j, P e_k>
    A = V * sw[:, None]
    Q, _ = numpy.linalg.qr(A)

    def project(X):
        X = X * sw[None]
        return X - (X @ Q) @ Q.T
    E = numpy.zeros((len(blocks), len(wall)))
    for j in range(len(blocks)):
        E[j, edges[j]:edges[j + 1]] = 1.
    PE = project(E)
    M = PE @ PE.T
    from itertools import product
    shifts = numpy.array(list(product(range(-2, 3), repeat=len(blocks))),
                         dtype=float)
    quad = 4 * numpy.pi ** 2 * numpy.einsum('sj,jk,sk->s', shifts, M, shifts)
    G = project(Y) @ PE.T                                     # (draws, blocks)
    cost = -4 * numpy.pi * G @ shifts.T + quad[None]          # (draws, combos)
    best = shifts[numpy.argmin(cost, axis=1)]
    Y = Y - 2 * numpy.pi * best @ E
    coef = numpy.linalg.lstsq(V * sw[:, None], (Y * sw[None]).T,
                              rcond=None)[0]
    R = Y - (V @ coef).T
    lam = draws[ok]
    ntr = int(train_fraction * len(ok))
    _, sv, Vh = numpy.linalg.svd(R[:ntr] * sw[None], full_matrices=False)
    Bw = Vh[:nshape]
    C = (R * sw[None]) @ Bw.T
    F = features(lam)
    em = numpy.linalg.lstsq(F[:ntr], C[:ntr], rcond=None)[0]
    # held out: the shapes' own truncation and the emulator
    Rt = R[ntr:]
    trunc = Rt - ((Rt * sw[None]) @ Bw.T @ Bw) / sw[None]
    emul = Rt - (F[ntr:] @ em @ Bw) / sw[None]
    # the shapes on the full grid: the weighted basis back in rad, by block, as a weighted Chebyshev polynomial of
    # degree ``smooth_degree`` over the fitted support (the SVD rows carry the bins' noise, whose derivative would
    # be the templates' clock), held at its edge values beyond: continuous phase, a clock step of the edge slope
    # over 2 pi (tens of s, against hundreds per bin of the chirp)
    shapes = {}
    for j, (k, b) in enumerate(zip(keys, blocks)):
        full = numpy.logical_or.reduce([pairs[i][k][1] for i in ok])
        S = numpy.zeros((nshape, len(f)))
        rows = Bw[:, edges[j]:edges[j + 1]] / sw[None, edges[j]:edges[j + 1]]
        fb = f[b]
        x0, x1 = fb[0], fb[-1]
        to_x = lambda v: 2 * (v - x0) / (x1 - x0) - 1
        wfit = sw[edges[j]:edges[j + 1]]
        for r in range(nshape):
            cheb = numpy.polynomial.chebyshev.Chebyshev.fit(to_x(fb), rows[r], smooth_degree, w=wfit,
                                                            domain=[-1, 1])
            # everywhere on the grid (a harmonic's support in a cell may reach past the training draws'): the
            # polynomial on the fitted support, its edge values beyond. Constant, not the edge line: f_ref lies below
            # the window, and the pin there would turn an edge slope into an arrival time of the whole template
            # (2e4 s with the edge line, the delay grid and the screen then far from the signal)
            S[r] = cheb(numpy.clip(to_x(f), -1, 1))
        shapes[k] = S
    shapes = _pin(shapes, f, f_ref)
    # the held-out emulator error with the smoothed, pinned shapes (up to the absorbed parts): fitted again
    smooth_err = None
    if len(Rt):
        Sst = numpy.concatenate([shapes[k][:, b] for k, b in zip(keys, blocks)], axis=1)   # (nshape, support)
        diff = Rt - F[ntr:] @ em @ Sst                                                    # coefficients x shapes
        cd = numpy.linalg.lstsq(V * sw[:, None], (diff * sw[None]).T, rcond=None)[0]
        smooth_err = float(numpy.abs(diff - (V @ cd).T).max())
    diagnostics = dict(draws=len(draws), ok=len(ok), train=ntr, emulator_smoothed_max_rad=smooth_err,
                       test=len(ok) - ntr,
                       singular_values_relative=(sv / sv[0])[:8].tolist(),
                       truncation_max_rad=float(numpy.abs(trunc).max())
                       if len(Rt) else None,
                       emulator_max_rad=float(numpy.abs(emul).max())
                       if len(Rt) else None,
                       emulator_median_max_rad=float(numpy.median(
                           numpy.abs(emul).max(axis=1))) if len(Rt) else None,
                       dpsi_max_rad=float(numpy.abs(R).max()))
    return PrecessionMainRow(keys, f, shapes, em, f_ref, diagnostics)


def _draw_pair(job):
    generator, lam = job
    try:
        P, A = generator(lam, True), generator(lam, False)
    except Exception:
        return None
    out = {}
    for k in A:
        if k not in P:
            continue
        ap, psp, hp = P[k]
        aa, psa, ha = A[k]
        have = hp & ha & (numpy.abs(aa[0]) > 0)
        if have.sum() < 10:
            continue
        Hp = ap[0] * numpy.exp(-1j * psp)
        Ha = aa[0] * numpy.exp(-1j * psa)
        R = numpy.where(have, Hp / numpy.where(have, Ha, 1), 1)
        dpsi = numpy.zeros(len(psp))
        idx = numpy.nonzero(have)[0]
        dpsi[idx] = -numpy.unwrap(numpy.angle(R[idx]))
        out[tuple(int(v) for v in k)] = (dpsi, have)
    return out


def isotropic_log_prior(chi_eff, chi1_perp, chi2_perp, q, a_max=0.99,
                        n=256):
    """ln of the density of ``(chi_eff, chi1_perp, chi2_perp)`` for spins of
    magnitude U[0, a_max] and isotropic directions (the in-plane azimuths
    uniform, so their difference too), the antisymmetric aligned part
    integrated out: with ``chi_i = (chi_i_perp, chi_iz)`` each of density
    ``chi_perp / (2 a_max (chi_perp^2 + chi_z^2))`` inside the ball, and
    ``chi_eff = (m1 chi_1z + m2 chi_2z) / M``, evaluated by its exact closed-form
    antiderivative (partial fractions into ``arctan`` and ``log``, with 32-pt
    Gauss-Legendre in ``arctan(z / s1)`` on the near-degenerate strip
    ``mu ~ 0, s1 ~ (w2 / w1) s2``). Normalised over chi_eff, chi1_perp,
    chi2_perp; ``q = m1 / m2``. ``n`` is retained only for API compatibility."""
    del n
    chi_eff, s1, s2, q = (numpy.atleast_1d(numpy.asarray(v, dtype=float))
                          for v in (chi_eff, chi1_perp, chi2_perp, q))
    chi_eff, s1, s2, q = numpy.broadcast_arrays(chi_eff, s1, s2, q)
    w1 = q / (1.0 + q)
    w2 = 1.0 / (1.0 + q)
    out = numpy.full(chi_eff.shape, -numpy.inf, dtype=float)
    zmax1 = numpy.sqrt(numpy.maximum(a_max ** 2 - s1 ** 2, 0.0))
    zmax2 = numpy.sqrt(numpy.maximum(a_max ** 2 - s2 ** 2, 0.0))
    zlo = numpy.maximum(-zmax1, (chi_eff - w2 * zmax2) / w1)
    zhi = numpy.minimum(zmax1, (chi_eff + w2 * zmax2) / w1)
    in_box = (zhi > zlo) & (s1 >= 0.0) & (s2 >= 0.0) & (s1 < a_max) & (s2 < a_max)
    if not numpy.any(in_box):
        return out

    # Exact s1 > 0, s2 > 0 interior (generic case)
    both_pos = in_box & (s1 > 0.0) & (s2 > 0.0)
    if numpy.any(both_pos):
        a = s1[both_pos]
        w1v = w1[both_pos]
        mu = chi_eff[both_pos] / w1v
        b = (w2[both_pos] / w1v) * s2[both_pos]
        zl = zlo[both_pos]
        zh = zhi[both_pos]
        # D = (mu^2 + (a - b)^2) (mu^2 + (a + b)^2)
        D = (mu ** 2 + (a - b) ** 2) * (mu ** 2 + (a + b) ** 2)
        scale4 = numpy.maximum((a + b) ** 4, 1e-300)
        near_deg = D < 1e-6 * scale4
        val = numpy.empty_like(a)
        g = ~near_deg
        if numpy.any(g):
            ag, bg, mg, zlg, zhg, Dg = a[g], b[g], mu[g], zl[g], zh[g], D[g]
            dt1 = numpy.arctan2(zhg, ag) - numpy.arctan2(zlg, ag)
            dt2 = numpy.arctan2(zhg - mg, bg) - numpy.arctan2(zlg - mg, bg)
            lterm = (numpy.log(zhg ** 2 + ag ** 2)
                     - numpy.log(zlg ** 2 + ag ** 2)
                     + numpy.log((zlg - mg) ** 2 + bg ** 2)
                     - numpy.log((zhg - mg) ** 2 + bg ** 2))
            val[g] = (mg * ag * bg * lterm
                      + bg * (mg ** 2 + bg ** 2 - ag ** 2) * dt1
                      + ag * (mg ** 2 + ag ** 2 - bg ** 2) * dt2) / Dg
        if numpy.any(near_deg):
            an, bn, mn = a[near_deg], b[near_deg], mu[near_deg]
            zln, zhn = zl[near_deg], zh[near_deg]
            tlo = numpy.arctan2(zln, an)
            thi = numpy.arctan2(zhn, an)
            xgl, wgl = numpy.polynomial.legendre.leggauss(32)
            t = 0.5 * (thi - tlo)[:, None] * xgl[None, :] + 0.5 * (thi + tlo)[:, None]
            z = an[:, None] * numpy.tan(t)
            integrand = bn[:, None] / ((z - mn[:, None]) ** 2 + bn[:, None] ** 2)
            val[near_deg] = numpy.sum(integrand * wgl[None, :], axis=1) * 0.5 * (thi - tlo)
        total = val / (4.0 * a_max ** 2 * w1v)
        with numpy.errstate(divide='ignore', invalid='ignore'):
            out[both_pos] = numpy.where(total > 0.0, numpy.log(total), -numpy.inf)

    # Exact boundary limits s1 == 0, s2 > 0 or s2 == 0, s1 > 0 (Dirac delta of Lorentzian)
    only_s2 = in_box & (s1 == 0.0) & (s2 > 0.0) & (zlo <= 0.0) & (zhi >= 0.0)
    if numpy.any(only_s2):
        w1v = w1[only_s2]
        mu = chi_eff[only_s2] / w1v
        b = (w2[only_s2] / w1v) * s2[only_s2]
        # endpoint weight: pi if strictly inside, pi/2 at an endpoint
        wt = numpy.where((zlo[only_s2] < 0.0) & (zhi[only_s2] > 0.0), numpy.pi, 0.5 * numpy.pi)
        total = wt * b / (4.0 * a_max ** 2 * w1v * (b ** 2 + mu ** 2))
        with numpy.errstate(divide='ignore'):
            out[only_s2] = numpy.where(total > 0.0, numpy.log(total), -numpy.inf)

    only_s1 = in_box & (s2 == 0.0) & (s1 > 0.0)
    if numpy.any(only_s1):
        w1v = w1[only_s1]
        mu = chi_eff[only_s1] / w1v
        a = s1[only_s1]
        in_z = (zlo[only_s1] <= mu) & (zhi[only_s1] >= mu)
        if numpy.any(in_z):
            wt = numpy.where((zlo[only_s1] < mu) & (zhi[only_s1] > mu), numpy.pi, 0.5 * numpy.pi)
            total = numpy.where(in_z, wt * a / (4.0 * a_max ** 2 * w1v * (a ** 2 + mu ** 2)), 0.0)
            with numpy.errstate(divide='ignore'):
                out[only_s1] = numpy.where(total > 0.0, numpy.log(total), -numpy.inf)

    return out


def isotropic_log_prior_sqrt(chi_eff, u1, u2, q, a_max=0.99):
    """ln of the isotropic spin prior density in quadratic in-plane coordinates
    ``(chi_eff, u1, u2)`` where ``chi1_perp = u1^2``, ``chi2_perp = u2^2`` on
    ``u_i in [0, sqrt(a_max)]``.

    Multiplying ``p(chi_eff, s1, s2)`` by the Jacobian ``4 u1 u2`` removes the
    ``1 / (w1 s1 + w2 s2)`` integrable pole at ``(chi_eff, s1, s2) = (0, 0, 0)``
    completely (bounded above everywhere by AM-GM), making climbs and AMR boxes
    well-conditioned across the entire prior box."""
    chi_eff, u1, u2, q = (numpy.atleast_1d(numpy.asarray(v, dtype=float))
                          for v in (chi_eff, u1, u2, q))
    chi_eff, u1, u2, q = numpy.broadcast_arrays(chi_eff, u1, u2, q)
    out = numpy.full(chi_eff.shape, -numpy.inf, dtype=float)
    pos = (u1 > 0.0) & (u2 > 0.0) & (u1 * u1 < a_max) & (u2 * u2 < a_max)
    if not numpy.any(pos):
        return out
    lp = isotropic_log_prior(chi_eff[pos], u1[pos] ** 2, u2[pos] ** 2, q[pos], a_max=a_max)
    with numpy.errstate(divide='ignore', invalid='ignore'):
        val = lp + numpy.log(4.0 * u1[pos] * u2[pos])
    out[pos] = numpy.where(numpy.isfinite(val), val, -numpy.inf)
    return out
