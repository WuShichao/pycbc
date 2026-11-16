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
"""The Bayes factor of a non-precessing signal with modes of several
angular patterns, marginalised over distance, inclination, polarisation,
coalescence phase and (for eccentric sources) mean anomaly.

A mode ``k = (l, m, n)`` (``m`` signed; ``n`` the eccentric harmonic, ``n =
m`` for circular sources) has the multipole moment ``h_k`` at the template's
reference distance, without any angular factor, and its response to a unit
plus and a unit cross polarisation, the two templates ``y_k = (y_k+,
y_kx)``. With pyEFPEHM's conventions (verified numerically: a phase shift
``dphi`` multiplies the mode by ``exp(i m dphi)``, a mean-anomaly shift
``dl0`` by ``exp(i (n - m) dl0)``; the viewing angles are ``Theta = pi -
iota``, ``Phi = 0``; the polarisations are ``h_k conj(Ap_lm)`` and ``h_k
conj(Ac_lm)``) the signal is

    s = h sum_k exp(i (m_k phi + c_k l0)) w_k(iota, psi)^T y_k,

``c_k = n_k - m_k``, ``h = scale / D`` and ``w_k = R(2 psi) (conj Ap_lm,
conj Ac_lm)``, the projector rotated by the polarisation angle as the
injections rotate the polarisations. With ``z_k = <y_k, d>`` and the Gram
``G[k, a, j, b] = <y_ka, y_jb>``, at fixed angles ``ln L = h X - h^2 g / 2``
with ``X = Re sum conj(v) z`` and ``g = v^H G v``, ``v_k = exp(i (m_k phi +
c_k l0)) w_k``.

The amplitude is integrated over the distance prior. A quarter turn of
``psi`` flips the sign of every ``w_k``, so ``psi`` runs over ``[0, pi / 2)``
and the amplitude integral takes ``cosh(h X)`` (both signs). Unlike a single
``m = 2`` pattern, a common phase is not a symmetry of a mixed-``m`` signal,
so ``phi`` is integrated numerically over ``[0, 2 pi)``; mean anomaly on its
uniform grid; inclination and polarisation on :class:`OrientationNodes`.
For modes that all have ``m = 2`` the result equals
:func:`pycbc.filter.mean_anomaly.mean_anomaly_log_bayes` (the ``phi``
integral of ``cosh`` is its ``I0``).
"""
import math

import numpy

__all__ = ['spin_weighted_m2', 'projector', 'pattern_weights', 'PatternSet',
           'hm_components', 'hm_log_bayes', 'phi_quadrature',
           'draw_hm_extrinsics']


def spin_weighted_m2(l, m, theta, phi=0.0):
    """The spin-weight -2 spherical harmonic ``_{-2}Y_lm(theta, phi)``, in
    LAL's convention (XLALSpinWeightedSphericalHarmonic's general sum)."""
    theta = numpy.asarray(theta, dtype=float)
    c, s = numpy.cos(theta / 2), numpy.sin(theta / 2)
    sw = 2
    fact = math.factorial
    pref = math.sqrt(fact(l + m) * fact(l - m) * fact(l + sw) * fact(l - sw))
    total = numpy.zeros(theta.shape)
    # LAL's XLALSpinWeightedSphericalHarmonic, general l (its sum is
    # written with s = 2 for the spin weight -2)
    for k in range(max(0, m - sw), min(l + m, l - sw) + 1):
        term = (-1) ** k * pref / (fact(l + m - k) * fact(l - sw - k)
                                   * fact(k) * fact(k + sw - m))
        total = total + term * c ** (2 * l + m - sw - 2 * k) \
            * s ** (2 * k + sw - m)
    return (-1) ** sw * math.sqrt((2 * l + 1) / (4 * math.pi)) * total \
        * numpy.exp(1j * m * numpy.asarray(phi, dtype=float))


def projector(l, m, iota, azimuth=0.0):
    """pyEFPEHM's plus and cross projector of the aligned mode ``(l, m)``
    (its inertial ``m' = m``) at inclination ``iota``: ``(Ap, Ac)`` with
    ``Ap = (Y_lm + (-1)^l conj Y_l-m) / 2`` and ``Ac = -i (Y_lm - (-1)^l conj
    Y_l-m) / 2`` at ``Theta = pi - iota``, ``Phi = azimuth``. A mode's
    polarisations are ``h conj(Ap)`` and ``h conj(Ac)``.

    With aligned spins pyEFPEHM's own ``phi_JN`` (the azimuth of the line of
    sight about J, undefined when J is along L) comes out 0 or -pi from
    rounding; it multiplies a mode by ``exp(-i m phi_JN)``, the same as a
    phase shift: a signal of ``phase`` and ``phi_JN`` is this model's with
    ``phi = phase - phi_JN`` and ``azimuth = 0``."""
    theta = numpy.pi - numpy.asarray(iota, dtype=float)
    y = spin_weighted_m2(l, m, theta, azimuth)
    ym = numpy.conj(spin_weighted_m2(l, -m, theta, azimuth)) * (-1) ** l
    return 0.5 * (y + ym), -0.5j * (y - ym)


def pattern_weights(l, m, iota, psi):
    """``w = R(2 psi) (conj Ap, conj Ac)``, shape ``(2,) + shape``: the
    weights of the unit-plus and unit-cross responses of mode ``(l, m)``."""
    ap, ac = projector(l, m, iota)
    plus, cross = numpy.conj(ap), numpy.conj(ac)
    c2, s2 = numpy.cos(2 * numpy.asarray(psi)), numpy.sin(2 * numpy.asarray(psi))
    return numpy.array([c2 * plus - s2 * cross, s2 * plus + c2 * cross])


class PatternSet(object):
    """The modes' angular patterns: ``keys`` ``(l, m, n)`` grouped by
    ``(l, m)``, with ``c_k = n - m`` and each pattern's ``m``."""
    def __init__(self, keys):
        self.keys = [tuple(int(v) for v in k) for k in keys]
        self.patterns = sorted({k[:2] for k in self.keys})
        self.group = numpy.array([self.patterns.index(k[:2])
                                  for k in self.keys])
        self.offsets = numpy.array([k[2] - k[1] for k in self.keys])
        self.m = numpy.array([p[1] for p in self.patterns])

    def weights(self, cos_iota, psi):
        """``(P, 2, nodes)``: every pattern's weights at the nodes."""
        iota = numpy.arccos(numpy.asarray(cos_iota, dtype=float))
        return numpy.array([pattern_weights(l, m, iota, psi)
                            for l, m in self.patterns])


def hm_components(z, gram, patterns, n_l0, xp=numpy):
    """Per pattern and mean anomaly: ``zg (n_l0, P, 2, *samples)`` with
    ``zg[j, g] = sum_{k in g} exp(-i c_k l0_j) z_k`` and ``Gg (n_l0, P, 2, P,
    2, *gram_samples)`` with ``Gg[j, g, :, g', :] = sum exp(i (c_k' - c_k)
    l0_j) G[k, :, k', :]``; and the grid ``l0_j = 2 pi j / n_l0``."""
    angles = 2 * numpy.pi * numpy.arange(int(n_l0)) / int(n_l0)
    K = len(patterns.keys)
    P = len(patterns.patterns)
    e = xp.asarray(numpy.exp(-1j * numpy.outer(angles, patterns.offsets)))
    member = numpy.zeros((P, K))
    member[patterns.group, numpy.arange(K)] = 1
    member = xp.asarray(member)
    # (n_l0, P, K): the phase of mode k in pattern g, zero elsewhere
    a = e[:, None, :] * member[None]
    zr = z.reshape((K, 2) + (-1,))
    zg = xp.einsum('jgk,kan->jgan', a, zr)
    zg = zg.reshape((len(angles), P, 2) + z.shape[2:])
    gr = gram.reshape((K, 2, K, 2, -1))
    # <v, v> holds conj(exp(i c_k l0)) exp(i c_k' l0) = a_k conj(a_k')
    Gg = xp.einsum('jgk,kalbn,jhl->jgahbn', a, gr, xp.conj(a))
    Gg = Gg.reshape((len(angles), P, 2, P, 2) + gram.shape[4:])
    return zg, Gg, angles


def _node_terms(zg, Gg, W, xp):
    """``x[j, g, node, s] = w_g^H zg`` and ``gam[j, g, h, node, s] = w_g^H
    G_gh w_h`` for weights ``W (P, 2, nodes)``."""
    Wc = xp.conj(W)
    x = xp.einsum('gan,jgas->jgns', Wc, zg)
    gam = xp.einsum('gan,jgahbs,hbn->jghns', Wc, Gg, W)
    return x, gam


def _phase_sums(x, gam, m, phis, xp):
    """``X(phi) = Re sum_g exp(-i m_g phi) x_g`` and ``g(phi) = Re sum_gh
    exp(i (m_h - m_g) phi) gam_gh`` on ``phis``: shapes ``(F,) + rest``."""
    m = xp.asarray(m)
    phis = xp.asarray(phis)
    ex = xp.exp(-1j * m[None, :] * phis[:, None])            # (F, P)
    shape = phis.shape
    X = xp.real(xp.tensordot(ex, x, axes=([1], [1])))         # (F, j, n, s)
    # exp(i (m_h - m_g) phi) = exp(-i m_g phi) conj(exp(-i m_h phi))
    eg = ex[:, :, None] * xp.conj(ex)[:, None, :]             # (F, g, h)
    g = xp.real(xp.einsum('fgh,jghns->fjns', eg, gam))
    return X, g


def _xg_at(x, gam, m, phi, xp, order=0):
    """``d^k X / dphi^k`` and ``d^k g / dphi^k`` (``k = order``) at angles
    ``phi (Q, n, s)``, for ``x (P, n, s)``, ``gam (P, P, n, s)``; summed
    pattern by pattern, so that nothing larger than ``(Q, n, s)`` is held
    (one einsum over the pattern pairs made a ``P^2 Q n s`` intermediate,
    13 GB at eight patterns)."""
    m = [float(v) for v in numpy.asarray(m)]
    P = len(m)
    e = [xp.exp(-1j * m[g] * phi) for g in range(P)]
    X = xp.zeros(phi.shape)
    g_ = xp.zeros(phi.shape)
    for a in range(P):
        X += xp.real((-1j * m[a]) ** order * e[a] * x[a][None])
        for b in range(P):
            d = (1j * (m[b] - m[a])) ** order
            if d == 0:
                continue
            g_ += xp.real(d * e[a] * xp.conj(e[b]) * gam[a, b][None])
    return X, g_


def _gh(n):
    t, w = numpy.polynomial.hermite.hermgauss(int(n))
    return t, w


def _phi_rule(x, gam, m, xp, amplitude, n_gh=10, n_uniform=64, newton=3,
              narrow=0.1):
    """Quadrature points ``phi (Q, n, s)`` and log weights (of ``dphi / 2
    pi``) for each node and sample: about each of the four peaks of
    ``X(phi)^2 / g(phi)`` (the dominant ``m = 2`` part's ``theta / 2 + k pi
    / 2``, refined by ``newton`` Newton steps of ``ln(X^2 / g)``),
    Gauss-Hermite with ``n_gh`` points scaled by ``sigma``, from the
    curvature of the amplitude marginal ``ln J(X(phi), g(phi))`` itself (a
    central difference; ``X^2 / 2 g`` overstates it where the prior
    dominates), where every peak is narrower than ``narrow`` (rad); the
    trapezoid rule on ``n_uniform`` points elsewhere (there the peaks
    overlap and are wide enough for it)."""
    m = numpy.asarray(m)
    shape = x.shape[1:]
    uni = 2 * numpy.pi * numpy.arange(int(n_uniform)) / int(n_uniform)
    phi_u = xp.broadcast_to(xp.asarray(uni)[:, None, None],
                            (len(uni),) + shape)
    lw_u = xp.full((len(uni),) + shape, -numpy.log(len(uni)))
    # the peaks: local maxima of F = X^2 / g on a scan of 64 points (F has
    # harmonics up to 2 m_max, so the scan resolves every maximum of it; the
    # integrand's peaks are narrow only because ln J grows with F), the
    # four highest, refined by Newton steps of ln F
    # F has harmonics up to 2 m_max = 8: 32 points resolve its maxima
    ns = 32
    scan = xp.asarray(2 * numpy.pi * numpy.arange(ns) / ns)
    Xs, gs = _xg_at(x, gam, m, xp.broadcast_to(scan[:, None, None],
                                               (ns,) + shape), xp)
    F = Xs * Xs / gs
    local = (F >= xp.roll(F, 1, axis=0)) & (F > xp.roll(F, -1, axis=0))
    Fl = xp.where(local, F, -numpy.inf)
    # the four highest maxima (a dominant m = 2 pattern has four); where a
    # fifth exists the trapezoid rule takes the cell (use below)
    order = xp.argsort(-Fl, axis=0)
    best = order[:4]                                                # (4,n,s)
    found = xp.take_along_axis(Fl, best, axis=0) > -numpy.inf
    fifth = xp.take_along_axis(Fl, order[4:5], axis=0)[0] > -numpy.inf
    phi = scan[best]
    for _ in range(int(newton) + 1):
        X, g = _xg_at(x, gam, m, phi, xp)
        X1, g1 = _xg_at(x, gam, m, phi, xp, 1)
        X2, g2 = _xg_at(x, gam, m, phi, xp, 2)
        d1 = 2 * X1 / X - g1 / g
        d2 = 2 * (X2 * X - X1 * X1) / (X * X) - (g2 * g - g1 * g1) / (g * g)
        if _ < newton:
            step = xp.where(d2 < 0, -d1 / d2, 0.)
            phi = phi + xp.clip(step, -0.05, 0.05)
    c = -0.5 * (X * X / g) * d2
    with numpy.errstate(divide='ignore', invalid='ignore'):
        est = xp.where(c > 0, 1 / xp.sqrt(c), numpy.inf)
    delta = xp.clip(0.5 * est, 1e-3, 0.05)
    L0 = amplitude(X, g)
    Xp, gp = _xg_at(x, gam, m, phi + delta, xp)
    Xm, gm = _xg_at(x, gam, m, phi - delta, xp)
    cj = -(amplitude(Xp, gp) + amplitude(Xm, gm) - 2 * L0) / (delta * delta)
    with numpy.errstate(divide='ignore', invalid='ignore'):
        sigma = xp.where(cj > 0, 1 / xp.sqrt(cj), numpy.inf)
    # Gauss-Hermite where every peak found is narrow and none overlaps
    # another (their spans of 5 sigma apart)
    use = xp.all(xp.where(found, sigma < narrow, True), axis=0) & ~fifth
    ordered = xp.sort(xp.where(found, xp.mod(phi, 2 * numpy.pi), xp.nan),
                      axis=0)
    gaps = xp.diff(ordered, axis=0)
    span = 10 * xp.max(xp.where(found, sigma, 0.), axis=0)
    use = use & xp.all(xp.where(xp.isfinite(gaps), gaps > span, True),
                       axis=0)
    sigma = xp.where(xp.isfinite(sigma), sigma, 1.)
    t, w = _gh(n_gh)
    t, w = xp.asarray(t), xp.asarray(w)
    phi_g = (phi[:, None] + numpy.sqrt(2) * sigma[:, None]
             * t[None, :, None, None]).reshape((-1,) + shape)
    lw_g = (xp.log(w)[None, :, None, None] + (t * t)[None, :, None, None]
            + xp.log(numpy.sqrt(2) * sigma)[:, None]
            - numpy.log(2 * numpy.pi)).reshape((-1,) + shape)
    lw_g = xp.where(use[None] & xp.repeat(found, len(t), axis=0), lw_g,
                    -numpy.inf)
    # the narrowest peak found, for the trapezoid rule where Gauss-Hermite
    # is not used
    sigma_min = xp.min(xp.where(found, sigma, numpy.inf), axis=0)
    return phi_g, lw_g, use, sigma_min


def _peaks_marginal(x, gam, m, xp, amplitude, logw, n_gh, n_uniform,
                    largest=4096, per_node=False):
    """``ln sum_nodes w int dphi / 2 pi J`` per sample: Gauss-Hermite
    about the peaks where :func:`_phi_rule` allows it; elsewhere the
    trapezoid rule with as many points (a power of two from ``n_uniform``
    to ``largest``) as resolve the narrowest peak found at that node and
    sample, ``2 pi / N <= 0.8 sigma``."""
    phi, lw, use, sigma_min = _phi_rule(x, gam, m, xp, amplitude, n_gh=n_gh,
                                        n_uniform=n_uniform)
    X, g = _xg_at(x, gam, m, phi, xp)
    terms = xp.where(xp.isfinite(lw), amplitude(X, g) + lw, -numpy.inf)
    top = xp.max(terms, axis=0)
    with numpy.errstate(invalid='ignore', divide='ignore'):
        node = top + xp.log(xp.sum(xp.exp(terms - top), axis=0))   # (n, s)
    node = xp.where(use, node, -numpy.inf)
    need = xp.where(xp.isfinite(sigma_min),
                    2 * numpy.pi / (0.8 * sigma_min), n_uniform)
    k = numpy.arange(int(numpy.log2(n_uniform)),
                     int(numpy.log2(largest)) + 1)
    sizes = 2 ** k
    rest = ~use
    for i, n in enumerate(sizes):
        sel = rest & (need <= n) if i < len(sizes) - 1 else rest
        rest = rest & ~sel
        idx = xp.nonzero(sel)
        if not len(idx[0]):
            continue
        xs = x[:, idx[0], idx[1]][..., None]                 # (P, E, 1)
        gs = gam[:, :, idx[0], idx[1]][..., None]
        uni = xp.asarray(2 * numpy.pi * numpy.arange(int(n)) / int(n))
        ph = xp.broadcast_to(uni[:, None, None], (len(uni),) + xs.shape[1:])
        Xu, gu = _xg_at(xs, gs, m, ph, xp)
        L = amplitude(Xu, gu)
        t = xp.max(L, axis=0)
        node[idx] = (t + xp.log(xp.mean(xp.exp(L - t), axis=0)))[:, 0]
    if per_node:
        return node
    terms = node + logw[:, None]
    top = xp.max(terms, axis=0)
    return top + xp.log(xp.sum(xp.exp(terms - top), axis=0))


def hm_log_bayes(z, gram, keys, prior, nodes, n_l0=32, n_phi=256,
                 order=64, xp=numpy, table=None, phi_rule='uniform',
                 sample_chunk=16, n_gh=10, n_uniform=64):
    """``ln B`` per sample, marginalised over distance, inclination and
    polarisation (``nodes``, an :class:`pycbc.filter.bstatistic.
    OrientationNodes`), coalescence phase and mean anomaly (uniform ``n_l0``
    points); ``z (K, 2, *samples)``, ``gram (K, 2, K, 2, *samples or
    none)``; ``keys`` the modes' ``(l, m, n)``.

    ``phi_rule='uniform'``: the trapezoid rule on ``n_phi`` points over ``[0,
    2 pi)``, exact (exponentially convergent) once they resolve the
    narrowest peak, about ``1 / (2 rho)``: the reference. ``'peaks'``:
    :func:`_phi_rule`, Gauss-Hermite about the four peaks where they are
    narrow, the trapezoid rule on ``n_uniform`` points where they are not.

    ``table``: an :class:`pycbc.filter.bstatistic.AmplitudeTable` of
    ``prior`` with ``kernel='cosh'`` for the amplitude integrals.

    Returns ``(log_b, log_b_by_angle (n_l0, *samples), angles)`` as
    :func:`pycbc.filter.mean_anomaly.mean_anomaly_log_bayes` does."""
    from pycbc.filter.bstatistic import log_amplitude_marginal
    if table is not None and getattr(table, 'kernel', 'i0') != 'cosh':
        raise ValueError('the amplitude table must have kernel cosh')

    def amplitude(X, g):
        if table is None:
            return log_amplitude_marginal(xp.abs(X), g, prior, order=order,
                                          xp=xp, kernel='cosh')
        return table.log_marginal(xp.abs(X), g, xp=xp)
    patterns = PatternSet(keys)
    shape = z.shape[2:]
    zg, Gg, angles = hm_components(z, gram, patterns, n_l0, xp=xp)
    S = int(numpy.prod(shape)) if len(shape) else 1
    zg = zg.reshape(zg.shape[:3] + (S,))
    if Gg.ndim == 5:
        Gg = Gg[..., None]
    Gg = xp.broadcast_to(Gg.reshape(Gg.shape[:5] + (-1,)),
                         Gg.shape[:5] + (S,))
    W = xp.asarray(patterns.weights(nodes.cos_iota, nodes.psi))
    logw = xp.log(xp.asarray(nodes.weights))
    out = xp.empty((len(angles), S))
    uni = xp.asarray(2 * numpy.pi * numpy.arange(int(n_phi)) / int(n_phi))
    for j in range(len(angles)):
        for c0 in range(0, S, int(sample_chunk)):
            c1 = min(S, c0 + int(sample_chunk))
            # one mean anomaly and a chunk of samples at a time: the
            # pattern Grams per node are (P, P, nodes, samples)
            x, gam = _node_terms(zg[j:j + 1, ..., c0:c1],
                                 Gg[j:j + 1, ..., c0:c1], W, xp)
            x, gam = x[0], gam[0]                         # (P,n,s) (P,P,n,s)
            if phi_rule == 'uniform':
                phi = xp.broadcast_to(uni[:, None, None],
                                      (len(uni),) + x.shape[1:])
                lw = xp.full(phi.shape, -numpy.log(len(uni)))
            elif phi_rule == 'peaks':
                out[j, c0:c1] = _peaks_marginal(
                    x, gam, patterns.m, xp, amplitude, logw, n_gh, n_uniform)
                continue
            else:
                raise ValueError('phi_rule must be uniform or peaks')
            X, g = _xg_at(x, gam, patterns.m, phi, xp)
            terms = amplitude(X, g) + lw + logw[None, :, None]
            terms = xp.where(xp.isfinite(lw), terms, -numpy.inf)
            top = xp.max(xp.max(terms, axis=0), axis=0)
            out[j, c0:c1] = top + xp.log(xp.sum(xp.exp(terms - top),
                                                axis=(0, 1)))
    top = xp.max(out, axis=0)
    log_b = top + xp.log(xp.mean(xp.exp(out - top), axis=0))
    return (log_b.reshape(shape), out.reshape((len(angles),) + shape),
            angles)


def phi_quadrature(z, gram, keys, prior, nodes, n_l0=32, n_phis=(64, 128,
                   256, 512), order=64, xp=numpy):
    """``ln B`` at each ``n_phi`` of ``n_phis``: its convergence."""
    return {int(n): hm_log_bayes(z, gram, keys, prior, nodes, n_l0=n_l0,
                                 n_phi=n, order=order, xp=xp)[0]
            for n in n_phis}


def draw_hm_extrinsics(z, gram, keys, prior, nodes, rng, n_l0=32,
                       n_phi=512, order=64):
    """Draw ``(l0, iota, psi, phi, h)`` from the posterior of one sample
    (``z (K, 2)``, ``gram (K, 2, K, 2)``): ``l0`` and ``(iota, psi, phi)`` by
    their grid weights, then jittered uniformly within the cell and
    importance-resampled against the exact density over 256 such draws,
    then ``h`` by inverting its conditional CDF. ``psi`` comes back in
    ``[0, pi)`` (the sign of ``h`` becomes a quarter turn of ``psi``) and
    ``phi`` in ``[0, 2 pi)``."""
    from pycbc.filter.bstatistic import (log_amplitude_marginal,
                                         draw_amplitude)
    patterns = PatternSet(keys)
    zg, Gg, angles = hm_components(z[..., None], gram[..., None], patterns,
                                   n_l0)
    W = patterns.weights(nodes.cos_iota, nodes.psi)
    x, gam = _node_terms(zg, Gg, W, numpy)
    phis = 2 * numpy.pi * numpy.arange(int(n_phi)) / int(n_phi)
    X, g = _phase_sums(x, gam, patterns.m, phis, numpy)         # (F,j,n,1)
    lj = log_amplitude_marginal(numpy.abs(X), g, prior, order=order,
                                kernel='cosh')[..., 0]
    terms = lj + numpy.log(nodes.weights)[None, None, :]
    p = numpy.exp(terms - terms.max())
    p = p / p.sum()
    pick = rng.choice(p.size, p=p.ravel())
    fi, jl, ni = numpy.unravel_index(pick, p.shape)
    step_l0 = 2 * numpy.pi / len(angles)
    step_phi = 2 * numpy.pi / len(phis)
    # jitter in the cell, importance-resampled against the exact density
    n = 256
    l0 = angles[jl] + (rng.random(n) - 0.5) * step_l0
    phi = phis[fi] + (rng.random(n) - 0.5) * step_phi
    ci = nodes.cos_iota[ni]
    k = numpy.searchsorted(nodes._cos_edges, ci) - 1
    lo, hi = nodes._cos_edges[k], nodes._cos_edges[k + 1]
    cos_iota = lo + rng.random(n) * (hi - lo)
    psi = nodes.psi[ni] + (rng.random(n) - 0.5) * nodes._psi_step
    Xs, gs = _exact_xg(z, gram, patterns, l0, numpy.arccos(cos_iota), psi,
                       phi)
    logd = log_amplitude_marginal(numpy.abs(Xs), gs, prior, order=order,
                                  kernel='cosh')
    # the proposal's density within the cell is uniform; its cell weight is
    # constant: the importance weight is the exact density
    wts = numpy.exp(logd - logd.max())
    i = rng.choice(n, p=wts / wts.sum())
    h = draw_amplitude(abs(Xs[i]), gs[i], prior, rng, kernel='cosh')
    psi_i = float(psi[i])
    # the sign of h: cosh(h X) = (exp(h X) + exp(-h X)) / 2; a negative X
    # half is a quarter turn of psi
    # P(+) = exp(h X) / (exp(h X) + exp(-h X))
    if rng.random() >= 1 / (1 + numpy.exp(-2 * h * Xs[i])):
        psi_i += numpy.pi / 2
    return dict(mean_anomaly=float(numpy.mod(l0[i], 2 * numpy.pi)),
                inclination=float(numpy.arccos(cos_iota[i])),
                polarization=float(numpy.mod(psi_i, numpy.pi)),
                coa_phase=float(numpy.mod(phi[i], 2 * numpy.pi)),
                amplitude=float(h))


def _exact_xg(z, gram, patterns, l0, iota, psi, phi):
    """``X`` and ``g`` at arbitrary angles (one sample, ``z (K, 2)``)."""
    K = len(patterns.keys)
    w = numpy.array([pattern_weights(*patterns.keys[k][:2], iota, psi)
                     for k in range(K)])                      # (K, 2, n)
    m = numpy.array([k[1] for k in patterns.keys])
    ph = numpy.exp(1j * (numpy.outer(m, phi)
                         + numpy.outer(patterns.offsets, l0)))  # (K, n)
    v = w * ph[:, None, :]
    X = numpy.real(numpy.einsum('kan,ka->n', numpy.conj(v), z))
    g = numpy.real(numpy.einsum('kan,kalb,lbn->n', numpy.conj(v), gram, v))
    return X, g


def hm_draw_batch(z, gram, keys, prior, nodes, rngs, n_l0=32, table=None,
                  xp=numpy, chunk=16, n_phi_draw=4096, proposals=64,
                  n_gh=10, n_uniform=64, prune_margin=None, table22=None):
    """For each of ``M`` samples (``z (K, 2, M)``, ``gram (K, 2, K, 2, M)``,
    one ``numpy.random.Generator`` each in ``rngs``): ``ln B`` marginalised
    as :func:`hm_log_bayes` (``phi_rule='peaks'``), and a draw of the
    marginalised angles and amplitude from their posterior.

    The draw: mean anomaly and orientation node by the per-node marginals;
    ``phi`` on ``n_phi_draw`` uniform points at that node; then ``proposals``
    jitters of all four angles within their cells, importance-resampled
    against the exact density; then ``h`` (:func:`pycbc.filter.bstatistic.
    draw_amplitude`, ``cosh`` kernel) and its sign, a negative one turning
    ``psi`` a quarter. Returns ``(log_b (M,), draws)``, each draw a dict of
    ``mean_anomaly``, ``inclination``, ``polarization`` (``[0, pi)``),
    ``coa_phase`` (``phi``, ``[0, 2 pi)``), ``amplitude`` and ``loglr`` (the
    log likelihood ratio at the drawn values).

    ``prune_margin``: the full ``phi`` quadrature only at the (mean anomaly,
    node) cells of a sample whose ``m = 2`` modes' own marginal (``I0``,
    exact for them: ``table22``, an ``i0`` AmplitudeTable of ``prior``) is
    within the margin of the sample's largest; the rest are left out (their
    weight below ``exp(-margin)``, provided the other modes move a cell's
    ``ln J`` by less than the margin, as for stBBHs, where they move ``ln L``
    by about 0.1). Without it every cell."""
    from pycbc.filter.bstatistic import (log_amplitude_marginal,
                                         draw_amplitude)

    def host(a):
        return a.get() if hasattr(a, 'get') else numpy.asarray(a)

    def amplitude(X, g):
        if table is None:
            return log_amplitude_marginal(xp.abs(X), g, prior, xp=xp,
                                          kernel='cosh')
        return table.log_marginal(xp.abs(X), g, xp=xp)
    patterns = PatternSet(keys)
    M = z.shape[-1]
    if gram.ndim == 4:
        gram = gram[..., None]
    gram = xp.broadcast_to(gram, gram.shape[:4] + (M,))
    zg, Gg, angles = hm_components(z, gram, patterns, n_l0, xp=xp)
    W = xp.asarray(patterns.weights(nodes.cos_iota, nodes.psi))
    logw = xp.log(xp.asarray(nodes.weights))
    L = numpy.full((len(angles), len(nodes.weights), M), -numpy.inf)
    keep = None
    if prune_margin is not None:
        # the m = 2 modes' marginal per cell: phi integrates to I0 for them
        two = numpy.flatnonzero(patterns.m == 2)
        L22 = numpy.empty_like(L)
        for j in range(len(angles)):
            for c0 in range(0, M, int(chunk) * 8):
                c1 = min(M, c0 + int(chunk) * 8)
                x, gam = _node_terms(zg[j:j + 1, ..., c0:c1],
                                     Gg[j:j + 1, ..., c0:c1], W, xp)
                x2 = xp.sum(x[0][two], axis=0)
                g2 = xp.real(xp.sum(gam[0][two][:, two], axis=(0, 1)))
                v = table22.log_marginal(xp.abs(x2), g2, xp=xp) \
                    if table22 is not None else log_amplitude_marginal(
                        xp.abs(x2), g2, prior, xp=xp)
                L22[j, :, c0:c1] = host(v)
        t22 = L22 + host(logw)[None, :, None]
        keep = t22 >= t22.max(axis=(0, 1))[None, None] - float(prune_margin)
    for j in range(len(angles)):
        for c0 in range(0, M, int(chunk)):
            c1 = min(M, c0 + int(chunk))
            x, gam = _node_terms(zg[j:j + 1, ..., c0:c1],
                                 Gg[j:j + 1, ..., c0:c1], W, xp)
            x, gam = x[0], gam[0]
            if keep is None:
                L[j, :, c0:c1] = host(_peaks_marginal(
                    x, gam, patterns.m, xp, amplitude, logw, n_gh, n_uniform,
                    per_node=True))
                continue
            ni, si = numpy.nonzero(keep[j, :, c0:c1])
            if not len(ni):
                continue
            for e0 in range(0, len(ni), 16 * len(nodes.weights) // 64):
                e1 = min(len(ni), e0 + 16 * len(nodes.weights) // 64)
                a, b = xp.asarray(ni[e0:e1]), xp.asarray(si[e0:e1])
                xs = x[:, a, b][:, :, None]
                gs = gam[:, :, a, b][..., None]
                v = _peaks_marginal(xs, gs, patterns.m, xp, amplitude,
                                    logw[a], n_gh, n_uniform, per_node=True)
                L[j, ni[e0:e1], c0 + si[e0:e1]] = host(v)[:, 0]
    terms = L + host(logw)[None, :, None]
    flat = terms.reshape(-1, M)
    top = flat.max(axis=0)
    log_b = top + numpy.log(numpy.sum(numpy.exp(flat - top), axis=0)) \
        - numpy.log(len(angles))
    zh, gh = host(z), host(gram)
    step_l0 = 2 * numpy.pi / len(angles)
    phis = 2 * numpy.pi * numpy.arange(int(n_phi_draw)) / int(n_phi_draw)
    step_phi = phis[1]
    Wh = host(W)
    m = patterns.m
    draws = []
    for i in range(M):
        rng = rngs[i]
        w = numpy.exp(flat[:, i] - flat[:, i].max())
        pick = rng.choice(len(w), p=w / w.sum())
        jl, ni = divmod(pick, len(nodes.weights))
        # phi on a fine grid at that node
        e = numpy.exp(-1j * patterns.offsets * angles[jl])
        zgi = numpy.zeros((len(patterns.patterns), 2), complex)
        Ggi = numpy.zeros((len(patterns.patterns), 2,
                           len(patterns.patterns), 2), complex)
        for k, gk in enumerate(patterns.group):
            zgi[gk] += e[k] * zh[k, :, i]
            for k2, gk2 in enumerate(patterns.group):
                Ggi[gk, :, gk2, :] += numpy.conj(e[k]) * e[k2] \
                    * gh[k, :, k2, :, i]
        wn = Wh[:, :, ni]                                   # (P, 2)
        xn = numpy.einsum('ga,ga->g', numpy.conj(wn), zgi)
        gamn = numpy.einsum('ga,gahb,hb->gh', numpy.conj(wn), Ggi, wn)
        ex = numpy.exp(-1j * numpy.outer(phis, m))          # (F, P)
        Xf = numpy.real(ex @ xn)
        gf = numpy.real(numpy.einsum('fg,gh,fh->f', ex, gamn,
                                     numpy.conj(ex)))
        lf = log_amplitude_marginal(numpy.abs(Xf), gf, prior,
                                    kernel='cosh') \
            if table is None else host(table.log_marginal(
                xp.asarray(numpy.abs(Xf)), xp.asarray(gf), xp=xp))
        wf = numpy.exp(lf - lf.max())
        fi = rng.choice(len(wf), p=wf / wf.sum())
        # jitter within the cells, importance-resampled
        n = int(proposals)
        l0 = angles[jl] + (rng.random(n) - 0.5) * step_l0
        phi = phis[fi] + (rng.random(n) - 0.5) * step_phi
        ci = nodes.cos_iota[ni]
        kc = numpy.searchsorted(nodes._cos_edges, ci) - 1
        lo, hi = nodes._cos_edges[kc], nodes._cos_edges[kc + 1]
        cos_iota = lo + rng.random(n) * (hi - lo)
        psi = nodes.psi[ni] + (rng.random(n) - 0.5) * nodes._psi_step
        Xs, gs = _exact_xg(zh[..., i], gh[..., i], patterns, l0,
                           numpy.arccos(cos_iota), psi, phi)
        ld = log_amplitude_marginal(numpy.abs(Xs), gs, prior, kernel='cosh')
        ws = numpy.exp(ld - ld.max())
        k = rng.choice(n, p=ws / ws.sum())
        h = draw_amplitude(abs(Xs[k]), gs[k], prior, rng, kernel='cosh')
        sign = 1.0 if rng.random() < 1 / (1 + numpy.exp(-2 * h * abs(Xs[k]))) \
            else -1.0
        # the sign that the draw took: +|X| means psi as is when X > 0
        flip = (sign * numpy.sign(Xs[k])) < 0
        psi_k = float(psi[k] + (numpy.pi / 2 if flip else 0.))
        loglr = sign * h * abs(Xs[k]) - 0.5 * h * h * gs[k]
        draws.append(dict(mean_anomaly=float(numpy.mod(l0[k], 2 * numpy.pi)),
                          inclination=float(numpy.arccos(cos_iota[k])),
                          polarization=float(numpy.mod(psi_k, numpy.pi)),
                          coa_phase=float(numpy.mod(phi[k], 2 * numpy.pi)),
                          amplitude=float(h), loglr=float(loglr)))
    return log_b, draws


def log_likelihood_batch(z, gram, keys, h, l0, iota, psi, phi):
    """``ln L = h X - h^2 g / 2`` of each of ``M`` samples at its own angles
    and amplitude (arrays (M,)): ``z (K, 2, M)``, ``gram (K, 2, K, 2, M)``
    (numpy)."""
    K = len(keys)
    v = numpy.empty((K, 2, len(h)), complex)
    for k, (l, m, n) in enumerate(keys):
        v[k] = pattern_weights(l, m, iota, psi) \
            * numpy.exp(1j * (m * phi + (n - m) * l0))[None]
    X = numpy.real(numpy.einsum('kam,kam->m', numpy.conj(v), z))
    g = numpy.real(numpy.einsum('kam,kalbm,lbm->m', numpy.conj(v), gram, v))
    return h * X - 0.5 * h * h * g
