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
"""The Bayes factor of a two-basis template, marginalised over amplitude.

A signal ``s = h exp(i phi) w(iota, psi)^T y`` has amplitude ``h``, phase
``phi`` and orientation weights ``w`` (see
:func:`pycbc.tmpltbank.fstatistic_metric.orientation_weights`). With the
correlations ``z_a = <y_a, d>`` and Gram matrix ``G_ab = <y_a, y_b>``, in
units where the noise has unit variance in the inner product,

    ln L = h Re(exp(-i phi) X) - h^2 g / 2,   X = w^H z,  g = w^H G w,

and the phase, uniform on a circle, integrates to ``ln I0(h |X|)``. This
module integrates the amplitude against a prior on the distance
``D = scale / h``, with no ceiling on the amplitude, and draws ``h`` and
``phi`` from their conditional posteriors for de-marginalisation.
"""

import numpy
from scipy.special import i0e, roots_legendre

# widths of a Gaussian peak, decay lengths of an exponential edge
_PEAK_WIDTHS, _EDGE_LENGTHS = 10.0, 30.0


def log_i0(x, xp=numpy):
    """``ln I0(x)`` without overflow, for ``x >= 0``: from the scaled Bessel
    function up to 700 and its asymptotic series, good to 1e-12, above
    (cupy's scaled Bessel function overflows beyond 709)."""
    if xp is numpy:
        scaled = i0e
    else:
        from cupyx.scipy.special import i0e as scaled
    small = x + xp.log(scaled(xp.minimum(x, 700.0)))
    big = xp.maximum(x, 700.0)
    series = 1 + 1 / (8 * big) + 9 / (128 * big ** 2) \
        + 225 / (3072 * big ** 3)
    large = big - 0.5 * xp.log(2 * numpy.pi * big) + xp.log(series)
    return xp.where(x <= 700.0, small, large)


class DistancePowerLaw(object):
    """A prior ``p(D) ~ D^power`` on ``[minimum, maximum]``, power 2 being
    uniform in Euclidean volume, as a prior on ``h = scale / D``.

    Parameters
    ----------
    minimum, maximum : float
        Distance bounds, in the units of ``scale``.
    power : float
        Exponent of the distance density.
    scale : float
        The amplitude at unit distance; 1 for templates generated at unit
        distance.
    """
    def __init__(self, minimum, maximum, power=2.0, scale=1.0):
        if not 0 < minimum < maximum:
            raise ValueError("need 0 < minimum < maximum")
        self.minimum, self.maximum = float(minimum), float(maximum)
        self.power, self.scale = float(power), float(scale)
        self.hmin, self.hmax = scale / maximum, scale / minimum
        # p(h) = c h^-(power + 2) on [hmin, hmax]
        n = self.power + 1
        if abs(n) < 1e-12:
            norm = numpy.log(self.hmax / self.hmin)
        else:
            norm = (self.hmin ** -n - self.hmax ** -n) / n
        self._log_norm = -numpy.log(norm)

    def log_pdf(self, h, xp=numpy):
        """``ln p(h)`` on the support."""
        return self._log_norm - (self.power + 2) * xp.log(h)


def _segments(x, g, prior, xp):
    """The interval holding the integrand's maximum on the prior's support:
    ten widths about the likelihood peak ``x / g``, or, when that lies off
    the support, thirty decay lengths ``1 / |x - g h|`` from the nearer
    end."""
    centre = xp.clip(x / g, prior.hmin, prior.hmax)
    width = 1 / xp.maximum(xp.sqrt(g) / _PEAK_WIDTHS,
                           xp.abs(x - g * centre) / _EDGE_LENGTHS)
    low = xp.clip(centre - width, prior.hmin, prior.hmax)
    high = xp.clip(centre + width, prior.hmin, prior.hmax)
    return low, high


def _log_nodes(lo, hi, order, xp):
    """Gauss-Legendre nodes and weights for ``dh`` on ``[lo, hi]`` in
    ``ln h``."""
    t, w = roots_legendre(order)
    t, w = xp.asarray(t), xp.asarray(w)
    a, b = xp.log(lo)[..., None], xp.log(hi)[..., None]
    h = xp.exp(0.5 * (a + b) + 0.5 * (b - a) * t)
    return h, 0.5 * (b - a) * w * h


def log_amplitude_integrand(h, x, g, prior, xp=numpy):
    """``ln [p(h) I0(h x) exp(-h^2 g / 2)]``."""
    return prior.log_pdf(h, xp=xp) + log_i0(h * x, xp=xp) - 0.5 * h * h * g


def log_amplitude_marginal(x, g, prior, order=64, chunk=65536, xp=numpy):
    """``ln int p(h) I0(h x) exp(-h^2 g / 2) dh`` over the prior.

    The prior's support is split into the interval about the integrand's
    maximum (ten widths of a peak, or thirty decay lengths of an edge),
    and the interval below it, where a steep prior can dominate a weak
    signal; both are integrated in ``ln h``, in which a power-law prior is
    smooth. Above the first interval the integrand is below ``exp(-50)`` of
    its maximum and is dropped.

    Parameters
    ----------
    x : array
        ``|X| = |w^H z|``, non-negative.
    g : array
        ``w^H G w``, positive, broadcastable with ``x``.
    prior : DistancePowerLaw
        The amplitude prior.
    order : int
        Gauss-Legendre nodes per interval.
    chunk : int
        Values integrated at a time, which bounds the memory to about
        ``chunk * 4 * order`` doubles whatever the size of ``x``.
    """
    x = xp.asarray(x, dtype=xp.float64)
    g = xp.asarray(g, dtype=xp.float64)
    x, g = xp.broadcast_arrays(x, g)
    shape = x.shape
    x, g = x.ravel(), g.ravel()
    out = xp.empty(x.shape, dtype=xp.float64)
    for k in range(0, x.size, chunk):
        out[k:k + chunk] = _log_amplitude_marginal(
            x[k:k + chunk], g[k:k + chunk], prior, order, xp)
    return out.reshape(shape)


def _log_amplitude_marginal(x, g, prior, order, xp):
    low, high = _segments(x, g, prior, xp)
    floor = xp.full(x.shape, prior.hmin)
    parts = []
    for lo, hi in ((floor, low), (low, high)):
        h, w = _log_nodes(lo, xp.maximum(hi, lo), order, xp)
        with numpy.errstate(divide='ignore'):
            f = log_amplitude_integrand(h, x[..., None], g[..., None],
                                        prior, xp=xp) + xp.log(w)
        f = xp.where(w > 0, f, -numpy.inf)
        parts.append(f)
    terms = xp.concatenate(parts, axis=-1)
    top = xp.max(terms, axis=-1, keepdims=True)
    return (top[..., 0]
            + xp.log(xp.sum(xp.exp(terms - top), axis=-1)))


def draw_amplitude(x, g, prior, rng, order=512):
    """Draw ``h`` from ``p(h | x, g)`` by inverting the CDF of the
    integrand on a fine logarithmic mesh of both intervals; scalar ``x``
    and ``g``."""
    low, high = _segments(numpy.array(x, float), numpy.array(g, float),
                          prior, numpy)
    tail = numpy.geomspace(prior.hmin, max(low, prior.hmin), order)
    peak = numpy.geomspace(low, max(high, low), order)
    h = numpy.unique(numpy.concatenate([tail, peak]))
    f = log_amplitude_integrand(h, x, g, prior)
    density = numpy.exp(f - f.max())
    cdf = numpy.concatenate([[0], numpy.cumsum(
        0.5 * (density[1:] + density[:-1]) * numpy.diff(h))])
    return float(numpy.interp(rng.random() * cdf[-1], cdf, h))


def draw_phase(h, X, rng):
    """Draw ``phi`` from ``p(phi | h, X) ~ exp(h Re(exp(-i phi) X))``, a
    von Mises distribution about ``arg X``, in ``[0, 2 pi)``."""
    phi = rng.vonmises(numpy.angle(X), h * abs(X))
    return float(numpy.mod(phi, 2 * numpy.pi))


class OrientationNodes(object):
    """Quadrature over an isotropic orientation: Gauss-Legendre in
    ``cos(iota)`` on ``[-1, 1]`` and the midpoint rule in ``psi`` over its
    period, ``pi / 2`` once the phase is marginalised (a quarter turn of
    ``psi`` only flips the sign of the weights). The weights sum to one.
    """
    def __init__(self, n_cos=16, n_psi=8):
        from pycbc.tmpltbank.fstatistic_metric import orientation_weights
        t, w = roots_legendre(int(n_cos))
        psi = (numpy.arange(int(n_psi)) + 0.5) * (numpy.pi / 2) / n_psi
        cos_iota, psi = numpy.meshgrid(t, psi, indexing='ij')
        self.cos_iota, self.psi = cos_iota.ravel(), psi.ravel()
        self.weights = numpy.repeat(0.5 * w, int(n_psi)) / n_psi
        self.vectors = orientation_weights(numpy.arccos(self.cos_iota),
                                           self.psi).T
        self.n_cos, self.n_psi = int(n_cos), int(n_psi)
        # cell edges for drawing inside a node's share of the sphere
        edges = numpy.concatenate([[-1], numpy.cumsum(w) - 1])
        self._cos_edges = edges
        self._psi_step = (numpy.pi / 2) / n_psi


def _project(z, g00, g11, g01, vectors, xp):
    """``|w^H z|`` and ``w^H G w`` for every sample and orientation."""
    v = xp.asarray(vectors)
    x = xp.abs(xp.conj(v[:, 0])[None, :] * z[0][:, None]
               + xp.conj(v[:, 1])[None, :] * z[1][:, None])
    g = (xp.abs(v[:, 0]) ** 2 * g00[:, None]
         + xp.abs(v[:, 1]) ** 2 * g11[:, None]
         + 2 * xp.real(xp.conj(v[:, 0]) * v[:, 1] * g01[:, None]))
    return x, g


def log_orientation_marginal(z, g00, g11, g01, prior, nodes, order=64,
                             margin=None, xp=numpy):
    """``ln B`` at each sample, marginalised over phase, amplitude and an
    isotropic orientation: ``ln sum_n weight_n B(w_n)``.

    ``z`` has shape ``(2, n)`` and the Gram elements ``(n,)``; see
    :func:`pycbc.filter.fstatistic.twof` for their meaning.

    With ``margin``, terms that cannot matter to the sum over all samples
    and nodes are skipped. A normalised amplitude prior gives
    ``ln B(w) <= max (h x - h^2 g / 2)`` over the prior's support, since
    ``ln I0(y) <= y``; the term with the largest bound is evaluated
    exactly, which bounds the total from below, and
    every term whose bound falls ``margin`` below that, less the log of
    the number of terms, is dropped. The dropped terms together are less
    than ``exp(-margin)`` of the total, so the sum over samples (an
    arrival-time marginal) keeps that relative accuracy; a sample all of
    whose terms are dropped gets ``-inf``.
    """
    x, g = _project(z, g00, g11, g01, nodes.vectors, xp)
    logw = xp.log(xp.asarray(nodes.weights))[None, :]
    if margin is None:
        terms = log_amplitude_marginal(x, g, prior, order=order, xp=xp) \
            + logw
    else:
        with numpy.errstate(divide='ignore', invalid='ignore'):
            h = xp.clip(x / g, prior.hmin, prior.hmax)
            bound = xp.where(g > 0, h * x - 0.5 * h * h * g, -numpy.inf) \
                + logw
        best = int(xp.argmax(bound))
        i, j = divmod(best, bound.shape[1])
        lower = float(log_amplitude_marginal(x[i, j], g[i, j], prior,
                                             order=order, xp=xp)) \
            + float(logw[0, j])
        keep = bound >= lower - margin - numpy.log(bound.size)
        terms = xp.full(bound.shape, -numpy.inf)
        terms[keep] = log_amplitude_marginal(x[keep], g[keep], prior,
                                             order=order, xp=xp) \
            + xp.broadcast_to(logw, bound.shape)[keep]
    top = xp.max(terms, axis=1, keepdims=True)
    with numpy.errstate(invalid='ignore'):
        out = top[:, 0] + xp.log(xp.sum(xp.exp(terms - top), axis=1))
    return xp.where(xp.isfinite(top[:, 0]), out, -numpy.inf)


def draw_orientation(z, gram, prior, nodes, rng, proposals=256):
    """Draw ``(cos(iota), psi)`` from their posterior given one sample's
    ``z`` (shape ``(2,)``) and Gram matrix ``(g00, g11, g01)``.

    Proposals are drawn uniformly inside the quadrature cells, a cell being
    chosen in proportion to its node's posterior mass, and one is kept with
    probability proportional to its posterior over its proposal density
    (sampling-importance resampling), so the draw is continuous and its
    error falls with ``proposals``.
    """
    from pycbc.tmpltbank.fstatistic_metric import orientation_weights
    g00, g11, g01 = (numpy.atleast_1d(numpy.asarray(v)) for v in gram)
    zz = numpy.asarray(z).reshape(2, 1)
    node = log_orientation_terms(zz, g00, g11, g01, prior, nodes)
    mass = numpy.exp(node - node.max())
    mass /= mass.sum()
    cells = rng.choice(len(mass), size=proposals, p=mass)
    i, j = numpy.divmod(cells, nodes.n_psi)
    lo, hi = nodes._cos_edges[i], nodes._cos_edges[i + 1]
    cos_iota = lo + (hi - lo) * rng.random(proposals)
    psi = (j + rng.random(proposals)) * nodes._psi_step
    vectors = orientation_weights(numpy.arccos(cos_iota), psi).T
    x, g = _project(zz, g00, g11, g01, vectors, numpy)
    target = log_amplitude_marginal(x[0], g[0], prior)
    # proposal density: the cell's mass over its area
    area = (hi - lo) * nodes._psi_step
    log_w = target - numpy.log(mass[cells] / area)
    keep = rng.choice(proposals, p=numpy.exp(log_w - log_w.max())
                      / numpy.sum(numpy.exp(log_w - log_w.max())))
    return float(cos_iota[keep]), float(psi[keep])


def log_orientation_terms(z, g00, g11, g01, prior, nodes, order=64):
    """``ln (weight_n B(w_n))`` of every node for one sample, the terms
    whose log-sum is :func:`log_orientation_marginal`."""
    x, g = _project(z, g00, g11, g01, nodes.vectors, numpy)
    return (log_amplitude_marginal(x[0], g[0], prior, order=order)
            + numpy.log(nodes.weights))


def converged_orientation_marginal(z, g00, g11, g01, prior, n_cos=16,
                                   n_psi=8, tolerance=1e-3, max_doublings=4,
                                   xp=numpy):
    """:func:`log_orientation_marginal`, doubling both node counts until
    no sample changes by more than ``tolerance``. Returns the values and
    the nodes reached; raises if ``max_doublings`` do not converge.

    The posterior over orientation narrows as the signal strengthens, so a
    fixed grid that suffices at low SNR does not at high SNR.
    """
    nodes = OrientationNodes(n_cos, n_psi)
    value = log_orientation_marginal(z, g00, g11, g01, prior, nodes, xp=xp)
    for _ in range(max_doublings):
        finer = OrientationNodes(2 * nodes.n_cos, 2 * nodes.n_psi)
        new = log_orientation_marginal(z, g00, g11, g01, prior, finer, xp=xp)
        change = float(xp.max(xp.abs(new - value)))
        nodes, value = finer, new
        if change <= tolerance:
            return value, nodes
    raise RuntimeError("orientation marginal did not converge to {} in {} "
                       "doublings".format(tolerance, max_doublings))
