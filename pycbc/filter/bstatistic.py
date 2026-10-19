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


def log_amplitude_marginal(x, g, prior, order=64, xp=numpy):
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
    """
    x = xp.asarray(x, dtype=xp.float64)
    g = xp.asarray(g, dtype=xp.float64)
    x, g = xp.broadcast_arrays(x, g)
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
