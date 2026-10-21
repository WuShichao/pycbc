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
    floor = xp.broadcast_to(xp.asarray(prior.hmin, dtype=xp.float64),
                            x.shape)
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
                             margin=None, table=None, xp=numpy):
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

    ``table``, an :class:`AmplitudeTable` of ``prior``, replaces the
    amplitude integrals by its interpolation.
    """
    if table is None:
        def amplitude(x, g):
            return log_amplitude_marginal(x, g, prior, order=order, xp=xp)
    else:
        def amplitude(x, g):
            return table.log_marginal(x, g, xp=xp)
    if margin is None and table is not None and xp is not numpy:
        return table.log_orientation_marginals(z, g00, g11, g01, [nodes],
                                               xp=xp)[0]
    x, g = _project(z, g00, g11, g01, nodes.vectors, xp)
    logw = xp.log(xp.asarray(nodes.weights))[None, :]
    if margin is None:
        terms = amplitude(x, g) + logw
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
        terms[keep] = amplitude(x[keep], g[keep]) \
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


class _UnitPowerLaw(object):
    """An unnormalised ``h^-(power + 2)`` with limits per value, for the
    tables' integrals."""
    def __init__(self, power, hmin, hmax):
        self.power, self.hmin, self.hmax = power, hmin, hmax
        self._log_norm = 0.0

    log_pdf = DistancePowerLaw.log_pdf


class AmplitudeTable(object):
    """:func:`log_amplitude_marginal` by interpolation in tables checked
    against it.

    With ``u = h sqrt(g)``, ``y = x / sqrt(g)``, ``a = h_min sqrt(g)`` and
    ``b = h_max sqrt(g)``, a prior ``p(h) = c h^-m`` gives

        ln J = ln c + (m - 1)/2 ln g + K(y, a),
        K = ln int_a^b f(u) du,   f = u^-m I0(u y) exp(-u^2 / 2),

    which does not depend on ``b`` once ``b > y + 12``. ``a`` is the SNR
    the template would have at the prior's largest distance.

    Below ``a = 1`` ``K`` is split at ``u = 1``: the part below, less
    ``ln(1 - a)``, is tabulated on ``y`` and ``ln a``, the part above, less
    ``y^2 / 2``, on ``y``, both read linearly and added in logs; below the
    table's smallest ``a`` the integrand is ``u^-m`` to ``a^2 y^2`` and that
    piece is added in closed form. From ``a = 1`` to ``a_max`` the prior's
    spike is off the support, and ``K`` less an envelope - the integral of
    ``exp(u y - u^2 / 2)`` beyond ``a``, with ``u^-m I0(u y) exp(-u y)``
    read at that Gaussian's mean beyond ``a`` - is tabulated on ``y`` and
    ``ln a``. Values off the tables - ``y`` beyond its largest, ``a`` beyond
    ``a_max`` or ``b`` within reach of the peak - are integrated exactly.
    The tables are refined until random checks against the exact integral
    agree to ``tolerance``.

    Parameters
    ----------
    prior : DistancePowerLaw
        The amplitude prior.
    y_max : float
        Largest ``y`` (peak SNR) tabulated.
    log_a_min : float
        Smallest ``ln a`` tabulated.
    a_max : float, optional
        Largest ``a`` tabulated; default ``y_max + 12``.
    tolerance : float
        Largest error in ``ln J`` accepted at the checks.
    """
    def __init__(self, prior, y_max=40.0, log_a_min=-12.0, a_max=None,
                 tolerance=1e-4, checks=4000, seed=0, xp=numpy):
        self.prior = prior
        self.m = prior.power + 2
        self.log_c = prior._log_norm
        self.y_max = float(y_max)
        a_max = self.y_max + 12 if a_max is None else float(a_max)
        self.log_a = (float(log_a_min), float(numpy.log(a_max)))
        self.tolerance = float(tolerance)
        self._xp = xp
        rng = numpy.random.default_rng(seed)
        y = rng.uniform(0, self.y_max, checks)
        half = checks // 2
        la = numpy.concatenate([rng.uniform(self.log_a[0], 0, half),
                                rng.uniform(0, self.log_a[1],
                                            checks - half)])
        exact = self._exact_scaled(y, numpy.exp(la))
        ny, na, nb = 200, 128, 64
        for _ in range(4):
            self._build(ny, na, nb)
            error = numpy.abs(self._lookup(y, la, numpy) - exact).max()
            self.error = float(error)
            if error <= self.tolerance:
                return
            ny, na, nb = 2 * ny, 2 * na, 2 * nb
        raise RuntimeError("the amplitude table did not reach {}: {}".format(
            tolerance, error))

    def _exact_scaled(self, y, a, b=numpy.inf, chunk=32768):
        """ln int_a^b u^-m I0(u y) exp(-u^2 / 2) du, from the exact
        integral at unit ``g``, computed on the array module the table was
        built with and returned on the host."""
        xp = getattr(self, '_xp', numpy)
        y, a = numpy.broadcast_arrays(numpy.asarray(y, dtype=float),
                                      numpy.asarray(a, dtype=float))
        shape = y.shape
        y, a = xp.asarray(y.ravel()), xp.asarray(a.ravel())
        b = xp.minimum(b, a * 1e12 + 1e3)
        out = xp.empty(y.shape)
        for k in range(0, y.size, chunk):
            limits = _UnitPowerLaw(self.prior.power, a[k:k + chunk],
                                   b[k:k + chunk])
            out[k:k + chunk] = _log_amplitude_marginal(
                y[k:k + chunk], xp.ones(len(limits.hmin)), limits, 96, xp)
        out = out.get() if hasattr(out, 'get') else out
        return out.reshape(shape)

    def write(self, path):
        """Store the tables in an HDF file, for :meth:`read`."""
        import h5py
        with h5py.File(path, 'w') as f:
            for name in ('y', 'la', 'lb', 'tail', 'peak', 'near'):
                f[name] = getattr(self, name)
            f.attrs['m'] = self.m
            f.attrs['y_max'] = self.y_max
            f.attrs['log_a'] = self.log_a
            f.attrs['tolerance'] = self.tolerance
            f.attrs['error'] = self.error

    @classmethod
    def read(cls, path, prior):
        """The tables of :meth:`write` for ``prior``, whose power they must
        share: in the scaled variables nothing else enters them."""
        import h5py
        table = cls.__new__(cls)
        with h5py.File(path, 'r') as f:
            m = float(f.attrs['m'])
            if abs(m - (prior.power + 2)) > 1e-12:
                raise ValueError("the table in {} is for power {}, not "
                                 "{}".format(path, m - 2, prior.power))
            for name in ('y', 'la', 'lb', 'tail', 'peak', 'near'):
                setattr(table, name, f[name][()])
            table.y_max = float(f.attrs['y_max'])
            table.log_a = tuple(float(v) for v in f.attrs['log_a'])
            table.tolerance = float(f.attrs['tolerance'])
            table.error = float(f.attrs['error'])
        table.prior, table.m, table.log_c = prior, m, prior._log_norm
        table._device = {}
        return table

    def envelope(self, y, a, xp=numpy):
        """The envelope of ``K`` for ``a >= 1``: ``y^2 / 2 + ln int_a^inf
        exp(-(u - y)^2 / 2) du + ln I0(u* y) - u* y - m ln u*``, with ``u*``
        the mean of that truncated Gaussian."""
        if xp is numpy:
            from scipy.special import erfc, erfcx
        else:
            from cupyx.scipy.special import erfc, erfcx
        s = (a - y) / numpy.sqrt(2)
        sp, sn = xp.maximum(s, 0), xp.minimum(s, 0)
        log_tail = 0.5 * numpy.log(numpy.pi / 2) + xp.where(
            s > 0, xp.log(erfcx(sp)) - sp * sp, xp.log(erfc(sn)))
        mills = numpy.sqrt(2 / numpy.pi) * xp.where(
            s > 0, 1 / erfcx(sp), xp.exp(-sn * sn) / erfc(sn))
        u = y + mills
        return 0.5 * y * y + log_tail + log_i0(u * y, xp=xp) - u * y \
            - self.m * xp.log(u)

    def _build(self, ny, na, nb):
        self.y = numpy.linspace(0.0, self.y_max, ny)
        self.la = numpy.linspace(self.log_a[0], 0.0, na)
        self.lb = numpy.linspace(0.0, self.log_a[1], nb)
        # a < 1: the integral split at u = 1, the tail below smooth in y and
        # ln a once its interval's length is taken out, the peak above a
        # function of y alone; their competition is left to logaddexp
        # rather than to the interpolation
        y, a = numpy.meshgrid(self.y, numpy.exp(self.la[:-1]),
                              indexing='ij')
        tail = self._exact_scaled(y, a, 1.0) - numpy.log1p(-a)
        # at a = 1 the tail's mean is f(1)
        self.tail = numpy.concatenate(
            [tail, (log_i0(self.y) - 0.5)[:, None]], axis=1)
        self.peak = self._exact_scaled(self.y, 1.0) - 0.5 * self.y ** 2
        # a >= 1: the whole integral against its envelope
        y, a = numpy.meshgrid(self.y, numpy.exp(self.lb), indexing='ij')
        self.near = self._exact_scaled(y, a) - self.envelope(y, a)
        self._device = {}

    def _tables(self, xp):
        if xp not in self._device:
            self._device[xp] = tuple(xp.ascontiguousarray(xp.asarray(v))
                                     for v in (self.tail, self.peak,
                                               self.near))
        return self._device[xp]

    def _lookup(self, y, la, xp):
        tail, peak, near = self._tables(xp)
        dy = self.y[1] - self.y[0]
        fy = xp.clip(y / dy, 0, len(self.y) - 1.000001)
        i = xp.floor(fy).astype(xp.int64)
        wy = fy - i

        def bilinear(table, grid):
            f = xp.clip((la - grid[0]) / (grid[1] - grid[0]), 0,
                        len(grid) - 1.000001)
            j = xp.floor(f).astype(xp.int64)
            w = f - j
            return ((1 - wy) * (1 - w) * table[i, j]
                    + wy * (1 - w) * table[i + 1, j]
                    + (1 - wy) * w * table[i, j + 1]
                    + wy * w * table[i + 1, j + 1])
        with numpy.errstate(divide='ignore'):
            low = bilinear(tail, self.la) + xp.log1p(
                -xp.exp(xp.minimum(la, 0.0)))
        high = (1 - wy) * peak[i] + wy * peak[i + 1] + 0.5 * y * y
        below = xp.logaddexp(low, high)
        above = bilinear(near, self.lb) + self.envelope(
            y, xp.exp(xp.maximum(la, 0.0)), xp=xp)
        return xp.where(la <= 0, below, above)

    def log_orientation_marginals(self, z, g00, g11, g01, node_sets,
                                  xp=numpy):
        """:func:`log_orientation_marginal` with this table for each of
        ``node_sets`` at once, shape ``(len(node_sets), n)``; on cupy one
        fused kernel projects, reads the table and sums over the nodes,
        and only samples with a node off the table are integrated
        exactly."""
        if xp is numpy:
            return numpy.stack([log_orientation_marginal(
                z, g00, g11, g01, self.prior, nodes, table=self, xp=xp)
                for nodes in node_sets])
        key = tuple((v.n_cos, v.n_psi) for v in node_sets)
        if key not in self._device:
            self._device[key] = (
                xp.asarray(numpy.ascontiguousarray(numpy.concatenate(
                    [v.vectors for v in node_sets]), dtype=numpy.complex64)),
                xp.asarray(numpy.log(numpy.concatenate(
                    [v.weights for v in node_sets]))),
                xp.asarray(numpy.cumsum(
                    [0] + [len(v.weights) for v in node_sets]),
                    dtype=xp.int32))
        vectors, logw, bounds = self._device[key]
        if 'orientation' not in self._device:
            self._device['orientation'] = xp.RawKernel(
                _ORIENTATION_KERNEL, 'orientation')
        tail, peak, near = self._tables(xp)
        z = xp.ascontiguousarray(z, dtype=xp.complex128)
        n = z.shape[1]
        out = xp.empty((len(node_sets), n), dtype=xp.float64)
        prior = self.prior
        if n:
            self._device['orientation']((n, len(node_sets)), (128,), (
                z, xp.ascontiguousarray(g00, dtype=xp.float64),
                xp.ascontiguousarray(g11, dtype=xp.float64),
                xp.ascontiguousarray(g01, dtype=xp.complex128),
                vectors, logw, bounds, numpy.int32(n), tail, peak, near,
                numpy.int32(len(self.y)), numpy.int32(len(self.la)),
                numpy.int32(len(self.lb)),
                numpy.float64(self.y[1] - self.y[0]),
                numpy.float64(self.la[0]),
                numpy.float64(self.la[1] - self.la[0]),
                numpy.float64(self.lb[1] - self.lb[0]),
                numpy.float64(self.y_max), numpy.float64(self.log_a[0]),
                numpy.float64(self.log_a[1]),
                numpy.float64(numpy.log(prior.hmin)),
                numpy.float64(prior.hmin), numpy.float64(prior.hmax),
                numpy.float64(self.m), numpy.float64(self.log_c), out))
        off = xp.isnan(out).any(axis=0)
        if bool(off.any()):
            rows = xp.nonzero(off)[0]
            exact = self._exact_orientation(
                z[:, rows], g00[rows], g11[rows], g01[rows], node_sets, xp)
            out[:, rows] = exact
        return out

    def _exact_orientation(self, z, g00, g11, g01, node_sets, xp):
        out = []
        for nodes in node_sets:
            x, g = _project(z, g00, g11, g01, nodes.vectors, xp)
            terms = self.log_marginal(x, g, xp=xp) \
                + xp.log(xp.asarray(nodes.weights))[None, :]
            top = xp.max(terms, axis=1, keepdims=True)
            with numpy.errstate(invalid='ignore'):
                v = top[:, 0] + xp.log(xp.sum(xp.exp(terms - top), axis=1))
            out.append(xp.where(xp.isfinite(top[:, 0]), v, -numpy.inf))
        return xp.stack(out)

    def log_marginal(self, x, g, xp=numpy):
        """``ln J(x, g)``, as :func:`log_amplitude_marginal`."""
        x = xp.asarray(x, dtype=xp.float64)
        g = xp.asarray(g, dtype=xp.float64)
        x, g = xp.broadcast_arrays(x, g)
        root = xp.sqrt(g)
        y = x / root
        a = self.prior.hmin * root
        b = self.prior.hmax * root
        la = xp.log(a)
        m = self.m
        K = self._lookup(y, xp.maximum(la, self.log_a[0]), xp)
        # below the table, int_a^a0 u^-m du in closed form
        a0 = numpy.exp(self.log_a[0])
        below = a < a0
        if bool(xp.any(below)):
            if abs(m - 1) < 1e-12:
                extra = xp.log(a0) - la
            else:
                extra = (a ** (1 - m) - a0 ** (1 - m)) / (m - 1)
            extra = xp.where(below, xp.maximum(extra, 0.0), 0.0)
            K = K + xp.log1p(extra * xp.exp(-K))
        out = self.log_c + 0.5 * (m - 1) * xp.log(g) + K
        off = (y > self.y_max) | (la > self.log_a[1]) | (b < y + 12)
        if bool(xp.any(off)):
            out[off] = log_amplitude_marginal(x[off], g[off], self.prior,
                                              xp=xp)
        return out


_ORIENTATION_KERNEL = r"""
#include <cupy/complex.cuh>
#define NEG_INF __longlong_as_double(0xfff0000000000000ULL)
#define NOT_A_NUMBER __longlong_as_double(0x7ff8000000000000ULL)
#define PI 3.141592653589793
#define SQRT_HALF 0.7071067811865476
#define SQRT_TWO_OVER_PI 0.7978845608028654
#define HALF_LOG_HALF_PI 0.22579135264472744

__device__ float log_i0_scaled(float x) {
  // ln I0(x) - x for x >= 0, to about 1e-6
  if (x < 15.0f) return logf(cyl_bessel_i0f(x)) - x;
  float r = 1.0f / x;
  float s = 1.0f + r * (0.125f + r * (0.0703125f + r * (0.0732421875f
      + r * 0.112152099609375f)));
  return -0.5f * logf(2.0f * (float)PI * x) + logf(s);
}

__device__ double bilinear(const double* table, int n, int i, double wy,
                           double f) {
  int j = (int)floor(f);
  double w = f - j;
  return (1 - wy) * (1 - w) * table[i * n + j]
      + wy * (1 - w) * table[(i + 1) * n + j]
      + (1 - wy) * w * table[i * n + j + 1]
      + wy * w * table[(i + 1) * n + j + 1];
}

extern "C" __global__ void orientation(
    const complex<double>* z, const double* g00, const double* g11,
    const complex<double>* g01, const complex<float>* vectors,
    const double* logw, const int* bounds, const int n, const double* tail,
    const double* peak, const double* near, const int ny, const int na,
    const int nb, const double dy, const double la0, const double da,
    const double db, const double y_max, const double log_a0,
    const double log_a1, const double log_hmin, const double hmin,
    const double hmax, const double m, const double log_c, double* out) {
  // out[set, sample] = ln sum_nodes w J(|w^H z|, w^H G w), the table's
  // log_marginal summed over one set of nodes; NaN where a node is off
  // the tables, for the caller to integrate exactly
  __shared__ double best[128], total[128];
  __shared__ int off[128];
  int sample = blockIdx.x, set = blockIdx.y, tid = threadIdx.x;
  // z comes from a single-precision correlation, so its projections are
  // taken in single precision too
  complex<float> z0(z[sample].real(), z[sample].imag());
  complex<float> z1(z[n + sample].real(), z[n + sample].imag());
  complex<float> G01(g01[sample].real(), g01[sample].imag());
  float G00 = (float)g00[sample], G11 = (float)g11[sample];
  double top = NEG_INF, sum = 0, a0 = exp(log_a0);
  int bad = 0;
  for (int v = bounds[set] + tid; v < bounds[set + 1]; v += blockDim.x) {
    complex<float> v0 = vectors[2 * v], v1 = vectors[2 * v + 1];
    float x2 = norm(conj(v0) * z0 + conj(v1) * z1);
    float g = norm(v0) * G00 + norm(v1) * G11
        + 2 * (conj(v0) * v1 * G01).real();
    // what is summed with values of order y^2 stays in double precision;
    // logarithms of order one and their corrections are single, an error
    // below 1e-6 in ln J
    float y2f = x2 / g;
    double y2 = y2f, y = sqrtf(y2f), lg = logf(g);
    double la = log_hmin + 0.5 * lg;
    if (!(g > 0) || !(y <= y_max) || la > log_a1
        || hmax * hmax * g < (y + 12) * (y + 12)) {
      bad = 1;
      continue;
    }
    double fy = fmin(fmax(y / dy, 0.0), ny - 1.000001);
    int i = (int)floor(fy);
    double wy = fy - i, K;
    if (la <= 0) {
      double fa = fmin(fmax((fmax(la, log_a0) - la0) / da, 0.0),
                       na - 1.000001);
      double low = bilinear(tail, na, i, wy, fa)
          + (double)log1pf(-expf((float)la));
      double high = (1 - wy) * peak[i] + wy * peak[i + 1] + 0.5 * y2;
      K = fmax(low, high) + (double)log1pf(expf((float)-fabs(low - high)));
      if (la < log_a0) {
        double a = exp(la);
        double extra = fabs(m - 1) < 1e-12 ? log_a0 - la
            : (pow(a, 1 - m) - pow(a0, 1 - m)) / (m - 1);
        K += log1p(fmax(extra, 0.0) * exp(-K));
      }
    } else {
      double fb = fmin(fmax(la / db, 0.0), nb - 1.000001);
      double a = hmin * sqrtf(g), s = (a - y) * SQRT_HALF, gauss;
      float sf = (float)s, mills;
      if (s > 0) {
        float ex = erfcxf(sf);
        // y^2 / 2 - s^2, the Gaussian's log beyond a, kept exact
        gauss = a * y - 0.5 * a * a + (double)logf(ex);
        mills = 1.0f / ex;
      } else {
        float ex = erfcf(sf);
        gauss = 0.5 * y2 + (double)logf(ex);
        mills = expf(-sf * sf) / ex;
      }
      double u = y + SQRT_TWO_OVER_PI * mills;
      K = bilinear(near, nb, i, wy, fb) + gauss + HALF_LOG_HALF_PI
          + (double)(log_i0_scaled((float)(u * y)) - (float)m * logf((float)u));
    }
    // the sum's relative weights need only single precision too
    double term = log_c + 0.5 * (m - 1) * lg + K + logw[v];
    if (term > top) {
      sum = sum * expf((float)(top - term)) + 1;
      top = term;
    } else {
      sum += expf((float)(term - top));
    }
  }
  best[tid] = top;
  total[tid] = sum;
  off[tid] = bad;
  __syncthreads();
  for (int h = blockDim.x / 2; h > 0; h >>= 1) {
    if (tid < h) {
      double t1 = best[tid], t2 = best[tid + h];
      double hi = fmax(t1, t2);
      if (hi > NEG_INF)
        total[tid] = total[tid] * expf((float)(t1 - hi))
            + total[tid + h] * expf((float)(t2 - hi));
      best[tid] = hi;
      off[tid] |= off[tid + h];
    }
    __syncthreads();
  }
  if (tid == 0)
    out[set * n + sample] = off[0] ? NOT_A_NUMBER
        : (best[0] > NEG_INF ? best[0] + log(total[0]) : NEG_INF);
}
"""
