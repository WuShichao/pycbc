# Copyright (C) 2026  The PyCBC contributors
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

r"""
The arrival-time, orientation and amplitude axes that the AMR Bayes factor
reduces away, restored as a joint draw.

WHAT THE PRODUCTION STATISTIC DOES
----------------------------------
The AMR cell Bayes factor is

    ln B_c = logsumexp over retained lags of
                 logsumexp over 128 orientation nodes of
                     ln w_g + ln Lbar(s_cg, y_cg) ,

where ``Lbar`` is the amplitude marginal

    Lbar(s, y) = int_1^U du u^-power exp(-u^2 s^2 / 2) I0(u s y)

on the registered prior ``p(u) ~ u^-power`` (``power = 4`` is the volumetric
distance prior), ``u = A / A_min``, and ``I0`` is the analytic marginalisation
over ``coa_phase``.  The orientation grid is 16 Gauss-Legendre nodes in
``cos(iota)`` on ``[-1, 1]`` times 8 nodes in ``psi`` at ``(j + 1/2) pi / 16``
on ``[0, pi/2)``, with weights ``w_i / 16``.  ``s = A_min sqrt(n)`` and
``y = |c^H D| / sqrt(n)`` are the template normalisation and the data
projection of one node at one lag.

The production code sums the 128 orientation values to one number per lag, then
sums the lags to one number per cell, and integrates the amplitude out first.
The summands and the amplitude integrand are destroyed.  This module re-derives
them from the archived sufficient statistics ``(D, Gram)`` per (cell, lag) and
draws **one sample per cell from the joint**

    p(lag, node, u | cell) = p(lag, node | cell) p(u | lag, node, cell) ,

with

    p(lag, node | cell)  proportional to  w_g Lbar(s, y)
    p(u | lag, node, cell) proportional to  u^-power exp(-u^2 s^2 / 2) I0(u s y)
                                           on [1, U] .

The first factor is the enumerated joint, the second is the exact conditional
integral at the chosen pair.  Drawing them in that order is an exact draw from
the joint: it is not the product of the two marginals, because ``u`` is drawn
conditional on the pair that was selected.  It is the same semantics
:py:meth:`pycbc.inference.models.tools.DistMarg.reconstruct` uses for a
marginalised distance -- a draw from ``p(theta_aux | theta_sampled)`` -- applied
to the whole reduced axis rather than to one parameter at a time.

The drawn columns are

    inclination     arccos(cos iota) of the drawn node, in radians
    polarization    psi of the same node, on [0, pi/2)
    delta_tc        the drawn lag, in seconds
    amplitude_u     u = A / A_min of the drawn amplitude, dimensionless
    distance_ratio  d/d_best = u_best / u of the same draw

``inclination``, ``polarization`` and ``delta_tc`` are one joint draw per cell,
so a corner plot of ``(inclination, amplitude_u)`` shows the degeneracy rather
than two marginals that have forgotten it.  ``delta_tc`` is now a draw: with
several lags retained per cell it takes more than one value where the archive
gives the lags comparable weight, which is the difference between a sample and
a report of the peak.

THE JOINT, NOT THE MARGINAL
---------------------------
The amplitude is the one genuine integral of the statistic, and its prior is
**registered and truncated**: ``u in [1, 5]`` with ``p(u) ~ u^-4``.  For the
stored controlled injection the likelihood mode is near ``u = 7.15`` and its
true amplitude is ``u = 6.717``, so the registered prior cuts the likelihood
before the ``(cos iota, ln u)`` ridge can develop.  At the registered ceiling
the correlation of ``cos iota`` with ``ln u`` is only about -0.36; at the
corrected ceiling 20 it is -0.94, and under a flat amplitude prior -0.95.  The
inclination *marginal* is therefore prior-dominated through the amplitude
prior: on the controlled injection its 68 per cent interval at ceiling 20 is
about 0.40 to 0.92 rad and does not contain the injected 1.0 rad, while
conditional on the amplitude the inclination is determined with a width of
about 0.055 rad.  A reader who takes the marginal as a measurement is wrong by
about four sigma on that injection, and by more than ten at the registered
ceiling.  **The joint, or the amplitude-conditional, is the object to quote;
the marginal is a prior-weighted projection of it.**  The amplitude ceiling and
the prior power are explicit arguments of :class:`AmplitudeMarginal`, and the
adapter records both in the file metadata so a reader knows which prior the
columns were drawn under.

THE AMPLITUDE COLUMN IS NOT A DISTANCE
--------------------------------------
``u`` and ``d/d_best`` are the only amplitude statements the partition supports.
An absolute distance would require the source's amplitude normalisation (the
intrinsic strain scale), which the partition does not carry.  Do not convert
``amplitude_u`` to megaparsecs; the metadata says so, and the column is named
``amplitude_u`` rather than ``distance`` for that reason.  ``u_best`` is the
amplitude that maximises the likelihood at the highest-weight (cell, lag, node)
of the partition, so ``distance_ratio = u_best / u`` is the distance relative
to the best-fit amplitude of this partition.

WHY THE TABLE LOOKS LIKE THE PRODUCTION ONE
-------------------------------------------
The amplitude marginal is evaluated on the same full axes as the production
table, ``ln s`` from ``ln 0.02`` to ``ln 30`` in 160 nodes and ``y`` from 0 to
45 in 901 nodes, and the sub-grid is built from rigorous bounds on ``s`` and
``y`` over every archived row and every orientation node.  The nodes are
therefore the production nodes and the bilinear interpolation is the
production interpolation.  At ``U = 5`` the rebuilt per-cell Bayes factor
reproduces the stored one (median absolute difference about ``3e-7``).
"""

import numpy
from scipy.special import ive

#: The ceiling of the registered amplitude prior, baked into the stored Bayes
#: factors.  Kept as a named constant so the comparison can be reproduced.
REGISTERED_CEILING = 5.0

#: The corrected default: wide enough that the amplitude likelihood is
#: contained (the stored injection's mode is at u = 7.15).
DEFAULT_CEILING = 20.0

#: The registered power of the volumetric amplitude prior, p(u) ~ u^-power.
AMPLITUDE_POWER = 4.0

#: The number of u nodes of the production amplitude table.
N_U = 2001

#: Orientation grid sizes.
N_CI = 16
N_PSI = 8
N_NODES = N_CI * N_PSI

#: The names the adapter writes for the restored joint draw.  The first three
#: are PyCBC's own parameter names (``pycbc.waveform.parameters``);
#: ``amplitude_u`` and ``distance_ratio`` are the new amplitude columns.
RECOVERED_PARAMETERS = ("inclination", "polarization", "delta_tc",
                        "amplitude_u", "distance_ratio")

#: The amplitude column: u = A / A_min, dimensionless.
AMPLITUDE_COLUMN = "amplitude_u"

#: The distance-ratio column: d / d_best = u_best / u, dimensionless.
DISTANCE_RATIO_COLUMN = "distance_ratio"

#: The full axes of the production amplitude table.  A table built on a slice
#: of these axes has the production values at the production nodes.
LS_AXIS = numpy.linspace(numpy.log(0.02), numpy.log(30.0), 160)
Y_AXIS = numpy.linspace(0.0, 45.0, 901)
DLS = LS_AXIS[1] - LS_AXIS[0]
DY = Y_AXIS[1] - Y_AXIS[0]


def logsumexp(values, axis):
    """Log of the sum of exponentials, stable along ``axis``."""
    values = numpy.asarray(values, dtype=float)
    vmax = numpy.max(values, axis=axis, keepdims=True)
    return (numpy.squeeze(vmax, axis=axis)
            + numpy.log(numpy.sum(numpy.exp(values - vmax), axis=axis)))


def _logsumexp_last(values):
    """Log-sum-exp along the last axis, without squeezing the others."""
    values = numpy.asarray(values, dtype=float)
    vmax = numpy.max(values, axis=-1, keepdims=True)
    return (numpy.squeeze(vmax, axis=-1)
            + numpy.log(numpy.sum(numpy.exp(values - vmax), axis=-1)))


def weighted_correlation(x, y, weights=None):
    """The weighted Pearson correlation of two one-dimensional arrays.

    ``weights`` are normalised internally.  With ``weights=None`` this is the
    ordinary correlation of the equal-weight sample, which is the reading of a
    resampled PyCBC file that has forgotten ``logwt``.
    """
    x = numpy.asarray(x, dtype=float)
    y = numpy.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(
            "the two columns must have the same shape, got {} and {}".format(
                x.shape, y.shape))
    if weights is None:
        weights = numpy.ones_like(x)
    weights = numpy.asarray(weights, dtype=float)
    if weights.shape != x.shape:
        raise ValueError(
            "the weights must have the same shape as the columns, got {} "
            "and {}".format(weights.shape, x.shape))
    total = float(weights.sum())
    if not numpy.isfinite(total) or total <= 0.0:
        raise ValueError("the weights must have a positive finite sum")
    w = weights / total
    mx = float(numpy.sum(w * x))
    my = float(numpy.sum(w * y))
    dx = x - mx
    dy = y - my
    cov = float(numpy.sum(w * dx * dy))
    sx = float(numpy.sqrt(numpy.sum(w * dx * dx)))
    sy = float(numpy.sqrt(numpy.sum(w * dy * dy)))
    if sx == 0.0 or sy == 0.0:
        return 0.0
    return cov / (sx * sy)


def weighted_quantile(values, weights, quantiles):
    """Posterior-weighted quantiles of one column.

    The written ensemble is the ``logwt``-weighted histogram of the cells, so
    a quantile reported in the file has to be taken under that weight rather
    than over the equal-weight set of cells.
    """
    values = numpy.asarray(values, dtype=float)
    weights = numpy.asarray(weights, dtype=float)
    if values.shape != weights.shape:
        raise ValueError(
            "values and weights must have the same shape, got {} and "
            "{}".format(values.shape, weights.shape))
    order = numpy.argsort(values)
    sorted_values = values[order]
    cumulative = numpy.cumsum(weights[order])
    cumulative /= cumulative[-1]
    return numpy.asarray(
        [numpy.interp(q, cumulative, sorted_values) for q in quantiles])


def orientation_grid():
    r"""The production orientation grid.

    Returns
    -------
    coefficients : (128, 2) complex ndarray
        The two polarisation amplitudes ``(r_p, r_c)`` of each node, in the
        node order ``g = i * 8 + j``.
    weights : (128,) ndarray
        The prior weight of each node, ``w_i / 16``.
    node_ci : (128,) ndarray
        ``cos(iota)`` of each node.
    node_psi : (128,) ndarray
        ``psi`` of each node, on ``[0, pi/2)``.
    ci, ci_weights, psi : ndarrays
        The one-dimensional axes and the Gauss-Legendre weights, for metadata.
    """
    ci, ci_weights = numpy.polynomial.legendre.leggauss(N_CI)
    psi = (numpy.arange(N_PSI) + 0.5) * numpy.pi / (2 * N_PSI)
    ci_grid, psi_grid = numpy.meshgrid(ci, psi, indexing="ij")
    hp = 0.5 * (1.0 + ci_grid ** 2)
    hc = -1j * ci_grid
    c2 = numpy.cos(2.0 * psi_grid)
    s2 = numpy.sin(2.0 * psi_grid)
    coefficients = numpy.stack(
        [c2 * hp - s2 * hc, s2 * hp + c2 * hc], axis=-1).reshape(-1, 2)
    weights = numpy.repeat(ci_weights / N_CI, N_PSI)
    return (coefficients, weights, numpy.repeat(ci, N_PSI),
            numpy.tile(psi, N_CI), ci, ci_weights, psi)


def _index_range(lo, hi, axis_min, step, nmax, margin=2):
    """The node index range of ``axis_min + step * k`` that covers [lo, hi]."""
    first = int(numpy.floor((lo - axis_min) / step)) - margin
    last = int(numpy.ceil((hi - axis_min) / step)) + margin
    return max(0, first), min(nmax - 1, last)


class AmplitudeMarginal(object):
    r"""The production amplitude marginal, for a configurable prior.

    ``Lbar(s, y) = ln int_1^U du u^-power exp(-u^2 s^2 / 2) I0(u s y)`` on the
    production ``(ln s, y)`` axes, bilinearly interpolated exactly as
    ``orientation_pilot.AmplitudeTable`` does.  The same class exposes the
    per-``u`` integrand that this integral sums away, because the joint draw
    needs ``p(u | s, y)`` rather than only ``Lbar``.

    Parameters
    ----------
    ceiling : float
        The upper limit ``U`` of ``u = A / A_min``.  Must exceed 1.
    power : float, optional
        The power of the amplitude prior.  Default is 4, the registered value.
        ``0`` is a flat prior and is used for diagnostics.
    n_u : int, optional
        Number of ``u`` nodes.  Default is 2001, the production number.
    ls_range, y_range : tuple of int, optional
        Inclusive node-index range of the ``ln s`` and ``y`` axes to build.
        When omitted the whole axis is built, which is slow; the adapter uses
        :meth:`from_data` or :meth:`from_bounds`.
    """

    def __init__(self, ceiling, power=AMPLITUDE_POWER, n_u=N_U,
                 ls_range=None, y_range=None):
        ceiling = float(ceiling)
        if not ceiling > 1.0:
            raise ValueError(
                "the amplitude ceiling must exceed 1 (u = A/A_min), got "
                "{}".format(ceiling))
        if not power >= 0:
            raise ValueError(
                "the amplitude power must be non-negative (0 is a flat "
                "prior, used for diagnostics)")
        self.ceiling = ceiling
        self.power = float(power)
        self.n_u = int(n_u)
        self.u = numpy.linspace(1.0, ceiling, self.n_u)
        self.log_u_weights = self._log_u_weights()
        self.ls_range = ((0, len(LS_AXIS) - 1) if ls_range is None
                         else (int(ls_range[0]), int(ls_range[1])))
        self.y_range = ((0, len(Y_AXIS) - 1) if y_range is None
                        else (int(y_range[0]), int(y_range[1])))
        i0, i1 = self.ls_range
        j0, j1 = self.y_range
        ls = LS_AXIS[i0:i1 + 1]
        yy = Y_AXIS[j0:j1 + 1]
        table = numpy.empty((len(ls), len(yy)))
        for a, sv in enumerate(numpy.exp(ls)):
            # (y, u): the same integrand the draw uses, with the prior measure
            # included, so logsumexp over u is Lbar including the normalisation
            arg = self.u[None, :] * sv * yy[:, None]
            values = (self.log_u_weights[None, :]
                      - 0.5 * (self.u[None, :] * sv) ** 2
                      + numpy.log(ive(0, arg)) + arg)
            table[a] = _logsumexp_last(values)
        self.table = table

    def _log_u_weights(self):
        """``ln(p(u) du)`` normalised on the grid, the quadrature measure."""
        pu = self.u ** (-self.power)
        du = numpy.gradient(self.u)
        return numpy.log(pu / numpy.sum(pu * du)) + numpy.log(du)

    @classmethod
    def from_data(cls, s, y, ceiling, power=AMPLITUDE_POWER, n_u=N_U,
                  margin=2):
        """Builds only the rows and columns that ``s`` and ``y`` touch."""
        s = numpy.asarray(s, dtype=float)
        y = numpy.asarray(y, dtype=float)
        if not numpy.all(s > 0):
            raise ValueError("s = A_min sqrt(n) must be positive everywhere")
        if not numpy.all(y >= 0):
            raise ValueError("y = |c^H D| / sqrt(n) must be non-negative")
        ls_range = _index_range(
            float(numpy.log(s).min()), float(numpy.log(s).max()),
            LS_AXIS[0], DLS, len(LS_AXIS), margin=margin)
        y_range = _index_range(
            float(y.min()), float(y.max()), Y_AXIS[0], DY, len(Y_AXIS),
            margin=margin)
        return cls(ceiling, power=power, n_u=n_u, ls_range=ls_range,
                   y_range=y_range)

    @classmethod
    def from_bounds(cls, s_lo, s_hi, y_lo, y_hi, ceiling,
                    power=AMPLITUDE_POWER, n_u=N_U, margin=2):
        """Builds the rows and columns that a rigorous ``(s, y)`` box touches.

        Used by the joint pass: the rows are all archived ``(cell, lag)`` rows
        and all 128 nodes, and their extreme ``s`` and ``y`` are bounded from
        the Gram matrices, so the table is built once and never clips.
        """
        if not s_lo > 0 or not s_hi >= s_lo:
            raise ValueError("the s bounds must satisfy 0 < s_lo <= s_hi")
        if not y_hi >= y_lo or y_lo < 0:
            raise ValueError("the y bounds must satisfy 0 <= y_lo <= y_hi")
        ls_range = _index_range(
            float(numpy.log(s_lo)), float(numpy.log(s_hi)),
            LS_AXIS[0], DLS, len(LS_AXIS), margin=margin)
        y_range = _index_range(
            float(y_lo), float(y_hi), Y_AXIS[0], DY, len(Y_AXIS),
            margin=margin)
        return cls(ceiling, power=power, n_u=n_u, ls_range=ls_range,
                   y_range=y_range)

    def __call__(self, s, y):
        """Bilinear interpolation in ``(ln s, y)``, clipped to the table."""
        s = numpy.asarray(s, dtype=float)
        y = numpy.asarray(y, dtype=float)
        i0, i1 = self.ls_range
        j0, j1 = self.y_range
        a = (numpy.clip(numpy.log(s), LS_AXIS[i0], LS_AXIS[i1])
             - LS_AXIS[0]) / DLS - i0
        b = (numpy.clip(y, Y_AXIS[j0], Y_AXIS[j1])
             - Y_AXIS[0]) / DY - j0
        nrows, ncols = self.table.shape
        a = numpy.clip(a, 0.0, nrows - 1.0 - 1e-12)
        b = numpy.clip(b, 0.0, ncols - 1.0 - 1e-12)
        i = numpy.minimum(a.astype(numpy.int64), nrows - 2)
        j = numpy.minimum(b.astype(numpy.int64), ncols - 2)
        fa = a - i
        fb = b - j
        table = self.table
        return ((1.0 - fa) * (1.0 - fb) * table[i, j]
                + fa * (1.0 - fb) * table[i + 1, j]
                + (1.0 - fa) * fb * table[i, j + 1]
                + fa * fb * table[i + 1, j + 1])

    def log_integrand(self, s, y):
        """``ln[p(u) du] + ln L(s, y, u)`` per ``u``, on the grid.

        The shape is ``s.shape + (n_u,)``; the prior measure is included, so
        :meth:`log_marginal` is the log of the same integral the table
        interpolates.
        """
        s = numpy.asarray(s, dtype=float)
        y = numpy.asarray(y, dtype=float)
        shape = numpy.broadcast_shapes(s.shape, y.shape)
        s = numpy.broadcast_to(s, shape)
        y = numpy.broadcast_to(y, shape)
        arg = self.u * s[..., None] * y[..., None]
        return (self.log_u_weights
                - 0.5 * (self.u * s[..., None]) ** 2
                + numpy.log(ive(0, arg)) + arg)

    def log_marginal(self, s, y):
        """``ln int du p(u) L(s, y, u)`` evaluated directly on the u grid."""
        return _logsumexp_last(self.log_integrand(s, y))

    def draw(self, s, y, rng, chunk=2048):
        """Draws one ``u`` per ``(s, y)`` entry from ``p(u | s, y)``.

        Parameters
        ----------
        s, y : array-like
            Equal-shaped arrays, normally the chosen ``(cell,)`` vectors.
        rng : numpy.random.Generator
            The generator, so the file is reproducible.
        chunk : int, optional
            How many entries to normalise at once, to bound memory.

        Returns
        -------
        ndarray
            ``u`` with the shape of ``s``.
        """
        s = numpy.asarray(s, dtype=float)
        y = numpy.asarray(y, dtype=float)
        if s.shape != y.shape:
            raise ValueError(
                "s and y must have the same shape, got {} and {}".format(
                    s.shape, y.shape))
        flat = s.reshape(-1)
        yflat = y.reshape(-1)
        out = numpy.empty(flat.shape)
        for start in range(0, flat.size, int(chunk)):
            stop = min(start + int(chunk), flat.size)
            logp = self.log_integrand(flat[start:stop], yflat[start:stop])
            logp = logp - _logsumexp_last(logp)[..., None]
            cdf = numpy.cumsum(numpy.exp(logp), axis=-1)
            cdf /= cdf[..., -1:]
            r = rng.random(cdf.shape[:-1])
            index = numpy.sum(r[..., None] > cdf, axis=-1)
            index = numpy.clip(index, 0, self.n_u - 1)
            out[start:stop] = self.u[index]
        return out.reshape(s.shape)

    def maximum_likelihood_u(self, s, y):
        """The ``u`` that maximises the likelihood (flat prior) at ``(s, y)``.

        The prior measure is removed, so this is the best-fit amplitude of the
        template at that point; it defines ``d/d_best = u_best / u``.
        """
        logp = self.log_integrand(numpy.atleast_1d(s),
                                  numpy.atleast_1d(y))
        logp = logp - self.log_u_weights
        value = self.u[numpy.argmax(logp, axis=-1)]
        return value if numpy.ndim(s) else float(value[0])

    def metadata(self):
        """The attributes that describe the table in the samples file."""
        return {
            "amr_amplitude_ceiling": self.ceiling,
            "amr_amplitude_power": self.power,
            "amr_amplitude_prior": (
                "p(u) proportional to u^-{} on [1, {}], u = A/A_min; the "
                "phase is marginalised analytically (I0) and the amplitude "
                "integral runs over the u grid linspace(1, ceiling, {})"
                .format(self.power, self.ceiling, self.n_u)),
            "amr_amplitude_table_rows": numpy.asarray(self.ls_range, dtype=int),
            "amr_amplitude_table_cols": numpy.asarray(self.y_range, dtype=int),
        }


class SufficientStatistics(object):
    """The archived ``(D, Gram)`` sufficient statistics, per (cell, lag).

    The archive is written by the research demarginalisation pass: for every
    retained lag of every stored cell it keeps ``D = <template, data>`` and
    ``Gram = <template, template>``, eight real numbers from which the whole
    128-node structure follows.  This class validates that the archive belongs
    to the partition, rebuilds the per-(cell, lag, node) statistic, and draws
    one sample per cell from the joint over ``(lag, node, u)``.

    Parameters
    ----------
    path : str
        The ``sufficient_full.npz`` archive.
    eta : (ncells, 6) ndarray
        The partition's cell centres.
    lnb : (ncells,) ndarray
        The partition's stored cell Bayes factors.
    """

    def __init__(self, path, eta, lnb):
        eta = numpy.asarray(eta, dtype=float)
        lnb = numpy.asarray(lnb, dtype=float)
        with numpy.load(path) as data:
            self.path = path
            self.D = numpy.asarray(data["D"])
            self.Gram = numpy.asarray(data["G"])
            self.lag = numpy.asarray(data["lag"], dtype=numpy.int64)
            self.two_f = numpy.asarray(data["twoF"], dtype=float)
            self.ptr = numpy.asarray(data["ptr"], dtype=numpy.int64)
            self.cells = numpy.asarray(data["cells"], dtype=numpy.int64)
            self.Delta = numpy.asarray(data["Delta"], dtype=float)
            self.A_min = float(data["A_min"])
            self.dt = float(data["Dt"])
            stored_eta = numpy.asarray(data["eta"], dtype=float)
            stored_lnb = numpy.asarray(data["lnB_stored"], dtype=float)
            truth = (numpy.asarray(data["truth"], dtype=float)
                     if "truth" in data.files else None)
        self.n_cells = len(eta)
        self.n_cells_archive = len(self.cells)
        self.n_rows = len(self.lag)
        if self.ptr.shape != (self.n_cells_archive + 1,):
            raise ValueError(
                "the sufficient-statistics archive is inconsistent: ptr has "
                "{} entries for {} cells".format(self.ptr.shape[0],
                                                 self.n_cells_archive))
        if not numpy.array_equal(numpy.sort(self.cells),
                                 numpy.arange(self.n_cells)):
            raise ValueError(
                "the sufficient-statistics archive does not cover this "
                "partition: it holds {} cells, the partition has {}. The "
                "recovered columns would be missing for some cells, so they "
                "are not written; re-run the pass with --all.".format(
                    self.n_cells_archive, self.n_cells))
        if not numpy.allclose(stored_eta, eta[self.cells],
                              rtol=0.0, atol=1e-9):
            raise ValueError(
                "the sufficient statistics were archived for different cell "
                "centres: max |d eta| = {:.3e}".format(
                    numpy.max(numpy.abs(stored_eta - eta[self.cells]))))
        if not numpy.allclose(stored_lnb, lnb[self.cells],
                              rtol=0.0, atol=1e-4):
            raise ValueError(
                "the sufficient statistics were archived for a different "
                "partition: max |d lnB| = {:.3e}".format(
                    numpy.max(numpy.abs(stored_lnb - lnb[self.cells]))))
        if truth is not None and truth.shape == (6,):
            self.truth = truth
        else:
            self.truth = None
        self.lnb_stored = stored_lnb
        self.nlag = numpy.diff(self.ptr)
        if numpy.any(self.nlag <= 0):
            raise ValueError("every archived cell must have at least one lag")

        coefficients, self.weights, self.node_ci, self.node_psi, \
            self.ci, self.ci_weights, self.psi = orientation_grid()
        self.coefficients = coefficients
        self.log_weights = numpy.log(self.weights)

        # one row per cell: the strongest retained lag, kept for the report and
        # for the historical peak-lag comparison
        best = numpy.empty(self.n_cells_archive, dtype=numpy.int64)
        for a in range(self.n_cells_archive):
            start, stop = self.ptr[a], self.ptr[a + 1]
            best[a] = start + int(numpy.argmax(self.two_f[start:stop]))
        self.best_row = best
        self.delta_tc_archive = self.Delta[self.lag[best]]
        self.two_f_peak = self.two_f[best]

        # the peak-row statistics drive the report and the clip count
        peak_s = numpy.empty((self.n_cells_archive, N_NODES))
        peak_y = numpy.empty((self.n_cells_archive, N_NODES))
        step = 4096
        for start in range(0, self.n_cells_archive, step):
            stop = min(start + step, self.n_cells_archive)
            rows = best[start:stop]
            peak_s[start:stop], peak_y[start:stop] = self._statistics(rows)
        self.s = peak_s
        self.y = peak_y
        self.clipped = int(numpy.sum(
            (peak_s < numpy.exp(LS_AXIS[0]))
            | (peak_s > numpy.exp(LS_AXIS[-1]))
            | (peak_y > Y_AXIS[-1])))

    @staticmethod
    def _check_positive(values, where):
        if not numpy.all(values > 0):
            raise ValueError(
                "the archived Gram matrix gives a non-positive template norm "
                "at {} ({} entries); the archive is not usable".format(
                    where, int(numpy.sum(values <= 0))))

    def _statistics(self, rows):
        """Per-node ``(s, y)`` for the given ``(cell, lag)`` rows.

        ``s = A_min sqrt(c^H Gram c)`` and ``y = |c^H D| / sqrt(c^H Gram c)``
        for the 128 nodes ``g = i * 8 + j``.  Returns two arrays with shape
        ``(len(rows), 128)``.
        """
        rows = numpy.asarray(rows, dtype=numpy.int64)
        D = self.D[rows]
        G = self.Gram[rows]
        coefficients = self.coefficients
        n = numpy.real(numpy.einsum(
            "ga,sab,gb->sg", numpy.conj(coefficients), G, coefficients))
        self._check_positive(n, "{} rows".format(rows.size))
        zg = numpy.abs(
            numpy.conj(coefficients)[None, :, 0] * D[:, 0, None]
            + numpy.conj(coefficients)[None, :, 1] * D[:, 1, None])
        root = numpy.sqrt(n)
        return self.A_min * root, zg / root

    def _row_statistics(self, start, stop):
        """Per-node ``(s, y)`` for the row slice ``[start, stop)``."""
        return self._statistics(numpy.arange(int(start), int(stop)))

    def report(self):
        """The numbers that justify the archive and the peak-lag choice."""
        return {
            "amr_sufficient_file": self.path,
            "amr_sufficient_cells": int(self.n_cells_archive),
            "amr_sufficient_rows": int(self.n_rows),
            "amr_delta_tc_reference": 0.0,
            "amr_delta_tc_grid_s": self.dt,
            "amr_delta_tc_distinct": int(
                numpy.unique(self.delta_tc_archive).size),
            "amr_amplitude_clipped_points": self.clipped,
        }

    def _table_bounds(self):
        """Rigorous ``(s, y)`` bounds over every row and every node.

        ``c`` is one of the 128 orientation vectors, so
        ``|c|^2 = ((1 + cos^2 i)/2)^2 + cos^2 i`` lies in ``[0.25, 2]``.
        Hence ``s^2 = A_min^2 c^H G c`` lies between ``0.25 A_min^2 lambda_min``
        and ``2 A_min^2 lambda_max``, and ``y = |c^H D| / sqrt(c^H G c)`` cannot
        exceed ``|D| / sqrt(lambda_min)``.  Both are evaluated on the archive
        without touching the orientation grid.
        """
        G = self.Gram
        a = G[:, 0, 0].real
        d = G[:, 1, 1].real
        b = G[:, 0, 1]
        trace = a + d
        determinant = a * d - (b.real ** 2 + b.imag ** 2)
        discriminant = numpy.sqrt(numpy.maximum(trace * trace
                                                - 4.0 * determinant, 0.0))
        lmax = 0.5 * (trace + discriminant)
        lmin = 0.5 * (trace - discriminant)
        if not numpy.all(lmin > 0):
            raise ValueError(
                "the archived Gram matrix is not positive definite for {} "
                "rows; the amplitude axis cannot be rebuilt".format(
                    int(numpy.sum(lmin <= 0))))
        abs_d = numpy.sqrt(numpy.abs(self.D[:, 0]) ** 2
                           + numpy.abs(self.D[:, 1]) ** 2)
        s_lo = self.A_min * numpy.sqrt(0.25 * float(lmin.min()))
        s_hi = self.A_min * numpy.sqrt(2.0 * float(lmax.max()))
        y_hi = float(abs_d.max()) / numpy.sqrt(float(lmin.min()))
        return s_lo, s_hi, 0.0, y_hi

    def columns(self, ceiling, power=AMPLITUDE_POWER, seed=20261018,
                weights=None, chunk=16384, n_u=N_U, return_marginal=False):
        """Draws the joint ``(lag, node, u)`` columns, one per cell.

        The per-(cell, lag, node) weights are
        ``w_g Lbar(s, y)``; they are normalised inside each cell, one pair is
        drawn, and ``u`` is then drawn from ``p(u | s, y)`` at that pair.  This
        is an exact joint draw, not a marginal draw followed by a conditional
        one: the pair and the amplitude are never separated.

        Parameters
        ----------
        ceiling : float
            The upper limit ``U`` of ``u = A / A_min``.
        power : float, optional
            The power of the amplitude prior.  Default is the registered 4.
        seed : int, optional
            Seed of the draw, so a file is reproducible.
        weights : array-like, optional
            Cell weights for the reported correlations (normally
            ``exp(logwt)``).  When omitted the correlations are equal-weight.
        chunk : int, optional
            Approximate number of ``(cell, lag)`` rows to hold at once.  The
            pass streams the archive in chunks, so its memory is set by this
            number rather than by the 2.1 million rows of the stored run.
        return_marginal : bool, optional
            Also return the per-cell orientation marginal
            ``p(node | cell) = sum_lag sum_u p(lag, node, u | cell)``.  It is
            the old marginal, offered so a test can show the joint reproduces
            it; it is not written to the file.

        Returns
        -------
        dict
            ``columns`` (partition order), ``diagnostics`` (archive order) and
            ``table``.
        """
        ceiling = float(ceiling)
        power = float(power)
        s_lo, s_hi, y_lo, y_hi = self._table_bounds()
        table = AmplitudeMarginal.from_bounds(
            s_lo, s_hi, y_lo, y_hi, ceiling, power=power, n_u=n_u)
        rng = numpy.random.default_rng(int(seed))

        draw_node = numpy.empty(self.n_cells_archive, dtype=numpy.int64)
        draw_lag = numpy.empty(self.n_cells_archive, dtype=numpy.int64)
        draw_row = numpy.empty(self.n_cells_archive, dtype=numpy.int64)
        draw_s = numpy.empty(self.n_cells_archive)
        draw_y = numpy.empty(self.n_cells_archive)
        lnb = numpy.empty(self.n_cells_archive)
        peak_lag = numpy.empty(self.n_cells_archive, dtype=numpy.int64)
        marginal = (numpy.zeros((self.n_cells_archive, N_NODES))
                    if return_marginal else None)

        best_value = -numpy.inf
        best_s = numpy.nan
        best_y = numpy.nan
        max_lag = int(self.nlag.max())
        chunk_cells = max(1, int(chunk) // max_lag)

        for c0 in range(0, self.n_cells_archive, chunk_cells):
            c1 = min(c0 + chunk_cells, self.n_cells_archive)
            r0 = int(self.ptr[c0])
            r1 = int(self.ptr[c1])
            s, y = self._row_statistics(r0, r1)
            nodes = table(s, y) + self.log_weights[None, :]
            row_lnb = _logsumexp_last(nodes)
            # the pair with the largest weight sets the best-fit amplitude
            flat = int(numpy.argmax(nodes))
            if nodes.flat[flat] > best_value:
                best_value = float(nodes.flat[flat])
                best_s = float(s.flat[flat])
                best_y = float(y.flat[flat])
            starts = self.ptr[c0:c1] - r0
            stops = self.ptr[c0 + 1:c1 + 1] - r0
            for k in range(c1 - c0):
                a = c0 + k
                start, stop = int(starts[k]), int(stops[k])
                cell_lnb = float(_logsumexp_last(row_lnb[start:stop]))
                lnb[a] = cell_lnb
                logp = nodes[start:stop] - cell_lnb
                probability = numpy.exp(logp)
                peak_lag[a] = self.lag[r0 + start
                                       + int(numpy.argmax(row_lnb[start:stop]))]
                if return_marginal:
                    marginal[a] = probability.sum(axis=0)
                cumulative = numpy.cumsum(probability.ravel())
                total = float(cumulative[-1])
                index = int(numpy.searchsorted(
                    cumulative, rng.random() * total))
                row_local, node = divmod(index, N_NODES)
                draw_node[a] = node
                draw_row[a] = r0 + start + row_local
                draw_lag[a] = self.lag[r0 + start + row_local]
                draw_s[a] = s[start + row_local, node]
                draw_y[a] = y[start + row_local, node]

        u = table.draw(draw_s, draw_y, rng)
        u_best = table.maximum_likelihood_u(best_s, best_y)
        inclination = numpy.arccos(self.node_ci[draw_node])
        polarization = self.node_psi[draw_node]
        delta_tc = self.Delta[draw_lag]
        amplitude = u
        distance_ratio = u_best / amplitude

        corr_inc_amp = weighted_correlation(inclination, amplitude, weights)
        corr_cos_lnu = weighted_correlation(
            self.node_ci[draw_node], numpy.log(amplitude), weights)
        quantiles = (0.05, 0.16, 0.5, 0.84, 0.95)
        diagnostics = {
            "lnb": lnb,
            "peak_lag_index": peak_lag,
            "drawn_lag_index": draw_lag,
            "node": draw_node,
            "s": draw_s,
            "y": draw_y,
            "amplitude": amplitude,
            "u_best": float(u_best),
            "corr_inclination_amplitude": float(corr_inc_amp),
            "corr_cos_lnu": float(corr_cos_lnu),
            "distinct_lags": int(numpy.unique(delta_tc).size),
            "cells_not_at_peak_lag": int(numpy.sum(draw_lag != peak_lag)),
            "amplitude_quantiles": numpy.quantile(
                amplitude, quantiles),
            "inclination_quantiles": numpy.quantile(
                inclination, quantiles),
            "n_cells": int(self.n_cells_archive),
        }

        def in_partition_order(values):
            order = numpy.empty(self.n_cells)
            order[self.cells] = values
            return order

        columns = {
            "inclination": in_partition_order(inclination),
            "polarization": in_partition_order(polarization),
            "delta_tc": in_partition_order(delta_tc),
            AMPLITUDE_COLUMN: in_partition_order(amplitude),
            DISTANCE_RATIO_COLUMN: in_partition_order(distance_ratio),
        }
        result = {"columns": columns, "diagnostics": diagnostics,
                  "table": table}
        if return_marginal:
            result["orientation_marginal"] = marginal
        return result

    def metadata(self, ceiling, power, seed, diagnostics=None):
        """The attributes that describe the restored joint draw."""
        info = self.report()
        diagnostics = {} if diagnostics is None else diagnostics
        info.update({
            "amr_recovered_params": " ".join(RECOVERED_PARAMETERS),
            "amr_recovered_seed": int(seed),
            "amr_recovered_kind": (
                "one joint draw per cell from p(lag, node, u | cell) = "
                "p(lag, node | cell) p(u | lag, node, cell), with "
                "p(lag, node | cell) proportional to w_g Lbar(s, y) over the "
                "retained lags and the 128 orientation nodes, and u drawn "
                "conditionally from the amplitude integrand at the chosen "
                "pair. The draw is a proper joint sample, not a marginal draw "
                "followed by a conditional one. The columns are restored "
                "conditionals, not sampled dimensions: they do not enter the "
                "prior, the evidence or the cell weights."),
            "amr_joint_components": "lag, node, u = A/A_min",
            "amr_orientation_grid": (
                "{} Gauss-Legendre nodes in cos(iota) on [-1, 1] times {} "
                "nodes in psi at (j+1/2) pi/{} on [0, pi/2); weights w_i/{}"
                .format(N_CI, N_PSI, 2 * N_PSI, N_CI)),
            "amr_inclination_units": "rad",
            "amr_polarization_range": "[0, pi/2); psi and psi+pi/2 are the "
                                      "same model point for this 22-mode "
                                      "statistic with the phase marginalised",
            "amr_delta_tc_units": "s, offset from the partition's arrival-time "
                                  "reference (recorded as 0)",
            "amr_amplitude_column": AMPLITUDE_COLUMN,
            "amr_amplitude_units": "u = A/A_min, dimensionless",
            "amr_amplitude_a_min": float(self.A_min),
            "amr_amplitude_u_best": float(diagnostics.get("u_best", numpy.nan)),
            "amr_distance_ratio_column": DISTANCE_RATIO_COLUMN,
            "amr_distance_ratio_definition": (
                "d/d_best = u_best/u, where u_best = {} is the amplitude that "
                "maximises the likelihood at the highest-weight (cell, lag, "
                "node) of the partition".format(
                    float(diagnostics.get("u_best", numpy.nan)))),
            "amr_distance_note": (
                "the absolute distance needs the source amplitude "
                "normalisation, which the partition does not carry: only "
                "u = A/A_min and d/d_best are available, and a distance in "
                "megaparsecs must not be invented"),
            "amr_drawn_corr_cos_lnu": float(
                diagnostics.get("corr_cos_lnu", numpy.nan)),
            "amr_drawn_corr_inclination_amplitude": float(
                diagnostics.get("corr_inclination_amplitude", numpy.nan)),
            "amr_drawn_distinct_lags": int(
                diagnostics.get("distinct_lags", 0)),
            "amr_drawn_cells_not_at_peak_lag": int(
                diagnostics.get("cells_not_at_peak_lag", 0)),
            "amr_drawn_amplitude_quantiles": numpy.asarray(
                diagnostics.get("amplitude_quantiles",
                                numpy.full(5, numpy.nan))),
            "amr_drawn_inclination_quantiles": numpy.asarray(
                diagnostics.get("inclination_quantiles",
                                numpy.full(5, numpy.nan))),
            "amr_drawn_quantile_levels": numpy.asarray(
                [0.05, 0.16, 0.5, 0.84, 0.95]),
        })
        return info
