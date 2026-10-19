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

The first factor is the joint over the lag and the orientation node, the
second is the exact conditional integral at the chosen pair.  Drawing them
in that order is an exact draw from the joint: it is not the product of
the two marginals, because ``u`` is drawn conditional on the pair that was
selected.  When the orientation interpolation is on, "node" means a node
of the fine 256 x 64 grid: ``Lbar`` there is the cubic interpolant of the
128 production values, the lag is drawn with the same fine-grid evidence,
and ``(s, y)`` for the amplitude conditional is evaluated exactly at the
drawn fine node.  The enumerated 128-node set remains the base of the
interpolation, and ``orientation-interpolation = false`` restores the old
enumerated draw.  It is the same semantics
:py:meth:`pycbc.inference.models.tools.DistMarg.reconstruct` uses for a
marginalised distance -- a draw from ``p(theta_aux | theta_sampled)`` -- applied
to the whole reduced axis rather than to one parameter at a time.

The drawn columns are

    inclination     arccos(cos iota) of the drawn fine node, in radians
    polarization    psi of the same fine node, on [0, pi/2)
    delta_tc        the drawn lag, in seconds
    amplitude_u     u = A / A_min of the drawn amplitude, dimensionless
    distance        the same draw in Mpc, d = K / u
    distance_ratio  d/d_best = u_best / u of the same draw

``inclination`` and ``polarization`` are one joint draw per cell from the
fine-grid conditional, and ``amplitude_u`` is the same cell's conditional
amplitude draw, so a corner plot of ``(inclination, amplitude_u)`` shows
the degeneracy rather than two marginals that have forgotten it.
``delta_tc`` is a draw from the enumerated lag grid: with several lags
retained per cell it takes more than one value where the archive gives the
lags comparable weight, which is the difference between a sample and a
report of the peak.  It is not interpolated; see the section below.

THE JOINT, NOT THE MARGINAL
---------------------------
The amplitude is the one genuine integral of the statistic, and its prior is
**registered and truncated**: ``u in [1, 5]`` with ``p(u) ~ u^-4``.  For the
stored controlled injection the likelihood mode is near ``u = 7.15`` and its
true amplitude is ``u = 6.673`` from the measured waveform norm (``6.717``
from the analytic inclination factor, 0.7 per cent higher), so the registered
prior cuts the likelihood
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

The three axes and the resolution each is recovered at.

The orientation is INTERPOLATED.  ``Lbar`` is a smooth function of
``(cos iota, psi)``, the search already forms it at the 16 x 8 nodes, and a
not-a-knot cubic spline in ``cos iota`` plus a periodic cubic spline of period
``pi/2`` in ``psi`` gives the between-node likelihood at zero additional
likelihood cost.  The recovery therefore forms the conditional over
``(cos iota, psi)`` on the resulting 256 x 64 grid, draws one fine node per
cell and dithers it uniformly inside its cell so the column is continuous
while the resolution stays the grid.  The interpolant was validated against
exact held-out evaluations at
1.27e-2 nats median, 0.254 at the 95th percentile and 0.751 worst; the
fine-grid marginal reproduces the converged density with L1 0.0081 and a
divergence from the prior of 1.350 nats against a converged 1.369.

The amplitude is SAMPLED FROM ITS OWN INTEGRAND.  It is the one genuine
integral inside the statistic, so its posterior is not a grid artefact: the
column is drawn from ``p(u | s, y)`` on the production 2001-node ``u`` grid, at
the exact ``(s, y)`` of the drawn fine node.  Drawing it conditionally on the
orientation is what preserves the ``(cos iota, ln u)`` ridge.

The arrival time is ENUMERATED, NOT INTERPOLATED.  The lag axis is a GRID
problem, not a resolution one: a local quadratic through the retained lags
misses its held-out lag by 7.87 nats median (19.5 at the 90th percentile and
28.4 worst), so no interpolant in lag is valid on this grid.  The fix is
oversampling the transform -- an FFT oversampling factor -- which changes the
grid rather than smoothing it, and that is not something a posterior pass can
do after the archive has been written.  ``delta_tc`` is therefore a draw from
the enumerated lag set, and its marginal is a histogram on that grid, not a
continuous density.

Every recovered quantity carries its resolution beside it in the file
metadata: ``amr_inclination_resolution``, ``amr_polarization_resolution``,
``amr_amplitude_resolution``, ``amr_distance_resolution`` and
``amr_delta_tc_resolution``, plus the measured interpolation error in
``amr_orientation_interpolation_error_*`` and the lag grid in
``amr_arrival_time_grid_s``.

FROM u TO A DISTANCE
--------------------
``u = A / A_min`` is dimensionless, and ``A_min`` is fixed by the search's own
calibration as the amplitude giving a face-on template an optimal SNR of 4 at
full overlap.  ``A`` multiplies the archived template, whose normalisation is
the SSB Gram's rather than the physical strain scale, so the archive's
``A_min`` alone does not fix a distance: the naive ``100 / A_min`` gives
7.0e10 Mpc, wrong by the ratio of the two scales.  The scale is recovered by
evaluating the waveform.  The partition's base point is recorded at 100 Mpc,
and under the search's own whitening the distance at which a face-on signal
reaches SNR 4 is

    d[Mpc] = K / u ,    K = 100 * rho_face(100 Mpc) / 4 .

``K`` is measured, not assumed.  Building the exact-TDI base waveform through
``pycbc.tdi.inference.source_from_parameters`` and whitening it exactly as the
search does (``ex.spectrum * sqrt(ex.weight)``) gives an optimal SNR of
``rho_face = 21.6840`` for the face-on base waveform at 100 Mpc, hence
``K = 542.1006 Mpc``.  A second determination evaluates the base orientation
instead (``rho = 9.4334`` at the base inclination 1.287) and extrapolates with
the two-polarization factor ``sqrt(((1 + c^2) / 2)^2 + c^2)``; it gives
548.9 Mpc, 1.3 per cent higher.  The two agree to that level, the residual
being the exact-TDI response's small departure from the analytic inclination
factor.  The controlled injection at ``rho = 16`` is at 81.24 Mpc, and the
drawn distance interval contains it.

``K`` is an option (``distance-constant``, in Mpc), defaulting to the measured
value.  The adapter raises rather than writing a distance when the constant is
unset: an unset constant would silently write ``u^-1``, which is a distance in
the wrong units.  ``u_best`` is the amplitude that maximises the likelihood at
the highest-weight (cell, lag, node) of the partition, so ``distance_ratio =
u_best / u`` is the distance relative to the best-fit amplitude of this
partition.

``distance`` is a deterministic function of the drawn ``amplitude_u``, so the
two columns are reciprocal to within ``K`` (``distance * amplitude_u = K``) and
the distance is correlated with the inclination and the amplitude by
construction.  It is a conditional draw, not a sampled dimension.  Its marginal
is controlled by the amplitude prior: a distance can be quoted only under an
amplitude prior whose support exceeds ``u`` of about 7 and which is flat over
five to nine.  The default draw uses the registered ``u^-4`` power, so a flat
prior is not the default; a user selects it with ``amplitude-power = 0`` (with
the default ceiling 20 when the archive is given).

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
from scipy.interpolate import CubicSpline

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


#: The fine grid the interpolated orientation draw is formed on: uniform nodes
#: in ``cos(iota)`` on ``[-1, 1]`` times uniform nodes in ``psi`` on
#: ``[0, pi/2)``.  The production grid is the 16 x 8 Gauss-Legendre set above;
#: the fine grid is a *sampling* grid, uniform because the prior is uniform, and
#: it is the resolution every recovered orientation column is drawn at.
N_CI_FINE = 256
N_PSI_FINE = 64
N_NODES_FINE = N_CI_FINE * N_PSI_FINE

#: A ``(cell, lag)`` row more than this many nats below its cell's strongest
#: row is not interpolated.  Its relative weight is below ``exp(-60) = 8.8e-27``,
#: so the cell's lag marginal and every drawn value are unchanged to machine
#: precision, while the pass avoids an exponential on rows that cannot be
#: drawn.  The rows remain part of the enumerated lag axis; only their
#: fine-grid conditional is skipped.
FINE_LAG_TRUNCATION_NATS = 60.0

#: Default sample used to measure the interpolation error written into the file
#: metadata.  The quoted values were validated on 200 cells and 3000 held-out
#: points; a recovery repeats the measurement on the archive it is writing, so
#: the number in the file is measured rather than copied.
INTERPOLATION_ERROR_CELLS = 200
INTERPOLATION_ERROR_POINTS = 2000
INTERPOLATION_ERROR_SEED = 20261019

#: The names the adapter writes for the restored joint draw.  The first three
#: are PyCBC's own parameter names (``pycbc.waveform.parameters``);
#: ``amplitude_u``, ``distance`` and ``distance_ratio`` are the amplitude and
#: distance columns.
RECOVERED_PARAMETERS = ("inclination", "polarization", "delta_tc",
                        "amplitude_u", "distance", "distance_ratio")

#: The amplitude column: u = A / A_min, dimensionless.
AMPLITUDE_COLUMN = "amplitude_u"

#: The distance column: d = K / u, in megaparsecs.
DISTANCE_COLUMN = "distance"

#: The distance-ratio column: d / d_best = u_best / u, dimensionless.
DISTANCE_RATIO_COLUMN = "distance_ratio"

#: The measured conversion constant: the distance in Mpc at which a face-on
#: signal has optimal SNR 4, i.e. u = 1.  Measured on 2026-10-18 from the
#: exact-TDI base waveform at the partition's recorded 100 Mpc under the
#: search's own whitening (rho_face = 21.684023211, so K = 100*rho_face/4).
#: A second determination from the base orientation with the analytic
#: inclination factor gives 548.9 Mpc, 1.3 per cent higher.  The value is a
#: default, not a law: a different partition or waveform path must measure it
#: again and pass ``distance-constant``.
DEFAULT_DISTANCE_CONSTANT_MPC = 542.1005802811611

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



def mu_weights(targets, mu_nodes):
    """Cubic-spline weights from the ``cos iota`` nodes to ``targets``.

    Returns ``W`` with shape ``(len(targets), len(mu_nodes))`` such that
    ``W @ values`` is the not-a-knot cubic spline through the node values,
    evaluated at each target.  The spline is linear in the values, so this is a
    fixed matrix and the per-row work is one small product.

    The nodes are Gauss-Legendre, so they are clustered at the ends of
    ``[-1, 1]`` and about 0.18 apart at the centre.  That is where the
    interpolation error lives, and it is measured rather than assumed; see
    :meth:`OrientationInterpolation.metadata`.
    """
    mu_nodes = numpy.asarray(mu_nodes, dtype=float)
    targets = numpy.asarray(targets, dtype=float)
    weights = numpy.zeros((len(targets), len(mu_nodes)))
    for j in range(len(mu_nodes)):
        basis = numpy.zeros(len(mu_nodes))
        basis[j] = 1.0
        weights[:, j] = CubicSpline(mu_nodes, basis)(targets)
    return weights


def psi_weights(targets, n):
    """Periodic cubic-spline weights from the ``psi`` nodes to ``targets``.

    ``psi`` is periodic with period ``pi/2`` for this statistic: ``psi`` and
    ``psi + pi/2`` are the same model point when the phase is marginalised, so
    the spline is built on three tiled periods and evaluated inside the middle
    one.  That makes the interpolant exactly periodic at the endpoints, and it
    is the rule the interpolation was validated with.  The production
    ``psi`` nodes are midpoints, so the tiling is the natural extension.
    """
    n = int(n)
    psi = (numpy.arange(n) + 0.5) * numpy.pi / (2 * n)
    period = 0.5 * numpy.pi
    x = numpy.concatenate([psi - period, psi, psi + period])
    targets = numpy.asarray(targets, dtype=float)
    weights = numpy.zeros((len(targets), n))
    for j in range(n):
        basis = numpy.zeros(3 * n)
        basis[j] = basis[j + n] = basis[j + 2 * n] = 1.0
        weights[:, j] = CubicSpline(x, basis)(targets)
    return weights


def fine_orientation_grid(n_mu=N_CI_FINE, n_psi=N_PSI_FINE):
    """The fine orientation grid: uniform nodes and their model coefficients.

    The prior on ``(cos iota, psi)`` is uniform on ``[-1, 1] x [0, pi/2)``, so
    the fine grid is uniform in both coordinates and every node carries the
    same prior mass ``1 / (n_mu n_psi)``.  The production grid is
    Gauss-Legendre because that is a good 16-node integral of the prior; the
    fine grid is uniform because it is a *sampling* grid, and a uniform grid is
    what makes the marginal a histogram of equal-width bars.

    Returns
    -------
    coefficients : (n_mu n_psi, 2) complex ndarray
        The two polarisation amplitudes of each node, in the order
        ``g = i * n_psi + j``.
    mu, psi : (n_mu n_psi,) ndarrays
        ``cos(iota)`` and ``psi`` of each node, flattened in the same order.
    mu_axis, psi_axis : (n_mu,), (n_psi,) ndarrays
        The one-dimensional axes.
    """
    n_mu = int(n_mu)
    n_psi = int(n_psi)
    mu_axis = numpy.linspace(-1.0, 1.0, n_mu)
    psi_axis = (numpy.arange(n_psi) + 0.5) * numpy.pi / (2 * n_psi)
    mu_grid, psi_grid = numpy.meshgrid(mu_axis, psi_axis, indexing="ij")
    hp = 0.5 * (1.0 + mu_grid ** 2)
    hc = -1j * mu_grid
    c2 = numpy.cos(2.0 * psi_grid)
    s2 = numpy.sin(2.0 * psi_grid)
    coefficients = numpy.stack(
        [c2 * hp - s2 * hc, s2 * hp + c2 * hc], axis=-1).reshape(-1, 2)
    return (coefficients, mu_grid.ravel(), psi_grid.ravel(),
            mu_axis, psi_axis)


class OrientationInterpolation(object):
    r"""Cubic interpolation of the per-node ``ln Lbar`` onto the fine grid.

    The production statistic evaluates the amplitude-marginalised likelihood at
    16 x 8 orientation nodes and sums them, then discards the summands.  Between
    the nodes the likelihood is a smooth function that is simply not evaluated,
    so it can be interpolated at zero extra likelihood cost: the interpolant is
    linear in the node values, and those values are already formed by the
    search.  The interpolant is separable -- a not-a-knot cubic spline in
    ``cos iota`` and a periodic cubic spline of period ``pi/2`` in ``psi`` -- so
    it is two fixed weight matrices and the per-row work is two small products.

    The interpolation error was measured against exact evaluations at held-out
    ``(cos iota, psi)`` on the stored archive: 1.27e-2 nats median, 0.254 at
    the 95th percentile and 0.751 worst.  Interpolating the 16 x 8 base onto
    this 256 x 64 grid reproduces the converged density with L1 0.0081 and a
    divergence from the prior of 1.350 nats against a converged 1.369, at zero
    additional likelihood evaluations.  The error is a property of the node
    spacing, not of the cell, and the recovery records a fresh measurement of
    it in the file metadata.

    Parameters
    ----------
    n_mu, n_psi : int, optional
        Fine-grid sizes.  The defaults, 256 and 64, are the grid the validation
        was run on.
    far_nats : float, optional
        Rows more than this many nats below their cell's strongest row are not
        interpolated; see :data:`FINE_LAG_TRUNCATION_NATS`.
    """

    def __init__(self, n_mu=N_CI_FINE, n_psi=N_PSI_FINE,
                 far_nats=FINE_LAG_TRUNCATION_NATS):
        self.n_mu = int(n_mu)
        self.n_psi = int(n_psi)
        self.n_nodes = self.n_mu * self.n_psi
        (self.coefficients, self.node_mu, self.node_psi,
         self.mu_axis, self.psi_axis) = fine_orientation_grid(
             self.n_mu, self.n_psi)
        ci = numpy.polynomial.legendre.leggauss(N_CI)[0]
        self.mu_weights = mu_weights(self.mu_axis, ci)
        self.psi_weights = psi_weights(self.psi_axis, N_PSI)
        # uniform prior: every fine node carries the same mass
        self.log_prior = -numpy.log(float(self.n_nodes))
        self.mu_spacing = float(self.mu_axis[1] - self.mu_axis[0])
        self.psi_spacing = 0.5 * numpy.pi / self.n_psi
        self.far_nats = float(far_nats)

    def log_weights(self, node_loglikelihood):
        """``ln[prior mass] + interpolated ln Lbar`` on the fine grid.

        ``node_loglikelihood`` has shape ``(rows, 128)`` in the production
        order ``g = i * 8 + j`` and must be the likelihood *without* the
        Gauss-Legendre prior weights: those belong to the production
        quadrature, while the fine grid integrates the uniform prior with equal
        node masses.  Returns ``(rows, 256 * 64)``.
        """
        values = numpy.asarray(node_loglikelihood, dtype=float)
        rows = values.shape[0]
        values = values.reshape(rows, N_CI, N_PSI)
        in_mu = numpy.einsum("rij,pi->rpj", values, self.mu_weights)
        in_psi = numpy.einsum("rpj,qj->rpq", in_mu, self.psi_weights)
        return in_psi.reshape(rows, self.n_nodes) + self.log_prior

    def log_evidence(self, node_loglikelihood):
        """The fine-grid log Bayes factor of each row."""
        return _logsumexp_last(self.log_weights(node_loglikelihood))

    def conditional(self, node_loglikelihood):
        """Normalised fine-grid conditional over the orientation, per row."""
        weights = self.log_weights(node_loglikelihood)
        weights = weights - _logsumexp_last(weights)[:, None]
        return numpy.exp(weights)

    def draw(self, node_loglikelihood, rng, chunk=512):
        """Draws one fine-grid orientation index per row from the conditional.

        Returns the flattened index, so ``self.node_mu[index]`` and
        ``self.node_psi[index]`` are the drawn values.
        """
        node_loglikelihood = numpy.asarray(node_loglikelihood, dtype=float)
        rows = node_loglikelihood.shape[0]
        index = numpy.empty(rows, dtype=numpy.int64)
        for start in range(0, rows, int(chunk)):
            stop = min(start + int(chunk), rows)
            probability = self.conditional(node_loglikelihood[start:stop])
            cumulative = numpy.cumsum(probability, axis=1)
            cumulative /= cumulative[:, -1:]
            value = rng.random(stop - start)
            flat = numpy.sum(cumulative < value[:, None], axis=1)
            index[start:stop] = numpy.clip(flat, 0, self.n_nodes - 1)
        return index

    def metadata(self):
        """The attributes that state the grid, its spacing and its error."""
        return {
            "amr_orientation_interpolation": (
                "the per-node log-likelihood Lbar is interpolated from the "
                "production 16 x 8 Gauss-Legendre orientation nodes onto a "
                "256 x 64 uniform grid in (cos iota, psi): not-a-knot cubic "
                "spline in cos iota, periodic cubic spline of period pi/2 in "
                "psi, equal prior mass 1/16384 per fine node.  The "
                "interpolant is linear in the node values and adds no "
                "likelihood evaluation; the drawn orientation is a sample "
                "from this fine-grid conditional, not from the enumerated "
                "node set.  The interpolation error is measured at 1.27e-2 "
                "nats median, 0.254 at the 95th percentile and 0.751 worst "
                "on the stored archive; see "
                "amr_orientation_interpolation_error_*."),
            "amr_orientation_fine_grid": (
                "{} uniform nodes in cos(iota) on [-1, 1] times {} uniform "
                "nodes in psi on [0, pi/2)".format(self.n_mu, self.n_psi)),
            "amr_orientation_fine_nodes": int(self.n_nodes),
            "amr_orientation_node_spacing": (
                "cos(iota) spacing {:.10g} ({} nodes over [-1, 1]); psi "
                "spacing {:.10g} rad ({} nodes over [0, pi/2))".format(
                    self.mu_spacing, self.n_mu, self.psi_spacing,
                    self.n_psi)),
            "amr_orientation_production_grid": (
                "{} Gauss-Legendre nodes in cos(iota) times {} midpoints in "
                "psi, the base the interpolation acts on".format(N_CI, N_PSI)),
            "amr_orientation_interpolation_cost": (
                "zero additional likelihood evaluations: the 16 x 8 node "
                "values are already formed by the search, and the interpolant "
                "is two fixed weight matrices.  The target grid costs one "
                "exponential per fine node per retained (cell, lag) row, "
                "which is a bookkeeping cost, not a template correlation."),
        }


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
        # the fine grid every interpolated orientation draw uses; built
        # once here because the weight matrices are fixed
        self.interpolation = OrientationInterpolation()

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

    def _statistics_at(self, coefficients, rows):
        """Per-node ``(s, y)`` at arbitrary orientation coefficients.

        ``coefficients`` has shape ``(m, 2)`` and ``rows`` shape ``(r,)``;
        returns ``s`` and ``y`` with shape ``(r, m)``.  The arithmetic is the
        same four real multiplies per element as :meth:`_statistics`, so the
        fine-grid draw and the interpolation-error measurement use the
        production response and not a second model of it.
        """
        coefficients = numpy.asarray(coefficients, dtype=numpy.complex128)
        rows = numpy.asarray(rows, dtype=numpy.int64)
        D = self.D[rows]
        G = self.Gram[rows]
        n = numpy.real(numpy.einsum(
            "ma,rab,mb->rm", numpy.conj(coefficients), G, coefficients))
        numpy.maximum(n, 1e-300, out=n)
        root = numpy.sqrt(n)
        s = self.A_min * root
        y = numpy.abs(
            numpy.conj(coefficients)[None, :, 0] * D[:, 0, None]
            + numpy.conj(coefficients)[None, :, 1] * D[:, 1, None]) / root
        return s, y

    def interpolation_error(self, table, n_cells=INTERPOLATION_ERROR_CELLS,
                            n_points=INTERPOLATION_ERROR_POINTS,
                            seed=INTERPOLATION_ERROR_SEED):
        """Measures exact minus interpolated ``ln Lbar`` at held-out points.

        The only honest way to quote an interpolation error is to evaluate the
        exact function where the interpolant did not see it.  The points are
        paired random draws uniform in ``(cos iota, psi)`` rather than a tensor
        grid, and the cells are the highest-weight ones, so the measurement is
        made where the posterior is.  No likelihood is evaluated: ``Lbar`` is
        the production amplitude table at the exact node statistic.

        Returns a dict with the median, 95th percentile and maximum absolute
        error in nats.  It is written into the file metadata by the recovery.
        """
        interpolator = self.interpolation
        rng = numpy.random.default_rng(int(seed))
        cells = numpy.argsort(self.two_f_peak)[::-1][:int(n_cells)]
        rows = self.best_row[cells]
        mu = rng.uniform(-1.0, 1.0, int(n_points))
        psi = rng.uniform(0.0, 0.5 * numpy.pi, int(n_points))
        c2 = numpy.cos(2.0 * psi)
        s2 = numpy.sin(2.0 * psi)
        hp = 0.5 * (1.0 + mu ** 2)
        hc = -1j * mu
        coefficients = numpy.stack(
            [c2 * hp - s2 * hc, s2 * hp + c2 * hc], axis=-1)
        exact_s, exact_y = self._statistics_at(coefficients, rows)
        exact = table(exact_s, exact_y)
        node_s, node_y = self._statistics(rows)
        node_lnl = table(node_s, node_y).reshape(len(rows), N_CI, N_PSI)
        ci = numpy.polynomial.legendre.leggauss(N_CI)[0]
        in_mu = numpy.einsum("rij,pi->rpj", node_lnl,
                             mu_weights(mu, ci))
        interpolated = numpy.einsum("rpj,pj->rp", in_mu,
                                    psi_weights(psi, N_PSI))
        error = exact - interpolated
        return {
            "n_cells": int(len(rows)),
            "n_points": int(n_points),
            "median_abs": float(numpy.median(numpy.abs(error))),
            "p95_abs": float(numpy.percentile(numpy.abs(error), 95)),
            "max_abs": float(numpy.max(numpy.abs(error))),
            "note": (
                "exact Lbar at random (cos iota, psi) minus the 16 x 8 cubic "
                "interpolant, on the highest-weight cells.  The error is a "
                "property of the node spacing, not of the cell; the validated "
                "values on the stored archive are 1.27e-2 nats median, 0.254 "
                "at the 95th percentile and 0.751 worst."),
        }

    def columns(self, ceiling, power=AMPLITUDE_POWER, seed=20261018,
                distance_constant=DEFAULT_DISTANCE_CONSTANT_MPC,
                weights=None, chunk=16384, n_u=N_U, return_marginal=False,
                orientation_interpolation=True,
                measure_interpolation_error=True):
        """Draws the joint ``(lag, orientation, u)`` columns, one per cell.

        This is the same joint draw as before -- one sample per cell from
        ``p(lag, orientation, u | cell)`` -- except that the orientation is no
        longer restricted to the enumerated 128-node set.  For every cell the
        conditional over ``(cos iota, psi)`` is formed on the interpolated
        256 x 64 grid, the lag is drawn with the fine-grid evidence of each
        retained row, ``(cos iota, psi)`` is drawn from that row's fine
        conditional and dithered uniformly inside the node's grid cell so
        the column is continuous, and only then is the amplitude drawn from
        its exact integrand at the drawn continuous point.  The three axes
        therefore keep their joint structure: the amplitude stays correlated
        with the orientation, and the lag stays an enumerated grid value
        rather than an interpolated one.

        Parameters
        ----------
        ceiling : float
            The upper limit ``U`` of ``u = A / A_min``.
        power : float, optional
            The power of the amplitude prior.  Default is the registered 4.
        distance_constant : float, optional
            The measured distance ``K`` in Mpc at which a face-on signal has
            optimal SNR 4, so ``distance = K / amplitude_u``.  Default is the
            measured 542.1006 Mpc.  ``None`` is refused, because an unset
            constant would write ``u^-1`` as if it were megaparsecs.
        seed : int, optional
            Seed of the draw, so a file is reproducible.
        weights : array-like, optional
            Cell weights for the reported correlations (normally
            ``exp(logwt)``).  When omitted the correlations are equal-weight.
        chunk : int, optional
            Approximate number of ``(cell, lag)`` rows to hold at once.
        n_u : int, optional
            Number of nodes of the amplitude grid.
        return_marginal : bool, optional
            Also return the per-cell orientation marginal
            ``p(node | cell) = sum_lag sum_u p(lag, node, u | cell)`` on the
            grid the draw used: the fine 256 x 64 grid when the interpolation
            is on, the enumerated 128 nodes when it is off.  It is offered so a
            test can show the draw follows that marginal; it is not written to
            the file and it is large, so use it on small archives only.
        orientation_interpolation : bool, optional
            When true (the default) the orientation is drawn from the cubic
            interpolant onto the fine grid.  ``False`` restores the enumerated
            node draw, which is kept so the old comb file and the old tests can
            be reproduced exactly.
        measure_interpolation_error : bool, optional
            Measure exact minus interpolated ``ln Lbar`` at held-out points on
            the archive being drawn and record it in the diagnostics.  The
            value is written into the file metadata by the adapter.

        Returns
        -------
        dict
            ``columns`` (partition order), ``diagnostics`` (archive order),
            ``table`` and, when requested, ``orientation_marginal``.
        """
        ceiling = float(ceiling)
        power = float(power)
        if distance_constant is None:
            raise ValueError(
                "distance-constant is not set: the conversion from u = A/A_min "
                "to a distance in Mpc is the measured constant K at which a "
                "face-on signal has optimal SNR 4 (542.1006 Mpc for the "
                "stored partition). Pass distance-constant in Mpc; without it "
                "the distance column would be u^-1 in the wrong units.")
        distance_constant = float(distance_constant)
        if (not numpy.isfinite(distance_constant)
                or distance_constant <= 0.0):
            raise ValueError(
                "distance-constant must be a positive finite distance in "
                "Mpc, got {}".format(distance_constant))
        s_lo, s_hi, y_lo, y_hi = self._table_bounds()
        table = AmplitudeMarginal.from_bounds(
            s_lo, s_hi, y_lo, y_hi, ceiling, power=power, n_u=n_u)
        rng = numpy.random.default_rng(int(seed))
        interpolator = (self.interpolation
                        if orientation_interpolation else None)
        n_draw_nodes = (N_NODES if interpolator is None
                        else interpolator.n_nodes)

        draw_node = numpy.empty(self.n_cells_archive, dtype=numpy.int64)
        draw_lag = numpy.empty(self.n_cells_archive, dtype=numpy.int64)
        draw_row = numpy.empty(self.n_cells_archive, dtype=numpy.int64)
        lnb = numpy.empty(self.n_cells_archive)
        peak_lag = numpy.empty(self.n_cells_archive, dtype=numpy.int64)
        marginal = (numpy.zeros((self.n_cells_archive, n_draw_nodes))
                    if return_marginal else None)

        best_value = -numpy.inf
        best_row = 0
        retained_rows = 0
        total_rows = 0
        max_lag = int(self.nlag.max())
        chunk_cells = max(1, int(chunk) // max_lag)
        far_nats = (FINE_LAG_TRUNCATION_NATS if interpolator is None
                    else interpolator.far_nats)

        for c0 in range(0, self.n_cells_archive, chunk_cells):
            c1 = min(c0 + chunk_cells, self.n_cells_archive)
            r0 = int(self.ptr[c0])
            r1 = int(self.ptr[c1])
            s, y = self._row_statistics(r0, r1)
            likelihood = table(s, y)
            nodes = likelihood + self.log_weights[None, :]
            row_lnb = _logsumexp_last(nodes)
            flat = int(numpy.argmax(row_lnb))
            if row_lnb[flat] > best_value:
                best_value = float(row_lnb[flat])
                best_row = r0 + flat
            starts = self.ptr[c0:c1] - r0
            stops = self.ptr[c0 + 1:c1 + 1] - r0
            for k in range(c1 - c0):
                a = c0 + k
                start, stop = int(starts[k]), int(stops[k])
                peak_lag[a] = self.lag[r0 + start
                                       + int(numpy.argmax(row_lnb[start:stop]))]
                if interpolator is None:
                    cell_lnb = float(_logsumexp_last(row_lnb[start:stop]))
                    lnb[a] = cell_lnb
                    probability = numpy.exp(nodes[start:stop] - cell_lnb)
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
                    continue
                # the fine conditional is formed only where the row can matter
                coarse = row_lnb[start:stop]
                keep = numpy.nonzero(coarse >= coarse.max() - far_nats)[0]
                retained_rows += int(keep.size)
                total_rows += int(stop - start)
                rows = start + keep
                fine_lnb = interpolator.log_evidence(likelihood[rows])
                cell_lnb = float(_logsumexp_last(fine_lnb))
                lnb[a] = cell_lnb
                if return_marginal:
                    for index, position in enumerate(keep):
                        marginal[a] += (
                            numpy.exp(fine_lnb[index] - cell_lnb)
                            * interpolator.conditional(
                                likelihood[start + position][None, :])[0])
                probability = numpy.exp(fine_lnb - cell_lnb)
                cumulative = numpy.cumsum(probability)
                total = float(cumulative[-1])
                position = int(numpy.searchsorted(
                    cumulative, rng.random() * total))
                position = min(position, keep.size - 1)
                row_local = int(keep[position])
                draw_row[a] = r0 + start + row_local
                draw_lag[a] = self.lag[r0 + start + row_local]
                draw_node[a] = int(interpolator.draw(
                    likelihood[start + row_local][None, :], rng)[0])

        rows = draw_row
        D = self.D[rows]
        G = self.Gram[rows]
        if interpolator is None:
            coefficients = self.coefficients[draw_node]
            node_mu_drawn = self.node_ci[draw_node]
            node_psi_drawn = self.node_psi[draw_node]
            drawn_mu = node_mu_drawn
            drawn_psi = node_psi_drawn
        else:
            node_mu_drawn = interpolator.node_mu[draw_node]
            node_psi_drawn = interpolator.node_psi[draw_node]
            # the fine grid is a piecewise-constant density: each node
            # carries the mass of its cell, so a continuous draw from it
            # is uniform inside the cell.  That is what makes the column
            # continuous without changing the node masses or the
            # resolution, which stay the 256 x 64 grid.
            dither_mu = ((rng.random(self.n_cells_archive) - 0.5)
                         * interpolator.mu_spacing)
            dither_psi = ((rng.random(self.n_cells_archive) - 0.5)
                          * interpolator.psi_spacing)
            drawn_mu = numpy.clip(node_mu_drawn + dither_mu, -1.0, 1.0)
            drawn_psi = numpy.mod(node_psi_drawn + dither_psi,
                                  0.5 * numpy.pi)
            hp = 0.5 * (1.0 + drawn_mu ** 2)
            hc = -1j * drawn_mu
            c2 = numpy.cos(2.0 * drawn_psi)
            s2 = numpy.sin(2.0 * drawn_psi)
            coefficients = numpy.stack(
                [c2 * hp - s2 * hc, s2 * hp + c2 * hc], axis=-1)
        norm = numpy.real(numpy.einsum(
            "ga,gab,gb->g", numpy.conj(coefficients), G, coefficients))
        numpy.maximum(norm, 1e-300, out=norm)
        root = numpy.sqrt(norm)
        draw_s = self.A_min * root
        draw_y = numpy.abs(
            numpy.conj(coefficients)[:, 0] * D[:, 0]
            + numpy.conj(coefficients)[:, 1] * D[:, 1]) / root
        u = table.draw(draw_s, draw_y, rng)

        best_rows = numpy.array([best_row], dtype=numpy.int64)
        best_s, best_y = self._statistics(best_rows)
        best_lnl = table(best_s, best_y)
        if interpolator is None:
            best_node = int(numpy.argmax(best_lnl[0] + self.log_weights))
            u_best = table.maximum_likelihood_u(
                best_s[0, best_node], best_y[0, best_node])
        else:
            fine = interpolator.log_weights(best_lnl)
            best_node = int(numpy.argmax(fine[0]))
            fine_s, fine_y = self._statistics_at(
                interpolator.coefficients[best_node][None, :], best_rows)
            u_best = table.maximum_likelihood_u(fine_s[0, 0], fine_y[0, 0])
        u_best = float(numpy.asarray(u_best).ravel()[0])

        inclination = numpy.arccos(drawn_mu)
        polarization = drawn_psi
        delta_tc = self.Delta[draw_lag]
        amplitude = u
        distance = distance_constant / amplitude
        distance_ratio = u_best / amplitude

        corr_inc_amp = weighted_correlation(inclination, amplitude, weights)
        corr_cos_lnu = weighted_correlation(
            drawn_mu, numpy.log(amplitude), weights)
        quantiles = (0.05, 0.16, 0.5, 0.84, 0.95)
        diagnostics = {
            "lnb": lnb,
            "peak_lag_index": peak_lag,
            "drawn_lag_index": draw_lag,
            "node": draw_node,
            "orientation_node_mu": node_mu_drawn,
            "orientation_node_psi": node_psi_drawn,
            "drawn_cos_iota": drawn_mu,
            "orientation_dither": bool(interpolator is not None),
            "s": draw_s,
            "y": draw_y,
            "amplitude": amplitude,
            "distance": distance,
            "distance_constant_mpc": float(distance_constant),
            "u_best": u_best,
            "corr_inclination_amplitude": float(corr_inc_amp),
            "corr_cos_lnu": float(corr_cos_lnu),
            "distinct_lags": int(numpy.unique(delta_tc).size),
            "cells_not_at_peak_lag": int(numpy.sum(draw_lag != peak_lag)),
            "amplitude_quantiles": numpy.quantile(
                amplitude, quantiles),
            "distance_quantiles": numpy.quantile(distance, quantiles),
            "inclination_quantiles": numpy.quantile(
                inclination, quantiles),
            "n_cells": int(self.n_cells_archive),
            "orientation_interpolation": bool(interpolator is not None),
            "orientation_fine_grid": (
                None if interpolator is None
                else (interpolator.n_mu, interpolator.n_psi)),
            "fine_rows_retained": int(retained_rows),
            "fine_rows_total": int(total_rows),
            "fine_lag_truncation_nats": float(far_nats),
        }
        if interpolator is not None and measure_interpolation_error:
            diagnostics["interpolation_error"] = self.interpolation_error(
                table)
        else:
            diagnostics["interpolation_error"] = None

        def in_partition_order(values):
            order = numpy.empty(self.n_cells)
            order[self.cells] = values
            return order

        columns = {
            "inclination": in_partition_order(inclination),
            "polarization": in_partition_order(polarization),
            "delta_tc": in_partition_order(delta_tc),
            AMPLITUDE_COLUMN: in_partition_order(amplitude),
            DISTANCE_COLUMN: in_partition_order(distance),
            DISTANCE_RATIO_COLUMN: in_partition_order(distance_ratio),
        }
        result = {"columns": columns, "diagnostics": diagnostics,
                  "table": table}
        if return_marginal:
            result["orientation_marginal"] = marginal
            if interpolator is not None:
                result["orientation_fine_axis"] = (
                    interpolator.mu_axis, interpolator.psi_axis)
        return result

    def metadata(self, ceiling, power, seed, diagnostics=None,
                 distance_constant=None):
        """The attributes that describe the restored joint draw.

        Every recovered quantity carries its own resolution beside it, so a
        reader of a panel knows what the panel can resolve: the orientation
        columns state the fine grid and its node spacing, the amplitude and
        distance state the amplitude grid, and the arrival time states the lag
        grid and that it is enumerated rather than interpolated.
        """
        info = self.report()
        diagnostics = {} if diagnostics is None else diagnostics
        interpolated = bool(diagnostics.get("orientation_interpolation",
                                            False))
        error = diagnostics.get("interpolation_error") or {}
        amplitude_nodes = int(N_U)
        amplitude_spacing = ((float(ceiling) - 1.0) / (amplitude_nodes - 1)
                             if amplitude_nodes > 1 else float("nan"))
        info.update({
            "amr_recovered_params": " ".join(RECOVERED_PARAMETERS),
            "amr_recovered_seed": int(seed),
            "amr_recovered_kind": (
                "one joint draw per cell from p(lag, orientation, u | cell) = "
                "p(lag, orientation | cell) p(u | lag, orientation, cell), "
                "with p(lag, orientation | cell) proportional to the "
                "orientation prior times Lbar on the interpolated fine "
                "orientation grid over the retained lags, and u drawn "
                "conditionally from the amplitude integrand at the chosen "
                "fine node. The draw is a proper joint sample, not a marginal "
                "draw followed by a conditional one. The columns are restored "
                "conditionals, not sampled dimensions: they do not enter the "
                "prior, the evidence or the cell weights."),
            "amr_joint_components": (
                "lag (enumerated), node (interpolated fine orientation), "
                "u = A/A_min (continuous)"),
            "amr_orientation_grid": (
                "production base: {} Gauss-Legendre nodes in cos(iota) on "
                "[-1, 1] times {} nodes in psi at (j+1/2) pi/{} on [0, "
                "pi/2); weights w_i/{}".format(
                    N_CI, N_PSI, 2 * N_PSI, N_CI)),
            "amr_inclination_units": "rad",
            "amr_polarization_range": "[0, pi/2); psi and psi+pi/2 are the "
                                      "same model point for this 22-mode "
                                      "statistic with the phase marginalised",
            "amr_delta_tc_units": "s, offset from the partition's arrival-time "
                                  "reference (recorded as 0)",
            "amr_amplitude_column": AMPLITUDE_COLUMN,
            "amr_amplitude_units": "u = A/A_min, dimensionless",
            "amr_amplitude_a_min": float(self.A_min),
            "amr_amplitude_u_best": float(diagnostics.get("u_best",
                                                          numpy.nan)),
            "amr_distance_column": DISTANCE_COLUMN,
            "amr_distance_units": "Mpc",
            "amr_distance_constant_mpc": float(
                distance_constant if distance_constant is not None
                else diagnostics.get("distance_constant_mpc", numpy.nan)),
            "amr_distance_definition": (
                "d[Mpc] = K/u, with u = amplitude_u and K = "
                "amr_distance_constant_mpc the distance at which a face-on "
                "signal has optimal SNR 4 (u = 1).  K is measured from the "
                "exact-TDI base waveform at the partition's recorded 100 Mpc "
                "under the search's whitening: K = 100*rho_face(100 Mpc)/4, "
                "rho_face = 21.6840, K = 542.1006 Mpc.  A second determination "
                "from the base orientation extrapolated with the analytic "
                "inclination factor gives 548.9 Mpc, 1.3 per cent higher; the "
                "two agree at that level."),
            "amr_amplitude_distance_relation": (
                "distance[Mpc] * amplitude_u = K to machine precision: the "
                "two columns are reciprocal to within the calibration "
                "constant K = amr_distance_constant_mpc, so a distance has "
                "exactly the information content of the amplitude it is "
                "derived from"),
            "amr_distance_note": (
                "distance is a deterministic function of the drawn "
                "amplitude_u, so it is correlated with inclination and "
                "amplitude by construction and is a conditional draw, not a "
                "sampled dimension.  Its marginal is controlled by the "
                "amplitude prior: quote a distance only under a prior whose "
                "support exceeds u of about 7 and which is flat over five to "
                "nine.  The default draw uses the registered u^-4 power, so a "
                "flat prior is NOT the default; select it with "
                "amplitude-power = 0."),
            "amr_distance_ratio_column": DISTANCE_RATIO_COLUMN,
            "amr_distance_ratio_definition": (
                "d/d_best = u_best/u, where u_best = {} is the amplitude that "
                "maximises the likelihood at the highest-weight (cell, lag, "
                "node) of the partition".format(
                    float(diagnostics.get("u_best", numpy.nan)))),
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
            "amr_drawn_distance_quantiles": numpy.asarray(
                diagnostics.get("distance_quantiles",
                                numpy.full(5, numpy.nan))),
            "amr_drawn_inclination_quantiles": numpy.asarray(
                diagnostics.get("inclination_quantiles",
                                numpy.full(5, numpy.nan))),
            "amr_drawn_quantile_levels": numpy.asarray(
                [0.05, 0.16, 0.5, 0.84, 0.95]),
        })

        # ------------------------------------------------------------------
        # the three axes, and why they are treated differently
        # ------------------------------------------------------------------
        if interpolated:
            info.update(self.interpolation.metadata())
            info.update({
                "amr_orientation_draw_kind": (
                    "interpolated: the fine-grid conditional over (cos iota, "
                    "psi) is formed from the cubic interpolant of the 16 x 8 "
                    "node log-likelihood, one fine node is drawn per cell "
                    "after the lag is drawn with the same fine-grid evidence, "
                    "and the orientation is then dithered uniformly within "
                    "the node's grid cell so the column is continuous"),
                "amr_orientation_dither": (
                    "after the fine node is drawn, cos(iota) is dithered "
                    "uniformly within half a node spacing and psi within "
                    "half a psi spacing (then wrapped into [0, pi/2)), so "
                    "the columns are continuous.  This renders the "
                    "piecewise-constant 256 x 64 density: the node masses "
                    "and the resolution are unchanged"),
                "amr_orientation_axis_note": (
                    "the orientation is INTERPOLATED because its per-node "
                    "log-likelihood is a smooth function of (cos iota, psi) "
                    "and the search already forms the node values, so the "
                    "between-node values are recoverable at zero additional "
                    "likelihood cost.  The amplitude is a continuous integral "
                    "inside the statistic, so its column is drawn from the "
                    "integrand on the u grid rather than from any grid of "
                    "orientation nodes.  The arrival time is NOT interpolated: "
                    "see amr_arrival_time_note."),
                "amr_inclination_resolution": (
                    "{} uniform nodes in cos(iota) on [-1, 1], spacing "
                    "{:.10g}; inclination = arccos(cos iota) of one of them, "
                    "so the column has at most {} distinct values and its "
                    "marginal is a {} -bar histogram of equal-width bars"
                    .format(self.interpolation.n_mu,
                            self.interpolation.mu_spacing,
                            self.interpolation.n_mu,
                            self.interpolation.n_mu)),
                "amr_polarization_resolution": (
                    "{} uniform nodes in psi on [0, pi/2), spacing {:.10g} "
                    "rad; the column has at most {} distinct values and its "
                    "marginal is a {} -bar histogram of equal-width bars"
                    .format(self.interpolation.n_psi,
                            self.interpolation.psi_spacing,
                            self.interpolation.n_psi,
                            self.interpolation.n_psi)),
            })
            if error:
                info.update({
                    "amr_orientation_interpolation_error_median_nats": float(
                        error["median_abs"]),
                    "amr_orientation_interpolation_error_p95_nats": float(
                        error["p95_abs"]),
                    "amr_orientation_interpolation_error_max_nats": float(
                        error["max_abs"]),
                    "amr_orientation_interpolation_error_cells": int(
                        error["n_cells"]),
                    "amr_orientation_interpolation_error_points": int(
                        error["n_points"]),
                    "amr_orientation_interpolation_error_note": str(
                        error["note"]),
                })
            else:
                info.update({
                    "amr_orientation_interpolation_error_note": (
                        "not measured on this pass; the validated values on "
                        "the stored archive are 1.27e-2 nats median, 0.254 at "
                        "the 95th percentile and 0.751 worst"),
                })
        else:
            info.update({
                "amr_orientation_draw_kind": (
                    "enumerated: one of the production 16 x 8 orientation "
                    "nodes, drawn from the node weights w_g Lbar(s, y)"),
                "amr_orientation_axis_note": (
                    "the orientation is ENUMERATED on this pass: one of the "
                    "128 production nodes is drawn per cell, so the "
                    "inclination and polarization columns take at most 16 and "
                    "8 distinct values and their marginals are combs"),
                "amr_inclination_resolution": (
                    "{} Gauss-Legendre nodes in cos(iota), no "
                    "interpolation: the column is a comb of at most "
                    "{} bars".format(N_CI, N_CI)),
                "amr_polarization_resolution": (
                    "{} midpoint nodes in psi, no interpolation: the "
                    "column is a comb of at most {} bars"
                    .format(N_PSI, N_PSI)),
                "amr_orientation_interpolation_error_note": (
                    "not applicable: the enumerated draw was selected"),
            })

        info.update({
            "amr_amplitude_resolution": (
                "continuous: u drawn on the {} -node grid linspace(1, {}, "
                "{}) with spacing {:.6g} in u; the prior p(u) ~ u^-{} is "
                "attached to that grid".format(
                    amplitude_nodes, float(ceiling), amplitude_nodes,
                    amplitude_spacing, float(power))),
            "amr_distance_resolution": (
                "K/u on the same amplitude grid, so the distance spacing is "
                "K du / u^2 and is not uniform; the column has exactly the "
                "resolution of amplitude_u"),
            "amr_delta_tc_resolution": (
                "enumerated on the arrival-time lag grid: one of the "
                "archive's lag values, spacing {:.10g} s; NOT interpolated "
                "(see amr_arrival_time_note)".format(self.dt)),
            "amr_arrival_time_grid_s": float(self.dt),
            "amr_arrival_time_distinct_lags": int(
                diagnostics.get("distinct_lags", 0)),
            "amr_arrival_time_reference_s": 0.0,
            "amr_arrival_time_note": (
                "the arrival time is a GRID problem, not a resolution one: a "
                "local quadratic through the retained lags misses its "
                "held-out lag by 7.87 nats median (19.5 at the 90th "
                "percentile and 28.4 worst), so no interpolant in lag is "
                "valid here and none is used.  The fix is OVERSAMPLING THE "
                "TRANSFORM (an FFT oversampling factor), not smoothing the "
                "enumerated lags.  delta_tc is therefore a draw from the "
                "enumerated lag grid, and its marginal is a histogram on that "
                "grid, not a continuous density."),
            "amr_recovered_resolutions": (
                "inclination: fine 256-node uniform cos(iota) grid, "
                "spacing {:.8g} | polarization: fine 64-node uniform psi "
                "grid, spacing {:.8g} rad | amplitude_u: continuous on the "
                "2001-node u grid | distance: K/u on the same grid | "
                "delta_tc: enumerated on the {:.6f} s lag grid"
                .format(self.interpolation.mu_spacing,
                        self.interpolation.psi_spacing, self.dt)),
        })
        return info
