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
The orientation and arrival-time axes that the AMR Bayes factor reduces away.

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

The production code sums the 128 orientation values to one number per lag and
then sums the lags to one number per arrival-time segment, so the summands are
destroyed.  This module re-derives them from the archived sufficient
statistics ``(D, Gram)`` per (cell, lag) and exposes:

    inclination   arccos(cos iota) of a drawn orientation node, in radians
    polarization  psi of the same node, on [0, pi/2)
    delta_tc      the cell's single strongest arrival-time lag, in seconds

``inclination`` and ``polarization`` are **one conditional draw per cell** from
the cell's own 128-node orientation posterior, which is the same conditional
restoration :py:meth:`pycbc.inference.models.tools.DistMarg.reconstruct` does
for a distance: a draw from ``p(theta_aux | theta_sampled)``.  The resulting
column is a valid draw from the joint posterior of the cell centre and the
restored parameter, but the restored parameter is **not a sampled dimension**:
it does not enter the prior, the evidence, or the cell weights.  ``delta_tc``
is not even a draw: it is the peak lag that carries the cell's arrival-time
mass, and the non-peak lag residual is recorded rather than guessed.

THE AMPLITUDE CEILING
---------------------
The registered prior support is ``u in [1, 5]``.  For the stored injection
about 89 per cent of the amplitude likelihood lies above ``u = 5`` (the
likelihood mode is near ``u = 7.15``), so the registered ceiling truncates the
marginal and drives the recovered orientation to the face-on corner.  The
ceiling is therefore an explicit argument of :class:`AmplitudeMarginal`::

    RegisteredCeiling = 5.0     # the value baked into the stored Bayes factors
    DEFAULT_CEILING   = 20.0    # wide enough that the likelihood is contained

Widening the ceiling keeps the same ``u^-power`` prior; it changes the prior's
normalisation and support, not its shape.  ``power`` is held at the registered
value 4.  Both ceilings are reproducible from the same archive, and the adapter
records which one produced a file.

WHY THE TABLE LOOKS LIKE THE PRODUCTION ONE
-------------------------------------------
The amplitude marginal is evaluated on the same full axes as the production
table, ``ln s`` from ``ln 0.02`` to ``ln 30`` in 160 nodes and ``y`` from 0 to
45 in 901 nodes, and only the rows and columns the data touch are built.  The
nodes are therefore the production nodes and the bilinear interpolation is the
production interpolation, so at ``U = 5`` the rebuilt per-cell Bayes factor
reproduces the stored one (median absolute difference about ``3e-7``, maximum
about 2 for the low-mass cells whose Bayes factor is spread over several lags,
because this module keeps only the cell's strongest lag).
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

#: The two names the adapter writes for the restored orientation, plus the
#: arrival-time offset.  ``inclination``, ``polarization`` and ``delta_tc`` are
#: PyCBC's own parameter names (``pycbc.waveform.parameters``).
RECOVERED_PARAMETERS = ("inclination", "polarization", "delta_tc")

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
    r"""The production amplitude marginal, for a configurable prior ceiling.

    ``Lbar(s, y) = ln int_1^U du u^-power exp(-u^2 s^2 / 2) I0(u s y)`` on the
    production ``(ln s, y)`` axes, bilinearly interpolated exactly as
    ``orientation_pilot.AmplitudeTable`` does.  Only the sub-grid the data
    touch is built; the nodes are the production nodes.

    Parameters
    ----------
    ceiling : float
        The upper limit ``U`` of ``u = A / A_min``.  Must exceed 1.
    power : float, optional
        The power of the amplitude prior.  Default is 4, the registered value.
        ``0`` is a flat prior and is used only for diagnostics.
    n_u : int, optional
        Number of ``u`` nodes.  Default is 2001, the production number.
    ls_range, y_range : tuple of int, optional
        Inclusive node-index range of the ``ln s`` and ``y`` axes to build.
        When omitted the whole axis is built, which is slow; the adapter uses
        :meth:`from_data`.
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
                "prior, used only for diagnostics)")
        self.ceiling = ceiling
        self.power = float(power)
        self.n_u = int(n_u)
        u = numpy.linspace(1.0, ceiling, self.n_u)
        pu = u ** (-self.power)
        du = numpy.gradient(u)
        lw = numpy.log(pu / numpy.sum(pu * du)) + numpy.log(du)
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
            arg = u[None, :] * sv * yy[:, None]
            values = (lw[None, :] - 0.5 * (u[None, :] * sv) ** 2
                      + numpy.log(ive(0, arg)) + arg)
            table[a] = logsumexp(values, axis=-1)
        self.table = table

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
    to the partition, reduces each cell to its strongest lag, and turns the
    per-node ``(s, y)`` into the restored columns.

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
            D = numpy.asarray(data["D"])
            Gram = numpy.asarray(data["G"])
            lag = numpy.asarray(data["lag"], dtype=numpy.int64)
            two_f = numpy.asarray(data["twoF"], dtype=float)
            ptr = numpy.asarray(data["ptr"], dtype=numpy.int64)
            cells = numpy.asarray(data["cells"], dtype=numpy.int64)
            Delta = numpy.asarray(data["Delta"], dtype=float)
            A_min = float(data["A_min"])
            dt = float(data["Dt"])
            stored_eta = numpy.asarray(data["eta"], dtype=float)
            stored_lnb = numpy.asarray(data["lnB_stored"], dtype=float)
            truth = (numpy.asarray(data["truth"], dtype=float)
                     if "truth" in data.files else None)
        self.n_cells_archive = len(cells)
        self.n_rows = len(lag)
        if ptr.shape != (self.n_cells_archive + 1,):
            raise ValueError(
                "the sufficient-statistics archive is inconsistent: ptr has "
                "{} entries for {} cells".format(ptr.shape[0],
                                                 self.n_cells_archive))
        if not numpy.array_equal(numpy.sort(cells),
                                 numpy.arange(len(eta))):
            raise ValueError(
                "the sufficient-statistics archive does not cover this "
                "partition: it holds {} cells, the partition has {}. The "
                "recovered columns would be missing for some cells, so they "
                "are not written; re-run the pass with --all.".format(
                    self.n_cells_archive, len(eta)))
        self.cells = cells
        if not numpy.allclose(stored_eta, eta[cells], rtol=0.0, atol=1e-9):
            raise ValueError(
                "the sufficient statistics were archived for different cell "
                "centres: max |d eta| = {:.3e}".format(
                    numpy.max(numpy.abs(stored_eta - eta[cells]))))
        if not numpy.allclose(stored_lnb, lnb[cells],
                              rtol=0.0, atol=1e-4):
            raise ValueError(
                "the sufficient statistics were archived for a different "
                "partition: max |d lnB| = {:.3e}".format(
                    numpy.max(numpy.abs(stored_lnb - lnb[cells]))))
        if truth is not None and truth.shape == (6,):
            self.truth = truth
        else:
            self.truth = None
        self.lnb_stored = stored_lnb
        self.Delta = Delta
        self.A_min = A_min
        self.dt = dt

        # one row per cell: the strongest retained lag, exactly the row the
        # recovery used.  The lags are ordered inside a cell.
        nlag = numpy.diff(ptr)
        if numpy.any(nlag <= 0):
            raise ValueError("every archived cell must have at least one lag")
        best = numpy.empty(self.n_cells_archive, dtype=numpy.int64)
        for a in range(self.n_cells_archive):
            start, stop = ptr[a], ptr[a + 1]
            best[a] = start + int(numpy.argmax(two_f[start:stop]))
        self.best_row = best
        self.delta_tc_archive = Delta[lag[best]]
        self.two_f_peak = two_f[best]

        coefficients, self.weights, self.node_ci, self.node_psi, \
            self.ci, self.ci_weights, self.psi = orientation_grid()
        self.log_weights = numpy.log(self.weights)
        Db = D[best]
        Gb = Gram[best]
        zg = numpy.abs(numpy.conj(coefficients[None, :, 0]) * Db[:, 0, None]
                       + numpy.conj(coefficients[None, :, 1]) * Db[:, 1, None])
        n = numpy.real(numpy.einsum(
            "ga,sab,gb->sg", numpy.conj(coefficients), Gb, coefficients))
        if not numpy.all(n > 0):
            raise ValueError(
                "the archived Gram matrix gives a non-positive template norm "
                "for {} (cell, node) entries; the archive is not usable"
                .format(int(numpy.sum(n <= 0))))
        self.s = A_min * numpy.sqrt(n)
        self.y = zg / numpy.sqrt(n)
        self.clipped = int(numpy.sum(
            (self.s < numpy.exp(LS_AXIS[0]))
            | (self.s > numpy.exp(LS_AXIS[-1]))
            | (self.y > Y_AXIS[-1])))
        del D, Gram

    def report(self):
        """The numbers that justify the archive and the peak-lag restriction."""
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

    def conditional(self, ceiling, power=AMPLITUDE_POWER, n_u=N_U):
        """The per-cell orientation conditional at one amplitude ceiling.

        Returns a dict with the table, the per-node log statistic, the per-cell
        ``lnB`` from the strongest lag, the normalised per-cell node
        probabilities and the non-peak-lag residual measured against the stored
        Bayes factor.
        """
        table = AmplitudeMarginal.from_data(
            self.s, self.y, ceiling, power=power, n_u=n_u)
        nodes = table(self.s, self.y) + self.log_weights[None, :]
        lnb = logsumexp(nodes, axis=1)
        # p(g | cell): the orientation conditional of one cell
        probability = numpy.exp(nodes - lnb[:, None])
        delta_lnb = lnb - self.lnb_stored
        # the fraction of the cell's arrival-time mass that its single
        # strongest lag does not carry, measured against the stored Bayes
        # factor.  Negative values are table arithmetic above the stored
        # number, which the archive's float32 D and Gram make possible.
        residual_raw = 1.0 - numpy.exp(numpy.clip(delta_lnb, -700.0, 700.0))
        return {
            "table": table,
            "nodes": nodes,
            "lnb": lnb,
            "probability": probability,
            "residual_raw": residual_raw,
            "residual": numpy.clip(residual_raw, 0.0, 1.0),
            "delta_lnb": delta_lnb,
        }

    def columns(self, probability, seed):
        """Draws the restored columns, in the partition's cell order.

        One node is drawn per cell from that cell's orientation conditional.
        The draw is from ``p(node | cell)``, so the equal-weight-resampled
        histogram of the columns is the recovered marginal; the columns are
        conditionals, not sampled dimensions.
        """
        rng = numpy.random.default_rng(int(seed))
        cdf = numpy.cumsum(probability, axis=1)
        cdf[:, -1] = 1.0
        draws = rng.random(self.n_cells_archive)[:, None]
        node = numpy.sum(draws > cdf, axis=1)
        node = numpy.clip(node, 0, probability.shape[1] - 1)
        inclination = numpy.arccos(self.node_ci[node])
        polarization = self.node_psi[node]
        columns = {
            "inclination": numpy.empty(len(self.cells)),
            "polarization": numpy.empty(len(self.cells)),
            "delta_tc": numpy.empty(len(self.cells)),
        }
        for name, values in (("inclination", inclination),
                             ("polarization", polarization),
                             ("delta_tc", self.delta_tc_archive)):
            columns[name][self.cells] = values
        return columns

    def metadata(self, ceiling, seed):
        """The attributes that describe the restored columns."""
        info = self.report()
        info.update({
            "amr_recovered_params": " ".join(RECOVERED_PARAMETERS),
            "amr_recovered_seed": int(seed),
            "amr_recovered_kind": (
                "inclination and polarization are one conditional draw per "
                "cell from that cell's 128-node orientation posterior; "
                "delta_tc is the cell's single strongest arrival-time lag. "
                "They are restored conditionals, not sampled dimensions: "
                "they do not enter the prior, the evidence or the cell "
                "weights."),
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
        })
        return info
