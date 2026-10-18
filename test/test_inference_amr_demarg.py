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

"""Tests for the restored joint draw of the AMR adapter.

The restored inclination, polarization, arrival time and amplitude are one
joint draw per cell from ``p(lag, node, u | cell)``.  These tests build small
synthetic archives with the production array layout, so they exercise the
parts that need no GPU and no research repository: the orientation grid, the
amplitude marginal with a configurable ceiling and power, the archive/partition
binding, the joint draw and its reproducibility, the lag variety, the
amplitude column's distribution against an independently integrated reference,
the agreement of the joint's orientation marginal with the old one, and the
ordinary-column write.

A test against the real stored archive runs only when
``PYCBC_AMR_SUFFICIENT`` and ``PYCBC_AMR_PARTITION`` are set; it is the check
that the drawn ``(cos iota, ln u)`` correlation reproduces the archived -0.94
at the corrected ceiling.
"""

import os

import numpy
import pytest
from scipy.special import ive

from pycbc.inference.io import loadfile
from pycbc.inference.models.analytic import TestNormal
from pycbc.inference.sampler.amr import AmrSampler, ShapeToPhysicalMap
from pycbc.inference.sampler.amr_demarg import (
    AMPLITUDE_COLUMN, AMPLITUDE_POWER, DISTANCE_RATIO_COLUMN,
    REGISTERED_CEILING, AmplitudeMarginal, SufficientStatistics,
    orientation_grid, weighted_correlation)
from pycbc.workflow import WorkflowConfigParser


MAP = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "examples", "inference",
    "amr-partition", "physical-map.json"))

#: The A_min of the stored run, so the synthetic s = A_min sqrt(n) lands in the
#: production table's ln s range.
A_MIN = 1.4246657895237213e-09
#: A template norm that puts s near 2 for the face-on node.
NORM = 1.0e18
#: The lag grid spacing of the stored run, in seconds.
DT = 142.6782667983213

#: The archived amplitude posterior of the stored run lives in this directory;
#: the two environment variables let the optional real-archive test find it.
REAL_ARCHIVE = os.environ.get("PYCBC_AMR_SUFFICIENT")
REAL_PARTITION = os.environ.get("PYCBC_AMR_PARTITION")
REAL_POSTERIORS = os.environ.get("PYCBC_AMR_POSTERIORS")


def make_model(params=("eta_0", "eta_1")):
    """A model whose only role is to supply the variable-parameter names."""
    return TestNormal(list(params), mean=[0.0] * len(params),
                      cov=[1.0] * len(params))


def make_partition(tmp_path, truth=None, ncells=4, eta=None):
    """A partition with hand-computable weights."""
    if eta is None:
        eta = numpy.zeros((ncells, 6))
        eta[:, 0] = numpy.arange(ncells) * 0.01
        eta[:, 1] = numpy.arange(ncells) * 0.005
    ncells = len(eta)
    widths = numpy.full((ncells, 6), 0.5)
    lnb = numpy.log(numpy.arange(1.0, ncells + 1.0))
    path = tmp_path / "posteriors.npz"
    arrays = dict(eta_L6=eta, w_L6=widths, lnB_L6=lnb)
    if truth is not None:
        arrays["truth"] = numpy.asarray(truth, dtype=float)
    numpy.savez(path, **arrays)
    return str(path), eta, widths, lnb


def make_archive(tmp_path, eta, lnb, lags=None, cells=None, delta=None,
                 truth=None, d_scale=None, direction=(1.0, 0.0),
                 name="sufficient.npz"):
    """A synthetic sufficient-statistics archive with the production layout.

    Each row has ``D = d sqrt(NORM) * direction`` and ``Gram = NORM * I``, so
    ``s = A_min sqrt(NORM) sqrt(|c_p|^2 + |c_c|^2)`` and
    ``y = d |c^H direction| / sqrt(|c_p|^2 + |c_c|^2)``.  With the default
    ``d = 5`` and a real direction the amplitude lands inside ``[1, 20]``; a
    direction along one orientation node and a larger ``d`` make the
    ``(cos iota, ln u)`` ridge of the stored run.
    """
    ncells = len(eta)
    direction = numpy.asarray(direction, dtype=numpy.complex128)
    direction = direction / numpy.linalg.norm(direction)
    if lags is None:
        lags = [[0]] * ncells
    if cells is None:
        cells = numpy.arange(ncells)
    if d_scale is None:
        d_scale = numpy.full(ncells, 5.0)
    d_scale = numpy.asarray(d_scale, dtype=float)
    cells = numpy.sort(numpy.asarray(cells, dtype=numpy.int64))
    if delta is None:
        delta = numpy.arange(-3, 4) * DT
    rows = sum(len(item) for item in lags)
    D = numpy.zeros((rows, 2), dtype=numpy.complex64)
    G = numpy.zeros((rows, 2, 2), dtype=numpy.complex64)
    lag = numpy.zeros(rows, dtype=numpy.int64)
    two_f = numpy.zeros(rows)
    ptr = [0]
    for position, item in enumerate(lags):
        for k, index in enumerate(item):
            row = ptr[-1] + k
            G[row] = NORM * numpy.eye(2)
            D[row] = d_scale[position] * numpy.sqrt(NORM) * direction
            lag[row] = index
            two_f[row] = 100.0 + 10.0 * k
        ptr.append(ptr[-1] + len(item))
    arrays = dict(
        D=D, G=G, lag=lag, twoF=two_f, ptr=numpy.asarray(ptr),
        cell=numpy.repeat(cells, [len(item) for item in lags]),
        cells=cells, cellsw=numpy.full(ncells, 1.0 / ncells),
        eta=eta[cells], lnB_stored=lnb[cells],
        Dt=DT, Delta=numpy.asarray(delta), A_min=A_MIN,
        mass=numpy.full(ncells, 1.0 / ncells),
        eta_all=eta, lnB_all=lnb, w_all=numpy.full((ncells, 6), 0.5),
        kappa=1.0)
    if truth is not None:
        arrays["truth"] = numpy.asarray(truth, dtype=float)
    path = tmp_path / name
    numpy.savez(path, **arrays)
    return str(path)


def make_config(partition, extra="", eta_columns="0 1"):
    """A [sampler] + [sampler-amr] config naming the adapter."""
    text = """
[model]
name = test_normal

[sampler]
name = amr

[sampler-amr]
enable-research = false
partition-file = {partition}
partition-level = last
evidence = quadrature
eta-columns = {eta_columns}
{extra}
""".format(partition=partition, eta_columns=eta_columns, extra=extra)
    cp = WorkflowConfigParser()
    cp.read_string(text)
    return cp


def run_sampler(tmp_path, cp, model=None, name="amr.hdf"):
    sampler = AmrSampler.from_config(
        cp, model if model is not None else make_model(),
        output_file=str(tmp_path / name))
    sampler.run()
    return sampler


def reference_orientation_amplitude(d_scale, ceiling=20.0, power=4.0,
                                    n_u=201, direction=(1.0, 0.0)):
    """The reference joint over (cos iota, u) from the archive's formula.

    This is a deliberately separate evaluation of the same production
    arithmetic: one cell at a time, the 128 nodes, and the amplitude integrand
    on a u grid, accumulated with equal cell weight.  It returns the per-cell
    orientation marginal and the joint mass over ``(cos iota node, u)``.
    """
    coefficients, weights, node_ci, node_psi, ci, ci_w, psi = \
        orientation_grid()
    direction = numpy.asarray(direction, dtype=numpy.complex128)
    direction = direction / numpy.linalg.norm(direction)
    n = numpy.real(numpy.einsum(
        "ga,ab,gb->g", numpy.conj(coefficients),
        NORM * numpy.eye(2), coefficients))
    s = A_MIN * numpy.sqrt(n)
    overlap = numpy.abs(
        numpy.conj(coefficients[:, 0]) * direction[0]
        + numpy.conj(coefficients[:, 1]) * direction[1])
    y_scale = overlap / numpy.sqrt(n / NORM)
    u = numpy.linspace(1.0, ceiling, n_u)
    pu = u ** (-power)
    du = numpy.gradient(u)
    lw = numpy.log(pu / numpy.sum(pu * du)) + numpy.log(du)
    node_index = numpy.repeat(numpy.arange(16), 8)
    joint = numpy.zeros((16, n_u))
    orientation = numpy.zeros(128)
    for value in d_scale:
        y = value * y_scale
        arg = u[None, :] * s[:, None] * y[:, None]
        logj = (lw[None, :] - 0.5 * (u[None, :] * s[:, None]) ** 2
                + numpy.log(ive(0, arg)) + arg)
        logj += numpy.log(weights)[:, None]
        total = logj.max()
        mass = numpy.exp(logj - total)
        mass /= mass.sum()
        orientation += mass.sum(axis=1)
        for node in range(128):
            joint[node_index[node]] += mass[node]
    orientation /= orientation.sum()
    return u, orientation, joint


def test_the_orientation_grid_is_the_production_grid():
    """16 Gauss-Legendre nodes in cos(iota) times 8 psi nodes, node g=i*8+j."""
    coefficients, weights, node_ci, node_psi, ci, ci_weights, psi = \
        orientation_grid()
    assert coefficients.shape == (128, 2)
    assert weights.shape == (128,)
    assert weights.sum() == pytest.approx(1.0)
    expected_ci, expected_wi = numpy.polynomial.legendre.leggauss(16)
    numpy.testing.assert_allclose(ci, expected_ci)
    numpy.testing.assert_allclose(
        numpy.unique(node_ci), expected_ci)
    numpy.testing.assert_allclose(
        psi, (numpy.arange(8) + 0.5) * numpy.pi / 16.0)
    numpy.testing.assert_allclose(
        weights, numpy.repeat(expected_wi / 16.0, 8))
    # g = i * 8 + j
    assert node_ci[8 * 3 + 5] == pytest.approx(expected_ci[3])
    assert node_psi[8 * 3 + 5] == pytest.approx(psi[5])


def test_the_amplitude_ceiling_must_exceed_one():
    with pytest.raises(ValueError, match="must exceed 1"):
        AmplitudeMarginal(1.0)
    with pytest.raises(ValueError, match="must exceed 1"):
        AmplitudeMarginal(0.5)


def test_the_amplitude_power_must_be_non_negative():
    with pytest.raises(ValueError, match="non-negative"):
        AmplitudeMarginal(5.0, power=-1.0)


def test_the_amplitude_marginal_changes_with_the_ceiling():
    """Widening the support lowers the normalised marginal density."""
    s = numpy.array([2.0, 2.0])
    y = numpy.array([4.0, 6.0])
    registered = AmplitudeMarginal.from_data(s, y, REGISTERED_CEILING)
    widened = AmplitudeMarginal.from_data(s, y, 20.0)
    assert widened.ceiling == 20.0
    assert numpy.all(numpy.isfinite(registered(s, y)))
    assert numpy.all(numpy.isfinite(widened(s, y)))
    assert not numpy.allclose(registered(s, y), widened(s, y))
    # the table metadata names both the ceiling and the prior
    meta = widened.metadata()
    assert meta["amr_amplitude_ceiling"] == 20.0
    assert "u^-4" in meta["amr_amplitude_prior"]


def test_the_amplitude_integrand_and_the_table_agree():
    """The per-u integrand sums to the interpolated marginal at the nodes."""
    s = numpy.array([1.2, 1.6, 2.4, 3.0])
    y = numpy.array([0.0, 2.0, 5.0, 9.0])
    table = AmplitudeMarginal.from_data(s, y, 20.0)
    direct = table.log_marginal(s, y)
    interpolated = table(s, y)
    numpy.testing.assert_allclose(direct, interpolated, atol=2e-3)


def test_the_restored_columns_are_ordinary_samples(tmp_path):
    partition, eta, widths, lnb = make_partition(tmp_path)
    archive = make_archive(tmp_path, eta, lnb)
    cp = make_config(partition, "sufficient-file = {}\namplitude-ceiling = 20"
                     "\n".format(archive))
    sampler = run_sampler(tmp_path, cp)
    for name in ("inclination", "polarization", "delta_tc",
                 AMPLITUDE_COLUMN, DISTANCE_RATIO_COLUMN):
        assert name in sampler.samples
        assert len(sampler.samples[name]) == len(eta)
    inclination = sampler.samples["inclination"]
    polarization = sampler.samples["polarization"]
    amplitude = sampler.samples[AMPLITUDE_COLUMN]
    ratio = sampler.samples[DISTANCE_RATIO_COLUMN]
    assert numpy.all(inclination >= 0.0) and numpy.all(inclination <= numpy.pi)
    assert numpy.all(polarization >= 0.0)
    assert numpy.all(polarization < numpy.pi / 2.0)
    assert numpy.all(amplitude >= 1.0) and numpy.all(amplitude <= 20.0)
    assert numpy.all(ratio > 0.0) and numpy.all(numpy.isfinite(ratio))
    # the arrival-time column is a lag of the archive's grid, but now a draw
    assert set(numpy.unique(sampler.samples["delta_tc"])) == {-3.0 * DT}
    # and the file is still self-consistent
    numpy.testing.assert_allclose(
        sampler.samples["logwt"],
        sampler.samples["loglikelihood"] + sampler.samples["logprior"])
    with loadfile(str(sampler.checkpoint_file), "r") as fp:
        names = ["inclination", "polarization", "delta_tc",
                 AMPLITUDE_COLUMN, DISTANCE_RATIO_COLUMN]
        raw = fp.read_raw_samples(names, raw_samples=True)
        assert set(raw) == set(names)
        post = fp.read_raw_samples(names, nsamples=100)
        assert len(post["inclination"]) == 100
        assert fp.attrs["amr_amplitude_ceiling"] == 20.0
        assert fp.attrs["amr_amplitude_power"] == 4.0
        assert fp.attrs["amr_recovered_params"] == \
            "inclination polarization delta_tc amplitude_u distance_ratio"
        assert "joint draw" in fp.attrs["amr_recovered_kind"]
        assert "lag, node, u" in fp.attrs["amr_joint_components"]
        assert "dimensionless" in fp.attrs["amr_amplitude_units"]
        assert "megaparsecs" in fp.attrs["amr_distance_note"]
        assert "must not be invented" in fp.attrs["amr_distance_note"]
        assert fp.attrs["amr_amplitude_column"] == AMPLITUDE_COLUMN
        assert fp.attrs["amr_distance_ratio_column"] == DISTANCE_RATIO_COLUMN


def test_the_registered_prior_keeps_the_stored_bayes_factors(tmp_path):
    partition, eta, widths, lnb = make_partition(tmp_path)
    archive = make_archive(tmp_path, eta, lnb)
    registered = run_sampler(
        tmp_path, make_config(
            partition, "sufficient-file = {}\namplitude-ceiling = 5"
            "\namplitude-power = 4\n".format(archive)))
    numpy.testing.assert_array_equal(registered.samples["loglikelihood"], lnb)
    widened = run_sampler(
        tmp_path, make_config(
            partition, "sufficient-file = {}\namplitude-ceiling = 20"
            "\n".format(archive)), name="widened.hdf")
    assert not numpy.allclose(widened.samples["loglikelihood"], lnb)
    # the eta columns are untouched by the prior
    numpy.testing.assert_array_equal(
        widened.samples["eta_0"], registered.samples["eta_0"])
    numpy.testing.assert_array_equal(
        widened.samples["logprior"], registered.samples["logprior"])
    # both priors are recorded
    info = registered.recovered_metadata()
    assert info["amr_amplitude_ceiling"] == 5.0
    assert info["amr_amplitude_power"] == 4.0
    assert widened.recovered_metadata()["amr_amplitude_ceiling"] == 20.0


def test_the_amplitude_power_is_configurable(tmp_path):
    partition, eta, widths, lnb = make_partition(tmp_path)
    archive = make_archive(tmp_path, eta, lnb)
    flat = run_sampler(
        tmp_path, make_config(
            partition, "sufficient-file = {}\namplitude-ceiling = 20"
            "\namplitude-power = 0\n".format(archive)), name="flat.hdf")
    assert flat.recovered_metadata()["amr_amplitude_power"] == 0.0
    with loadfile(str(flat.checkpoint_file), "r") as fp:
        assert fp.attrs["amr_amplitude_power"] == 0.0
        assert "u^-0" in fp.attrs["amr_amplitude_prior"]
    # a changed power without the archive is refused
    with pytest.raises(ValueError, match="amplitude-power"):
        AmrSampler(make_model(), partition_file=partition, amplitude_power=0.0,
                   amplitude_ceiling=REGISTERED_CEILING)


def test_the_draw_is_reproducible_and_seeded(tmp_path):
    partition, eta, widths, lnb = make_partition(tmp_path)
    archive = make_archive(tmp_path, eta, lnb)
    first = run_sampler(
        tmp_path, make_config(
            partition, "sufficient-file = {}\nrecovered-seed = 11\n"
            .format(archive)), name="first.hdf")
    second = run_sampler(
        tmp_path, make_config(
            partition, "sufficient-file = {}\nrecovered-seed = 11\n"
            .format(archive)), name="second.hdf")
    other = run_sampler(
        tmp_path, make_config(
            partition, "sufficient-file = {}\nrecovered-seed = 12\n"
            .format(archive)), name="other.hdf")
    for name in ("inclination", "polarization", AMPLITUDE_COLUMN):
        numpy.testing.assert_array_equal(
            first.samples[name], second.samples[name])
    assert not numpy.array_equal(
        first.samples["inclination"], other.samples["inclination"])
    assert not numpy.array_equal(
        first.samples[AMPLITUDE_COLUMN], other.samples[AMPLITUDE_COLUMN])


def test_the_drawn_lag_takes_more_than_one_value(tmp_path):
    """With several equally weighted lags the drawn lag is not the peak."""
    eta = numpy.zeros((256, 6))
    eta[:, 0] = numpy.arange(256) * 0.01
    partition, eta, widths, lnb = make_partition(tmp_path, eta=eta)
    lags = [[0, 1, 2]] * len(eta)
    archive = make_archive(tmp_path, eta, lnb, lags=lags)
    statistics = SufficientStatistics(archive, eta, lnb)
    result = statistics.columns(20.0, power=4.0, seed=20261018, n_u=201)
    delta = result["columns"]["delta_tc"]
    values = numpy.unique(delta)
    # the four archive lags are Delta[0], Delta[1], Delta[2] and Delta[3]
    assert len(values) > 1
    assert set(values) <= set(numpy.arange(-3, 4) * DT)
    assert result["diagnostics"]["cells_not_at_peak_lag"] > 0
    assert result["diagnostics"]["distinct_lags"] > 1


def test_the_joint_draw_reproduces_the_archive_correlation(tmp_path):
    """The drawn (cos iota, ln u) correlation matches an independent integral."""
    ncells = 800
    # the archive's d must not be drawn from the same seed as the columns: the
    # same uniform would pick the node and set the cell's amplitude
    rng = numpy.random.default_rng(12345)
    eta = numpy.zeros((ncells, 6))
    eta[:, 0] = numpy.arange(ncells) * 0.01
    partition, eta, widths, lnb = make_partition(tmp_path, eta=eta)
    coefficients, _, _, _, _, _, _ = orientation_grid()
    direction = coefficients[12 * 8 + 3]
    d_scale = 12.0 * (0.85 + 0.3 * rng.random(ncells))
    archive = make_archive(tmp_path, eta, lnb, d_scale=d_scale,
                           direction=direction, name="ridge.npz")
    statistics = SufficientStatistics(archive, eta, lnb)
    result = statistics.columns(20.0, power=4.0, seed=20261018, n_u=201)

    u, orientation, joint = reference_orientation_amplitude(
        d_scale, direction=direction)
    ci = numpy.arccos(numpy.polynomial.legendre.leggauss(16)[0])
    expected = weighted_correlation(
        numpy.repeat(ci, len(u)), numpy.tile(u, 16), joint.ravel())

    drawn = weighted_correlation(
        result["columns"]["inclination"], result["columns"][AMPLITUDE_COLUMN])
    # inclination = arccos(cos iota), so this is the positive-sign reading of
    # the negative (cos iota, ln u) ridge
    assert expected > 0.6
    assert drawn == pytest.approx(expected, abs=0.10)
    # and the cos(iota)/ln u form is the number the archive quotes
    drawn_cos_lnu = weighted_correlation(
        numpy.cos(result["columns"]["inclination"]),
        numpy.log(result["columns"][AMPLITUDE_COLUMN]))
    expected_cos_lnu = weighted_correlation(
        numpy.repeat(numpy.polynomial.legendre.leggauss(16)[0], len(u)),
        numpy.tile(numpy.log(u), 16), joint.ravel())
    assert expected_cos_lnu < -0.6
    assert drawn_cos_lnu == pytest.approx(expected_cos_lnu, abs=0.10)


def test_the_inclination_marginal_matches_the_old_marginal(tmp_path):
    """Summing the joint over u reproduces the orientation marginal, and the
    drawn inclination follows it to resampling noise."""
    ncells = 4000
    eta = numpy.zeros((ncells, 6))
    partition, eta, widths, lnb = make_partition(tmp_path, eta=eta)
    archive = make_archive(tmp_path, eta, lnb)
    statistics = SufficientStatistics(archive, eta, lnb)
    result = statistics.columns(20.0, power=4.0, seed=20261018, n_u=201,
                                return_marginal=True)
    marginal = result["orientation_marginal"]
    # every cell is identical, so the old orientation marginal is one row
    numpy.testing.assert_allclose(
        marginal, numpy.broadcast_to(marginal[0], marginal.shape), atol=1e-12)

    # independent evaluation: sum over u first, then read the node marginal
    rows = numpy.arange(int(statistics.ptr[0]), int(statistics.ptr[1]))
    s, y = statistics._statistics(rows)
    table = AmplitudeMarginal.from_bounds(
        float(s.min()) * 0.9, float(s.max()) * 1.1, 0.0, float(y.max()) * 1.1,
        20.0, power=4.0, n_u=201)
    old = table.log_marginal(s[0], y[0]) + numpy.log(statistics.weights)
    old = numpy.exp(old - old.max())
    old /= old.sum()
    numpy.testing.assert_allclose(marginal[0], old, atol=5e-3,
                                  rtol=0.0)

    # the empirical inclination CDF follows the expected atom CDF
    inclination = result["columns"]["inclination"]
    ci = numpy.polynomial.legendre.leggauss(16)[0]
    p_cos = old.reshape(16, 8).sum(axis=1)
    atoms = numpy.arccos(ci)
    order = numpy.argsort(atoms)
    atoms = atoms[order]
    expected_cdf = numpy.cumsum(p_cos[order])
    empirical_cdf = numpy.searchsorted(
        numpy.sort(inclination), atoms, side="right") / ncells
    d_statistic = numpy.max(numpy.abs(empirical_cdf - expected_cdf))
    assert d_statistic < 0.04


def test_the_amplitude_column_matches_the_archived_posterior(tmp_path):
    """The drawn amplitude follows the integrated amplitude posterior."""
    ncells = 4000
    eta = numpy.zeros((ncells, 6))
    partition, eta, widths, lnb = make_partition(tmp_path, eta=eta)
    coefficients, _, _, _, _, _, _ = orientation_grid()
    direction = coefficients[12 * 8 + 3]
    archive = make_archive(tmp_path, eta, lnb, d_scale=numpy.full(ncells, 12.0),
                           direction=direction)
    statistics = SufficientStatistics(archive, eta, lnb)
    result = statistics.columns(20.0, power=4.0, seed=20261018, n_u=201)
    u, orientation, joint = reference_orientation_amplitude(
        numpy.array([12.0]), n_u=201, direction=direction)
    p_u = joint.sum(axis=0)
    p_u = p_u / p_u.sum()
    cdf = numpy.cumsum(p_u)
    cdf /= cdf[-1]
    draws = numpy.sort(result["columns"][AMPLITUDE_COLUMN])
    # compare at the support values: with ties, the empirical CDF has to be
    # evaluated after the whole tie group, not inside it
    support = numpy.unique(draws)
    expected_cdf = numpy.interp(support, u, cdf)
    empirical_cdf = numpy.searchsorted(draws, support, side="right") / draws.size
    d_statistic = float(numpy.max(numpy.abs(expected_cdf - empirical_cdf)))
    assert d_statistic < 0.04


def test_the_joint_reproduces_the_real_archive_correlation():
    """The drawn corr(cos iota, ln u) against the archived -0.94."""
    if not (REAL_ARCHIVE and REAL_PARTITION):
        pytest.skip("set PYCBC_AMR_SUFFICIENT and PYCBC_AMR_PARTITION")
    with numpy.load(REAL_PARTITION) as data:
        levels = sorted(int(key.rsplit("_L", 1)[1]) for key in data.files
                        if key.startswith("eta_L"))
        level = levels[-1]
        eta = numpy.asarray(data["eta_L%d" % level], dtype=float)
        widths = numpy.asarray(data["w_L%d" % level], dtype=float)
        lnb = numpy.asarray(data["lnB_L%d" % level], dtype=float)
    statistics = SufficientStatistics(REAL_ARCHIVE, eta, lnb)
    result = statistics.columns(20.0, power=4.0, seed=20261018)
    columns = result["columns"]
    volume = numpy.prod(widths[:, [0, 1, 2, 3]], axis=1)
    logmass = lnb + numpy.log(volume)
    weights = numpy.exp(logmass - logmass.max())
    corr = weighted_correlation(
        numpy.cos(columns["inclination"]),
        numpy.log(columns[AMPLITUDE_COLUMN]), weights)
    assert corr == pytest.approx(-0.94, abs=0.03)
    assert result["diagnostics"]["distinct_lags"] > 1


def test_the_real_amplitude_column_matches_the_archived_posterior():
    """The drawn amplitude follows the archived registered-prior posterior."""
    if not (REAL_ARCHIVE and REAL_PARTITION and REAL_POSTERIORS):
        pytest.skip("set PYCBC_AMR_SUFFICIENT, PYCBC_AMR_PARTITION and "
                    "PYCBC_AMR_POSTERIORS")
    with numpy.load(REAL_PARTITION) as data:
        levels = sorted(int(key.rsplit("_L", 1)[1]) for key in data.files
                        if key.startswith("eta_L"))
        level = levels[-1]
        eta = numpy.asarray(data["eta_L%d" % level], dtype=float)
        lnb = numpy.asarray(data["lnB_L%d" % level], dtype=float)
    with numpy.load(REAL_ARCHIVE) as data:
        cellsw = numpy.asarray(data["cellsw"], dtype=float)
    statistics = SufficientStatistics(REAL_ARCHIVE, eta, lnb)
    weights = numpy.empty(len(eta))
    weights[statistics.cells] = cellsw
    result = statistics.columns(5.0, power=4.0, seed=20261018,
                                weights=weights)
    amplitude = result["columns"][AMPLITUDE_COLUMN]
    with numpy.load(REAL_POSTERIORS) as data:
        u = numpy.asarray(data["u"], dtype=float)
        pu = numpy.asarray(data["pu"], dtype=float)
    order = numpy.argsort(amplitude)
    sorted_amplitude = amplitude[order]
    sorted_weights = weights[order]
    cw = numpy.cumsum(sorted_weights) / sorted_weights.sum()
    index = numpy.searchsorted(sorted_amplitude, u, side="right") - 1
    empirical = numpy.where(index >= 0,
                            cw[numpy.clip(index, 0, None)], 0.0)
    archived = numpy.cumsum(pu) / pu.sum()
    ks = float(numpy.max(numpy.abs(empirical - archived)))
    assert ks < 0.05


def test_an_archive_for_another_partition_is_refused(tmp_path):
    partition, eta, widths, lnb = make_partition(tmp_path)
    wrong_eta = eta.copy()
    wrong_eta[0, 0] += 0.1
    archive = make_archive(tmp_path, wrong_eta, lnb)
    sampler = AmrSampler(
        make_model(), partition_file=partition, sufficient_file=archive,
        amplitude_ceiling=5)
    with pytest.raises(ValueError, match="different cell centres"):
        sampler._sufficient_statistics(eta, lnb)

    archive = make_archive(tmp_path, eta, lnb)
    sampler = AmrSampler(
        make_model(), partition_file=partition, sufficient_file=archive,
        amplitude_ceiling=5)
    with pytest.raises(ValueError, match="different partition"):
        sampler._sufficient_statistics(eta, lnb + 1.0)


def test_a_wide_ceiling_needs_the_archive(tmp_path):
    partition, _, _, _ = make_partition(tmp_path)
    with pytest.raises(ValueError, match="sufficient-file"):
        AmrSampler(make_model(), partition_file=partition,
                   amplitude_ceiling=20)
    cp = make_config(partition, "amplitude-ceiling = 20")
    with pytest.raises(ValueError, match="sufficient-file"):
        AmrSampler.from_config(
            cp, make_model(), output_file=str(tmp_path / "amr.hdf"))


def test_the_archive_cannot_be_combined_with_the_research_path(tmp_path):
    with pytest.raises(ValueError, match="enable-research"):
        AmrSampler(make_model(), partition_file="unused.npz",
                   sufficient_file=str(tmp_path / "archive.npz"),
                   enable_research=True, amplitude_ceiling=5)


def test_the_physical_and_sky_columns_and_the_injections_group(tmp_path):
    shape_map = ShapeToPhysicalMap.from_file(MAP)
    partition, eta, widths, lnb = make_partition(
        tmp_path, truth=shape_map.injection["eta"], ncells=4)
    archive = make_archive(tmp_path, eta, lnb, truth=shape_map.injection["eta"])
    cp = make_config(
        partition,
        "sufficient-file = {}\nphysical-map = {}\n".format(archive, MAP),
        eta_columns="0 1 2 3")
    sampler = run_sampler(
        tmp_path, cp,
        model=make_model(("eta_0", "eta_1", "eta_2", "eta_3")))
    for name in ("mchirp", "q", "spin1z", "spin2z",
                 "eclipticlongitude", "eclipticlatitude"):
        assert name in sampler.samples
    with loadfile(str(sampler.checkpoint_file), "r") as fp:
        injection = fp.read_injections()
        for name in ("inclination", "polarization", "delta_tc",
                     "mchirp", "q", "spin1z", "spin2z",
                     "eclipticlongitude", "eclipticlatitude"):
            assert name in injection.fieldnames
        assert injection["inclination"][0] == pytest.approx(1.0)
        assert injection["polarization"][0] == pytest.approx(0.7)
        assert injection["delta_tc"][0] == pytest.approx(0.0)
