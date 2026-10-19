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
    AMPLITUDE_COLUMN, AMPLITUDE_POWER, DEFAULT_DISTANCE_CONSTANT_MPC,
    DISTANCE_COLUMN, DISTANCE_RATIO_COLUMN, N_CI_FINE, N_PSI_FINE,
    REGISTERED_CEILING, AmplitudeMarginal, OrientationInterpolation,
    SufficientStatistics, fine_orientation_grid, mu_weights,
    orientation_grid, psi_weights, weighted_correlation)
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

#: The measured distance constant of the stored partition, in Mpc, and the
#: distance of its controlled injection (rho = 16, inclination 1.0).  Both are
#: measured by the research calibration pass; the optional real-archive test
#: checks that the drawn 68 per cent distance interval contains the injection.
DISTANCE_CONSTANT = DEFAULT_DISTANCE_CONSTANT_MPC
INJECTED_DISTANCE = 81.24030874912705

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
                 AMPLITUDE_COLUMN, DISTANCE_COLUMN, DISTANCE_RATIO_COLUMN):
        assert name in sampler.samples
        assert len(sampler.samples[name]) == len(eta)
    inclination = sampler.samples["inclination"]
    polarization = sampler.samples["polarization"]
    amplitude = sampler.samples[AMPLITUDE_COLUMN]
    distance = sampler.samples[DISTANCE_COLUMN]
    ratio = sampler.samples[DISTANCE_RATIO_COLUMN]
    numpy.testing.assert_array_equal(
        distance, DISTANCE_CONSTANT / amplitude)
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
                 AMPLITUDE_COLUMN, DISTANCE_COLUMN, DISTANCE_RATIO_COLUMN]
        raw = fp.read_raw_samples(names, raw_samples=True)
        assert set(raw) == set(names)
        post = fp.read_raw_samples(names, nsamples=100)
        assert len(post["inclination"]) == 100
        assert fp.attrs["amr_amplitude_ceiling"] == 20.0
        assert fp.attrs["amr_amplitude_power"] == 4.0
        assert fp.attrs["amr_recovered_params"] == \
            "inclination polarization delta_tc amplitude_u distance distance_ratio"
        assert "joint draw" in fp.attrs["amr_recovered_kind"]
        assert "lag" in fp.attrs["amr_joint_components"]
        assert "node" in fp.attrs["amr_joint_components"]
        assert fp.attrs["amr_orientation_fine_nodes"] == \
            N_CI_FINE * N_PSI_FINE
        assert "NOT interpolated" in fp.attrs[
            "amr_delta_tc_resolution"]
        assert fp.attrs[
            "amr_orientation_interpolation_error_median_nats"] >= 0.0
        assert "dimensionless" in fp.attrs["amr_amplitude_units"]
        assert fp.attrs["amr_distance_column"] == DISTANCE_COLUMN
        assert fp.attrs["amr_distance_units"] == "Mpc"
        assert fp.attrs["amr_distance_constant_mpc"] == pytest.approx(
            DISTANCE_CONSTANT, rel=0.0, abs=0.0)
        assert "reciprocal" in fp.attrs["amr_amplitude_distance_relation"]
        assert "flat" in fp.attrs["amr_distance_note"]
        assert "not a sampled dimension" in fp.attrs["amr_distance_note"]
        assert fp.attrs["amr_amplitude_column"] == AMPLITUDE_COLUMN
        assert fp.attrs["amr_distance_ratio_column"] == DISTANCE_RATIO_COLUMN


def test_the_distance_column_is_the_constant_over_the_amplitude(tmp_path):
    """distance = K / amplitude_u to machine precision, and the file says so."""
    partition, eta, widths, lnb = make_partition(tmp_path)
    archive = make_archive(tmp_path, eta, lnb)
    statistics = SufficientStatistics(archive, eta, lnb)
    result = statistics.columns(
        20.0, power=0.0, seed=20261018,
        distance_constant=DISTANCE_CONSTANT)
    distance = result["columns"][DISTANCE_COLUMN]
    amplitude = result["columns"][AMPLITUDE_COLUMN]
    assert numpy.all(distance > 0.0)
    numpy.testing.assert_array_equal(distance, DISTANCE_CONSTANT / amplitude)
    info = statistics.metadata(
        20.0, 0.0, 20261018, result["diagnostics"],
        distance_constant=DISTANCE_CONSTANT)
    assert info["amr_distance_column"] == DISTANCE_COLUMN
    assert info["amr_distance_units"] == "Mpc"
    assert info["amr_distance_constant_mpc"] == DISTANCE_CONSTANT
    assert "reciprocal" in info["amr_amplitude_distance_relation"]
    # the measured constant is the default of the draw, and a caller can
    # override it without touching the amplitude column
    default = statistics.columns(20.0, power=0.0, seed=20261018)
    numpy.testing.assert_array_equal(default["columns"][DISTANCE_COLUMN],
                                    distance)
    other = statistics.columns(
        20.0, power=0.0, seed=20261018, distance_constant=2.0 * DISTANCE_CONSTANT)
    numpy.testing.assert_array_equal(
        other["columns"][DISTANCE_COLUMN], 2.0 * distance)
    numpy.testing.assert_array_equal(
        other["columns"][AMPLITUDE_COLUMN], amplitude)


def test_the_distance_constant_must_be_set(tmp_path):
    """An unset constant is refused; it must not silently write u^-1 as Mpc."""
    partition, eta, widths, lnb = make_partition(tmp_path)
    archive = make_archive(tmp_path, eta, lnb)
    with pytest.raises(ValueError, match="distance-constant is not set"):
        AmrSampler(make_model(), partition_file=partition,
                   sufficient_file=archive, distance_constant=None)
    statistics = SufficientStatistics(archive, eta, lnb)
    with pytest.raises(ValueError, match="distance-constant is not set"):
        statistics.columns(20.0, distance_constant=None)
    with pytest.raises(ValueError, match="positive finite"):
        AmrSampler(make_model(), partition_file=partition,
                   sufficient_file=archive, distance_constant=0.0)


def test_the_file_metadata_records_the_distance_constant(tmp_path):
    """The written file names K, its units and the reciprocal relation."""
    partition, eta, widths, lnb = make_partition(tmp_path)
    archive = make_archive(tmp_path, eta, lnb)
    sampler = run_sampler(
        tmp_path, make_config(
            partition,
            "sufficient-file = {}\namplitude-ceiling = 20\n"
            "amplitude-power = 0\ndistance-constant = {}\n".format(
                archive, DISTANCE_CONSTANT)))
    with loadfile(str(sampler.checkpoint_file), "r") as fp:
        assert fp.attrs["amr_distance_constant_mpc"] == pytest.approx(
            DISTANCE_CONSTANT, rel=0.0, abs=0.0)
        assert fp.attrs["amr_distance_units"] == "Mpc"
        assert "d[Mpc] = K/u" in fp.attrs["amr_distance_definition"]
        assert "reciprocal" in fp.attrs["amr_amplitude_distance_relation"]
        assert "flat" in fp.attrs["amr_distance_note"]
        assert "amplitude-power = 0" in fp.attrs["amr_distance_note"]
    numpy.testing.assert_array_equal(
        sampler.samples[DISTANCE_COLUMN],
        DISTANCE_CONSTANT / sampler.samples[AMPLITUDE_COLUMN])


def test_the_real_distance_interval_contains_the_injection():
    """The controlled injection's drawn 68 per cent distance set contains it."""
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
    # the flat amplitude prior is the one under which the marginal is
    # statistically consistent; the default draw keeps u^-4
    result = statistics.columns(20.0, power=0.0, seed=20261018,
                                distance_constant=DISTANCE_CONSTANT)
    distance = result["columns"][DISTANCE_COLUMN]
    volume = numpy.prod(widths[:, [0, 1, 2, 3]], axis=1)
    logmass = lnb + numpy.log(volume)
    weights = numpy.exp(logmass - logmass.max())
    order = numpy.argsort(distance)
    cumulative = numpy.cumsum(weights[order]) / numpy.sum(weights)
    low = distance[order][numpy.searchsorted(cumulative, 0.16)]
    high = distance[order][numpy.searchsorted(cumulative, 0.84)]
    median = distance[order][numpy.searchsorted(cumulative, 0.50)]
    assert low < INJECTED_DISTANCE < high
    assert 0.7 < median / INJECTED_DISTANCE < 1.3


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


def test_the_fine_orientation_grid_is_uniform_and_carries_equal_mass():
    """The draw grid is uniform in both angles, with 1/N prior mass per node."""
    coefficients, mu, psi, mu_axis, psi_axis = fine_orientation_grid()
    n = N_CI_FINE * N_PSI_FINE
    assert coefficients.shape == (n, 2)
    assert mu_axis.shape == (N_CI_FINE,)
    assert psi_axis.shape == (N_PSI_FINE,)
    numpy.testing.assert_allclose(
        mu_axis, numpy.linspace(-1.0, 1.0, N_CI_FINE))
    numpy.testing.assert_allclose(
        psi_axis, (numpy.arange(N_PSI_FINE) + 0.5) * numpy.pi
        / (2 * N_PSI_FINE))
    numpy.testing.assert_allclose(mu, numpy.repeat(mu_axis, N_PSI_FINE))
    numpy.testing.assert_allclose(psi, numpy.tile(psi_axis, N_CI_FINE))
    interpolator = OrientationInterpolation()
    weights = numpy.exp(interpolator.log_weights(numpy.zeros((1, 128))))
    numpy.testing.assert_allclose(weights, numpy.full((1, n), 1.0 / n),
                                  rtol=1e-12, atol=0.0)
    # grid and node spacing are the resolution the columns are drawn at
    assert interpolator.mu_spacing == pytest.approx(
        2.0 / (N_CI_FINE - 1))
    assert interpolator.psi_spacing == pytest.approx(
        0.5 * numpy.pi / N_PSI_FINE)


def test_the_splines_are_exact_at_their_own_nodes():
    """The weights are an interpolant: identity at the production nodes."""
    ci = numpy.polynomial.legendre.leggauss(16)[0]
    numpy.testing.assert_allclose(mu_weights(ci, ci), numpy.eye(16),
                                  atol=1e-12)
    psi = (numpy.arange(8) + 0.5) * numpy.pi / 16.0
    numpy.testing.assert_allclose(psi_weights(psi, 8), numpy.eye(8),
                                  atol=1e-12)


def test_the_interpolation_preserves_the_inclination_expectation(tmp_path):
    """E[cos iota] on the interpolated grid equals the exact fine-grid value.

    This is the test the interpolation has to pass: it may smooth between the
    nodes, but it must not move the first moment of the orientation marginal.
    The exact comparison evaluates Lbar at the fine nodes directly, so the
    difference is the interpolation error alone, not the quadrature difference
    between the 16 Gauss-Legendre nodes and the fine uniform grid.
    """
    eta = numpy.zeros((1, 6))
    partition, eta, widths, lnb = make_partition(tmp_path, eta=eta)
    coefficients, _, _, _, _, _, _ = orientation_grid()
    direction = coefficients[12 * 8 + 3]
    archive = make_archive(tmp_path, eta, lnb, d_scale=numpy.array([12.0]),
                           direction=direction)
    statistics = SufficientStatistics(archive, eta, lnb)
    rows = numpy.array([int(statistics.ptr[0])])
    node_s, node_y = statistics._statistics(rows)
    s_lo, s_hi, y_lo, y_hi = statistics._table_bounds()
    table = AmplitudeMarginal.from_bounds(
        s_lo, s_hi, y_lo, y_hi, 20.0, power=0.0, n_u=201)
    coarse = table(node_s, node_y)
    interpolator = statistics.interpolation
    interpolated = interpolator.conditional(coarse)[0]
    mu_mass = interpolated.reshape(N_CI_FINE, N_PSI_FINE).sum(axis=1)
    interpolated_mu = float(mu_mass @ interpolator.mu_axis)
    interpolated_iota = float(
        mu_mass @ numpy.arccos(interpolator.mu_axis))

    exact_s, exact_y = statistics._statistics_at(
        interpolator.coefficients, rows)
    exact = table(exact_s, exact_y)[0]
    exact = numpy.exp(exact - exact.max())
    exact /= exact.sum()
    exact_mu_mass = exact.reshape(N_CI_FINE, N_PSI_FINE).sum(axis=1)
    exact_mu = float(exact_mu_mass @ interpolator.mu_axis)
    exact_iota = float(exact_mu_mass @ numpy.arccos(interpolator.mu_axis))

    assert abs(interpolated_mu - exact_mu) < 2e-3
    assert abs(interpolated_iota - exact_iota) < 2e-3


def test_the_drawn_inclination_follows_the_fine_grid(tmp_path):
    """The empirical inclination CDF follows the interpolated fine marginal.

    The cells are identical, so there is one expected distribution and 4000
    independent draws from it.  This checks the draw, not the interpolation:
    the expected conditional is the one the draw was taken from.
    """
    ncells = 4000
    eta = numpy.zeros((ncells, 6))
    partition, eta, widths, lnb = make_partition(tmp_path, eta=eta)
    coefficients, _, _, _, _, _, _ = orientation_grid()
    direction = coefficients[12 * 8 + 3]
    archive = make_archive(tmp_path, eta, lnb,
                           d_scale=numpy.full(ncells, 12.0),
                           direction=direction)
    statistics = SufficientStatistics(archive, eta, lnb)
    result = statistics.columns(20.0, power=0.0, seed=20261018, n_u=201)
    inclination = result["columns"]["inclination"]
    rows = numpy.array([int(statistics.ptr[0])])
    node_s, node_y = statistics._statistics(rows)
    s_lo, s_hi, y_lo, y_hi = statistics._table_bounds()
    table = AmplitudeMarginal.from_bounds(
        s_lo, s_hi, y_lo, y_hi, 20.0, power=0.0, n_u=201)
    probability = statistics.interpolation.conditional(
        table(node_s, node_y))[0]
    mu_mass = probability.reshape(N_CI_FINE, N_PSI_FINE).sum(axis=1)
    atoms = numpy.arccos(statistics.interpolation.mu_axis)
    order = numpy.argsort(atoms)
    atoms = atoms[order]
    expected_cdf = numpy.cumsum(mu_mass[order])
    empirical_cdf = numpy.searchsorted(
        numpy.sort(inclination), atoms, side="right") / ncells
    d_statistic = float(numpy.max(numpy.abs(empirical_cdf - expected_cdf)))
    assert d_statistic < 0.05
    # the column is no longer an eight-bar comb
    assert numpy.unique(numpy.round(inclination, 10)).size > 50


def test_the_interpolation_error_is_recorded(tmp_path):
    """The recovery measures and records the interpolation error."""
    partition, eta, widths, lnb = make_partition(tmp_path)
    archive = make_archive(tmp_path, eta, lnb)
    statistics = SufficientStatistics(archive, eta, lnb)
    result = statistics.columns(20.0, power=0.0, seed=20261018, n_u=201)
    error = result["diagnostics"]["interpolation_error"]
    assert error is not None
    for key in ("median_abs", "p95_abs", "max_abs", "n_cells", "n_points"):
        assert key in error
    assert 0.0 <= error["median_abs"] <= error["p95_abs"] \
        <= error["max_abs"]
    assert error["n_cells"] > 0 and error["n_points"] > 0
    assert result["diagnostics"]["orientation_interpolation"] is True
    assert result["diagnostics"]["orientation_fine_grid"] == (
        N_CI_FINE, N_PSI_FINE)

    info = statistics.metadata(20.0, 0.0, 20261018,
                               result["diagnostics"],
                               distance_constant=DISTANCE_CONSTANT)
    assert info["amr_orientation_interpolation_error_median_nats"] == \
        pytest.approx(error["median_abs"], rel=0.0, abs=0.0)
    assert info["amr_orientation_interpolation_error_p95_nats"] == \
        pytest.approx(error["p95_abs"], rel=0.0, abs=0.0)
    assert info["amr_orientation_interpolation_error_max_nats"] == \
        pytest.approx(error["max_abs"], rel=0.0, abs=0.0)
    assert info["amr_orientation_fine_nodes"] == N_CI_FINE * N_PSI_FINE
    assert "0.007843" in info["amr_orientation_node_spacing"]
    assert "NOT interpolated" in info["amr_delta_tc_resolution"]
    assert "OVERSAMPLING" in info["amr_arrival_time_note"]
    assert info["amr_arrival_time_grid_s"] == pytest.approx(DT)


def test_the_enumerated_draw_can_still_be_selected(tmp_path):
    """orientation-interpolation = false restores the old comb, and says so."""
    partition, eta, widths, lnb = make_partition(tmp_path)
    archive = make_archive(tmp_path, eta, lnb)
    statistics = SufficientStatistics(archive, eta, lnb)
    result = statistics.columns(20.0, power=0.0, seed=20261018, n_u=201,
                                orientation_interpolation=False)
    assert result["diagnostics"]["orientation_interpolation"] is False
    assert result["diagnostics"]["interpolation_error"] is None
    info = statistics.metadata(20.0, 0.0, 20261018,
                               result["diagnostics"],
                               distance_constant=DISTANCE_CONSTANT)
    assert "ENUMERATED" in info["amr_orientation_axis_note"]
    assert "comb" in info["amr_inclination_resolution"]


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
