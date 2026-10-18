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

"""Tests for the restored columns of the AMR adapter.

The restored inclination, polarization and delta_tc are conditional columns
read from an archived sufficient-statistics file.  These tests build a small
synthetic archive with the production array layout, so they exercise the parts
that need no GPU and no research repository: the orientation grid, the
amplitude marginal with a configurable ceiling, the archive/partition binding,
the peak-lag choice, the conditional draw's determinism, the ceiling switch and
the ordinary-column write.
"""

import json
import os

import numpy
import pytest

from pycbc.inference.io import loadfile
from pycbc.inference.models.analytic import TestNormal
from pycbc.inference.sampler.amr import AmrSampler, ShapeToPhysicalMap
from pycbc.inference.sampler.amr_demarg import (
    AmplitudeMarginal, REGISTERED_CEILING, SufficientStatistics,
    orientation_grid)
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


def make_model(params=("eta_0", "eta_1")):
    """A model whose only role is to supply the variable-parameter names."""
    return TestNormal(list(params), mean=[0.0] * len(params),
                      cov=[1.0] * len(params))


def make_partition(tmp_path, truth=None):
    """A four-cell partition with hand-computable weights."""
    eta = numpy.zeros((4, 6))
    eta[:, 0] = [0.0, 1.0, 0.0, 1.0]
    eta[:, 1] = [0.0, 0.0, 1.0, 1.0]
    widths = numpy.full((4, 6), 0.5)
    lnb = numpy.log(numpy.arange(1.0, 5.0))
    path = tmp_path / "posteriors.npz"
    arrays = dict(eta_L6=eta, w_L6=widths, lnB_L6=lnb)
    if truth is not None:
        arrays["truth"] = numpy.asarray(truth, dtype=float)
    numpy.savez(path, **arrays)
    return str(path), eta, widths, lnb


def make_archive(tmp_path, eta, lnb, lags=None, cells=None, delta=None,
                 truth=None):
    """A synthetic sufficient-statistics archive with the production layout.

    Each row has ``D = (5 sqrt(NORM), 0)`` and ``Gram = NORM * I``, so
    ``s = A_min sqrt(NORM) sqrt(|c_p|^2 + |c_c|^2)`` is close to 2 and
    ``y = 5 |c_p| / sqrt(|c_p|^2 + |c_c|^2)`` is between 0 and 4.  The absolute
    values do not matter for these tests; the array layout and the binding do.
    """
    if lags is None:
        lags = [[0]] * len(eta)
    if cells is None:
        cells = numpy.arange(len(eta))
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
            D[row] = (5.0 * numpy.sqrt(NORM), 0.0)
            lag[row] = index
            two_f[row] = 100.0 + 10.0 * k
        ptr.append(ptr[-1] + len(item))
    arrays = dict(
        D=D, G=G, lag=lag, twoF=two_f, ptr=numpy.asarray(ptr),
        cell=numpy.repeat(cells, [len(item) for item in lags]),
        cells=cells, cellsw=numpy.full(len(eta), 1.0 / len(eta)),
        eta=eta[cells], lnB_stored=lnb[cells],
        Dt=DT, Delta=numpy.asarray(delta), A_min=A_MIN,
        mass=numpy.full(len(eta), 1.0 / len(eta)),
        eta_all=eta, lnB_all=lnb, w_all=numpy.full((len(eta), 6), 0.5),
        kappa=1.0)
    if truth is not None:
        arrays["truth"] = numpy.asarray(truth, dtype=float)
    path = tmp_path / "sufficient.npz"
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


def test_the_restored_columns_are_ordinary_samples(tmp_path):
    partition, eta, widths, lnb = make_partition(tmp_path)
    archive = make_archive(tmp_path, eta, lnb)
    cp = make_config(partition, "sufficient-file = {}\namplitude-ceiling = 20"
                     "\n".format(archive))
    sampler = run_sampler(tmp_path, cp)
    for name in ("inclination", "polarization", "delta_tc"):
        assert name in sampler.samples
        assert len(sampler.samples[name]) == len(eta)
    inclination = sampler.samples["inclination"]
    polarization = sampler.samples["polarization"]
    assert numpy.all(inclination >= 0.0) and numpy.all(inclination <= numpy.pi)
    assert numpy.all(polarization >= 0.0)
    assert numpy.all(polarization < numpy.pi / 2.0)
    # the arrival-time column is a lag of the archive's grid
    assert set(numpy.unique(sampler.samples["delta_tc"])) <= {
        -3.0 * DT, 0.0}
    # and the file is still self-consistent
    numpy.testing.assert_allclose(
        sampler.samples["logwt"],
        sampler.samples["loglikelihood"] + sampler.samples["logprior"])
    with loadfile(str(sampler.checkpoint_file), "r") as fp:
        raw = fp.read_raw_samples(
            ["inclination", "polarization", "delta_tc"], raw_samples=True)
        assert set(raw) == {"inclination", "polarization", "delta_tc"}
        post = fp.read_raw_samples(
            ["inclination", "polarization", "delta_tc"], nsamples=100)
        assert len(post["inclination"]) == 100
        assert fp.attrs["amr_amplitude_ceiling"] == 20.0
        assert fp.attrs["amr_recovered_params"] == \
            "inclination polarization delta_tc"
        assert "conditional draw" in fp.attrs["amr_recovered_kind"]


def test_the_registered_ceiling_keeps_the_stored_bayes_factors(tmp_path):
    partition, eta, widths, lnb = make_partition(tmp_path)
    archive = make_archive(tmp_path, eta, lnb)
    registered = run_sampler(
        tmp_path, make_config(
            partition, "sufficient-file = {}\namplitude-ceiling = 5"
            "\n".format(archive)))
    numpy.testing.assert_array_equal(registered.samples["loglikelihood"], lnb)
    widened = run_sampler(
        tmp_path, make_config(
            partition, "sufficient-file = {}\namplitude-ceiling = 20"
            "\n".format(archive)), name="widened.hdf")
    assert not numpy.allclose(widened.samples["loglikelihood"], lnb)
    # the eta columns are untouched by the ceiling
    numpy.testing.assert_array_equal(
        widened.samples["eta_0"], registered.samples["eta_0"])
    numpy.testing.assert_array_equal(
        widened.samples["logprior"], registered.samples["logprior"])


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
    numpy.testing.assert_array_equal(
        first.samples["inclination"], second.samples["inclination"])
    numpy.testing.assert_array_equal(
        first.samples["polarization"], second.samples["polarization"])
    assert not numpy.array_equal(
        first.samples["inclination"], other.samples["inclination"])


def test_the_peak_lag_is_used_per_cell(tmp_path):
    """delta_tc is the cell's strongest lag, in the partition's row order."""
    partition, eta, widths, lnb = make_partition(tmp_path)
    # cell 0 has two lags and the second is stronger; cells 1-3 follow
    lags = [[1, 2], [4], [5], [6]]
    archive = make_archive(tmp_path, eta, lnb, lags=lags)
    cp = make_config(partition, "sufficient-file = {}".format(archive))
    sampler = run_sampler(tmp_path, cp)
    delta = sampler.samples["delta_tc"]
    expected = numpy.array([2, 4, 5, 6]) * DT - 3.0 * DT
    numpy.testing.assert_allclose(delta, expected)
    # polarization/inclination are node values of the grid
    assert numpy.all(sampler.samples["polarization"] >= 0.0)


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

    archive = make_archive(tmp_path, eta, lnb + 1.0)
    sampler = AmrSampler(
        make_model(), partition_file=partition, sufficient_file=archive,
        amplitude_ceiling=5)
    with pytest.raises(ValueError, match="different partition"):
        sampler._sufficient_statistics(eta, lnb)


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
        tmp_path, truth=shape_map.injection["eta"])
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
