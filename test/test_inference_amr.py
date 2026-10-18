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

"""Tests for the AMR cell-partition adapter.

The object under test is a deterministic grid quadrature, not a sampler, so the
tests check the parts that can be checked without a GPU: the registry entry,
the config parsing, the samples file's format and its self-consistency
(``logwt = loglikelihood + logprior``, ``logsumexp(logwt) = log_evidence``),
the weighted moments against a hand-computed partition, the determinism of the
resampling, and the refinement-criterion switch against a stub posterior that
implements the research interface.  A live research run needs cupy, a CUDA GPU
and the companion repository and is not part of the test suite.
"""

import json
import types

import numpy
import pytest

from pycbc.inference.io import loadfile
from pycbc.inference.sampler import samplers
from pycbc.inference.sampler.amr import (AmrSampler, ShapeToPhysicalMap,
                                         _logsumexp)
from pycbc.inference.models.analytic import TestNormal as NormalModel
from pycbc.workflow import WorkflowConfigParser


def make_model(params=("eta_0", "eta_1")):
    """A model whose only role is to supply the variable-parameter names."""
    return NormalModel(list(params), mean=[0.0] * len(params),
                      cov=[1.0] * len(params))


def make_partition(tmp_path, cells=4):
    """A tiny weighted partition with hand-computable weights.

    The centres sit on the corners of the unit square, every cell has volume
    0.25, and the Bayes factors are 1, 2, 3, 4, so the posterior weights are
    0.1, 0.2, 0.3 and 0.4 and the evidence is log(2.5).  This is exactly the
    arithmetic :py:meth:`AmrSampler._set_samples` must reproduce.
    """
    eta = numpy.zeros((cells, 6))
    eta[:, 0] = [0.0, 1.0, 0.0, 1.0]
    eta[:, 1] = [0.0, 0.0, 1.0, 1.0]
    widths = numpy.full((cells, 6), 0.5)
    lnb = numpy.log(numpy.arange(1.0, cells + 1.0))
    path = tmp_path / "posteriors.npz"
    numpy.savez(path, eta_L6=eta, w_L6=widths, lnB_L6=lnb)
    return str(path)


def make_config(tmp_path, partition_file, extra=""):
    """A [sampler] + [sampler-amr] config naming the adapter."""
    text = """
[model]
name = test_normal

[sampler]
name = amr

[sampler-amr]
enable-research = false
partition-file = {}
partition-level = last
evidence = quadrature
eta-columns = 0 1
{}
""".format(partition_file, extra)
    cp = WorkflowConfigParser()
    cp.read_string(text)
    return cp


def test_amr_is_in_the_sampler_registry():
    assert "amr" in samplers
    assert samplers["amr"] is AmrSampler


def test_from_config_run_and_file_contents(tmp_path):
    partition = make_partition(tmp_path)
    cp = make_config(tmp_path, partition)
    output = str(tmp_path / "amr.hdf")
    sampler = AmrSampler.from_config(cp, make_model(), output_file=output)
    assert sampler.name == "amr"
    assert sampler.source == "partition"
    assert sampler.eta_columns == [0, 1]
    sampler.run()

    with loadfile(str(sampler.checkpoint_file), "r") as fp:
        assert fp.attrs["filetype"] == "amr_file"
        assert fp.attrs["sampler"] == "amr"
        # the nature of the object is recorded in the file itself
        assert not fp.attrs["stochastic_sampler"]
        assert fp.attrs["evidence_error"] == "none"
        assert "deterministic grid quadrature" in fp.attrs["posterior_kind"]
        assert list(fp.variable_params) == ["eta_0", "eta_1"]

        raw = fp.read_raw_samples(
            ["eta_0", "eta_1", "loglikelihood", "logprior", "logwt", "snr"],
            raw_samples=True)
        # one row per cell, weighted, not resampled
        assert len(raw["logwt"]) == 4
        numpy.testing.assert_allclose(
            raw["logwt"], numpy.log([0.25, 0.5, 0.75, 1.0]))
        numpy.testing.assert_allclose(
            raw["logprior"], numpy.full(4, numpy.log(0.25)))
        numpy.testing.assert_allclose(
            raw["loglikelihood"], numpy.log([1.0, 2.0, 3.0, 4.0]))
        # the file is self-consistent: logwt = loglikelihood + logprior
        numpy.testing.assert_allclose(
            raw["logwt"], raw["loglikelihood"] + raw["logprior"])
        # the effective SNR is stored explicitly and is sqrt(2 ln B)
        assert "snr" in fp["samples"]
        numpy.testing.assert_allclose(
            raw["snr"], numpy.sqrt(2.0 * numpy.clip(
                raw["loglikelihood"], 0.0, None)))
        # and the evidence is the Riemann sum
        assert fp.attrs["log_evidence"] == pytest.approx(numpy.log(2.5))
        assert numpy.isnan(fp.attrs["dlog_evidence"])


def test_weighted_moments_match_the_partition(tmp_path):
    partition = make_partition(tmp_path)
    cp = make_config(tmp_path, partition)
    sampler = AmrSampler.from_config(
        cp, make_model(), output_file=str(tmp_path / "amr.hdf"))
    sampler.run()
    weights = numpy.array([0.1, 0.2, 0.3, 0.4])
    eta = sampler.samples
    assert numpy.dot(weights, eta["eta_0"]) == pytest.approx(0.6)
    assert numpy.dot(weights, eta["eta_1"]) == pytest.approx(0.7)


def test_resampling_is_deterministic_and_raw_is_not(tmp_path):
    partition = make_partition(tmp_path)
    cp = make_config(tmp_path, partition)
    sampler = AmrSampler.from_config(
        cp, make_model(), output_file=str(tmp_path / "amr.hdf"))
    sampler.run()
    with loadfile(str(sampler.checkpoint_file), "r") as fp:
        first = fp.read_raw_samples(["eta_0"], seed=7)
        second = fp.read_raw_samples(["eta_0"], seed=7)
        other = fp.read_raw_samples(["eta_0"], seed=8)
        raw = fp.read_raw_samples(["eta_0"], raw_samples=True)
    numpy.testing.assert_array_equal(first["eta_0"], second["eta_0"])
    assert not numpy.array_equal(first["eta_0"], other["eta_0"])
    # the raw read keeps one entry per cell and no resampling
    assert len(raw["eta_0"]) == 4
    # a resampled posterior can be any size
    with loadfile(str(sampler.checkpoint_file), "r") as fp:
        many = fp.read_raw_samples(["eta_0"], nsamples=1000)
    assert len(many["eta_0"]) == 1000
    # every posterior sample is a cell centre, not an interpolation
    assert set(numpy.unique(first["eta_0"])) <= {0.0, 1.0}


def test_unknown_options_and_bad_columns_are_refused(tmp_path):
    partition = make_partition(tmp_path)
    cp = make_config(tmp_path, partition, extra="refinment = both")
    with pytest.raises(RuntimeError, match="refinment"):
        AmrSampler.from_config(
            cp, make_model(), output_file=str(tmp_path / "amr.hdf"))

    cp = make_config(tmp_path, partition, extra="")
    with pytest.raises(ValueError, match="eta-columns"):
        AmrSampler.from_config(
            cp, make_model(("eta_0", "eta_1", "eta_2")),
            output_file=str(tmp_path / "amr.hdf"))

    # without a research run there is no way to obtain the Bayes factor
    cp = WorkflowConfigParser()
    cp.read_string("[sampler]\nname = amr\n[sampler-amr]\n"
                   "enable-research = false\n")
    with pytest.raises(ValueError, match="partition-file"):
        AmrSampler.from_config(
            cp, make_model(), output_file=str(tmp_path / "amr.hdf"))


def test_options_are_read_from_the_sampler_section_too(tmp_path):
    """PyCBC's own samplers read [sampler]; [sampler-amr] still wins."""
    partition = make_partition(tmp_path)
    text = """
[model]
name = test_normal

[sampler]
name = amr
partition-file = {partition}
eta-columns = 0 1
max-level = 3

[sampler-amr]
max-level = 4
""".format(partition=partition)
    cp = WorkflowConfigParser()
    cp.read_string(text)
    sampler = AmrSampler.from_config(
        cp, make_model(), output_file=str(tmp_path / "amr.hdf"))
    assert sampler.max_level == 4
    assert sampler.eta_columns == [0, 1]


def test_refinement_criterion_switches_which_cells_are_refined():
    """`refinement` chooses the mass criterion, the 2F criterion or both."""

    class StubPosterior:
        """The research interface, without the GPU or the physics."""

        def __init__(self):
            self.n_eval = 0

        def lnB(self, etas, keys=None):
            # cell at eta_0 = 0 has a large Bayes factor, the other does not
            lnb = numpy.where(numpy.asarray(etas)[:, 0] == 0.0, 10.0, 0.0)
            fmax = numpy.where(numpy.asarray(etas)[:, 0] == 0.0, 0.0, 100.0)
            return lnb, fmax

        def logmass(self, lnb, widths):
            return lnb + numpy.log(numpy.prod(widths[:, :4], axis=1))

        def split(self, eta, widths):
            offs = numpy.array([[-0.25] + [0.0] * 5, [0.25] + [0.0] * 5])
            child_eta = (eta[:, None, :] + offs[None, :, :]).reshape(-1, 6)
            child_w = numpy.repeat(widths / 2.0, 2, axis=0)
            return child_eta, child_w

    module = types.SimpleNamespace(logsumexp=_logsumexp)
    eta = numpy.array([[0.0] * 6, [10.0] * 6])
    widths = numpy.ones((2, 6))

    def refine(refinement):
        sampler = AmrSampler(
            make_model(), partition_file="unused.npz",
            refinement=refinement, max_level=1, max_cells=1000)
        return sampler._refine(StubPosterior(), eta, widths, module)[0]

    mass = refine("mass")
    assert mass["n_mass"] == 1 and mass["n_fmax"] == 0
    assert mass["n_refined"] == 1

    fmax = refine("fmax")
    assert fmax["n_mass"] == 0 and fmax["n_fmax"] == 1
    assert fmax["n_refined"] == 1

    both = refine("both")
    assert both["n_mass"] == 1 and both["n_fmax"] == 1
    assert both["n_refined"] == 2


def test_the_research_path_names_what_is_missing(tmp_path):
    """Without the companion repository the error says what to do instead."""
    sampler = AmrSampler(
        make_model(), partition_file="unused.npz",
        enable_research=True, amr_root=str(tmp_path / "not_a_repo"))
    with pytest.raises(OSError, match="companion repository"):
        sampler._import_research()


def make_physical_map(tmp_path, eta_columns=(0, 1), injection=None,
                      lam=None, E=None):
    """A small physical map with a hand-computable projection.

    ``E`` is the identity and ``lam = (4, 1, 1, 1, 1, 1)``, so
    ``M = diag(1/2, 1, 1, 1, 1, 1)`` and ``xi = (eta_0/2, eta_1, 0, 0, 0, 0)``.
    The injection ``xi = (1, 0, ...)`` is therefore ``eta = (2, 0, ...)``.
    """
    spec = {
        "eta_columns": list(eta_columns),
        "eta_static": {"4": 0.0, "5": 0.0},
        "lam": [4.0, 1.0, 1.0, 1.0, 1.0, 1.0] if lam is None else lam,
        "E": numpy.eye(6).tolist() if E is None else E,
        "parameters": [
            {"name": "mchirp", "xi_index": 0, "kind": "identity",
             "reference": 10.0, "scale": 0.1},
            {"name": "q", "xi_index": 1, "kind": "reciprocal",
             "reference": 2.0, "scale": 0.01},
        ],
        "injection": injection if injection is not None else {
            "xi": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "eta": [2.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
    }
    path = tmp_path / "physical-map.json"
    path.write_text(json.dumps(spec))
    return str(path)


def test_the_physical_map_inverts_the_chart_and_applies_the_q_convention(
        tmp_path):
    mapped = ShapeToPhysicalMap.from_file(make_physical_map(tmp_path))
    # the matrix is E diag(1/sqrt(lam)); with the identity basis it is diag(1/2, 1, ...)
    numpy.testing.assert_allclose(
        mapped.matrix, numpy.diag([0.5, 1.0, 1.0, 1.0, 1.0, 1.0]))
    # xi = M eta, so eta_0 = 2 is one shape unit of mchirp
    numpy.testing.assert_allclose(
        mapped.xi(numpy.array([[2.0, 0.0]]))[0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # p = p_ref + sigma xi, and q is stored as PyCBC's m1/m2 = 1/q_proj
    values = mapped.physical(numpy.array([[2.0, 1.0], [0.0, 0.0]]))
    numpy.testing.assert_allclose(values["mchirp"], [10.1, 10.0])
    numpy.testing.assert_allclose(values["q"], 1.0 / numpy.array([2.01, 2.0]))
    # the truth is the injection's own xi, not the chart image of its eta
    assert mapped.truth()["mchirp"] == pytest.approx(10.1)
    assert mapped.truth()["q"] == pytest.approx(0.5)


def test_the_physical_map_refuses_a_matrix_that_misses_the_injection(tmp_path):
    # eta = (1, 0, ...) would give xi = (0.5, 0, ...), not the recorded (1, ...)
    bad = {"xi": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
           "eta": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
    with pytest.raises(ValueError, match="does not reproduce"):
        ShapeToPhysicalMap.from_file(make_physical_map(tmp_path, injection=bad))


def test_the_physical_map_refuses_a_non_orthogonal_basis(tmp_path):
    with pytest.raises(ValueError, match="orthogonal"):
        ShapeToPhysicalMap.from_file(
            make_physical_map(tmp_path, E=(2.0 * numpy.eye(6)).tolist()))


def test_the_sampler_writes_the_physical_columns_beside_eta(tmp_path):
    partition = make_partition(tmp_path)
    spec = make_physical_map(tmp_path)
    cp = make_config(tmp_path, partition, extra="physical-map = {}\n".format(spec))
    sampler = AmrSampler.from_config(
        cp, make_model(), output_file=str(tmp_path / "amr.hdf"))
    sampler.run()
    with loadfile(str(sampler.checkpoint_file), "r") as fp:
        fields = list(fp[fp.samples_group].keys())
        # the eta columns stay: they are the chart the covering is defined in
        assert "eta_0" in fields and "eta_1" in fields
        assert "mchirp" in fields and "q" in fields
        raw = fp.read_raw_samples(
            ["eta_0", "eta_1", "mchirp", "q"], raw_samples=True)
        numpy.testing.assert_allclose(
            raw["mchirp"], 10.0 + 0.1 * 0.5 * raw["eta_0"])
        # q is the reciprocal of the linear coordinate, so it is not linear
        numpy.testing.assert_allclose(
            raw["q"], 1.0 / (2.0 + 0.01 * raw["eta_1"]))
        # the transformation is recorded so a reader can rebuild it
        assert fp.attrs["amr_physical_params"] == "mchirp q"
        numpy.testing.assert_allclose(
            numpy.asarray(fp.attrs["amr_shape_matrix"]),
            numpy.diag([0.5, 1.0, 1.0, 1.0, 1.0, 1.0]))
        numpy.testing.assert_allclose(
            numpy.asarray(fp.attrs["amr_shape_reference"]), [10.0, 2.0])
        numpy.testing.assert_allclose(
            numpy.asarray(fp.attrs["amr_shape_scale"]), [0.1, 0.01])
        # and the injection is checked, not assumed
        numpy.testing.assert_allclose(
            numpy.asarray(fp.attrs["amr_injection_xi"]),
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_a_map_that_disagrees_with_eta_columns_is_refused(tmp_path):
    partition = make_partition(tmp_path)
    spec = make_physical_map(tmp_path, eta_columns=(0, 1, 2))
    cp = make_config(tmp_path, partition, extra="physical-map = {}\n".format(spec))
    with pytest.raises(ValueError, match="eta columns"):
        AmrSampler.from_config(
            cp, make_model(), output_file=str(tmp_path / "amr.hdf"))


def test_unrefined_columns_are_recorded_as_a_rank_deficiency(tmp_path):
    """A six-column partition that only subdivides four of them is rank four.

    The research covering grid refines only the stiff directions and leaves
    one cell across the others, so a six-column file can be a four-dimensional
    posterior.  The adapter must say so, in the sampler and in the file,
    rather than letting a reader take the column count as the dimension.
    """
    eta = numpy.zeros((4, 6))
    eta[:, 0] = [0.0, 1.0, 0.0, 1.0]
    eta[:, 1] = [0.0, 0.0, 1.0, 1.0]
    eta[:, 2] = [0.0, 1.0, 1.0, 0.0]
    eta[:, 3] = [1.0, 0.0, 1.0, 0.0]
    # eta_4 and eta_5 are the unrefined directions: one centre, one width
    widths = numpy.full((4, 6), 0.5)
    lnb = numpy.log(numpy.arange(1.0, 5.0))
    path = tmp_path / "posteriors_six.npz"
    numpy.savez(path, eta_L6=eta, w_L6=widths, lnB_L6=lnb)

    cp = WorkflowConfigParser()
    cp.read_string("""
[model]
name = test_normal

[sampler]
name = amr

[sampler-amr]
enable-research = false
partition-file = {}
partition-level = last
evidence = quadrature
eta-columns = 0 1 2 3 4 5
""".format(path))
    names = ("eta_0", "eta_1", "eta_2", "eta_3", "eta_4", "eta_5")
    sampler = AmrSampler.from_config(
        cp, make_model(names), output_file=str(tmp_path / "amr.hdf"))
    sampler.run()

    assert sampler.column_info["rank"] == 4
    assert sampler.column_info["refined"] == [0, 1, 2, 3]
    assert sampler.column_info["degenerate"] == [4, 5]
    assert sampler.level_info["posterior_rank"] == 4
    with loadfile(str(sampler.checkpoint_file), "r") as fp:
        assert int(fp.attrs["amr_posterior_rank"]) == 4
        numpy.testing.assert_array_equal(
            numpy.asarray(fp.attrs["amr_degenerate_columns"]), [4, 5])
        numpy.testing.assert_array_equal(
            numpy.asarray(fp.attrs["amr_refined_columns"]), [0, 1, 2, 3])
        numpy.testing.assert_array_equal(
            numpy.asarray(fp.attrs["amr_column_n_centres"]),
            [2, 2, 2, 2, 1, 1])
