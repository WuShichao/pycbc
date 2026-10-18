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

"""Tests for the physical-parameter map of the AMR adapter.

These cover the two directions the map needs -- shape to physical for the
stored columns, physical to shape for an injection -- and the guard that ties
a map to the partition it is applied to.  The forward map, the q convention
and the geometry re-check are the other half and are exercised by
``test_inference_amr.py``.
"""

import os

import numpy
import pytest

from pycbc.inference.models.analytic import TestNormal
from pycbc.inference.sampler.amr import AmrSampler, ShapeToPhysicalMap

MAP = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "examples", "inference",
    "amr-partition", "physical-map.json"))


def make_model(params=("eta_0", "eta_1", "eta_2", "eta_3")):
    """A model whose only role is to supply the variable-parameter names."""
    return TestNormal(list(params), mean=[0.0] * len(params),
                      cov=[1.0] * len(params))


def make_partition(tmp_path, truth=None):
    """A two-cell partition, optionally carrying an injection's eta."""
    eta = numpy.zeros((2, 6))
    eta[:, 0] = [0.1, -0.1]
    eta[:, 1] = [0.05, -0.05]
    widths = numpy.full((2, 6), 0.1)
    lnb = numpy.log([1.0, 2.0])
    path = str(tmp_path / "posteriors.npz")
    arrays = dict(eta_L6=eta, w_L6=widths, lnB_L6=lnb)
    if truth is not None:
        arrays["truth"] = numpy.asarray(truth, dtype=float)
    numpy.savez(path, **arrays)
    return path, eta, widths, lnb


def test_the_inverse_returns_the_recorded_injection():
    """p -> xi must undo p = p_ref + sigma xi, including the q convention."""
    shape_map = ShapeToPhysicalMap.from_file(MAP)
    xi = shape_map.xi_from_physical(shape_map.truth())
    numpy.testing.assert_allclose(xi, shape_map.injection["xi"], atol=1e-8)


def test_the_physical_columns_are_the_map_applied_to_eta(tmp_path):
    shape_map = ShapeToPhysicalMap.from_file(MAP)
    path, eta, widths, lnb = make_partition(tmp_path)
    sampler = AmrSampler(make_model(), partition_file=path,
                         physical_map=MAP, eta_columns=[0, 1, 2, 3])
    sampler._set_samples(eta, widths, lnb)
    expected = shape_map.physical(eta[:, [0, 1, 2, 3]])
    for name in shape_map.physical_names():
        numpy.testing.assert_allclose(sampler.samples[name], expected[name])


def test_a_map_from_another_run_is_refused(tmp_path):
    shape_map = ShapeToPhysicalMap.from_file(MAP)
    truth = numpy.asarray(shape_map.injection["eta"]) + 1e-3
    path, _, _, _ = make_partition(tmp_path, truth=truth)
    sampler = AmrSampler(make_model(), partition_file=path,
                         physical_map=MAP, eta_columns=[0, 1, 2, 3])
    with pytest.raises(ValueError, match="different geometry or run"):
        sampler._load_partition()


def test_a_matching_partition_is_accepted(tmp_path):
    shape_map = ShapeToPhysicalMap.from_file(MAP)
    path, eta, widths, lnb = make_partition(
        tmp_path, truth=shape_map.injection["eta"])
    sampler = AmrSampler(make_model(), partition_file=path,
                         physical_map=MAP, eta_columns=[0, 1, 2, 3])
    got_eta, got_widths, got_lnb = sampler._load_partition()
    numpy.testing.assert_array_equal(got_eta, eta)
    numpy.testing.assert_array_equal(got_widths, widths)
    numpy.testing.assert_array_equal(got_lnb, lnb)
