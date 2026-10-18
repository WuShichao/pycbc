"""Tests for the exact box-ellipsoid containment test in ``pycbc.tdi.exact_inside``."""

import numpy as np

from pycbc.tdi.exact_inside import (
    active_set_min,
    certified,
    diagonal_clamp,
    inside_exact,
    metric,
)


def _random_geometry(rng, n=3, count=64):
    """A tilted positive-definite metric and a batch of random boxes."""
    A = np.eye(n) + 0.6 * rng.normal(size=(n, n))
    c = rng.uniform(-2.0, 2.0, (count, n))
    half = rng.uniform(0.05, 0.6, (count, n))
    return A, c, half


def test_exact_inside_agrees_with_active_set_brute_force():
    rng = np.random.default_rng(20261018)
    A, c, half = _random_geometry(rng)
    M = metric(A)

    keep = inside_exact(A, c, half)
    q_min = active_set_min(c, half, M)          # independent 3^n enumeration
    assert np.array_equal(keep, q_min <= 1.0)

    out = certified(c, half, M)
    assert np.all(out["q_lb"] <= q_min + 1e-12)
    assert np.all(q_min <= out["q_ub"] + 1e-12)
    assert np.all(keep[out["keep_value"]])


def test_diagonal_closed_form_agrees_with_exact_minimum():
    rng = np.random.default_rng(7)
    A = np.diag(np.sqrt(rng.uniform(0.5, 2.0, 3)))
    M = metric(A)
    c = rng.uniform(-2.0, 2.0, (32, 3))
    half = rng.uniform(0.05, 0.6, (32, 3))
    assert np.allclose(diagonal_clamp(c, half, M),
                       active_set_min(c, half, M), rtol=1e-12, atol=1e-12)


def test_dropped_boxes_contain_no_sampled_point():
    rng = np.random.default_rng(11)
    A, c, half = _random_geometry(rng, count=12)
    keep = inside_exact(A, c, half)
    M = metric(A)
    for row in np.nonzero(~keep)[0]:
        x = c[row] + rng.uniform(-1.0, 1.0, (2000, 3)) * half[row]
        q = np.einsum("ni,ij,nj->n", x, M, x)
        assert q.min() > 1.0


def test_singular_metric_keeps_box_containing_a_zero():
    A = np.array([[1.0, 1.0]])
    out = certified([[0.5, -0.5]], [0.2, 0.2], metric(A))
    assert out["keep"][0]
    assert np.isfinite(out["q_lb"][0])
    assert not out["unresolved"][0]


def test_nonfinite_bound_is_kept_unresolved():
    out = certified([[np.inf, 0.0]], [0.2, 0.2], np.eye(2))
    assert out["keep"][0]
    assert out["unresolved"][0]
    out = certified([[0.0, 0.0]], [0.2, 0.2], np.full((2, 2), np.nan))
    assert out["keep"][0]
    assert out["unresolved"][0]
