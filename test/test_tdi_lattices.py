"""Tests for the D4 lattice nearest-point map in ``pycbc.tdi.lattices``."""

import numpy as np

from pycbc.tdi.lattices import (
    BCC_HOLES,
    BCC_R,
    D4_HOLES,
    D4_R,
    d4_brute,
    d4_in_voronoi,
    d4_nearest,
)

SCALE = 0.08837


def test_d4_nearest_agrees_with_brute_force():
    rng = np.random.default_rng(20261018)
    y = np.concatenate([
        rng.uniform(-3.0, 3.0, (128, 4)) * SCALE,
        (rng.integers(-3, 3, (64, 4)) + 0.5) * SCALE,   # ties on the cell walls
    ])

    point = d4_nearest(y, SCALE)
    ref, ref_distance, _ = d4_brute(y / SCALE, span=2)

    assert np.allclose(np.linalg.norm(y - point, axis=1),
                       ref_distance * SCALE, atol=1e-12)
    assert np.all(np.abs(point - np.rint(point / SCALE) * SCALE) < 1e-9)


def test_all_twenty_four_deep_holes_sit_at_the_covering_radius():
    assert D4_HOLES.shape == (24, 4)
    assert np.allclose(np.linalg.norm(D4_HOLES, axis=1), D4_R, atol=1e-12)
    # the origin is a nearest lattice point of every hole (ties allowed)
    assert np.all(d4_in_voronoi(D4_HOLES))

    nearest = d4_nearest(D4_HOLES * SCALE, SCALE)
    assert np.allclose(np.linalg.norm(D4_HOLES * SCALE - nearest, axis=1),
                       D4_R * SCALE, atol=1e-12)
    # independently: brute force reaches the same covering radius at every hole
    _, brute_distance, _ = d4_brute(D4_HOLES, span=2)
    assert np.allclose(brute_distance, D4_R, atol=1e-12)


def test_bcc_all_vertices_include_each_coordinate_permutation():
    assert BCC_HOLES.shape == (24, 3)
    assert len(np.unique(BCC_HOLES, axis=0)) == 24
    assert np.allclose(np.linalg.norm(BCC_HOLES, axis=1), BCC_R, atol=1e-12)
    for vertex in BCC_HOLES:
        assert np.any(np.all(BCC_HOLES == -vertex, axis=1))
    # Omitting the permutations with the large component on axis 2 misses this.
    quadratic = np.einsum("ni,ij,nj->n", BCC_HOLES,
                          np.diag([1.0, 1.0, 100.0]), BCC_HOLES)
    assert np.isclose(quadratic.max(), 25.0625)
