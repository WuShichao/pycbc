"""The n-dimensional lattices of pycbc.tmpltbank.lattice_utils."""
import itertools

import numpy
import pytest

from pycbc.tmpltbank.lattice_utils import (Lattice, lattice_points_in_box,
                                           unit_ball_volume)

CASES = [('Zn', 2), ('Zn', 4), ('Dn', 3), ('Dn', 4), ('Dn', 5),
         ('Anstar', 2), ('Anstar', 3), ('Anstar', 4), ('Anstar', 5)]


def brute_nearest(lattice, points, reach=2):
    inverse = numpy.linalg.inv(lattice.generator)
    offsets = numpy.array(list(itertools.product(range(-reach, reach + 1),
                                                 repeat=lattice.dim)))
    out = []
    for p in points:
        k = numpy.round(p @ inverse) + offsets
        candidates = k @ lattice.generator
        out.append(candidates[numpy.argmin(
            numpy.sum((candidates - p) ** 2, axis=1))])
    return numpy.array(out)


@pytest.mark.parametrize('name,dim', CASES)
def test_nearest_point_is_the_closest_lattice_point(name, dim):
    lattice = Lattice(name, dim)
    points = numpy.random.default_rng(dim).normal(size=(300, dim)) * 3
    near = lattice.nearest(points)
    expected = brute_nearest(lattice, points)
    assert numpy.allclose(numpy.sum((points - near) ** 2, axis=1),
                          numpy.sum((points - expected) ** 2, axis=1))
    # the nearest points are lattice points
    coeff = near @ numpy.linalg.inv(lattice.generator)
    assert numpy.allclose(coeff, numpy.round(coeff), atol=1e-9)


@pytest.mark.parametrize('name,dim', CASES)
def test_covering_radius_is_one(name, dim):
    lattice = Lattice(name, dim)
    points = numpy.random.default_rng(1).random((200000, dim)) \
        @ lattice.generator
    distance = numpy.linalg.norm(points - lattice.nearest(points), axis=1)
    assert distance.max() <= 1 + 1e-9
    assert distance.max() > 0.85


# Conway and Sloane, Sphere Packings, Lattices and Groups, tables 2.1 and 2.3
@pytest.mark.parametrize('name,dim,thickness,moment', [
    ('Zn', 2, numpy.pi / 2, 1 / 12.),
    ('Zn', 4, numpy.pi ** 2 / 2, 1 / 12.),
    ('Dn', 3, 2.0944, 0.0787451),
    ('Dn', 4, 2.4674, 0.0766032),
    ('Anstar', 2, 1.2092, 0.0801875),
    ('Anstar', 3, 1.4635, 0.0785433),
    ('Anstar', 4, 1.7655, 0.0775587),
])
def test_thickness_and_second_moment_match_the_tables(name, dim, thickness,
                                                       moment):
    lattice = Lattice(name, dim)
    assert lattice.thickness == pytest.approx(thickness, rel=1e-4)
    estimate, error = lattice.normalized_second_moment(nsamples=400000)
    assert abs(estimate - moment) < 4 * error


def test_unit_ball_volume():
    assert unit_ball_volume(2) == pytest.approx(numpy.pi)
    assert unit_ball_volume(3) == pytest.approx(4 * numpy.pi / 3)


def test_voronoi_cell_membership():
    lattice = Lattice('Anstar', 4)
    rng = numpy.random.default_rng(2)
    points = rng.normal(size=(2000, 4))
    inside = lattice.in_voronoi_cell(points)
    near = lattice.nearest(points)
    assert numpy.array_equal(inside, numpy.all(numpy.isclose(near, 0),
                                               axis=1))
    shifted = lattice.in_voronoi_cell(points + lattice.generator[1],
                                      centre=lattice.generator[1])
    assert numpy.array_equal(inside, shifted)


@pytest.mark.parametrize('name', ['Zn', 'Anstar'])
def test_points_cover_a_box_under_a_metric(name):
    metric = numpy.array([[4.0, 1.2, 0.0], [1.2, 1.0, 0.3], [0.0, 0.3, 0.5]])
    lower, upper = numpy.array([-1., 0., 2.]), numpy.array([1., 3., 4.])
    lattice = Lattice(name, 3)
    bank = lattice_points_in_box(lattice, lower, upper, 0.05, metric=metric)
    rng = numpy.random.default_rng(3)
    probes = lower + rng.random((5000, 3)) * (upper - lower)
    diff = probes[:, None, :] - bank[None, :, :]
    mismatch = numpy.einsum('pbi,ij,pbj->pb', diff, metric, diff).min(axis=1)
    assert mismatch.max() <= 0.05 * (1 + 1e-9)


def test_unknown_lattice_and_dimension_are_refused():
    with pytest.raises(ValueError):
        Lattice('E8', 8)
    with pytest.raises(ValueError):
        Lattice('Dn', 2)
