"""Nested cells over an ellipsoid, pycbc.tmpltbank.adaptive_cell_partition."""
import numpy
import pytest

from pycbc.tmpltbank.adaptive_cell_partition import EllipsoidCells
from pycbc.tmpltbank.box_ellipsoid import active_set_min, metric
from pycbc.tmpltbank.fstatistic_metric import MetricChart

RNG = numpy.random.default_rng(17)
B = RNG.normal(size=(4, 4))
# a tilted metric with one soft direction
Q, _ = numpy.linalg.qr(B)
CHART = MetricChart(Q @ numpy.diag([400, 60, 5, 0.002]) @ Q.T)
RADII = numpy.array([1.0, 0.8, 0.5, 2.0])


def cells():
    return EllipsoidCells(CHART, RADII, max_mismatch=0.2)


def inside_region(n):
    """Points uniform in the region, in chart coordinates."""
    u = RNG.normal(size=(n, 4))
    u *= (RNG.random(n) ** 0.25 / numpy.linalg.norm(u, axis=1))[:, None]
    return CHART.to_chart(u * RADII)


def covered(points, centres, widths, chunk=200):
    """Whether each point lies in some box, a chunk of points at a time."""
    lo, hi = centres - widths / 2, centres + widths / 2
    out = numpy.zeros(len(points), dtype=bool)
    for k in range(0, len(points), chunk):
        p = points[k:k + chunk, None, :]
        out[k:k + chunk] = numpy.all((p >= lo[None]) & (p <= hi[None]),
                                     axis=2).any(axis=1)
    return out


def test_finest_cells_respect_the_mismatch():
    c = cells()
    n = c.stiff.sum()
    assert 1 <= n < 4
    corner = n * (c.finest_width / 2) ** 2
    assert corner == pytest.approx(0.2)


def test_every_level_covers_the_region_and_children_nest():
    c = cells()
    points = inside_region(3000)
    centres, widths = c.level(2)
    assert covered(points, centres, widths).all()
    for _ in range(2):
        kids, kwidths, parents = c.children(centres, widths)
        assert covered(points, kids, kwidths).all()
        # children lie inside their parents
        assert numpy.all(numpy.abs(kids - centres[parents])
                         <= (widths[parents] - kwidths) / 2 + 1e-12)
        centres, widths = kids, kwidths
    # the coarse grid divides the region evenly, so the finest cells are
    # at most, and more than half, the finest width
    stiff = widths[:, c.stiff]
    assert numpy.all(stiff <= c.finest_width * (1 + 1e-12))
    assert numpy.all(stiff > c.finest_width / 2)


def test_kept_cells_meet_the_region_and_dropped_ones_do_not():
    c = cells()
    width = c.finest_width * 2
    axes = [numpy.arange(-e - width, e + width, width) if s else numpy.zeros(1)
            for e, s in zip(c.half_extent, c.stiff)]
    grid = numpy.array(numpy.meshgrid(*axes, indexing='ij')).reshape(4, -1).T
    widths = numpy.where(c.stiff, width, 2 * c.half_extent + 1e-12)
    widths = numpy.repeat(widths[None, :], len(grid), axis=0)
    keep = c.keep(grid, widths)
    exact = active_set_min(grid, widths / 2, metric(c.A)) <= 1.0
    assert numpy.array_equal(keep, exact)


def test_bad_radii_are_refused():
    with pytest.raises(ValueError):
        EllipsoidCells(CHART, [1.0, 1.0], 0.05)
