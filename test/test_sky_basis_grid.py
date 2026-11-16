"""SkyBasis.project through a GridSkyResponse's nodes equals projecting each point's interpolated response."""
import unittest

import numpy

from pycbc.filter.nufft_correlation import SkyBasis
from pycbc.tdi.ssb_yearly_harmonics import GridSkyResponse


class _Family(object):
    xp = numpy

    def __init__(self, grid, nk=3):
        self._grid = grid
        self.orders = numpy.arange(nk)
        self.R0 = grid(numpy.zeros(2))

    def responses(self, pts):
        return numpy.stack([self._grid(p) for p in numpy.atleast_2d(pts)])


class TestSkyBasisGrid(unittest.TestCase):
    def test_fast_equals_rows(self):
        rng = numpy.random.default_rng(1)
        ax = numpy.linspace(-1, 1, 21)
        shape = (3, 2, 2, 16)

        def f(sky):
            k = numpy.arange(numpy.prod(shape)).reshape(shape) / 50.
            return numpy.exp(1j * (k * sky[0] + 0.7 * k ** 1.3 * sky[1])) * (1 + 0.1 * sky[0] * sky[1])
        grid = GridSkyResponse((ax, ax), function=f)
        fam = _Family(grid)
        weight = rng.uniform(0.5, 1.5, (3, 1, 1, 16))
        train = rng.uniform(-0.3, 0.3, (60, 2))
        basis = SkyBasis(fam, train, tolerance=1e-3, weight=weight)
        pts = rng.uniform(-0.45, 0.45, (200, 2))
        C0, e0 = basis.project(pts)
        fam.grid_response = grid
        C1, e1 = basis.project(pts)
        self.assertLess(numpy.abs(C1 - C0).max(), 1e-10 * numpy.abs(C0).max())
        self.assertLess(numpy.abs(e1 - e0).max(), 1e-7)
        # a second batch that widens the node range rebuilds the cache
        pts2 = rng.uniform(-0.9, 0.9, (50, 2))
        fam.grid_response = None
        C0, e0 = basis.project(pts2)
        fam.grid_response = grid
        C1, e1 = basis.project(pts2)
        self.assertLess(numpy.abs(C1 - C0).max(), 1e-10 * numpy.abs(C0).max())
        self.assertLess(numpy.abs(e1 - e0).max(), 1e-7)


if __name__ == '__main__':
    unittest.main()
