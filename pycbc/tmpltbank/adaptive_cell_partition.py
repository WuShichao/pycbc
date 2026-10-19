# Copyright (C) 2026  Shichao Wu
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
"""Nested cells covering a search region in metric coordinates.

The region is an ellipsoid in the shape coordinates ``xi`` of a template
family, ``sum_a (xi_a / s_a)^2 <= 1``, seen in the eigen-coordinates
``eta = sqrt(lambda) E^T xi`` of the metric, where the mismatch is
``|eta|^2`` (see :class:`pycbc.tmpltbank.fstatistic_metric.MetricChart`).
Cells are boxes in ``eta``. Along a direction the region is no wider than
the finest cell (a soft direction) there is a single cell; along the stiff
directions the coarsest level divides the region's extent and each level
halves the cells' widths until they reach the finest width, set by the
largest mismatch allowed at a cell's corner,

    width = 2 sqrt(max_mismatch / n_stiff).

A cell is kept only if it meets the region, decided with the certified test
of :mod:`pycbc.tmpltbank.box_ellipsoid`, so the kept cells of every level
cover the region and a child lies inside its parent.
"""

import itertools

import numpy

from pycbc.tmpltbank.box_ellipsoid import inside_exact


class EllipsoidCells(object):
    """Nested boxes covering an ellipsoid in metric eigen-coordinates.

    Parameters
    ----------
    chart : pycbc.tmpltbank.fstatistic_metric.MetricChart
        The metric chart of the shape coordinates.
    radii : array_like
        The region's semi-axis ``s_a`` along each shape coordinate.
    max_mismatch : float
        The largest mismatch at the corner of a finest cell.
    """
    def __init__(self, chart, radii, max_mismatch):
        lam = numpy.maximum(chart.eigenvalues, 0)
        E = chart.eigenvectors
        radii = numpy.asarray(radii, dtype=float)
        if radii.shape != (chart.dim,) or numpy.any(radii <= 0):
            raise ValueError("need a positive radius per shape coordinate")
        # xi = E eta / sqrt(lam), so the region is |A eta| <= 1
        with numpy.errstate(divide='ignore'):
            inv = numpy.where(lam > 0, 1 / numpy.sqrt(lam), 0.0)
        self.A = (E / radii[:, None]) * inv[None, :]
        self.chart = chart
        # the region's half-extent along each eta axis
        self.half_extent = numpy.sqrt(lam) * numpy.sqrt(
            numpy.sum((E * radii[:, None]) ** 2, axis=0))
        n = chart.dim
        width = None
        for _ in range(n):
            stiff = (2 * self.half_extent > (width or 0))
            width = 2 * numpy.sqrt(max_mismatch / max(1, stiff.sum()))
            if numpy.array_equal(stiff, 2 * self.half_extent > width):
                break
        self.stiff = 2 * self.half_extent > width
        self.finest_width = width
        self.max_mismatch = float(max_mismatch)

    def keep(self, centres, widths):
        """Whether each box meets the region."""
        return inside_exact(self.A, centres, widths / 2.0)

    def level(self, depth):
        """The kept boxes of the grid ``2**depth`` finest widths wide along
        the stiff directions, as ``(centres, widths)``."""
        width = self.finest_width * 2 ** int(depth)
        axes, sizes = [], []
        for extent, stiff in zip(self.half_extent, self.stiff):
            if stiff:
                n = int(numpy.ceil(2 * extent / width))
                step = 2 * extent / n
                axes.append(-extent + (numpy.arange(n) + 0.5) * step)
                sizes.append(step)
            else:
                axes.append(numpy.zeros(1))
                sizes.append(max(2 * extent, 1e-12))
        centres = numpy.array(list(itertools.product(*axes)))
        widths = numpy.repeat(numpy.array(sizes)[None, :], len(centres),
                              axis=0)
        keep = self.keep(centres, widths)
        return centres[keep], widths[keep]

    def children(self, centres, widths):
        """The kept halves of each box along its stiff directions wider
        than the finest width, as ``(centres, widths, parents)``."""
        split = self.stiff[None, :] & (widths > self.finest_width * 1.0001)
        kids, sizes, parents = [], [], []
        patterns, which = numpy.unique(split, axis=0, return_inverse=True)
        for k, pattern in enumerate(patterns):
            rows = numpy.nonzero(which.ravel() == k)[0]
            offsets = numpy.array(list(itertools.product(
                *[(-0.25, 0.25) if s else (0.0,) for s in pattern])))
            kids.append((centres[rows][:, None, :] + offsets[None, :, :]
                         * widths[rows][:, None, :]).reshape(
                             -1, centres.shape[1]))
            sizes.append(numpy.repeat(numpy.where(pattern, widths[rows] / 2,
                                                  widths[rows]),
                                      len(offsets), axis=0))
            parents.append(numpy.repeat(rows, len(offsets)))
        kids = numpy.concatenate(kids)
        sizes = numpy.concatenate(sizes)
        parents = numpy.concatenate(parents)
        keep = self.keep(kids, sizes)
        return kids[keep], sizes[keep], parents[keep]

    def shape_coordinates(self, eta):
        """``xi`` of chart coordinates ``eta``."""
        return self.chart.from_chart(eta)

    def write(self, group, centres, widths):
        """Write the region and a set of cells to an HDF group."""
        group.attrs['max_mismatch'] = self.max_mismatch
        group['metric'] = self.chart.metric
        group['A'] = self.A
        group['stiff'] = self.stiff
        group['centres'] = centres
        group['widths'] = widths
