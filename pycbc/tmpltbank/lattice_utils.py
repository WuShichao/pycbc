# Copyright (C) 2013 Ian W. Harry
#
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

import logging
import copy
import numpy


logger = logging.getLogger('pycbc.tmpltbank.lattice_utils')

def generate_hexagonal_lattice(maxv1, minv1, maxv2, minv2, mindist):
    """
    This function generates a 2-dimensional lattice of points using a hexagonal
    lattice.

    Parameters
    -----------
    maxv1 : float
        Largest value in the 1st dimension to cover
    minv1 : float
        Smallest value in the 1st dimension to cover
    maxv2 : float
        Largest value in the 2nd dimension to cover
    minv2 : float
        Smallest value in the 2nd dimension to cover
    mindist : float
        Maximum allowed mismatch between a point in the parameter space and the
        generated bank of points.

    Returns
    --------
    v1s : numpy.array
        Array of positions in the first dimension
    v2s : numpy.array
        Array of positions in the second dimension
    """
    if minv1 > maxv1:
        raise ValueError("Invalid input to function.")
    if minv2 > maxv2:
        raise ValueError("Invalid input to function.")
    # Place first point
    v1s = [minv1]
    v2s = [minv2]
    initPoint = [minv1,minv2]
    # Place first line
    initLine = [initPoint]
    tmpv1 = minv1
    while (tmpv1 < maxv1):
        tmpv1 = tmpv1 + (3 * mindist)**(0.5)
        initLine.append([tmpv1,minv2])
        v1s.append(tmpv1)
        v2s.append(minv2)
    initLine = numpy.array(initLine)
    initLine2 = copy.deepcopy(initLine)
    initLine2[:,0] += 0.5 * (3*mindist)**0.5
    initLine2[:,1] += 1.5 * (mindist)**0.5
    for i in range(len(initLine2)):
        v1s.append(initLine2[i,0])
        v2s.append(initLine2[i,1])
    tmpv2_1 = initLine[0,1]
    tmpv2_2 = initLine2[0,1]
    while tmpv2_1 < maxv2 and tmpv2_2 < maxv2:
        tmpv2_1 = tmpv2_1 + 3.0 * (mindist)**0.5
        tmpv2_2 = tmpv2_2 + 3.0 * (mindist)**0.5
        initLine[:,1] = tmpv2_1
        initLine2[:,1] = tmpv2_2
        for i in range(len(initLine)):
            v1s.append(initLine[i,0])
            v2s.append(initLine[i,1])
        for i in range(len(initLine2)):
            v1s.append(initLine2[i,0])
            v2s.append(initLine2[i,1])
    v1s = numpy.array(v1s)
    v2s = numpy.array(v2s)
    return v1s, v2s

def generate_anstar_3d_lattice(maxv1, minv1, maxv2, minv2, maxv3, minv3, \
                               mindist):
    """
    This function calls into LAL routines to generate a 3-dimensional array
    of points using the An^* lattice.

    Parameters
    -----------
    maxv1 : float
        Largest value in the 1st dimension to cover
    minv1 : float
        Smallest value in the 1st dimension to cover
    maxv2 : float
        Largest value in the 2nd dimension to cover
    minv2 : float
        Smallest value in the 2nd dimension to cover
    maxv3 : float
        Largest value in the 3rd dimension to cover
    minv3 : float
        Smallest value in the 3rd dimension to cover
    mindist : float
        Maximum allowed mismatch between a point in the parameter space and the
        generated bank of points.

    Returns
    --------
    v1s : numpy.array
        Array of positions in the first dimension
    v2s : numpy.array
        Array of positions in the second dimension
    v3s : numpy.array
        Array of positions in the second dimension
    """
    # Lal/Lalpulsar are not a requirement for the rest of pycbc, so check if we have it
    # here in this function.
    try:
        import lal
        import lalpulsar
    except:
        raise ImportError(
            "A SWIG-wrapped install of lalpulsar is needed to use "
            "the anstar tiling functionality."
        )

    tiling = lalpulsar.CreateLatticeTiling(3)
    lalpulsar.SetLatticeTilingConstantBound(tiling, 0, minv1, maxv1)
    lalpulsar.SetLatticeTilingConstantBound(tiling, 1, minv2, maxv2)
    lalpulsar.SetLatticeTilingConstantBound(tiling, 2, minv3, maxv3)
    # Make a 3x3 Euclidean lattice
    a = lal.gsl_matrix(3,3)
    a.data[0,0] = 1
    a.data[1,1] = 1
    a.data[2,2] = 1
    try:
        # old versions of lalpulsar used an enumeration
        lattice = lalpulsar.TILING_LATTICE_ANSTAR
    except AttributeError:
        # newer versions of lalpulsar use a string
        lattice = 'An-star'
    lalpulsar.SetTilingLatticeAndMetric(tiling, lattice, a, mindist)
    try:
        iterator = lalpulsar.CreateLatticeTilingIterator(tiling, 3)
    except TypeError:
        # old versions of lalpulsar required the flags argument
        # (set to 0 for defaults)
        iterator = lalpulsar.CreateLatticeTilingIterator(tiling, 3, 0)

    vs1 = []
    vs2 = []
    vs3 = []
    curr_point = lal.gsl_vector(3)
    while (lalpulsar.NextLatticeTilingPoint(iterator, curr_point) > 0):
        vs1.append(curr_point.data[0])
        vs2.append(curr_point.data[1])
        vs3.append(curr_point.data[2])
    return vs1, vs2, vs3



def _round_half_away(x):
    return numpy.where(x >= 0, numpy.floor(x + 0.5), numpy.ceil(x - 0.5))


def _nearest_dn(x):
    """Nearest points of D_n, the integer vectors with even sum (Conway and
    Sloane 1982, algorithm 2)."""
    f = _round_half_away(x)
    odd = numpy.mod(f.sum(axis=1), 2) != 0
    if odd.any():
        rows = numpy.nonzero(odd)[0]
        error = x[rows] - f[rows]
        worst = numpy.argmax(numpy.abs(error), axis=1)
        step = numpy.where(error[numpy.arange(len(rows)), worst] > 0, 1, -1)
        f[rows, worst] += step
    return f


def _nearest_an(x):
    """Nearest points of A_n, the integer vectors with zero sum, for points
    on the plane of zero sum (Conway and Sloane 1982, algorithm 3)."""
    f = _round_half_away(x)
    deficiency = f.sum(axis=1).astype(int)
    order = numpy.argsort(x - f, axis=1)
    for row in numpy.nonzero(deficiency)[0]:
        d = deficiency[row]
        if d > 0:
            f[row, order[row, :d]] -= 1
        else:
            f[row, order[row, d:]] += 1
    return f


def unit_ball_volume(dim):
    """Volume of the unit ball in ``dim`` dimensions."""
    from scipy.special import gamma
    return numpy.pi ** (dim / 2) / gamma(dim / 2 + 1)


class Lattice(object):
    """A lattice in ``dim`` dimensions, scaled so its covering radius is one.

    Supported lattices are the hypercubic ``Zn``, the checkerboard ``Dn``
    (``dim`` at least 3) and ``Anstar``, the dual of the root lattice A_n,
    which covers space most thinly of all lattices known in dimensions up to
    23 (Conway and Sloane, Sphere Packings, Lattices and Groups, chapter 2).
    Points are found with the nearest-point algorithms of Conway and Sloane,
    IEEE Trans. Inf. Theory 28, 227 (1982).

    The coordinates are Euclidean: to tile a space with metric ``g``, place
    the lattice in ``y = C x`` with ``C^T C = g``, as
    :func:`lattice_points_in_box` does.

    Parameters
    ----------
    name : {'Zn', 'Dn', 'Anstar'}
        The lattice.
    dim : int
        The dimension.
    """
    names = ('Zn', 'Dn', 'Anstar')

    def __init__(self, name, dim):
        if name not in self.names:
            raise ValueError("unknown lattice {}, expected one of {}".format(
                name, ', '.join(self.names)))
        dim = int(dim)
        if dim < 1 or (name == 'Dn' and dim < 3):
            raise ValueError("{} needs a larger dimension than {}".format(
                name, dim))
        self.name, self.dim = name, dim
        n = dim
        if name == 'Zn':
            basis = numpy.eye(n)
            radius = numpy.sqrt(n) / 2
            self._embed = None
        elif name == 'Dn':
            basis = numpy.zeros((n, n))
            basis[0, :2] = -1
            for i in range(1, n):
                basis[i, i - 1], basis[i, i] = 1, -1
            radius = max(1.0, numpy.sqrt(n) / 2)
            self._embed = None
        else:
            # A_n* lives on the plane of zero sum in n + 1 dimensions
            full = numpy.zeros((n, n + 1))
            for i in range(n - 1):
                full[i, 0], full[i, i + 1] = 1, -1
            full[n - 1, :] = 1.0 / (n + 1)
            full[n - 1, 0] = -n / (n + 1.0)
            _, _, vt = numpy.linalg.svd(numpy.ones((1, n + 1)))
            self._embed = vt[1:].T
            self._glue = -full[n - 1]
            basis = full @ self._embed
            radius = numpy.sqrt(n * (n + 2) / (12.0 * (n + 1)))
        self._scale = radius
        self.generator = basis / radius
        self.determinant = abs(numpy.linalg.det(self.generator))
        self.covering_radius = 1.0
        self.thickness = unit_ball_volume(n) / self.determinant

    def nearest(self, points):
        """The lattice points nearest to each of ``points``, an array of
        shape ``(npoints, dim)``."""
        x = numpy.atleast_2d(numpy.asarray(points, dtype=float)) * self._scale
        if self.name == 'Zn':
            near = _round_half_away(x)
        elif self.name == 'Dn':
            near = _nearest_dn(x)
        else:
            x = x @ self._embed.T
            best, dist = None, None
            for i in range(self.dim + 1):
                shift = i * self._glue
                candidate = _nearest_an(x - shift) + shift
                d = numpy.sum((x - candidate) ** 2, axis=1)
                if best is None:
                    best, dist = candidate, d
                else:
                    closer = d < dist
                    best[closer], dist[closer] = candidate[closer], d[closer]
            near = best @ self._embed
        return near / self._scale

    def in_voronoi_cell(self, points, centre=None):
        """Whether each point lies in the Voronoi cell of ``centre`` (the
        origin by default), ties on the boundary counted as inside."""
        points = numpy.atleast_2d(numpy.asarray(points, dtype=float))
        if centre is not None:
            points = points - numpy.asarray(centre, dtype=float)
        near = self.nearest(points)
        return (numpy.sum(points ** 2, axis=1)
                <= numpy.sum((points - near) ** 2, axis=1) + 1e-12)

    def normalized_second_moment(self, nsamples=200000, seed=0):
        """Monte Carlo estimate of the normalized second moment G, the mean
        squared quantisation error per dimension in units of the cell
        volume. Returns the estimate and its standard error."""
        rng = numpy.random.default_rng(seed)
        cell = rng.random((nsamples, self.dim)) @ self.generator
        error = numpy.sum((cell - self.nearest(cell)) ** 2, axis=1)
        scale = self.dim * self.determinant ** (2.0 / self.dim)
        return (float(error.mean() / scale),
                float(error.std() / numpy.sqrt(nsamples) / scale))


def lattice_points_in_box(lattice, lower, upper, max_mismatch, metric=None):
    """Points of ``lattice`` covering the box ``[lower, upper]``, so that
    every point of the box is within ``max_mismatch`` of one of them. A few
    points whose cells only reach the box's whitened bounding box are kept
    too.

    Parameters
    ----------
    lattice : Lattice
        The lattice.
    lower, upper : array_like
        Corners of the box.
    max_mismatch : float
        The largest metric distance squared, ``dx^T g dx``, from a point of
        the box to its nearest lattice point.
    metric : array_like, optional
        The constant metric ``g``. Default is the identity.

    Returns
    -------
    numpy.ndarray
        The points, of shape ``(npoints, dim)``.
    """
    lower = numpy.asarray(lower, dtype=float)
    upper = numpy.asarray(upper, dtype=float)
    if lower.shape != (lattice.dim,) or upper.shape != (lattice.dim,):
        raise ValueError("lower and upper must have {} entries".format(
            lattice.dim))
    if numpy.any(upper < lower) or max_mismatch <= 0:
        raise ValueError("need upper >= lower and a positive mismatch")
    metric = (numpy.eye(lattice.dim) if metric is None
              else numpy.asarray(metric, dtype=float))
    # y = C x with C^T C = g; the lattice has covering radius sqrt(mismatch)
    whiten = numpy.linalg.cholesky(metric).T
    generator = lattice.generator * numpy.sqrt(max_mismatch)
    corners = numpy.array(numpy.meshgrid(*zip(lower, upper),
                                         indexing='ij'))
    corners = corners.reshape(lattice.dim, -1).T @ whiten.T
    # integer coefficients reaching one covering radius beyond the box
    margin = numpy.sqrt(max_mismatch)
    coeff = corners @ numpy.linalg.inv(generator)
    reach = margin * numpy.linalg.norm(numpy.linalg.inv(generator), axis=0)
    low = numpy.floor(coeff.min(axis=0) - reach)
    high = numpy.ceil(coeff.max(axis=0) + reach)
    axes = [numpy.arange(a, b + 1) for a, b in zip(low, high)]
    k = numpy.array(numpy.meshgrid(*axes, indexing='ij'))
    y = k.reshape(lattice.dim, -1).T @ generator
    # keep the points within one covering radius of the box's bounding box
    # in the whitened frame, a superset of those whose cells meet the box
    keep = numpy.all((y >= corners.min(axis=0) - margin)
                     & (y <= corners.max(axis=0) + margin), axis=1)
    return numpy.linalg.solve(whiten, y[keep].T).T
