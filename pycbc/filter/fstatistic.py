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
"""The F-statistic of a two-basis template.

With ``z_a = <y_a, d>`` the correlations of the data with the plus and cross
bases and ``G_ab = <y_a, y_b>`` their Gram matrix, maximising the likelihood
over the four real amplitudes gives ``2F = z^H G^-1 z``. Where the template
barely overlaps the data ``G`` loses rank, so each value is classified by
the eigenvalues ``l1 >= l2`` of ``G``:

====  ==========================  =====================================
code  condition                   2F
====  ==========================  =====================================
0     full rank                   ``z^H G^-1 z``
1     ``l2 <= rtol l1``           ``|u1^H z|^2 / l1``, the pseudo-inverse
2     ``l1 <= ctol scale``        ``-inf``: the template misses the data
3     non-finite, or indefinite   ``nan``: unresolved, never discarded
====  ==========================  =====================================

``scale`` is the largest trace of ``G`` over the values given. The arithmetic
is in double precision, on numpy or cupy arrays alike.
"""

import numpy

FULL_RANK, RANK_ONE, NO_COVERAGE, UNRESOLVED = 0, 1, 2, 3


def twof(z, g00, g11, g01, kappa=1.0, rank_tolerance=1e-6,
         coverage_tolerance=1e-6, scale=None, xp=numpy):
    """The F-statistic and its rank class at every sample.

    Parameters
    ----------
    z : array, complex, shape ``(2, n)``
        Correlations with the plus and cross bases.
    g00, g11 : array, real, shape ``(n,)``
        The diagonal of the Gram matrix.
    g01 : array, complex, shape ``(n,)``
        Its off-diagonal element ``<y_0, y_1>``.
    kappa : float
        A calibration factor multiplying 2F.
    rank_tolerance, coverage_tolerance : float
        ``rtol`` and ``ctol`` of the module docstring.
    scale : float, optional
        Reference for the coverage test; default the largest trace given.
    xp : module
        numpy or cupy, as :func:`pycbc.scheme.array_module` returns.

    Returns
    -------
    twof : array, shape ``(n,)``
    code : array of int8, shape ``(n,)``
    """
    z0 = z[0].astype(xp.complex128)
    z1 = z[1].astype(xp.complex128)
    a = g00.astype(xp.float64)
    d = g11.astype(xp.float64)
    c = g01.astype(xp.complex128)
    if scale is None:
        scale = float(xp.max(a + d))
    finite = (xp.isfinite(a) & xp.isfinite(d) & xp.isfinite(c)
              & xp.isfinite(z0) & xp.isfinite(z1))
    c2 = xp.abs(c) ** 2
    trace = a + d
    l1 = 0.5 * (trace + xp.sqrt((a - d) ** 2 + 4 * c2))
    det = a * d - c2
    with numpy.errstate(divide='ignore', invalid='ignore'):
        l2 = xp.where(l1 > 0, det / l1, trace - l1)
        full = (d * xp.abs(z0) ** 2 + a * xp.abs(z1) ** 2
                - 2 * xp.real(c * xp.conj(z0) * z1)) / det
        # eigenvector of l1: (c, l1 - a), or (l1 - d, conj c) when that is 0
        u0, u1 = c, (l1 - a) + 0j
        small = xp.abs(u0) ** 2 + xp.abs(u1) ** 2 < 1e-300
        u0 = xp.where(small, (l1 - d) + 0j, u0)
        u1 = xp.where(small, xp.conj(c), u1)
        norm = xp.sqrt(xp.abs(u0) ** 2 + xp.abs(u1) ** 2)
        single = xp.abs(xp.conj(u0) * z0 + xp.conj(u1) * z1) ** 2 \
            / (norm ** 2 * l1)
    missing = ~(l1 > coverage_tolerance * scale)
    bad = (l2 < -rank_tolerance * l1) | (a < 0) | (d < 0)
    rank_one = l2 <= rank_tolerance * l1
    code = xp.full(a.shape, FULL_RANK, dtype=xp.int8)
    code = xp.where(rank_one, RANK_ONE, code)
    code = xp.where(bad, UNRESOLVED, code)
    code = xp.where(missing, NO_COVERAGE, code)
    code = xp.where(finite, code, UNRESOLVED).astype(xp.int8)
    value = xp.where(code == FULL_RANK, full, single)
    value = xp.where(code == NO_COVERAGE, -numpy.inf, value)
    value = xp.where(code == UNRESOLVED, numpy.nan, value)
    return kappa * value, code
