# Copyright (C) 2026 Shichao Wu
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
"""The yearly harmonics of the response over a block of the sky, from the
exact response.

A family's first-order sky model ``R0 exp(delta . L)`` holds to about 1% within
0.01 rad of its reference and fails beyond a few hundredths of a radian, while
the exact response costs about half a second a point. Over a block of the sky
the exact harmonics are close to a low-rank set: an SVD basis of them at a
few hundred points, and each point's coordinates on it as a two-dimensional
Chebyshev series in the sky, give the harmonics anywhere in the block at the
cost of a small matrix product. Points held out of the fit give the error.
"""
import logging

import numpy
from numpy.polynomial import chebyshev

from pycbc.tdi.ssb_yearly_harmonics import ORBITAL_PERIOD, ssb_delay


class SkyBlockResponse(object):
    """The response harmonics over a block of the sky.

    The block is ``|lon - lon0| cos(lat0) <= radius`` and
    ``|lat - lat0| <= radius``. Offsets are in the family's shape units,
    ``(lon - lon0) / scales[0]`` and ``(lat - lat0) / scales[1]``, so that
    :meth:`responses` stands in for the family's own.

    Parameters
    ----------
    orbit, terms
        The orbit and the TDI channel terms of the response.
    frequencies : array
        The response's coarse frequencies.
    centre : tuple
        Ecliptic longitude and latitude of the block's centre, the
        reference sky of the family, rad.
    scales : tuple
        The family's sky shape units, rad.
    radius : float
        The block's half-width, rad.
    kmax : int
        Yearly harmonics kept, ``-kmax .. kmax``.
    samples : int
        Response samples per orbit.
    ratio : array, optional
        Factor of each channel at each coarse frequency, shape
        ``(nchannels, nfrequencies)``, as the family carries.
    weight : array, optional
        Weight of each coarse frequency in the error norm; default uniform.
    points, test_points : int
        Points of the block the basis and the series are fitted to, and
        points held out to measure the error.
    tolerance : float
        Largest relative error of a point's harmonics accepted from the
        basis; the series' degree grows until the held-out points meet it
        too, or ``max_degree`` is reached.
    exact_frequencies : int, optional
        Compute the exact response at this many of the frequencies, evenly
        spread, and interpolate the basis linearly to the rest: the
        response changes slowly with frequency. Its error at the held-out
        points is measured at every frequency. Default all.
    """
    def __init__(self, orbit, terms, frequencies, centre, scales, radius,
                 kmax=8, samples=32, ratio=None, weight=None, points=400,
                 test_points=100, tolerance=1e-4, max_degree=24, seed=0,
                 exact_frequencies=None):
        from pycbc.tdi.onthefly import MultiChannelTermGeometry
        self.output_frequencies = numpy.asarray(frequencies, dtype=float)
        n_out = len(self.output_frequencies)
        pick = numpy.arange(n_out) if exact_frequencies is None or \
            exact_frequencies >= n_out else numpy.unique(numpy.linspace(
                0, n_out - 1, int(exact_frequencies)).round().astype(int))
        self.frequencies = self.output_frequencies[pick]
        self.centre = (float(centre[0]), float(centre[1]))
        self.scales = (float(scales[0]), float(scales[1]))
        self.radius = float(radius)
        self.kmax, self.samples = int(kmax), int(samples)
        if 2 * self.kmax + 1 > self.samples:
            raise ValueError("too few response samples for kmax")
        self.orders = numpy.arange(-self.kmax, self.kmax + 1)
        self.tolerance = float(tolerance)
        nf = len(self.frequencies)
        # the times of pycbc.tdi.ssb_yearly_harmonics.yearly_response_samples
        t = (numpy.arange(self.samples)[:, None] * ORBITAL_PERIOD
             / self.samples + 1e-3 * numpy.arange(nf)[None])
        self._times = t.ravel()
        self._geometry = MultiChannelTermGeometry(orbit, self._times, terms)
        self._shift = numpy.exp(-2j * numpy.pi * numpy.tile(
            self.frequencies, self.samples) * ssb_delay(
                orbit, self._times, *self.centre))
        self._ratio = None if ratio is None else numpy.asarray(ratio)[:, pick]
        w = numpy.ones(n_out) if weight is None else \
            numpy.asarray(weight, float)
        # every frequency kept in the basis, however little power it holds:
        # the basis is interpolated between the exact frequencies
        w = numpy.maximum(w, 1e-6 * w.max())
        w_out = w
        w = w[pick]
        self._root = numpy.sqrt(w / w.sum())
        rng = numpy.random.default_rng(seed)
        train = rng.uniform(-1, 1, (int(points), 2))
        test = rng.uniform(-1, 1, (int(test_points), 2))
        A = self._rows(self.exact(self._offsets(train)))
        B = self._rows(self.exact(self._offsets(test)))
        # the basis: leading right singular vectors of the training rows
        _, sig, vt = numpy.linalg.svd(A, full_matrices=False)
        coords = A @ numpy.conj(vt.T)
        norm2 = numpy.sum(numpy.abs(A) ** 2, axis=1)
        left = norm2[:, None] - numpy.cumsum(numpy.abs(coords) ** 2, axis=1)
        err = numpy.sqrt(numpy.maximum(left, 0) / norm2[:, None]).max(axis=0)
        good = numpy.nonzero(err <= self.tolerance / 3)[0]
        self.rank = int(good[0]) + 1 if len(good) else len(sig)
        self._vt = vt[:self.rank]
        coords = coords[:, :self.rank]
        # the coordinates as Chebyshev series in the block, the degree
        # raised until the held-out points meet the tolerance
        for degree in range(2, int(max_degree) + 1, 2):
            V = chebyshev.chebvander2d(train[:, 0], train[:, 1],
                                       [degree, degree])
            if V.shape[1] > len(train) // 2:
                break
            self._series = numpy.linalg.lstsq(V, coords, rcond=None)[0]
            self.degree = degree
            self.error = float(self._error(test, B))
            if self.error <= self.tolerance:
                break
        # the basis on every frequency, unweighted, by linear interpolation
        nk = len(self.orders)
        inverse = numpy.where(self._root > 0, 1 / numpy.maximum(self._root,
                                                                 1e-300), 0)
        V = (self._vt.reshape(self.rank, nk, 2, -1, nf) * inverse)
        if len(pick) < n_out:
            V = numpy.apply_along_axis(
                lambda v: numpy.interp(self.output_frequencies,
                                       self.frequencies, v.real)
                + 1j * numpy.interp(self.output_frequencies,
                                    self.frequencies, v.imag), -1, V)
            # the error at every frequency, at a few held-out points
            full = SkyBlockResponse.__new__(SkyBlockResponse)
            full.__dict__.update(self.__dict__)
            full.frequencies = self.output_frequencies
            full._ratio = None if ratio is None else numpy.asarray(ratio)
            t = (numpy.arange(self.samples)[:, None] * ORBITAL_PERIOD
                 / self.samples + 1e-3 * numpy.arange(n_out)[None])
            full._geometry = MultiChannelTermGeometry(orbit, t.ravel(), terms)
            full._shift = numpy.exp(-2j * numpy.pi * numpy.tile(
                self.output_frequencies, self.samples) * ssb_delay(
                    orbit, t.ravel(), *self.centre))
            some = test[:min(8, len(test))]
            exact = full.exact(self._offsets(some))
            root = numpy.sqrt(w_out / w_out.sum())
            approx = numpy.einsum('mr,r...->m...', self._coords(some), V)
            diff = ((approx - exact) * root).reshape(len(some), -1)
            ref = (exact * root).reshape(len(some), -1)
            self.interpolation_error = float(numpy.max(
                numpy.linalg.norm(diff, axis=1)
                / numpy.linalg.norm(ref, axis=1)))
            self.error = max(self.error, self.interpolation_error)
        self._basis = V
        logging.info("sky block at (%.4f, %.4f) of radius %.3f rad: rank "
                     "%d, Chebyshev degree %d, held-out error %.2g",
                     self.centre[0], self.centre[1], self.radius, self.rank,
                     self.degree, self.error)
        if self.error > self.tolerance:
            logging.warning("sky block error %.2g above the tolerance %.2g: "
                            "more points or a smaller block", self.error,
                            self.tolerance)

    def _offsets(self, unit):
        """Family offsets of points of the unit square."""
        lon0, lat0 = self.centre
        lon = lon0 + unit[:, 0] * self.radius / numpy.cos(lat0)
        lat = lat0 + unit[:, 1] * self.radius
        return numpy.column_stack([(lon - lon0) / self.scales[0],
                                   (lat - lat0) / self.scales[1]])

    def _unit(self, offsets):
        lat0 = self.centre[1]
        d = numpy.atleast_2d(numpy.asarray(offsets, dtype=float))
        return numpy.column_stack([
            d[:, 0] * self.scales[0] * numpy.cos(lat0) / self.radius,
            d[:, 1] * self.scales[1] / self.radius])

    def _rows(self, rho):
        """Weighted, flattened harmonics ``(M, D)``."""
        return (rho * self._root).reshape(len(rho), -1)

    def exact(self, offsets):
        """The exact harmonics at family ``offsets`` ``(M, 2)``, shape ``(M, 2
        kmax + 1, 2, nchannels, nfrequencies)``."""
        from pycbc.tdi.onthefly import frequency_response_factors
        offsets = numpy.atleast_2d(numpy.asarray(offsets, dtype=float))
        nf = len(self.frequencies)
        f = numpy.tile(self.frequencies, self.samples)
        out = []
        for d in offsets:
            lon = self.centre[0] + d[0] * self.scales[0]
            lat = self.centre[1] + d[1] * self.scales[1]
            factors = frequency_response_factors(self._geometry, f, lon, lat)
            names = self._geometry.channel_names
            r = numpy.array([[factors[c][a] for c in names]
                             for a in range(2)]) * self._shift
            r = r.reshape(2, len(names), self.samples, nf).transpose(2, 0, 1,
                                                                     3)
            if self._ratio is not None:
                r = r * self._ratio[None, None]
            rho = numpy.fft.fft(r, axis=0) / self.samples
            out.append(rho[self.orders % self.samples])
        return numpy.array(out)

    def _coords(self, unit):
        """The series' coordinates at points of the unit square."""
        n = self.degree + 1
        return numpy.stack([chebyshev.chebval2d(
            unit[:, 0], unit[:, 1], self._series[:, r].reshape(n, n))
            for r in range(self.rank)], axis=1)

    def _error(self, unit, rows):
        approx = self._coords(unit) @ self._vt
        return numpy.max(numpy.linalg.norm(approx - rows, axis=1)
                         / numpy.linalg.norm(rows, axis=1))

    def responses(self, offsets, xp=numpy):
        """The harmonics at family ``offsets`` ``(M, 2)`` from the basis,
        shape ``(M, 2 kmax + 1, 2, nchannels, nfrequencies)``."""
        c = self._coords(self._unit(offsets))
        return xp.asarray(numpy.einsum('mr,r...->m...', c, self._basis))

    def inside(self, offsets):
        """Whether each point lies in the block."""
        return numpy.all(numpy.abs(self._unit(offsets)) <= 1, axis=1)
