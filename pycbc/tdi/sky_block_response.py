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


def _device_geometry(geometry, xp):
    """The arrays of a :class:`~pycbc.tdi.onthefly.MultiChannelTermGeometry`
    the frequency-domain response needs, on ``xp``, kept on the geometry."""
    cache = getattr(geometry, '_sky_block_arrays', None)
    if cache is not None and cache[0] is xp:
        return cache[1]
    g = dict(coefficient=geometry.coefficient[:, 0], ltt=geometry.ltt,
             chain_delay=geometry.chain_delay, n_hat=geometry.n_hat,
             r_emit=geometry.r_emit, r_recv=geometry.r_recv)
    if geometry.velocity_order:
        g.update(v_emit=geometry.v_emit, v_recv=geometry.v_recv,
                 n_dot_v_recv=geometry.n_dot_v_recv,
                 n_dot_v_mix=geometry.n_dot_v_mix)
    g = {k: xp.asarray(v) for k, v in g.items()}
    g['velocity_order'] = geometry.velocity_order
    g['channels'] = [xp.asarray(numpy.nonzero(geometry.term_channel == i)[0])
                     for i in range(len(geometry.channel_names))]
    geometry._sky_block_arrays = (xp, g)
    return g


def _response_factors(g, omega, lon, lat, xp):
    """:func:`pycbc.tdi.onthefly.frequency_response_factors` at several sky
    points at once: shape ``(M, 2, nchannels, ngrid)``, plus then cross."""
    from pycbc.tdi.response import C_SI
    lon, lat = xp.asarray(lon), xp.asarray(lat)
    sl, cl, sb, cb = xp.sin(lon), xp.cos(lon), xp.sin(lat), xp.cos(lat)
    zero = xp.zeros_like(lon)
    # polarization_basis, one column per point
    u = xp.stack([sl, -cl, zero])
    v = xp.stack([-sb * cl, -sb * sl, cb])
    k = xp.stack([-cb * cl, -cb * sl, -sb])

    def dot(a, w):
        # (T, G, 3) against (3, M) -> (M, T, G)
        t, n = a.shape[:2]
        return xp.moveaxis((a.reshape(-1, 3) @ w).reshape(t, n, -1), -1, 0)
    nu, nv, nk = dot(g['n_hat'], u), dot(g['n_hat'], v), dot(g['n_hat'], k)
    # antenna_prefactor, its removable singularity near n = k elementwise
    plus, cross = nu ** 2 - nv ** 2, 2 * nu * nv
    gap = 1 - nk
    limit = gap < 1e-4
    transverse = nu ** 2 + nv ** 2
    resolved = transverse > (64 * numpy.finfo(float).eps) ** 2
    scale = xp.where(resolved, 0.5 * (1 + nk)
                     / xp.where(resolved, transverse, 1.0), 0.0)
    half = 1 / (2 * xp.where(limit, 1.0, gap))
    pp = xp.where(limit, scale * plus, plus * half)
    pc = xp.where(limit, scale * cross, cross * half)
    del nu, nv, plus, cross, gap, limit, transverse, resolved, scale, half
    tau_emit = g['ltt'][None] + dot(g['r_emit'], k) / C_SI
    tau_recv = dot(g['r_recv'], k) / C_SI
    chain = g['chain_delay'][None]
    phase = -1j * omega[None, None]
    transfer = xp.exp(phase * (chain + tau_emit))
    recv = xp.exp(phase * (chain + tau_recv))
    if g['velocity_order']:
        transfer *= 1 + (-dot(g['v_emit'], k) + g['n_dot_v_recv'][None]) / C_SI
        recv *= 1 + (-dot(g['v_recv'], k) + g['n_dot_v_mix'][None]) / C_SI
    transfer -= recv
    del recv, tau_emit, tau_recv
    transfer *= g['coefficient'][None, :, None]
    out = xp.empty((len(lon), 2, len(g['channels']), transfer.shape[-1]),
                   dtype=transfer.dtype)
    for c, sel in enumerate(g['channels']):
        out[:, 0, c] = (transfer[:, sel] * pp[:, sel]).sum(axis=1)
        out[:, 1, c] = (transfer[:, sel] * pc[:, sel]).sum(axis=1)
    return out


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
    exact_frequencies : int or array of int, optional
        Compute the exact response at this many of the frequencies, evenly
        spread, or at these indices of them (the first and last are always
        added), and interpolate the basis linearly to the rest: the
        response changes slowly with frequency. Default all.
    interpolation_check : {'midpoints', 'all'}
        Where the interpolation error is measured, at a few held-out
        points: the midpoints between exact frequencies (default), or every
        frequency that is not exact. A narrow feature between exact
        frequencies (the source's power at a TDI-2 zero, say) can hide from
        the midpoints; 'all' costs the exact response at every frequency at
        those few points.
    """
    def __init__(self, orbit, terms, frequencies, centre, scales, radius,
                 kmax=8, samples=32, ratio=None, weight=None, points=400,
                 test_points=100, tolerance=1e-4, max_degree=24, seed=0,
                 exact_frequencies=None, xp=None,
                 interpolation_check='midpoints'):
        from pycbc.tdi.onthefly import MultiChannelTermGeometry
        self.output_frequencies = numpy.asarray(frequencies, dtype=float)
        n_out = len(self.output_frequencies)
        if interpolation_check not in ('midpoints', 'all'):
            raise ValueError("interpolation_check is 'midpoints' or 'all'")
        if exact_frequencies is None:
            pick = numpy.arange(n_out)
        elif numpy.ndim(exact_frequencies) == 0:
            pick = numpy.arange(n_out) if exact_frequencies >= n_out else \
                numpy.unique(numpy.linspace(0, n_out - 1, int(
                    exact_frequencies)).round().astype(int))
        else:
            pick = numpy.asarray(exact_frequencies, dtype=int)
            if pick.min() < 0 or pick.max() >= n_out:
                raise ValueError("exact frequency index out of range")
            pick = numpy.unique(numpy.concatenate([pick, [0, n_out - 1]]))
        self.frequencies = self.output_frequencies[pick]
        self.centre = (float(centre[0]), float(centre[1]))
        self.scales = (float(scales[0]), float(scales[1]))
        self.radius = float(radius)
        self.kmax, self.samples = int(kmax), int(samples)
        if 2 * self.kmax + 1 > self.samples:
            raise ValueError("too few response samples for kmax")
        self.orders = numpy.arange(-self.kmax, self.kmax + 1)
        self.tolerance = float(tolerance)
        self._xp = numpy if xp is None else xp
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
        # the basis: leading right singular vectors of the training rows,
        # from the eigenvectors of the small matrix A A^H rather than a full
        # SVD of A, whose work space is several times A
        values, vectors = numpy.linalg.eigh(A @ numpy.conj(A.T))
        order = numpy.argsort(values)[::-1]
        values, vectors = numpy.maximum(values[order], 0), vectors[:, order]
        keep = int(numpy.sum(values > values[0] * 1e-14))
        coords = vectors[:, :keep] * numpy.sqrt(values[:keep])[None]
        norm2 = numpy.sum(numpy.abs(A) ** 2, axis=1)
        left = norm2[:, None] - numpy.cumsum(numpy.abs(coords) ** 2, axis=1)
        # the training rows' worst relative error at each rank
        err = numpy.sqrt(numpy.maximum(left, 0) / norm2[:, None]).max(axis=0)
        rank = self._rank_at(err, self.tolerance)
        vt = (numpy.conj(vectors[:, :rank].T) @ A) / numpy.sqrt(
            values[:rank])[:, None]
        del A
        # what a looser block (:meth:`coarsened`) is fitted from: a looser
        # tolerance keeps the leading part of the same basis
        self._fit_state = dict(train=train, test=test, B=B, err=err,
                               coords=coords[:, :rank], vt=vt,
                               max_degree=int(max_degree), n_out=n_out,
                               pick=pick, w_out=w_out)
        if len(pick) < n_out:
            # the exact harmonics at a few held-out points: at the
            # midpoints between the exact frequencies, where linear
            # interpolation is worst for a smooth response, or at every
            # frequency that is not exact
            if interpolation_check == 'all':
                mid = numpy.delete(self.output_frequencies, pick)
            else:
                mid = 0.5 * (self.frequencies[1:] + self.frequencies[:-1])
            full = SkyBlockResponse.__new__(SkyBlockResponse)
            full.__dict__.update(self.__dict__)
            full.frequencies = mid
            full._ratio = None if ratio is None else numpy.array([
                numpy.interp(mid, self.output_frequencies, r)
                for r in numpy.asarray(ratio)])
            t = (numpy.arange(self.samples)[:, None] * ORBITAL_PERIOD
                 / self.samples + 1e-3 * numpy.arange(len(mid))[None])
            full._geometry = MultiChannelTermGeometry(orbit, t.ravel(), terms)
            full._shift = numpy.exp(-2j * numpy.pi * numpy.tile(
                mid, self.samples) * ssb_delay(orbit, t.ravel(), *self.centre))
            some = test[:min(8, len(test))]
            self._fit_state.update(mid=mid, some=some,
                                   mid_exact=full.exact(self._offsets(some)))
            del full
        self._fit(rank)

    @staticmethod
    def _rank_at(err, tolerance):
        good = numpy.nonzero(err <= tolerance / 3)[0]
        return int(good[0]) + 1 if len(good) else len(err)

    def _fit(self, rank):
        """The block at ``rank`` from the stored fit: the coordinates as
        Chebyshev series in the block, the basis on every frequency and the
        held-out error."""
        st = self._fit_state
        self.rank = int(rank)
        self._vt = st['vt'][:self.rank]
        coords = st['coords'][:, :self.rank]
        train, test = st['train'], st['test']
        # the coordinates as Chebyshev series in the block, the degree
        # raised until the held-out points meet the tolerance
        for degree in range(2, st['max_degree'] + 1, 2):
            V = chebyshev.chebvander2d(train[:, 0], train[:, 1],
                                       [degree, degree])
            if V.shape[1] > len(train) // 2:
                break
            self._series = numpy.linalg.lstsq(V, coords, rcond=None)[0]
            self.degree = degree
            self.error = float(self._error(test, st['B']))
            if self.error <= self.tolerance:
                break
        # the basis on every frequency, unweighted, by linear interpolation
        nk, nf = len(self.orders), len(self.frequencies)
        inverse = numpy.where(self._root > 0, 1 / numpy.maximum(self._root,
                                                                 1e-300), 0)
        V = (self._vt.reshape(self.rank, nk, 2, -1, nf) * inverse)
        if len(st['pick']) < st['n_out']:
            V = numpy.apply_along_axis(
                lambda v: numpy.interp(self.output_frequencies,
                                       self.frequencies, v.real)
                + 1j * numpy.interp(self.output_frequencies,
                                    self.frequencies, v.imag), -1, V)
            mid, some, exact = st['mid'], st['some'], st['mid_exact']
            w_out = st['w_out']
            root = numpy.sqrt(numpy.interp(mid, self.output_frequencies,
                                           w_out / w_out.sum()))
            at_mid = numpy.apply_along_axis(
                lambda v: numpy.interp(mid, self.output_frequencies, v.real)
                + 1j * numpy.interp(mid, self.output_frequencies, v.imag),
                -1, V)
            approx = numpy.einsum('mr,r...->m...', self._coords(some), at_mid)
            diff = ((approx - exact) * root).reshape(len(some), -1)
            ref = (exact * root).reshape(len(some), -1)
            self.interpolation_error = float(numpy.max(
                numpy.linalg.norm(diff, axis=1)
                / numpy.linalg.norm(ref, axis=1)))
            self.error = max(self.error, self.interpolation_error)
        self._basis = V
        self.__dict__.pop('_device_basis', None)
        logging.info("sky block at (%.4f, %.4f) of radius %.3f rad: rank "
                     "%d, Chebyshev degree %d, held-out error %.2g",
                     self.centre[0], self.centre[1], self.radius, self.rank,
                     self.degree, self.error)
        if self.error > self.tolerance:
            logging.warning("sky block error %.2g above the tolerance %.2g: "
                            "more points or a smaller block", self.error,
                            self.tolerance)

    def coarsened(self, tolerance):
        """The same block at a looser ``tolerance``, from this fit: the
        leading part of its basis, the series refitted. The same as fitting
        the block afresh at ``tolerance`` (same points, same seed) without
        computing the exact response again."""
        st = getattr(self, '_fit_state', None)
        if st is None:
            raise ValueError("a block keeps its fit only until it is saved, "
                             "loaded or released")
        if tolerance < self.tolerance:
            raise ValueError("a coarsened block cannot be tighter")
        out = SkyBlockResponse.__new__(SkyBlockResponse)
        out.__dict__.update(self.__dict__)
        out.__dict__.pop('_device_basis', None)
        out.tolerance = float(tolerance)
        out._fit(self._rank_at(st['err'], out.tolerance))
        return out

    def release(self):
        """Drop what :meth:`coarsened` needs (the held-out rows, the
        coordinates of the training points)."""
        self.__dict__.pop('_fit_state', None)

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

    def exact(self, offsets, chunk=None):
        """The exact harmonics at family ``offsets`` ``(M, 2)``, shape ``(M, 2
        kmax + 1, 2, nchannels, nfrequencies)``: the response at the block's
        times and frequencies, ``chunk`` points at a time (by default as many
        as fit in about 256 MB of work space), on the block's array module
        ``xp``."""
        offsets = numpy.atleast_2d(numpy.asarray(offsets, dtype=float))
        xp = getattr(self, '_xp', numpy)
        g = _device_geometry(self._geometry, xp)
        nf, ns = len(self.frequencies), self.samples
        nt, ng = g['ltt'].shape
        omega = xp.asarray(2 * numpy.pi * numpy.tile(self.frequencies, ns))
        shift = xp.asarray(self._shift)
        ratio = None if self._ratio is None else xp.asarray(self._ratio)
        pick = xp.asarray(self.orders % ns)
        if chunk is None:
            chunk = max(1, int(256 * 1024 ** 2 // (160 * nt * ng)))
        lon = self.centre[0] + offsets[:, 0] * self.scales[0]
        lat = self.centre[1] + offsets[:, 1] * self.scales[1]
        out = []
        for i in range(0, len(offsets), chunk):
            r = _response_factors(g, omega, lon[i:i + chunk],
                                  lat[i:i + chunk], xp) * shift
            r = r.reshape(len(r), 2, -1, ns, nf).transpose(0, 3, 1, 2, 4)
            if ratio is not None:
                r = r * ratio[None, None, None]
            rho = xp.fft.fft(r, axis=1)[:, pick] / ns
            out.append(rho.get() if hasattr(rho, 'get') else rho)
        return numpy.concatenate(out)

    def _coords(self, unit):
        """The series' coordinates at points of the unit square: one
        product of the points' Chebyshev Vandermonde rows with the series'
        coefficients, column ``i (degree + 1) + j`` of the one being the
        coefficient ``[i, j]`` of the other."""
        unit = numpy.clip(numpy.atleast_2d(unit), -1, 1)
        k = numpy.arange(self.degree + 1)
        # T_k(x) = cos(k arccos x), the Vandermonde rows as an outer product
        tx = numpy.cos(k * numpy.arccos(unit[:, :1]))
        ty = numpy.cos(k * numpy.arccos(unit[:, 1:2]))
        V = (tx[:, :, None] * ty[:, None, :]).reshape(len(unit), -1)
        return V @ self._series[:, :self.rank]

    def _error(self, unit, rows):
        approx = self._coords(unit) @ self._vt
        return numpy.max(numpy.linalg.norm(approx - rows, axis=1)
                         / numpy.linalg.norm(rows, axis=1))

    def responses(self, offsets, xp=numpy, dtype=None):
        """The harmonics at family ``offsets`` ``(M, 2)`` from the basis,
        shape ``(M, 2 kmax + 1, 2, nchannels, nfrequencies)``; on the
        device in ``dtype`` (the basis's own by default), the basis kept
        there in it."""
        c = self._coords(self._unit(offsets))
        if xp is numpy:
            out = numpy.einsum('mr,r...->m...', c, self._basis)
            return out if dtype is None else out.astype(dtype)
        dtype = numpy.dtype(self._basis.dtype if dtype is None else dtype)
        cache = getattr(self, '_device_basis', None)
        if cache is None or cache[0] is not xp or cache[1].dtype != dtype:
            cache = (xp, xp.asarray(self._basis.reshape(self.rank, -1),
                                    dtype=dtype))
            self._device_basis = cache
        rows = xp.asarray(c, dtype=dtype) @ cache[1]
        return rows.reshape((len(c),) + self._basis.shape[1:])

    _SAVED = ('output_frequencies', 'frequencies', 'centre', 'scales',
              'radius', 'kmax', 'samples', 'orders', 'tolerance', 'rank',
              '_series', 'degree', 'error', '_basis', '_root', '_vt')

    def save(self, path):
        """Store the fitted block (not its orbit geometry, which only the
        fit needs) in an ``.npz`` file."""
        numpy.savez(path, **{k: numpy.asarray(getattr(self, k))
                             for k in self._SAVED})

    @classmethod
    def load(cls, path):
        """A block stored by :meth:`save`; :meth:`exact` is not available
        on it."""
        data = numpy.load(path)
        self = cls.__new__(cls)
        for k in cls._SAVED:
            v = data[k]
            setattr(self, k, v.item() if v.ndim == 0 else v)
        self.centre, self.scales = tuple(self.centre), tuple(self.scales)
        self.kmax, self.samples = int(self.kmax), int(self.samples)
        self.rank, self.degree = int(self.rank), int(self.degree)
        self._geometry = None
        self._xp = numpy
        return self

    @classmethod
    def cached(cls, directory, *args, **kwargs):
        """The block of ``args`` and ``kwargs`` (as the constructor takes
        them), fitted once and kept in ``directory`` under a hash of every
        input to the fit; a block depends on no data."""
        import os
        path = cls._cache_path(directory, args, kwargs)
        if os.path.exists(path):
            return cls.load(path)
        block = cls(*args, **kwargs)
        os.makedirs(directory, exist_ok=True)
        block.save(path)
        return block

    @staticmethod
    def _cache_path(directory, args, kwargs):
        """The file of a block in ``directory``: a hash of every input to
        the fit but the array module it runs on."""
        import hashlib
        import os
        h = hashlib.sha256()
        for v in list(args[2:]) + [kwargs[k] for k in sorted(kwargs)
                                   if k != 'xp']:
            h.update(numpy.ascontiguousarray(
                numpy.asarray(v, dtype=float) if v is not None
                else numpy.zeros(0)).tobytes())
        h.update(repr(args[1]).encode())
        return os.path.join(directory, 'sky_block_%s.npz' % h.hexdigest()[:24])

    def inside(self, offsets):
        """Whether each point lies in the block."""
        return numpy.all(numpy.abs(self._unit(offsets)) <= 1, axis=1)


def sky_block(*args, cache=None, **kwargs):
    """A :class:`SkyBlockResponse`, from ``cache`` (a directory; by default
    the environment's ``PYCBC_SKY_BLOCK_CACHE``) when it has been fitted
    before."""
    import os
    cache = cache or os.environ.get('PYCBC_SKY_BLOCK_CACHE')
    if cache:
        return SkyBlockResponse.cached(cache, *args, **kwargs)
    return SkyBlockResponse(*args, **kwargs)


def sky_blocks(*args, tolerances=(1e-4,), cache=None, **kwargs):
    """The blocks of ``args`` and ``kwargs`` (as :class:`SkyBlockResponse`
    takes them) at each of ``tolerances``, from one fit at the tightest
    (:meth:`SkyBlockResponse.coarsened`), as a two-stage search wants its
    fine and coarse blocks; from ``cache`` (as :func:`sky_block`) when all
    have been fitted before, and stored there."""
    import os
    cache = cache or os.environ.get('PYCBC_SKY_BLOCK_CACHE')
    tolerances = [float(t) for t in tolerances]
    paths = [SkyBlockResponse._cache_path(cache, args, dict(kwargs,
                                                            tolerance=t))
             for t in tolerances] if cache else None
    if paths and all(os.path.exists(f) for f in paths):
        return [SkyBlockResponse.load(f) for f in paths]
    tight = min(tolerances)
    block = SkyBlockResponse(*args, tolerance=tight, **kwargs)
    out = [block if t == tight else block.coarsened(t) for t in tolerances]
    for b in out:
        b.release()
    if paths:
        os.makedirs(cache, exist_ok=True)
        for b, f in zip(out, paths):
            if not os.path.exists(f):
                b.save(f)
    return out


def exact_frequency_count(frequencies, spacing, minimum=5):
    """Exact frequencies for :class:`SkyBlockResponse` spaced by at most
    ``spacing`` Hz over the band of ``frequencies``, at least ``minimum``,
    at most all of them.

    The count follows the band: a long-tau reference's band of 1e-4 Hz
    needs a handful where a short one of 1e-2 Hz needs hundreds, and a fixed
    count spends the same exact response on both.
    """
    f = numpy.asarray(frequencies, dtype=float)
    n = int(numpy.ceil((f[-1] - f[0]) / float(spacing))) + 1
    return int(min(len(f), max(int(minimum), n)))


def exact_frequency_indices(frequencies, spacing, minimum=5, dense=None):
    """Indices of exact frequencies for :class:`SkyBlockResponse`: spaced by
    at most ``spacing`` Hz over the band (:func:`exact_frequency_count`),
    plus every frequency inside the bands ``dense``, a sequence of
    ``(low, high)`` Hz.

    Next to a notched TDI-2 zero the weighted response changes faster than
    anywhere else in the band, and the interpolation error of a block
    concentrates there; dense exact frequencies over a band a little wider
    than the notch remove it at a fraction of the cost of spacing the whole
    band that finely.
    """
    f = numpy.asarray(frequencies, dtype=float)
    pick = numpy.linspace(0, len(f) - 1, exact_frequency_count(
        f, spacing, minimum)).round().astype(int)
    extra = [numpy.nonzero((f >= lo) & (f <= hi))[0]
             for lo, hi in (dense or ())]
    return numpy.unique(numpy.concatenate([pick] + extra))
