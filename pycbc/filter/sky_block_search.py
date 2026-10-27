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
"""A search over blocks of the sky, one reference waveform each.

The PN-coefficient family is exact in the intrinsic phase over the whole
prior, so a cell is set by the sky: each block of the sky gets the model
recentred there (a sky block of the exact response, the data resampled to SSB
time at its centre), and one
:func:`pycbc.filter.nufft_correlation.phase_grid_search` covers the whole
intrinsic region, the block's sky points and the arrival times. A source seen
from several blocks is kept once, from the loudest.
"""
import logging
import time

import numpy

from pycbc.filter.nufft_correlation import (PhaseNUFFTCorrelation,
                                            phase_grid_search)


def sky_block_centres(longitudes, latitudes, radius):
    """Centres of blocks of half-width ``radius`` (``|lon - lon0| cos(lat0)
    <= radius``, ``|lat - lat0| <= radius``) covering the ecliptic
    ``longitudes`` and ``latitudes`` ranges, rad: latitude bands of width
    ``2 radius``, each cut in longitude at the spacing its edge nearer the
    equator needs."""
    lat_lo, lat_hi = latitudes
    lon_lo, lon_hi = longitudes
    nlat = max(1, int(numpy.ceil((lat_hi - lat_lo) / (2 * radius) - 1e-9)))
    centres = []
    for i in range(nlat):
        lat = lat_lo + (i + 0.5) * (lat_hi - lat_lo) / nlat
        edge = min(abs(lat) - (lat_hi - lat_lo) / (2 * nlat), numpy.pi / 2)
        width = 2 * radius / max(numpy.cos(max(edge, 0.0)), 1e-6)
        nlon = max(1, int(numpy.ceil((lon_hi - lon_lo) / width - 1e-9)))
        for j in range(nlon):
            centres.append((lon_lo + (j + 0.5) * (lon_hi - lon_lo) / nlon,
                            lat))
    return numpy.array(centres)


def block_sky_points(model, radius, step):
    """Sky offsets, in the model's shape units, of a square lattice of
    spacing ``step`` (shape units) over the model's sky block of half-width
    ``radius``."""
    lat0 = model.reference['eclipticlatitude']
    half = (radius / numpy.cos(lat0) / model.scales['eclipticlongitude'],
            radius / model.scales['eclipticlatitude'])
    axes = [numpy.arange(-numpy.floor(h / step), numpy.floor(h / step) + 1)
            * step for h in half]
    return numpy.array(numpy.meshgrid(*axes, indexing='ij')).reshape(2, -1).T


def search_sky_block(model, region, sky, top=20, sky_tolerance=1e-4,
                     band_margin=5 * 86400.0, **grid_options):
    """The loudest ``top`` candidates of one block: the model's intrinsic
    ``region`` (points ``(M, 4)``: mchirp, q, spin1z, spin2z) at the sky
    offsets ``sky``, every arrival time of the model, over the frequencies
    observed within ``band_margin`` of the data window (see
    :class:`pycbc.filter.nufft_correlation.PhaseNUFFTCorrelation`)."""
    fam = model.surrogate
    xp = fam.xp
    corr = PhaseNUFFTCorrelation(fam, model.data_spectrum, sky=(0.0, 0.0),
                                 sky_points=sky, sky_tolerance=sky_tolerance,
                                 band_margin=band_margin)

    def gram(T, offset):
        g = model.daily_gram(fam.b0, fam.t0, fam._response(offset))
        return model.daily_gram.at_delays(g, xp.asarray(T))
    out = phase_grid_search(corr, region,
                            (model.delays.min(), model.delays.max()), gram,
                            model.noise_variance, sky=sky, top=top,
                            **grid_options)
    names = ('eclipticlongitude', 'eclipticlatitude')
    for c in out['candidates']:
        c['longitude'] = model.reference[names[0]] \
            + c['sky'][0] * model.scales[names[0]]
        c['latitude'] = model.reference[names[1]] \
            + c['sky'][1] * model.scales[names[1]]
    return out


def fft_search_sky_block(model, thetas, sky, delay_range=None, top=20,
                         anchors=5, lag_chunk=1 << 18, refine=0,
                         subsamples=8, coefficients=None, sky_set=None,
                         lag_shift=None, oversample=1, gram_reuse=1,
                         basis_chunk=16, kernels=None, share_spectra=True,
                         decimate=1, threshold=None):
    """The loudest ``top`` candidates of one sky block by one FFT per
    intrinsic template and sky-basis vector.

    Each template's correlation with the data for each of the block's
    ``R`` basis vectors comes from one inverse FFT, every arrival time at
    once; the block's ``P`` sky points ``sky`` are combinations of those, so
    a template costs ``R`` FFTs whatever ``P``. The Gram matrix of each sky
    point comes from ``anchors x anchors`` Chebyshev nodes of the block on a
    daily grid, interpolated in the sky (2F within about 1.7% of its peak at
    5 x 5 for a 0.07 rad block).

    ``thetas`` are rows (mchirp, q, spin1z, spin2z); ``delay_range`` the
    arrival-time offsets searched, s, by default the model's. The loudest
    ``refine`` candidates are refined by :func:`refine_candidate`. With
    ``coefficients`` (rows of the 16 basis-coefficient changes, as
    :func:`phase_lattice` gives) the templates are those points of the
    coefficient space and ``thetas`` only label them. ``sky`` may be a list
    of sky sets, each template searching the set ``sky_set`` gives it (as
    :func:`joint_lattice` lays them out), with its lags shifted by
    ``lag_shift`` lag steps. ``oversample`` pads the transform so the lag
    step is the grid's over that factor."""
    from numpy.polynomial import chebyshev
    fam, dg = model.surrogate, model.daily_gram
    xp, blk = fam.xp, fam.sky_block
    if blk is None:
        raise ValueError("the model has no sky block")
    var = model.noise_variance
    lo, hi = (float(model.delays.min()), float(model.delays.max())) \
        if delay_range is None else delay_range
    oversample, decimate = int(oversample), int(decimate)
    if decimate > 1 and (oversample > 1 or fam.size % decimate):
        raise ValueError("decimate needs oversample 1 and to divide the "
                         "grid")
    dt = fam.grid.delta_t * decimate / oversample
    size = fam.size * oversample // decimate

    def fold(a):
        # every decimate-th lag: the spectrum's folds summed, a transform
        # decimate times shorter
        if decimate == 1:
            return a
        return a.reshape(a.shape[:-1] + (decimate, size)).sum(axis=-2)
    lags = numpy.arange(numpy.ceil(lo / dt), numpy.floor(hi / dt) + 1)
    index = xp.asarray(lags.astype(numpy.int64) % size)
    delays = lags * dt
    day = dg.day
    days = numpy.arange(numpy.floor(lo / day), numpy.ceil(hi / day) + 1) * day
    lag_day = xp.asarray(numpy.clip(((delays - days[0]) / day).round(), 0,
                                    len(days) - 1).astype(numpy.int64))
    # a template's correlation conjugates its response: each sky point weighs
    # the basis vectors' correlations by its coordinates' conjugates
    if isinstance(sky, (list, tuple)) and numpy.ndim(sky[0]) == 2:
        sets = [numpy.asarray(v, dtype=float) for v in sky]
    else:
        sets = [numpy.atleast_2d(numpy.asarray(sky, dtype=float))]
    U = xp.asarray(blk._basis)
    nodes = numpy.cos(numpy.pi * (numpy.arange(anchors) + 0.5) / anchors)
    unit = numpy.array([[a, b] for a in nodes for b in nodes])
    degree = [anchors - 1, anchors - 1]
    inverse = numpy.linalg.inv(chebyshev.chebvander2d(*unit.T, degree))
    Cs = [xp.asarray(numpy.conj(blk._coords(blk._unit(p)))) for p in sets]
    Ws = [xp.asarray(chebyshev.chebvander2d(*blk._unit(p).T, degree)
                     @ inverse) for p in sets]
    shifts = [0.0] * len(sets) if lag_shift is None else list(lag_shift)
    # a shift of the lags is a phase ramp on the spectrum
    ramps = [None if v == 0 else xp.exp(2j * numpy.pi * v / size
                                        * xp.arange(fam.size))
             for v in shifts]
    anchor_rho = fam.responses(blk._offsets(unit))
    fast = xp is not numpy and fam.dtype == numpy.complex64 \
        if kernels is None else bool(kernels)
    if fast:
        from pycbc.filter.sky_block_kernels import (correlation_spectra,
                                                    multiply_fold, twof_max)
        Ub = xp.asarray(blk._basis, dtype=xp.complex64)
        lag_day = lag_day.astype(xp.int32)
        shared = None
        if coefficients is not None and share_spectra:
            shared = xp.empty((blk.rank, 2, fam.size), dtype=xp.complex64)
            for r0 in range(0, blk.rank, basis_chunk):
                nr = min(basis_chunk, blk.rank - r0)
                correlation_spectra(fam, model.data_spectrum, fam.b0, fam.t0,
                                    Ub, r0, nr, out=shared[r0:r0 + nr])
    found, start, survivors = [], time.time(), []
    for n, theta in enumerate(numpy.atleast_2d(thetas)):
        ks = 0 if sky_set is None else int(sky_set[n])
        sky, C, W = sets[ks], Cs[ks], Ws[ks]
        if coefficients is not None:
            # a template of the coefficient space, labelled by ``theta``
            dC = numpy.asarray(coefficients[n], dtype=float)
            b, t = fam.source_from_coefficients(dC[:8], dC[8:])
        else:
            point = dict(model.static_params)
            point.update(dict(zip(['mchirp', 'q', 'spin1z', 'spin2z'],
                                  theta)))
            point.update({k: model.reference[k] for k in
                          ('eclipticlongitude', 'eclipticlatitude')})
            xi = model._shape_coordinates(point)
            b, t = fam._source(xi[:fam.nintrinsic])
        # the Gram at the anchors, a day at a time, interpolated to the sky
        if n % max(1, int(gram_reuse)) == 0:
            g = [[], [], []]
            for r in anchor_rho:
                q = dg.prepare(b, t, r)
                for c, v in enumerate(dg.at_prepared(q, days)):
                    g[c].append(xp.asarray(v))
            g = [xp.stack(v) for v in g]
        G = [W @ v / var for v in g]                                # (P, days)
        det = G[0] * G[1] - xp.abs(G[2]) ** 2
        # z(Delta) = sum_m a_m exp(2 pi i m Delta / T): the inverse FFT
        # times its length
        ramp = ramps[ks]
        if fast:
            Z = xp.empty((blk.rank, 2, len(lags)), dtype=xp.complex64)
            if shared is not None:
                factor = xp.conj(fam.coefficient_factor(dC[:8], dC[8:]))
                if ramp is not None:
                    factor = factor * ramp
            for r0 in range(0, blk.rank, basis_chunk):
                nr = min(basis_chunk, blk.rank - r0)
                if shared is not None:
                    # the template's spectra are the reference's times its
                    # factor, to second order in the harmonics' shift
                    a = multiply_fold(shared[r0:r0 + nr], factor, decimate)
                else:
                    a = correlation_spectra(fam, model.data_spectrum, b, t,
                                            Ub, r0, nr)
                    if ramp is not None:
                        a *= ramp
                    a = fold(a)
                Z[r0:r0 + nr] = xp.fft.ifft(a, n=size, axis=-1)[
                    :, :, index] * size
                del a
            tb, ta = twof_max(Z, C, G[0], G[1], G[2], det, lag_day, 1 / var)
            if threshold is not None:
                ps, tiles = xp.nonzero(tb > threshold)
                if len(ps):
                    js = _host(ta[ps, tiles])
                    survivors.append(numpy.column_stack([
                        numpy.full(len(js), n), _host(ps), js,
                        _host(tb[ps, tiles]), numpy.full(len(js), ks)]))
            k = int(xp.argmax(tb))
            p, tile = divmod(k, tb.shape[1])
            best, where = float(tb[p, tile]), (p, int(ta[p, tile]))
            del Z
        else:
            Z = xp.stack([xp.fft.ifft(fold(fam.correlation_spectrum(
                model.data_spectrum, b, t, U[r])
                * (1 if ramp is None else ramp)), n=size, axis=-1)[
                    :, index] * size for r in range(blk.rank)])     # (R, 2, L)
            best, where = -numpy.inf, None
        for c0 in (range(0, len(lags), lag_chunk) if not fast else []):
            cut = slice(c0, c0 + lag_chunk)
            z = (C @ Z[:, :, cut].reshape(blk.rank, -1)).reshape(
                len(sky), 2, -1) / var
            d = lag_day[cut]
            g00, g11, g01, dd = G[0][:, d], G[1][:, d], G[2][:, d], det[:, d]
            twof = (g11 * xp.abs(z[:, 0]) ** 2 + g00 * xp.abs(z[:, 1]) ** 2
                    - 2 * xp.real(xp.conj(z[:, 0]) * g01 * z[:, 1])) / dd
            k = int(xp.argmax(twof))
            v = float(twof.reshape(-1)[k])
            if v > best:
                best, where = v, (k // twof.shape[1], c0 + k % twof.shape[1])
        p, j = where
        found.append(dict(twof=best, theta=list(map(float, theta)),
                          coefficients=(None if coefficients is None else
                                        [float(v) for v in coefficients[n]]),
                          sky=[float(x) for x in sky[p]],
                          time=float(delays[j] + shifts[ks] * dt),
                          longitude=float(
                              model.reference['eclipticlongitude']
                              + sky[p][0] * model.scales['eclipticlongitude']),
                          latitude=float(model.reference['eclipticlatitude']
                                         + sky[p][1]
                                         * model.scales['eclipticlatitude'])))
        if (n + 1) % 50 == 0:
            logging.info("FFT block search: %d/%d templates, %.0f s",
                         n + 1, len(thetas), time.time() - start)
    found.sort(key=lambda c: -c['twof'])
    for c in found[:refine]:
        refine_candidate(model, c, anchors=anchors, subsamples=subsamples)
    if refine:
        found.sort(key=lambda c: -c['twof'])
    points = sum(len(sets[0 if sky_set is None else int(sky_set[n])])
                 for n in range(len(found)))
    out = dict(candidates=found[:top], templates=points * len(lags))
    if threshold is not None:
        # rows: template, sky point, lag index, 2F, sky set; the lag's
        # time is delays[lag] plus the set's shift
        out['survivors'] = numpy.vstack(survivors) if survivors else \
            numpy.zeros((0, 5))
        out['delays'] = delays
    return out


def refine_candidate(model, candidate, anchors=5, subsamples=8, chunk=1 << 18):
    """Refine a candidate of :func:`fft_search_sky_block` in place: its
    arrival time between the grid's lags and its sky offset continuous in
    the block, for its intrinsic template. The correlations of each sky
    basis vector at ``subsamples`` fractions of a lag either side of the
    candidate's lag come from their spectra directly, ``z(Delta) = sum_m a_m
    exp(2 pi i m Delta / T)``; the sky point's correlation and Gram are
    smooth in its offset (Chebyshev coordinates, Chebyshev anchors), so the
    sky offset is optimised for each arrival time and the largest 2F kept."""
    from numpy.polynomial import chebyshev
    from scipy.optimize import minimize
    fam, dg = model.surrogate, model.daily_gram
    xp, blk = fam.xp, fam.sky_block
    var, dt, size = model.noise_variance, fam.grid.delta_t, fam.size
    point = dict(model.static_params)
    point.update(dict(zip(['mchirp', 'q', 'spin1z', 'spin2z'],
                          candidate['theta'])))
    point.update({k: model.reference[k] for k in
                  ('eclipticlongitude', 'eclipticlatitude')})
    if candidate.get('coefficients') is not None:
        dC = numpy.asarray(candidate['coefficients'])
        b, t = fam.source_from_coefficients(dC[:8], dC[8:])
    else:
        xi = model._shape_coordinates(point)
        b, t = fam._source(xi[:fam.nintrinsic])
    delays = candidate['time'] + dt * numpy.linspace(-1, 1, 2 * subsamples + 1)
    phase = xp.asarray(numpy.exp(2j * numpy.pi * numpy.outer(
        numpy.arange(size), delays / (size * dt))).astype(fam.dtype))
    U = xp.asarray(blk._basis)
    Z = []
    for r in range(blk.rank):
        a = fam.correlation_spectrum(model.data_spectrum, b, t, U[r])
        z = 0
        for c0 in range(0, size, chunk):
            z = z + a[:, c0:c0 + chunk] @ phase[c0:c0 + chunk]
        Z.append(z)
    Z = xp.stack(Z) / var                                    # (R, 2, delays)
    nodes = numpy.cos(numpy.pi * (numpy.arange(anchors) + 0.5) / anchors)
    unit = numpy.array([[u, v] for u in nodes for v in nodes])
    degree = [anchors - 1, anchors - 1]
    inverse = numpy.linalg.inv(chebyshev.chebvander2d(*unit.T, degree))
    g = [[], [], []]
    for rho in fam.responses(blk._offsets(unit)):
        q = dg.prepare(b, t, rho)
        for c, v in enumerate(dg.at_prepared(q, delays)):
            g[c].append(xp.asarray(v))
    g = [xp.stack(v) / var for v in g]                        # (A^2, delays)

    def twof(offset):
        u = blk._unit(numpy.atleast_2d(offset))
        if numpy.any(numpy.abs(u) > 1):
            return -numpy.inf, 0
        c = xp.asarray(numpy.conj(blk._coords(u)))[0]         # (R,)
        w = xp.asarray(chebyshev.chebvander2d(*u.T, degree) @ inverse)[0]
        z = xp.tensordot(c, Z, axes=1)                         # (2, delays)
        g00, g11, g01 = (w @ v for v in g)
        f = (g11 * xp.abs(z[0]) ** 2 + g00 * xp.abs(z[1]) ** 2
             - 2 * xp.real(xp.conj(z[0]) * g01 * z[1])) / (
                 g00 * g11 - xp.abs(g01) ** 2)
        k = int(xp.argmax(f))
        return float(f[k]), k
    res = minimize(lambda o: -twof(o)[0], numpy.asarray(candidate['sky']),
                   method='Nelder-Mead',
                   options=dict(xatol=1e-4, fatol=1e-3, maxiter=200))
    value, k = twof(res.x)
    if value > candidate['twof']:
        candidate.update(
            twof=value, time=float(delays[k]), sky=[float(v) for v in res.x],
            longitude=float(model.reference['eclipticlongitude']
                            + res.x[0] * model.scales['eclipticlongitude']),
            latitude=float(model.reference['eclipticlatitude']
                           + res.x[1] * model.scales['eclipticlatitude']),
            refined=True)
    return candidate


def aligned_lattice(s, metric, mean_mismatch, reach=1.0):
    """A cubic lattice covering the points ``s`` (M, n) of a space with
    constant ``metric``, aligned with their principal axes in the metric,
    with mean mismatch ``mean_mismatch``; nodes farther than ``reach``
    covering radii from every point are dropped. Returns the nodes, the
    index of the point nearest each and the spacings."""
    from scipy.spatial import cKDTree
    ndir = s.shape[1]
    # metric coordinates u = L^T s, then the region's principal axes in them
    L = numpy.linalg.cholesky(metric)
    u = (s - s.mean(axis=0)) @ L
    _, _, axes_t = numpy.linalg.svd(u, full_matrices=False)
    u = u @ axes_t.T
    lo, hi = u.min(axis=0), u.max(axis=0)
    extent, centre = hi - lo, 0.5 * (lo + hi)
    spread = numpy.mean((u - centre) ** 2, axis=0)
    # an axis narrower than the spacing gets one layer, and its spread is
    # mismatch the filled axes' spacing need not supply
    h = numpy.sqrt(12 * mean_mismatch / ndir)
    for _ in range(ndir):
        thin = extent <= h
        budget = max(mean_mismatch - spread[thin].sum(), 0.25 * mean_mismatch)
        h = numpy.sqrt(12 * budget / max(1, int(numpy.sum(~thin))))
    axes = []
    for d in range(ndir):
        n = max(1, int(numpy.ceil(extent[d] / h)))
        axes.append(centre[d] + (numpy.arange(n) - (n - 1) / 2.0) * h)
    nodes = numpy.array(numpy.meshgrid(*axes, indexing='ij')).reshape(
        ndir, -1).T
    tree = cKDTree(u)
    half = numpy.array([0.5 * h if len(a) > 1 else 0.5 * e
                        for a, e in zip(axes, extent)])
    covering = numpy.sqrt(numpy.sum(half ** 2))
    dist, near = tree.query(nodes)
    keep = dist <= reach * covering
    nodes, near = nodes[keep], near[keep]
    vectors = numpy.linalg.solve(L.T, axes_t.T)            # u -> s
    return s.mean(axis=0) + nodes @ vectors.T, near, numpy.full(ndir, h)


# normalised second moments (Conway and Sloane, Sphere Packings, Lattices
# and Groups, table 2.3) of the cubic lattice and of Z^n with its cells'
# centres added: bcc in three dimensions, D4* (similar to D4) in four, D5*
_SECOND_MOMENT = {'cubic': 1 / 12.0, 3: 0.0785433, 4: 0.0766032,
                  5: 0.0756254}


def joint_lattice(s, metric, sky_metric, sky_half, mean_mismatch=None,
                  time_step=None, reach=1.0, lattice='centred'):
    """One lattice over a region's phase coordinates, a sky block and,
    optionally, the arrival time, so the mismatch budget is shared between
    them and the best lattice quantizer of the joint dimension covers all.

    ``s`` (M, n) are the region's phase coordinates and ``metric`` the loss
    of 2F per unit of them squared; ``sky_metric`` (2, 2) the loss per
    shape unit squared of the sky offsets and ``sky_half`` the block's
    half-widths, shape units. Along the region's principal axes (as
    :func:`aligned_lattice`) an axis narrower than a layer gets one and its
    spread is taken from the budget; the axes it fills, the two sky axes
    and, with ``time_step`` (the search's lag step, whitened: the root of
    the loss per lag step squared), the arrival time carry the lattice.

    ``lattice`` is ``'cubic'`` or ``'centred'``: Z^n with its cells'
    centres, every index of a point of one parity, which is bcc in three
    dimensions and D4* in four. A centred lattice gives each intrinsic node
    one of two sky grids, the second shifted half a cell, and the lags of
    the second shifted half a lag step: the search still shares its FFTs
    over the sky, and the shift is a phase ramp.

    The scale is set by ``mean_mismatch`` or, with the time axis, by the lag
    step. Returns the intrinsic nodes (phase coordinates), the index of the
    region point nearest each, each node's set (0 or 1), the sets' sky
    offsets (shape units) and lag shifts (lag steps), the index step, the
    lattice and its mean mismatch."""
    from scipy.spatial import cKDTree
    ndir = s.shape[1]
    L = numpy.linalg.cholesky(metric)
    u = (s - s.mean(axis=0)) @ L
    _, _, axes_t = numpy.linalg.svd(u, full_matrices=False)
    u = u @ axes_t.T
    lo, hi = u.min(axis=0), u.max(axis=0)
    extent, centre = hi - lo, 0.5 * (lo + hi)
    spread = numpy.mean((u - centre) ** 2, axis=0)
    timed = time_step is not None
    filled = numpy.ones(ndir, dtype=bool)
    for _ in range(ndir + 1):
        n = int(filled.sum()) + 2 + int(timed)
        kind = n if (lattice == 'centred' and n in _SECOND_MOMENT) \
            else 'cubic'
        # index step c: a point per c^n (cubic) or per (2 c)^n / 2
        if timed:
            c = time_step if kind == 'cubic' else 0.5 * time_step
        else:
            budget = max(mean_mismatch - spread[~filled].sum(),
                         0.25 * mean_mismatch)
            volume = (budget / (n * _SECOND_MOMENT[kind])) ** (n / 2.0)
            c = volume ** (1.0 / n) if kind == 'cubic' else \
                0.5 * (2 * volume) ** (1.0 / n)
        layer = c if kind == 'cubic' else 2 * c
        new = extent > layer
        if (new == filled).all():
            break
        filled = new
    volume = c ** n if kind == 'cubic' else (2 * c) ** n / 2.0
    mean = n * _SECOND_MOMENT[kind] * volume ** (2.0 / n) \
        + spread[~filled].sum()
    # the deep holes: a cell's corner (cubic), (1/2, 1/4, 0) of a side 2 c
    # cell (bcc), (1/2, 1/2, 0, ...) (four and five dimensions)
    covering = 0.5 * c * numpy.sqrt(n) if kind == 'cubic' else \
        (0.5 * c * numpy.sqrt(5) if n == 3 else c * numpy.sqrt(2))
    axes = []
    for d in range(ndir):
        if filled[d]:
            k = numpy.arange(numpy.floor((lo[d] - covering - centre[d]) / c),
                             numpy.ceil((hi[d] + covering - centre[d]) / c)
                             + 1)
            axes.append(centre[d] + k * c)
        else:
            axes.append(numpy.array([centre[d]]))
    nodes = numpy.array(numpy.meshgrid(*axes, indexing='ij')).reshape(
        ndir, -1).T
    index = numpy.round((nodes - centre) / c).astype(int)[:, filled]
    parity = numpy.mod(index[:, 0], 2) if index.shape[1] else \
        numpy.zeros(len(nodes), dtype=int)
    if kind == 'cubic':
        parity = numpy.zeros(len(nodes), dtype=int)
        same = numpy.ones(len(nodes), dtype=bool)
    else:
        same = numpy.all(numpy.mod(index, 2) == parity[:, None], axis=1)
    dist, near = cKDTree(u).query(nodes)
    keep = same & (dist <= reach * covering)
    nodes, near, parity = nodes[keep], near[keep], parity[keep]
    if kind != 'cubic' and not filled.any():
        # no intrinsic axis: the sky and time carry both parities
        nodes, near = numpy.repeat(nodes, 2, axis=0), numpy.repeat(near, 2)
        parity = numpy.tile([0, 1], len(parity))
    vectors = numpy.linalg.solve(L.T, axes_t.T)
    # the sky, whitened by its metric: the grid of index step c inside the
    # block widened by the covering radius, each set the points of its
    # parity
    Ls = numpy.linalg.cholesky(sky_metric)
    half = numpy.asarray(sky_half, dtype=float)
    widen = covering * numpy.sqrt(numpy.diag(numpy.linalg.inv(sky_metric)))
    corners = numpy.array([[a, b] for a in (-1, 1) for b in (-1, 1)]) \
        * (half + widen)
    w = corners @ Ls
    j = [numpy.arange(numpy.floor(w[:, d].min() / c),
                      numpy.ceil(w[:, d].max() / c) + 1) for d in range(2)]
    grid = numpy.array(numpy.meshgrid(*j, indexing='ij')).reshape(
        2, -1).T.astype(int)
    offsets = (grid * c) @ numpy.linalg.inv(Ls)
    inside = numpy.all(numpy.abs(offsets) <= half + widen, axis=1)
    sets, shifts = [], []
    for p in (0, 1):
        mine = inside if kind == 'cubic' else inside & numpy.all(
            numpy.mod(grid, 2) == p, axis=1)
        sets.append(offsets[mine])
        shifts.append(0.5 * p if (timed and kind != 'cubic') else 0.0)
    name = 'cubic' if kind == 'cubic' else \
        {3: 'bcc', 4: 'D4*', 5: 'D5*'}[n]
    return (s.mean(axis=0) + nodes @ vectors.T, near, parity, sets, shifts,
            c, name, mean)


def phase_frame(model, region, ndir=3, band_margin=5 * 86400.0):
    """The phase frame of ``region`` (points ``(M, 4)``: mchirp, q,
    spin1z, spin2z) over the observed band: the points' phase coordinates,
    the metric in them from the template power with the arrival time
    projected out, the frame and its correlator."""
    fam = model.surrogate
    corr = PhaseNUFFTCorrelation(fam, model.data_spectrum, sky=(0.0, 0.0),
                                 band_margin=band_margin)
    region = numpy.atleast_2d(numpy.asarray(region, dtype=float))
    frame = corr.frame(region, ndir)
    ndir = frame['directions'].shape[1]
    s, _, _ = corr.project(region, frame)
    parts = corr.nodes(frame)
    rows = [_host(corr.node_row(parts, d, 0)) for d in range(ndir)] \
        + [_host(corr.node_row(parts, -1, 0))]
    b0 = numpy.abs(_host(fam.b0)) ** 2
    power = b0 if corr._sel is None else b0[corr._sel]
    power = power / power.sum()
    x = numpy.array(rows)
    x = x - (x @ power)[:, None]
    g = (x * power) @ x.T
    gs = g[:ndir, :ndir] - numpy.outer(g[:ndir, -1], g[-1, :ndir]) / g[-1, -1]
    return s, gs, frame, corr


def phase_lattice(model, region, mean_mismatch, ndir=3,
                  band_margin=5 * 86400.0, reach=1.0, full=False):
    """A template bank of the PN family in its phase coordinates, where the
    phase is linear in the basis coefficients and the mismatch therefore an
    exact quadratic form: the frame of ``region`` (points ``(M, 4)``:
    mchirp, q, spin1z, spin2z) over the observed band, its metric from the
    template power with the arrival time projected out (the search
    maximises over it), and an :func:`aligned_lattice` of mean mismatch
    ``mean_mismatch``.

    The region's image is far thinner than it is long (over two years the
    mass ratio and spins move the phase by a fraction of a radian), so the
    lattice follows its principal axes: an axis it spans less than a
    spacing gets one layer. Only nodes within ``reach`` covering radii of a
    region point are kept; every point is nearer one of them than any
    dropped node.

    Returns the nodes' 16 basis-coefficient changes, the region point
    nearest each (its label), and the spacings; with ``full``, also the
    nodes' phase coordinates, the frame and its correlator."""
    region = numpy.atleast_2d(numpy.asarray(region, dtype=float))
    s, gs, frame, corr = phase_frame(model, region, ndir, band_margin)
    s_nodes, near, h = aligned_lattice(s, gs, mean_mismatch, reach)
    dC = s_nodes @ frame['directions'].T
    # the reference's own coefficients carry over unchanged at s = 0: the
    # frame's s are changes from the reference
    if full:
        return dC, region[near], h, s_nodes, frame, corr
    return dC, region[near], h


def _host(a):
    return a.get() if hasattr(a, 'get') else numpy.asarray(a)


def merge_candidates(candidates, sky_distance, time_distance):
    """Each source once: from the loudest down, a candidate within
    ``sky_distance`` (rad, on the sphere) and ``time_distance`` (s) of one
    kept is the same source seen from another block or sky point."""
    kept = []
    for c in sorted(candidates, key=lambda c: -c['twof']):
        n = _unit(c['longitude'], c['latitude'])
        same = False
        for k in kept:
            angle = numpy.arccos(numpy.clip(n @ _unit(k['longitude'],
                                                      k['latitude']), -1, 1))
            if angle <= sky_distance and \
                    abs(c['time'] - k['time']) <= time_distance:
                same = True
                break
        if not same:
            kept.append(c)
    return kept


def _unit(lon, lat):
    return numpy.array([numpy.cos(lat) * numpy.cos(lon),
                        numpy.cos(lat) * numpy.sin(lon), numpy.sin(lat)])


def sky_block_search(model, centres, radius, region, sky_step, top=20,
                     sky_distance=None, time_distance=600.0,
                     sky_block_points=600, route='fft', **options):
    """Search the blocks of ``centres`` (from :func:`sky_block_centres`),
    each with ``model`` recentred there, and merge the candidates.

    ``route`` is ``'fft'`` (:func:`fft_search_sky_block`: ``region`` lists
    the intrinsic templates; every arrival time at once, the route for
    full-length data) or ``'grid'`` (:func:`search_sky_block`: ``region``
    spans the intrinsic region, a NUFFT grid over it; for a short window of
    arrival times); ``options`` go to the route.

    Returns the merged candidates, loudest first, each with its block, and
    the blocks' statistics (seconds, templates, sky points)."""
    found, blocks = [], []
    sky_distance = 2 * radius if sky_distance is None else sky_distance
    start = time.time()
    for i, (lon, lat) in enumerate(centres):
        tick = time.time()
        block = model.recentred(lon, lat, radius,
                                sky_block_points=sky_block_points)
        sky = block_sky_points(block, radius, sky_step)
        if route == 'fft':
            out = fft_search_sky_block(block, region, sky, top=top, **options)
        elif route == 'grid':
            out = search_sky_block(block, region, sky, top=top, **options)
        else:
            raise ValueError("route must be 'fft' or 'grid'")
        for c in out['candidates']:
            c['block'] = i
        found.extend(out['candidates'])
        seconds = time.time() - tick
        blocks.append(dict(centre=(float(lon), float(lat)), seconds=seconds,
                           templates=out['templates'], sky_points=len(sky),
                           sky_error=float(block.surrogate.sky_block.error)))
        done = time.time() - start
        logging.info("sky block %d/%d (%.4f, %.4f): %d sky points, loudest "
                     "2F %.1f, %.0f s; about %.0f s left", i + 1,
                     len(centres), lon, lat, len(sky),
                     max([c['twof'] for c in out['candidates']] or [0]),
                     seconds, done / (i + 1) * (len(centres) - i - 1))
        del block
    return merge_candidates(found, sky_distance, time_distance), blocks
