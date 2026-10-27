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
                         subsamples=8, coefficients=None):
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
    coefficient space and ``thetas`` only label them."""
    from numpy.polynomial import chebyshev
    fam, dg = model.surrogate, model.daily_gram
    xp, blk = fam.xp, fam.sky_block
    if blk is None:
        raise ValueError("the model has no sky block")
    var = model.noise_variance
    lo, hi = (float(model.delays.min()), float(model.delays.max())) \
        if delay_range is None else delay_range
    dt, size = fam.grid.delta_t, fam.size
    lags = numpy.arange(numpy.ceil(lo / dt), numpy.floor(hi / dt) + 1)
    index = xp.asarray(lags.astype(numpy.int64) % size)
    delays = lags * dt
    day = dg.day
    days = numpy.arange(numpy.floor(lo / day), numpy.ceil(hi / day) + 1) * day
    lag_day = xp.asarray(numpy.clip(((delays - days[0]) / day).round(), 0,
                                    len(days) - 1).astype(numpy.int64))
    # a template's correlation conjugates its response: each sky point weighs
    # the basis vectors' correlations by its coordinates' conjugates
    C = xp.asarray(numpy.conj(blk._coords(blk._unit(sky))))         # (P, R)
    U = xp.asarray(blk._basis)
    nodes = numpy.cos(numpy.pi * (numpy.arange(anchors) + 0.5) / anchors)
    unit = numpy.array([[a, b] for a in nodes for b in nodes])
    degree = [anchors - 1, anchors - 1]
    W = xp.asarray(chebyshev.chebvander2d(*blk._unit(sky).T, degree)
                   @ numpy.linalg.inv(chebyshev.chebvander2d(*unit.T,
                                                             degree)))
    anchor_rho = fam.responses(blk._offsets(unit))
    found, start = [], time.time()
    for n, theta in enumerate(numpy.atleast_2d(thetas)):
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
        g = [[], [], []]
        for r in anchor_rho:
            q = dg.prepare(b, t, r)
            for c, v in enumerate(dg.at_prepared(q, days)):
                g[c].append(xp.asarray(v))
        G = [W @ xp.stack(v) / var for v in g]                     # (P, days)
        det = G[0] * G[1] - xp.abs(G[2]) ** 2
        # z(Delta) = sum_m a_m exp(2 pi i m Delta / T): the inverse FFT
        # times its length
        Z = xp.stack([xp.fft.ifft(fam.correlation_spectrum(
            model.data_spectrum, b, t, U[r]), axis=-1)[:, index] * size
            for r in range(blk.rank)])                              # (R, 2, L)
        best, where = -numpy.inf, None
        for c0 in range(0, len(lags), lag_chunk):
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
                          time=float(delays[j]), longitude=float(
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
    return dict(candidates=found[:top],
                templates=len(found) * len(sky) * len(lags))


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


def phase_lattice(model, region, mean_mismatch, ndir=3,
                  band_margin=5 * 86400.0, reach=2.0, full=False):
    """A template bank of the PN family in its phase coordinates, where the
    phase is linear in the basis coefficients and the mismatch therefore an
    exact quadratic form: the frame of ``region`` (points ``(M, 4)``:
    mchirp, q, spin1z, spin2z) over the observed band, its metric from the
    template power with the arrival time projected out (the search
    maximises over it), and a lattice along the metric's eigenvectors with
    the spacing that gives the cubic lattice a mean mismatch of
    ``mean_mismatch``. Only nodes within ``reach`` spacings, in the metric,
    of a region point are kept: the physical part of the frame.

    Returns the nodes' 16 basis-coefficient changes, the region point
    nearest each (its label), and the spacings; with ``full``, also the
    nodes' phase coordinates, the frame and its correlator."""
    from scipy.spatial import cKDTree
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
    values, vectors = numpy.linalg.eigh(gs)
    values = numpy.maximum(values, 1e-300)
    h = numpy.sqrt(12 * mean_mismatch / (ndir * values))
    u = (s - s.mean(axis=0)) @ vectors                       # eigen coordinates
    axes = []
    for d in range(ndir):
        lo, hi = u[:, d].min(), u[:, d].max()
        n = int(numpy.ceil((hi - lo) / h[d]))
        axes.append(0.5 * (lo + hi) + (numpy.arange(n + 1) - n / 2) * h[d]
                    if n > 0 else numpy.array([0.5 * (lo + hi)]))
    nodes = numpy.array(numpy.meshgrid(*axes, indexing='ij')).reshape(
        ndir, -1).T
    # in metric units, the region's points and the nodes; nodes near none dropped
    scale = numpy.sqrt(values)
    tree = cKDTree(u * scale)
    dist, near = tree.query(nodes * scale)
    keep = dist <= reach * numpy.sqrt(numpy.sum(values * h ** 2))
    nodes, near = nodes[keep], near[keep]
    s_nodes = s.mean(axis=0) + nodes @ vectors.T
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
