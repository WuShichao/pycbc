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
                         anchors=5, lag_chunk=1 << 18):
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
    arrival-time offsets searched, s, by default the model's."""
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
        point = dict(model.static_params)
        point.update(dict(zip(['mchirp', 'q', 'spin1z', 'spin2z'], theta)))
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
    return dict(candidates=found[:top],
                templates=len(found) * len(sky) * len(lags))


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
