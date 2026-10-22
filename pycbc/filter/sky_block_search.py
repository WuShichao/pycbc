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
                     **grid_options):
    """The loudest ``top`` candidates of one block: the model's intrinsic
    ``region`` (points ``(M, 4)``: mchirp, q, spin1z, spin2z) at the sky
    offsets ``sky``, every arrival time of the model."""
    fam = model.surrogate
    xp = fam.xp
    corr = PhaseNUFFTCorrelation(fam, model.data_spectrum, sky=(0.0, 0.0),
                                 sky_points=sky, sky_tolerance=sky_tolerance)

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
                     sky_block_points=600, **grid_options):
    """Search the blocks of ``centres`` (from :func:`sky_block_centres`),
    each with ``model`` recentred there, and merge the candidates.

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
        out = search_sky_block(block, region, sky, top=top, **grid_options)
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
