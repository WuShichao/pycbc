# Copyright (C) 2026 Shichao Wu
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
"""Shared FFT correlations for the response basis of one sky block.

The source and stationary time are evaluated once per intrinsic template.
The response basis needs R FFTs, independent of the number of sky points.
Combining the resulting fields is streamed; no P-by-N sky/time array is kept.
This module does not approximate a Gram matrix or choose a search threshold.
"""

import numpy


class SkyBlockFFT(object):
    """Correlate one intrinsic template over a fixed collection of sky offsets.

    Parameters
    ----------
    family : LinearExponentSurrogate
        A family carrying a trained ``sky_block``.
    data : array_like
        SSB data spectrum, in the same convention as ``family.correlate``.
    offsets : array_like, shape (P, 2)
        Sky offsets in the family's shape coordinates, inside the block.
    field_memory_limit : int
        Maximum bytes for the R-by-2-by-N correlation fields alone. FFT work
        space and the input arrays also need memory; callers cap the allocator.
    """

    def __init__(self, family, data, offsets, field_memory_limit=2 * 1024 ** 3):
        self.family, self.xp = family, family.xp
        block = getattr(family, 'sky_block', None)
        if block is None:
            raise ValueError('the family needs a sky response block')
        offsets = numpy.asarray(offsets, dtype=float)
        if offsets.ndim != 2 or offsets.shape[1] != 2 or not len(offsets):
            raise ValueError('sky offsets must have shape (P, 2)')
        unit = block._unit(offsets)
        # Converting an exact corner to offsets and back can exceed one by a
        # few ulps. This is a coordinate roundoff allowance, not a wider block.
        edge_tolerance = 32 * numpy.finfo(float).eps
        if (not numpy.isfinite(offsets).all() or not numpy.isfinite(unit).all()
                or not (numpy.abs(unit) <= 1 + edge_tolerance).all()):
            raise ValueError('sky offsets lie outside the response block')
        if not numpy.array_equal(family.orders, block.orders):
            raise ValueError('family and response block harmonics differ')
        if tuple(data.shape) != (block._basis.shape[-2], family.size):
            raise ValueError('data and template network shapes differ')
        required = block.rank * 2 * family.size * numpy.dtype(family.dtype).itemsize
        if required > field_memory_limit:
            raise MemoryError('response correlation fields need {} bytes'.format(required))
        self.data = self.xp.asarray(data, dtype=family.dtype)
        self.offsets = offsets.copy()
        # Responses are C @ U, but the template argument of <h,d> is antilinear.
        coords = block._coords(unit)
        self.weights = self.xp.asarray(numpy.conj(coords), dtype=family.dtype)
        self.basis = self.xp.asarray(block._basis, dtype=family.dtype)
        self.rank, self.required_field_bytes = block.rank, required

    def basis_correlations(self, intrinsic):
        """Return the shared fields and the exact source pieces used for them.

        Gram evaluation must use these same b,t, not the reference template's.
        ``family.correlate`` supplies the inverse FFT normalization.
        """
        fam, xp = self.family, self.xp
        b, t = fam._source(intrinsic)
        return self.correlate_basis(b, t), b, t

    def correlate_basis(self, b, t):
        """Shared FFTs of already evaluated source pieces b,t."""
        fam, xp = self.family, self.xp
        fields = xp.empty((self.rank, 2, fam.size), dtype=fam.dtype)
        for r in range(self.rank):
            fields[r] = fam.correlate(self.data, b, t, self.basis[r])
        return fields

    def chunks(self, fields, lag_indices=None, sky_chunk=4, lag_chunk=4096):
        """Yield (sky slice, lag indices, correlations of shape (Pchunk,2,L)).

        Indices refer to the full periodic FFT. The caller supplies the search
        domain's own delays and mask, including negative-delay wraparound.
        """
        xp, fam = self.xp, self.family
        if fields.shape != (self.rank, 2, fam.size):
            raise ValueError('unexpected shared field shape')
        if sky_chunk < 1 or lag_chunk < 1:
            raise ValueError('chunk sizes must be positive')
        if lag_indices is None:
            lag_indices = numpy.arange(fam.size)
        lag_indices = numpy.asarray(lag_indices)
        if lag_indices.ndim != 1 or not numpy.issubdtype(lag_indices.dtype, numpy.integer):
            raise ValueError('lag indices must be a one-dimensional integer array')
        if numpy.any(lag_indices < 0) or numpy.any(lag_indices >= fam.size):
            raise ValueError('lag index outside the FFT')
        for start in range(0, len(lag_indices), lag_chunk):
            selected = lag_indices[start:start + lag_chunk]
            flat = fields[:, :, xp.asarray(selected)].reshape(self.rank, -1)
            for first in range(0, len(self.offsets), sky_chunk):
                last = min(first + sky_chunk, len(self.offsets))
                value = self.weights[first:last] @ flat
                yield slice(first, last), selected, value.reshape(last - first, 2, -1)


def network_twof(z, gram, noise_variance=1.0, xp=numpy):
    """Evaluate z^H G^-1 z, returning explicit rank/validity labels.

    z has shape (...,2), Gram (...,2,2); both are unscaled raw statistics.
    Status 0 is full rank, 1 exact rank one, 2 zero coverage, 3 unresolved.
    A search must retain/report unresolved entries, not treat them as negatives.
    No positive eigenvalue is truncated. Negative eigenvalues are unresolved.
    """
    if not numpy.isfinite(noise_variance) or noise_variance <= 0:
        raise ValueError('noise variance must be positive and finite')
    z = xp.asarray(z, dtype=xp.complex128)
    gram = xp.asarray(gram, dtype=xp.complex128)
    if gram.shape != z.shape[:-1] + (2, 2) or z.shape[-1] != 2:
        raise ValueError('correlation and Gram shapes differ')
    finite = xp.isfinite(z).all(axis=-1) & xp.isfinite(gram).all(axis=(-2, -1))
    hermitian = xp.max(xp.abs(gram - xp.swapaxes(gram.conj(), -2, -1)), axis=(-2, -1))
    scale = xp.max(xp.abs(gram), axis=(-2, -1))
    valid = finite & (hermitian <= 32 * numpy.finfo(float).eps * scale)
    safe = xp.where(valid[..., None, None], gram, xp.eye(2))
    eig, vec = xp.linalg.eigh(safe)
    projection = xp.einsum('...ij,...i->...j', vec.conj(), z)
    positive = eig > 0
    null_ok = xp.all(positive | (xp.abs(projection) == 0), axis=-1)
    valid &= xp.all(eig >= 0, axis=-1) & null_ok
    divisor = xp.sqrt(xp.where(positive, eig, 1))
    terms = xp.where(positive, xp.abs(projection) / divisor, 0)
    value = xp.sum(terms * terms, axis=-1) / noise_variance
    valid &= xp.isfinite(value)
    rank = xp.sum(positive, axis=-1)
    status = xp.where(valid, xp.where(rank == 2, 0, xp.where(rank == 1, 1, 2)), 3)
    return xp.where(valid, value, xp.nan), status


def search_sky_block_fft(engine, intrinsics, prepare_gram, lag_indices,
                         noise_variance=1.0, top=20, sky_chunk=4,
                         lag_chunk=4096):
    """Search a finite template/sky/delay grid without pruning any entries.

    ``prepare_gram(index, intrinsic, b, t)`` returns a callable taking a sky
    slice and FFT indices and returning raw Gram matrices of shape (P,L,2,2).
    The caller owns its approximation and physical-time convention. In
    particular, it must use this template's b,t and the data's normalization.

    ``lag_indices`` is the explicit time-domain mask; it must contain unique
    integer FFT indices. The output lists the largest ``top`` finite 2F
    values, with template, sky and lag indices. Rank counts cover EVERY grid
    entry, not just returned candidates. Any unresolved entry sets
    ``complete=False``: finite candidates are retained for diagnosis but
    cannot certify the global maximum. This is not a continuous-domain or
    false-alarm guarantee.
    """
    from time import perf_counter

    xp = engine.xp
    points = numpy.asarray(intrinsics, dtype=float)
    lags = numpy.asarray(lag_indices)
    if (points.ndim != 2 or not len(points) or
            not numpy.isfinite(points).all()):
        raise ValueError('intrinsics must be a finite, nonempty matrix')
    if (lags.ndim != 1 or not len(lags) or
            not numpy.issubdtype(lags.dtype, numpy.integer) or
            len(numpy.unique(lags)) != len(lags) or
            numpy.any(lags < 0) or numpy.any(lags >= engine.family.size)):
        raise ValueError('time domain needs unique, valid integer FFT indices')
    if (not all(isinstance(v, (int, numpy.integer)) and v > 0
                for v in (top, sky_chunk, lag_chunk))):
        raise ValueError('top and chunk sizes must be positive integers')
    lags = numpy.sort(lags)

    def host(value):
        return value.get() if hasattr(value, 'get') else numpy.asarray(value)

    def synchronize():
        if xp is not numpy:
            xp.cuda.get_current_stream().synchronize()

    counts = {str(k): 0 for k in range(4)}
    timings = dict(source_seconds=0., fft_seconds=0., gram_prepare_seconds=0.,
                   scan_seconds=0.)
    candidates = []
    began = perf_counter()
    for index, intrinsic in enumerate(points):
        synchronize()
        tick = perf_counter()
        b, t = engine.family._source(intrinsic)
        synchronize()
        timings['source_seconds'] += perf_counter() - tick
        tick = perf_counter()
        fields = engine.correlate_basis(b, t)
        synchronize()
        timings['fft_seconds'] += perf_counter() - tick
        tick = perf_counter()
        gram_at = prepare_gram(index, intrinsic, b, t)
        synchronize()
        timings['gram_prepare_seconds'] += perf_counter() - tick
        tick = perf_counter()
        for sky, selected, values in engine.chunks(
                fields, lags, sky_chunk=sky_chunk, lag_chunk=lag_chunk):
            # The network component axis is last for the quadratic form.
            z = xp.moveaxis(values, 1, -1)
            matrix = gram_at(sky, selected)
            twof, status = network_twof(z, matrix, noise_variance, xp=xp)
            for label in counts:
                counts[label] += int(xp.count_nonzero(status == int(label)))
            finite = xp.flatnonzero(xp.isfinite(twof.ravel()))
            if not int(finite.size):
                continue
            # Include ties deterministically. Only scalar candidates return
            # to the host; no full sky-by-time statistic array is stored.
            order = xp.argsort(-twof.ravel()[finite], kind='stable')[:top]
            locations = host(finite[order])
            strengths = host(twof.ravel()[finite[order]])
            for location, strength in zip(locations, strengths):
                p, j = divmod(int(location), len(selected))
                candidates.append(dict(template=index, sky=sky.start + p,
                                       lag=int(selected[j]), twof=float(strength)))
            candidates.sort(key=lambda c: (-c['twof'], c['template'],
                                            c['sky'], c['lag']))
            del candidates[top:]
        synchronize()
        timings['scan_seconds'] += perf_counter() - tick
        del fields
    expected = len(points) * len(engine.offsets) * len(lags)
    if sum(counts.values()) != expected:
        raise RuntimeError('rank counts do not account for the search domain')
    return dict(complete=counts['3'] == 0, candidates=candidates,
                rank_counts=counts, evaluated=expected, templates=len(points),
                sky_points=len(engine.offsets), lag_count=len(lags),
                timings=timings, elapsed_seconds=perf_counter() - began)
