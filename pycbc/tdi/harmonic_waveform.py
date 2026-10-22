# Copyright (C) 2026 Shichao Wu
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
"""Frequency shifts for a time-periodic response acting on a source spectrum."""
import numpy


def harmonic_spectrum(source, responses, orders, periods=1, xp=numpy):
    r"""Assemble exact harmonic shifts on a grid of whole orbital periods.

    If ``R(t,f) = sum_k rho_k(f) exp(i k Omega t)``, its action on a
    Fourier density is ``Y(f) = sum_k rho_k(f-k/T) H(f-k/T)``. This retains
    the source phase curvature, instead of replacing each shift by the
    stationary-time phase ``exp(i k Omega t(f))``.

    ``responses`` has shape ``(nk, ..., nfrequency)`` and ``source`` has
    shape ``(nfrequency,)``. Input outside the supplied frequency interval
    is zero, not periodic: callers must supply enough physical guard band
    and verify its convergence. No source, detector window or whitening is
    supplied here. The Fourier epoch must be zero for the response samples;
    otherwise the caller must include its harmonic phase in ``responses``.
    """
    orders = numpy.asarray(orders)
    if (orders.ndim != 1 or not len(orders) or
            not numpy.issubdtype(orders.dtype, numpy.integer) or
            len(numpy.unique(orders)) != len(orders)):
        raise ValueError('harmonic orders must be unique integers')
    if not isinstance(periods, (int, numpy.integer)) or periods < 1:
        raise ValueError('periods must be a positive integer')
    source, responses = xp.asarray(source), xp.asarray(responses)
    if (source.ndim != 1 or not source.size or responses.ndim < 2 or
            responses.shape[0] != len(orders) or
            responses.shape[-1] != source.size):
        raise ValueError('source and response frequency shapes differ')
    if not bool(xp.isfinite(source).all()) or not bool(xp.isfinite(responses).all()):
        raise ValueError('source and responses must be finite')
    output = xp.zeros(responses.shape[1:], dtype=xp.result_type(source, responses, complex))
    n = source.size
    for i, k in enumerate(orders):
        shift = int(k) * int(periods)
        if abs(shift) >= n:
            continue
        if shift >= 0:
            output[..., shift:] += responses[i, ..., :n - shift] * source[:n - shift]
        else:
            output[..., :n + shift] += responses[i, ..., -shift:] * source[-shift:]
    return output


def harmonic_correlation_spectrum(source, responses, orders, data,
                                   periods=1, xp=numpy):
    r"""FFT spectrum of correlations using exact response frequency shifts.

    ``data`` is the adjoint-projected detector data, shape ``(channels,N)``.
    Responses have shape ``(harmonics,bases,channels,N)``. Returning ``a``
    gives ``z(Delta_j) = N ifft(a)[j] exp(2 pi i f0 Delta_j)`` on the native
    time grid. f0 is its first physical Fourier frequency, not zero.

    Both source and response are evaluated at input frequency j; data is
    at j+k*periods. Missing frequencies are zero, never circularly wrapped.
    This handles the complex correlations only. The Gram still belongs to
    the original detector window/band and must be evaluated consistently.
    """
    orders = numpy.asarray(orders)
    if (orders.ndim != 1 or not len(orders) or
            not numpy.issubdtype(orders.dtype, numpy.integer) or
            len(numpy.unique(orders)) != len(orders)):
        raise ValueError('harmonic orders must be unique integers')
    if not isinstance(periods, (int, numpy.integer)) or periods < 1:
        raise ValueError('periods must be a positive integer')
    source, responses, data = xp.asarray(source), xp.asarray(responses), xp.asarray(data)
    if (source.ndim != 1 or not source.size or responses.ndim != 4 or data.ndim != 2 or
            responses.shape[0] != len(orders) or responses.shape[-2:] != data.shape or
            data.shape[-1] != source.size):
        raise ValueError('harmonic correlation shapes differ')
    if not all(bool(xp.isfinite(x).all()) for x in (source, responses, data)):
        raise ValueError('harmonic correlation inputs must be finite')
    n = source.size
    out = xp.zeros((responses.shape[1], n), dtype=xp.result_type(source, responses, data, complex))
    for i, k in enumerate(orders):
        shift = int(k) * int(periods)
        if abs(shift) >= n:
            continue
        begin, end = max(0, -shift), min(n, n - shift)
        contribution = xp.einsum('acf,cf->af', responses[i, ..., begin:end].conj(),
                                 data[:, begin + shift:end + shift])
        out[:, begin:end] += source[begin:end].conj() * contribution
    return out
