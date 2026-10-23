# Copyright (C) 2026 Shichao Wu
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
"""CPU harmonic shifts without a dense harmonic-by-frequency response."""
import numpy


class InterpolatedHarmonicResponse:
    """Interpolate one response harmonic at a time on a fine frequency grid.

    Parameters
    ----------
    orders : array of int
        Unique harmonic orders, in the same order as ``responses``.
    coarse_frequencies, frequencies : array
        Increasing grids. The fine grid must lie inside the coarse grid;
        response extrapolation is not performed.
    responses : array
        Shape (harmonics, bases, channels, coarse frequencies). An owned,
        read-only copy is retained. No full fine-grid response is cached.
    max_block_bytes : int
        Largest allowed fine-grid response for a single harmonic.
    """

    def __init__(self, orders, coarse_frequencies, frequencies, responses,
                 max_block_bytes=512 * 1024**2):
        orders = numpy.array(orders, copy=True)
        coarse = numpy.array(coarse_frequencies, dtype=float, copy=True)
        fine = numpy.array(frequencies, dtype=float, copy=True)
        values = numpy.array(responses, dtype=complex, copy=True)
        if (orders.ndim != 1 or not len(orders) or
                not numpy.issubdtype(orders.dtype, numpy.integer) or
                len(numpy.unique(orders)) != len(orders)):
            raise ValueError('harmonic orders must be unique integers')
        if (coarse.ndim != 1 or fine.ndim != 1 or len(coarse) < 2 or
                not len(fine) or not numpy.isfinite(coarse).all() or
                not numpy.isfinite(fine).all() or
                numpy.any(numpy.diff(coarse) <= 0) or
                numpy.any(numpy.diff(fine) <= 0) or
                fine[0] < coarse[0] or fine[-1] > coarse[-1]):
            raise ValueError('fine frequencies must lie inside the coarse grid')
        if (values.ndim != 4 or values.shape[0] != len(orders) or
                values.shape[-1] != len(coarse) or
                min(values.shape[1:3]) < 1 or
                not numpy.isfinite(values).all()):
            raise ValueError('finite coarse two-polarization responses required')
        required = int(numpy.prod(values.shape[1:3])) * len(fine) * values.dtype.itemsize
        if not isinstance(max_block_bytes, int) or max_block_bytes < 1:
            raise ValueError('a positive per-harmonic memory limit is required')
        if required > max_block_bytes:
            raise MemoryError('one fine-grid response exceeds its memory limit')
        self.orders, self.coarse_frequencies = orders, coarse
        self.frequencies, self.coarse_responses = fine, values
        self.dtype = values.dtype
        self.shape = values.shape[:-1] + (len(fine),)
        for array in (orders, coarse, fine, values):
            array.setflags(write=False)

    def at_index(self, index):
        """Return just one (bases, channels, fine frequencies) response."""
        if (not isinstance(index, (int, numpy.integer)) or
                not 0 <= index < len(self.orders)):
            raise IndexError('unknown harmonic index')
        output = numpy.empty(self.shape[1:], dtype=self.dtype)
        source = self.coarse_responses[index].reshape(-1, len(self.coarse_frequencies))
        for target, row in zip(output.reshape(-1, len(self.frequencies)), source):
            target.real = numpy.interp(self.frequencies, self.coarse_frequencies, row.real)
            target.imag = numpy.interp(self.frequencies, self.coarse_frequencies, row.imag)
        return output


def _inputs(source, responses, orders, periods):
    orders = numpy.asarray(orders)
    source = numpy.asarray(source)
    if (orders.ndim != 1 or not len(orders) or
            not numpy.issubdtype(orders.dtype, numpy.integer) or
            len(numpy.unique(orders)) != len(orders)):
        raise ValueError('harmonic orders must be unique integers')
    if not isinstance(periods, (int, numpy.integer)) or periods < 1:
        raise ValueError('periods must be a positive integer')
    if (source.ndim != 1 or not source.size or
            not numpy.isfinite(source).all() or
            not callable(getattr(responses, 'at_index', None)) or
            len(responses.shape) != 4 or
            responses.shape[0] != len(orders) or
            responses.shape[-1] != source.size or
            not numpy.array_equal(responses.orders, orders)):
        raise ValueError('source and lazy response shapes or orders differ')
    return source, orders


def _read(responses, index):
    value = numpy.asarray(responses.at_index(index))
    if (value.shape != responses.shape[1:] or
            not numpy.isfinite(value).all()):
        raise ValueError('invalid fine-grid response harmonic')
    return value


def streamed_harmonic_spectrum(source, responses, orders, periods=1):
    """The same finite harmonic shifts as harmonic_spectrum, on the CPU.

    Memory holds the output and one response harmonic, not all harmonics.
    Missing source frequencies are zero, never circularly wrapped. Source
    support, detector windowing and whitening remain the caller's job.
    """
    source, orders = _inputs(source, responses, orders, periods)
    output = numpy.zeros(responses.shape[1:], dtype=numpy.result_type(
        source.dtype, responses.dtype, complex))
    size = source.size
    for index, order in enumerate(orders):
        shift = int(order) * int(periods)
        if abs(shift) >= size:
            continue
        response = _read(responses, index)
        if shift >= 0:
            output[..., shift:] += response[..., :size-shift] * source[:size-shift]
        else:
            output[..., :size+shift] += response[..., -shift:] * source[-shift:]
        del response
    return output


def streamed_correlation_spectrum(source, responses, orders, data, periods=1):
    """Share exact harmonic-shift correlations without a dense response.

    ``data`` is the original-band adjoint-projected detector data. This
    returns a complex correlation spectrum, not a Gram approximation.
    """
    source, orders = _inputs(source, responses, orders, periods)
    data = numpy.asarray(data)
    if (data.shape != responses.shape[-2:] or
            not numpy.isfinite(data).all()):
        raise ValueError('finite adjoint data with the response shape required')
    output = numpy.zeros((responses.shape[1], source.size),
                         dtype=numpy.result_type(source.dtype, responses.dtype,
                                                 data.dtype, complex))
    for index, order in enumerate(orders):
        shift = int(order) * int(periods)
        if abs(shift) >= source.size:
            continue
        response = _read(responses, index)
        begin, end = max(0, -shift), min(source.size, source.size-shift)
        contribution = numpy.einsum('acf,cf->af', response[..., begin:end].conj(),
                                   data[:, begin+shift:end+shift])
        output[:, begin:end] += source[begin:end].conj() * contribution
        del response, contribution
    return output
