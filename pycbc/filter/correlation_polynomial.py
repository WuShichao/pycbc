# Copyright (C) 2026 Shichao Wu
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
"""Finite Fourier correlations on a declared physical time interval."""
import numpy


def interval_weights(times):
    """Trapezoid prior masses covering exactly the supplied interval.

    Both physical prior endpoints must be supplied. These weights do not
    select a peak, omit a tail, or renormalize a searched subset.
    """
    times = numpy.asarray(times, dtype=float)
    if (times.ndim != 1 or len(times) < 2
            or not numpy.isfinite(times).all()
            or not numpy.all(numpy.diff(times) > 0)):
        raise ValueError('finite increasing interval nodes required')
    gaps = numpy.diff(times)
    widths = numpy.empty(len(times))
    widths[0], widths[-1] = gaps[0] / 2, gaps[-1] / 2
    widths[1:-1] = (gaps[:-1] + gaps[1:]) / 2
    return widths / (times[-1] - times[0])


class CorrelationPolynomial(object):
    """Sum of coefficients times exp(2 pi i (first + j spacing) time).

    Coefficients already contain the chosen data inner product. This class
    neither changes the noise model nor assumes the template Gram is
    constant. Its derivative bounds concern this finite polynomial only,
    not waveform-model errors or a floating-point interval certificate.
    """

    def __init__(self, coefficients, first_frequency, spacing):
        values = numpy.array(coefficients, dtype=complex, copy=True)
        if (values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1
                or not numpy.isfinite(values).all()
                or not numpy.isfinite([first_frequency, spacing]).all()
                or spacing <= 0):
            raise ValueError('finite coefficient rows and positive spacing required')
        self.coefficients = values
        self.coefficients.setflags(write=False)
        self.first_frequency = float(first_frequency)
        self.spacing = float(spacing)

    def _offsets(self, reference_frequency):
        if not numpy.isfinite(reference_frequency):
            raise ValueError('reference frequency must be finite')
        return (self.first_frequency - reference_frequency
                + numpy.arange(self.coefficients.shape[1]) * self.spacing)

    def derivative_bound(self, order, reference_frequency=0.):
        """L1 coefficient bound on each demodulated derivative magnitude."""
        if not isinstance(order, (int, numpy.integer)) or order < 0:
            raise ValueError('derivative order must be a nonnegative integer')
        frequencies = self._offsets(reference_frequency)
        bound = numpy.abs(self.coefficients) @ (2 * numpy.pi * numpy.abs(frequencies))**order
        if not numpy.isfinite(bound).all():
            raise ValueError('nonfinite derivative bound')
        return bound

    def interval(self, start, stop, count, derivatives=(0,),
                 reference_frequency=0., max_output_bytes=512 * 1024**2):
        """Uniform endpoint-inclusive values and analytic derivatives.

        Returns ``(times, fields)``; ``fields[p]`` has shape (bases, times)
        and differentiates exp(-2 pi i reference_frequency time) times the
        physical correlation. A common reference phase disappears only in
        statistics that marginalize the corresponding phase; callers still
        own the physical time prior and non-wrapping observation convention.
        """
        from scipy.signal import ZoomFFT

        if (not numpy.isfinite([start, stop]).all() or stop <= start
                or not isinstance(count, (int, numpy.integer)) or count < 2):
            raise ValueError('finite increasing interval and count >= 2 required')
        derivatives = tuple(derivatives)
        if (not derivatives or len(set(derivatives)) != len(derivatives)
                or any(not isinstance(p, (int, numpy.integer)) or p < 0
                       for p in derivatives)):
            raise ValueError('distinct nonnegative derivative orders required')
        if not isinstance(max_output_bytes, (int, numpy.integer)) or max_output_bytes < 1:
            raise ValueError('output memory limit must be positive')
        required = (len(derivatives) * self.coefficients.shape[0] * int(count)
                    * numpy.dtype(complex).itemsize)
        if required > max_output_bytes:
            raise MemoryError('interval fields exceed their output memory limit')
        frequencies = self._offsets(reference_frequency)
        times = numpy.linspace(start, stop, int(count))
        # ZoomFFT uses a negative-exponent transform. Reversed normalized
        # endpoints give the positive-exponent physical correlation sum.
        transform = ZoomFFT(self.coefficients.shape[1],
            [-start * self.spacing, -stop * self.spacing],
            m=int(count), fs=1., endpoint=True)
        carrier = numpy.exp(2j * numpy.pi
                            * (self.first_frequency - reference_frequency) * times)
        fields = {}
        for order in derivatives:
            values = transform(self.coefficients * (2j * numpy.pi * frequencies)**order)
            fields[order] = values * carrier
            if not numpy.isfinite(fields[order]).all():
                raise ValueError('nonfinite interval correlations')
        return times, fields

    def hermite(self, start, stop, count, reference_frequency=0.,
                max_output_bytes=512 * 1024**2):
        """Cubic Hermite field and a real/imaginary interpolation bound.

        The returned per-basis bound applies separately to either component
        between adjacent nodes. A complex absolute bound is sqrt(2) times
        this value. Transform roundoff is not included in that analytic
        interpolation remainder and must be checked independently.
        """
        from scipy.interpolate import CubicHermiteSpline

        times, fields = self.interval(start, stop, count, derivatives=(0, 1),
            reference_frequency=reference_frequency, max_output_bytes=max_output_bytes)
        bound = (numpy.max(numpy.diff(times))**4 / 384
                 * self.derivative_bound(4, reference_frequency))
        return CubicHermiteSpline(times, fields[0], fields[1], axis=-1,
                                 extrapolate=False), bound
