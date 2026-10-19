# Copyright (C) 2026  Shichao Wu
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""Correlations at arrival times between the samples of an FFT grid.

A correlation computed by one inverse FFT of length ``N`` over a duration
``T``,

    z(Delta) = sum_m a_m exp(2 pi i m Delta / T),   m = 0 .. N - 1,

is known at every ``Delta`` once its spectrum ``a_m`` is: the samples at
``Delta = j T / N`` are only one evaluation of it. Here ``m`` is the
baseband frequency index of a one-sided band, so the interpolant is the
modulated cardinal sine, not the plain one. The statistic between samples
loses ``(2 pi sigma_f delta)^2`` of its value at an offset ``delta`` from
the peak, with ``sigma_f`` the rms bandwidth of the template's power, so
evaluating on a grid ``P`` times finer bounds the loss by
``(pi sigma_f delta_t / P)^2``.
"""

import numpy


def correlation_at(spectrum, delays, duration, chunk=16, xp=numpy):
    """``z(Delta)`` at arbitrary ``delays`` (s) from the spectrum, shape
    ``(..., N)``; returns shape ``(..., len(delays))``. The direct sum costs
    ``N`` per delay, for a few delays."""
    delays = xp.asarray(numpy.atleast_1d(delays), dtype=xp.float64)
    m = xp.arange(spectrum.shape[-1], dtype=xp.float64)
    out = []
    for lo in range(0, len(delays), chunk):
        phase = xp.exp(2j * numpy.pi * xp.outer(delays[lo:lo + chunk], m)
                       / duration)
        out.append(spectrum @ phase.T)
    return xp.concatenate(out, axis=-1)


def fractional_correlations(spectrum, oversampling, xp=numpy):
    """``z`` on the grid ``Delta = (j + p / P) T / N`` for every sample
    ``j`` and every ``p < P``; shape ``(P,) + spectrum.shape``. One inverse
    FFT per fraction, exact."""
    n = spectrum.shape[-1]
    m = xp.arange(n, dtype=xp.float64)
    out = []
    for p in range(int(oversampling)):
        ramp = xp.exp(2j * numpy.pi * m * p / (oversampling * n))
        out.append(xp.fft.ifft(spectrum * ramp, axis=-1) * n)
    return xp.stack(out)


def rms_bandwidth(frequencies, power):
    """The rms bandwidth ``sigma_f`` of a template's power spectrum."""
    frequencies = numpy.asarray(frequencies, dtype=float)
    power = numpy.asarray(power, dtype=float)
    mean = numpy.sum(power * frequencies) / numpy.sum(power)
    return float(numpy.sqrt(numpy.sum(power * (frequencies - mean) ** 2)
                            / numpy.sum(power)))


def oversampling_for_loss(sigma_f, delta_t, max_loss):
    """The smallest ``P`` for which a peak half a fine step from the
    nearest fine sample loses at most ``max_loss`` of the statistic."""
    return max(1, int(numpy.ceil(numpy.pi * sigma_f * delta_t
                                 / numpy.sqrt(max_loss))))


def refine_peak(statistic, low, high, tolerance=1e-3):
    """The maximum of ``statistic(delay)`` on ``[low, high]`` by
    golden-section search, for a peak known to be the only one there.

    Returns
    -------
    delay : float
    value : float
    """
    ratio = (numpy.sqrt(5) - 1) / 2
    a, b = float(low), float(high)
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = statistic(c), statistic(d)
    while b - a > tolerance:
        if fc >= fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = statistic(c)
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = statistic(d)
    return (c, fc) if fc >= fd else (d, fd)
