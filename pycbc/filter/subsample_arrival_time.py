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


def interpolate_twice_sampled(even, odd, step, targets, half_width=24,
                              beta=14.0, xp=numpy):
    """A correlation at arbitrary ``targets`` from its samples at half the
    grid step.

    ``even[..., j]`` is ``z(2 j step)`` and ``odd[..., j]`` is ``z((2 j +
    1) step)``, with ``step = T / (2 N)`` half the step of an ``N``-point
    grid over ``T``: two inverse FFTs, the second of the spectrum turned by
    half a step. The band of ``z`` fills the grid's Nyquist interval, so
    demodulated at its centre, ``1 / (4 step)``, it fills half of the twice
    finer one, and a Kaiser-windowed cardinal sine over ``half_width``
    samples either side reconstructs it to about ``exp(-beta)`` (5e-8 at
    the defaults). The samples are periodic, as an inverse FFT's are;
    ``targets`` are in seconds from sample 0. The taps are computed where
    ``targets`` live, so a few targets on the host cost no device work but
    the final gather and sum.
    """
    tp = numpy if isinstance(targets, numpy.ndarray) else xp
    targets = tp.asarray(targets, dtype=tp.float64)
    x = targets / step
    base = tp.floor(x).astype(tp.int64)
    taps = tp.arange(-half_width + 1, half_width + 1)
    index = base[:, None] + taps[None, :]
    offset = x[:, None] - index
    index = index % (2 * even.shape[-1])
    # Kaiser window over the taps
    u = offset / half_width
    window = tp.where(tp.abs(u) < 1, _bessel_i0(
        beta * tp.sqrt(tp.maximum(1 - u * u, 0)), tp)
        / _bessel_i0(tp.asarray(beta), tp), 0)
    # demodulation at the band centre: phase pi / 2 per sample
    kernel = tp.sinc(offset) * window * tp.exp(0.5j * numpy.pi * offset)
    kernel = xp.asarray(kernel)
    half, parity = xp.asarray(index // 2), xp.asarray(index % 2 == 1)
    values = xp.where(parity, odd[..., half], even[..., half])
    return xp.sum(values * kernel, axis=-1)


def _bessel_i0(x, xp):
    if xp is numpy:
        from scipy.special import i0
        return i0(x)
    from cupyx.scipy.special import i0
    return i0(x)
