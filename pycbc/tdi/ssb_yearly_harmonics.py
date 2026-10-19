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
"""Data and response of a heliocentric space detector in SSB time.

Resampling the data from detector time ``t`` to solar-system-barycentre time
``t' = t + n.x(t)/c`` at a reference sky ``n`` removes the yearly Doppler
modulation of a source at that sky. What remains of the response is then
periodic with the orbit, so it is a short sum of yearly harmonics, and a
shift of the arrival time by ``Delta`` multiplies harmonic ``k`` by
``exp(i k Omega Delta)``. On a frequency grid whose length is a whole number
of orbital periods, harmonic ``k`` is a shift by a whole number of bins, so
one FFT per basis correlates a template with the data at every arrival time.

``x(t)`` is the constellation centre, the mean of the spacecraft positions
of any orbit in :mod:`pycbc.coordinates.space_orbit`.
"""

import math

import numpy

from pycbc.constants import C_SI

# 2 pi / sidereal year, as pycbc.coordinates.space.space_position_ssb
ORBITAL_ANGULAR_FREQUENCY = 1.99098659277e-7
ORBITAL_PERIOD = 2 * math.pi / ORBITAL_ANGULAR_FREQUENCY


def constellation_centre(orbit, times):
    """The mean spacecraft position, shape ``(3, ntimes)`` in metres."""
    times = numpy.atleast_1d(numpy.asarray(times, dtype=float))
    positions = numpy.asarray(orbit.compute_position(times, sc=(1, 2, 3)),
                              dtype=float)
    return positions.reshape(len(times), 3, 3).mean(axis=1).T


def ssb_delay(orbit, times, longitude, latitude):
    """``n.x(t)/c``: SSB time minus the constellation-centre arrival time
    of a plane wave from ecliptic ``longitude`` and ``latitude``."""
    n = numpy.array([math.cos(latitude) * math.cos(longitude),
                     math.cos(latitude) * math.sin(longitude),
                     math.sin(latitude)])
    return n @ constellation_centre(orbit, times) / C_SI


def detector_times(orbit, ssb_times, longitude, latitude, iterations=4):
    """The detector times whose SSB times at the given sky are
    ``ssb_times``, by fixed-point iteration (the delay changes by under
    1e-4 s per second)."""
    ssb_times = numpy.asarray(ssb_times, dtype=float)
    t = ssb_times.copy()
    for _ in range(iterations):
        t = ssb_times - ssb_delay(orbit, t, longitude, latitude)
    return t


def _smooth_length(need):
    """The smallest 2^a 3^b 5^c not below ``need``."""
    best = 2 ** int(math.ceil(math.log2(need)))
    p3 = 1
    while p3 < best:
        p5 = p3
        while p5 < best:
            n = p5 * 2 ** max(0, int(math.ceil(math.log2(need / p5))))
            best = min(best, n)
            p5 *= 5
        p3 *= 3
    return best


class SSBGrid(object):
    """A uniform SSB time grid over a whole number of orbital periods.

    The grid starts at the data start and spans ``periods`` orbital periods,
    so its frequency resolution is ``1 / (periods * period)`` and harmonic
    ``k`` of the orbit is a shift of ``k * periods`` bins. It has enough
    samples to hold the band of ``nbins`` data bins of width ``delta_f``.

    Parameters
    ----------
    start, stop : float
        The data's first and last time, s.
    first_frequency : float
        Frequency of the first data bin, the baseband origin, Hz.
    nbins : int
        Number of data bins in the band.
    delta_f : float
        The data's frequency resolution, ``1 / (stop - start)``.
    periods : int
        Length of the grid in orbital periods. It must exceed the data plus
        the in-band duration of the templates, or correlations wrap round.
    length : {'smooth', 'pow2'}
        A 5-smooth or a power-of-two number of samples.
    period : float, optional
        The orbital period, s. Default the sidereal year.
    """
    def __init__(self, start, stop, first_frequency, nbins, delta_f,
                 periods=5, length='smooth', period=ORBITAL_PERIOD):
        if length not in ('smooth', 'pow2'):
            raise ValueError("length must be 'smooth' or 'pow2'")
        self.start, self.stop = float(start), float(stop)
        self.period = float(period)
        self.duration = int(periods) * self.period
        need = nbins * self.duration / (self.stop - self.start)
        self.size = (_smooth_length(need) if length == 'smooth'
                     else 2 ** int(math.ceil(math.log2(need))))
        self.delta_t = self.duration / self.size
        self.delta_f = 1.0 / self.duration
        self.first_frequency = float(first_frequency)
        self.frequencies = (self.first_frequency
                            + numpy.arange(self.size) * self.delta_f)
        self.times = self.start + numpy.arange(self.size) * self.delta_t
        self.in_data = self.times < self.stop
        self.harmonic_shift = int(periods)
        self.data_delta_f = float(delta_f)
        self.data_nbins = int(nbins)


def resample_to_ssb(grid, bins, orbit, longitude, latitude, epsilon=1e-10):
    """Detector-frame frequency bins to an SSB-time series on ``grid``.

    ``bins`` are the whitened data, shape ``(nchannels, grid.data_nbins)``,
    on frequencies ``grid.first_frequency + j * grid.data_delta_f`` with
    phases referenced to ``grid.start``. The series at SSB time ``t'`` is
    the detector-frame signal at the detector time ``t(t')``, basebanded by
    ``exp(-2 pi i f0 (t' - start))``; it is zero after the data end.

    Requires `finufft <https://finufft.readthedocs.io>`_.
    """
    try:
        import finufft
    except ImportError as err:
        raise ImportError("resample_to_ssb needs finufft") from err
    bins = numpy.atleast_2d(numpy.asarray(bins, dtype=complex))
    nbins = grid.data_nbins
    if bins.shape[1] != nbins:
        raise ValueError("expected {} bins per channel".format(nbins))
    n_in = int(grid.in_data.sum())
    ssb = grid.times[:n_in]
    t = detector_times(orbit, ssb, longitude, latitude)
    dfd = grid.data_delta_f
    x = numpy.mod(2 * numpy.pi * dfd * (t - grid.start), 2 * numpy.pi)
    padded = nbins + (nbins % 2)
    centre = padded // 2
    carrier = (numpy.exp(2j * numpy.pi * (grid.first_frequency + centre * dfd)
                         * (t - ssb))
               * numpy.exp(2j * numpy.pi * centre * dfd * (ssb - grid.start)))
    out = numpy.zeros((bins.shape[0], grid.size), dtype=complex)
    for ch, row in enumerate(bins):
        modes = numpy.zeros(padded, dtype=complex)
        modes[:nbins] = row
        out[ch, :n_in] = finufft.nufft1d2(x, modes, isign=1,
                                          eps=epsilon) * carrier
    return out


def ssb_frame_response(orbit, terms, times, frequencies, longitude, latitude,
                       ref_longitude, ref_latitude):
    """The adiabatic response at paired ``(times, frequencies)`` with the
    reference sky's SSB delay removed, shape ``(2, nchannels, n)`` for the
    plus and cross bases.

    ``terms`` maps channel names to TDI terms, as
    :func:`pycbc.tdi.inference.channel_terms` returns; ``times`` must
    increase.
    """
    from pycbc.tdi.onthefly import (MultiChannelTermGeometry,
                                    frequency_response_factors)
    times = numpy.asarray(times, dtype=float)
    frequencies = numpy.asarray(frequencies, dtype=float)
    geometry = MultiChannelTermGeometry(orbit, times, terms)
    factors = frequency_response_factors(geometry, frequencies, longitude,
                                         latitude)
    response = numpy.array([[factors[c][a] for c in geometry.channel_names]
                            for a in range(2)])
    delay = ssb_delay(orbit, times, ref_longitude, ref_latitude)
    return response * numpy.exp(-2j * numpy.pi * frequencies * delay)


def yearly_response_samples(orbit, terms, frequencies, longitude, latitude,
                            ref_longitude, ref_latitude, samples=32,
                            period=ORBITAL_PERIOD):
    """The SSB-frame response at ``samples`` equally spaced times over one
    period, shape ``(samples, 2, nchannels, nfrequencies)``."""
    frequencies = numpy.asarray(frequencies, dtype=float)
    # times one millisecond apart keep the geometry's grid increasing
    stagger = 1e-3 * numpy.arange(len(frequencies))
    return numpy.array([
        ssb_frame_response(orbit, terms, j * period / samples + stagger,
                           frequencies, longitude, latitude, ref_longitude,
                           ref_latitude)
        for j in range(samples)])


def harmonic_coefficients(samples, kmax):
    """Harmonics ``-kmax..kmax`` of response samples over one period:
    ``rho_k = (1/N) sum_j R(t_j) exp(-i k Omega t_j)``. Returns the orders
    and the coefficients, shape ``(2 kmax + 1,) + samples.shape[1:]``."""
    samples = numpy.asarray(samples)
    n = samples.shape[0]
    if 2 * kmax + 1 > n:
        raise ValueError("{} samples cannot resolve {} harmonics".format(
            n, 2 * kmax + 1))
    orders = numpy.arange(-kmax, kmax + 1)
    return orders, (numpy.fft.fft(samples, axis=0) / n)[orders % n]


def sky_log_derivatives(orbit, terms, frequencies, longitude, latitude,
                        scales=(1.0, 1.0), step=1e-4, samples=32,
                        period=ORBITAL_PERIOD):
    """Derivatives of the log response samples with respect to longitude
    and latitude, times ``scales``, at a reference sky equal to the source
    sky; shape ``(2, samples, 2, nchannels, nfrequencies)``.

    A small sky offset ``delta`` then multiplies the samples by
    ``exp(delta . L)``.
    """
    out = []
    for axis, scale in enumerate(scales):
        offset = numpy.zeros(2)
        offset[axis] = step
        plus = yearly_response_samples(
            orbit, terms, frequencies, longitude + offset[0],
            latitude + offset[1], longitude, latitude, samples, period)
        minus = yearly_response_samples(
            orbit, terms, frequencies, longitude - offset[0],
            latitude - offset[1], longitude, latitude, samples, period)
        out.append(numpy.log(plus / minus) / (2 * step) * scale)
    return numpy.array(out)


def interpolate_in_frequency(values, coarse, fine):
    """Linear interpolation of complex ``values`` on the uniformly spaced
    ``coarse`` frequencies (last axis) to the ``fine`` ones, real and
    imaginary parts separately; constant beyond the ends."""
    coarse = numpy.asarray(coarse, dtype=float)
    x = (numpy.asarray(fine, dtype=float) - coarse[0]) \
        / (coarse[1] - coarse[0])
    i = numpy.clip(numpy.floor(x).astype(int), 0, len(coarse) - 2)
    w = numpy.clip(x - i, 0, 1)
    return values[..., i] * (1 - w) + values[..., i + 1] * w
