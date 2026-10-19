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
