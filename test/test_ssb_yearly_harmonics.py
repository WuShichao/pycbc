"""SSB-time grid, delay and resampling of pycbc.tdi.ssb_yearly_harmonics."""
import math

import numpy
import pytest

from pycbc.coordinates.space import space_position_ssb
from pycbc.coordinates.space_orbit import LisaEqualArmOrbit
from pycbc.tdi.ssb_yearly_harmonics import (ORBITAL_PERIOD, SSBGrid,
                                            _smooth_length, detector_times,
                                            resample_to_ssb, ssb_delay)

ORBIT = LisaEqualArmOrbit(armlength=2.5e9, t0=0.0)
SKY = (3.57, 0.34)
DAY = 86400.0


def test_delay_is_that_of_the_guiding_centre():
    t = numpy.linspace(0, 3e7, 11)
    p, _ = space_position_ssb(t, t0=0.0)
    x, y = numpy.asarray(p[0, 0], float), numpy.asarray(p[1, 0], float)
    lam, beta = SKY
    expected = math.cos(beta) * (math.cos(lam) * x + math.sin(lam) * y) \
        / 299792458.0
    assert numpy.allclose(ssb_delay(ORBIT, t, *SKY), expected,
                          rtol=0, atol=1e-9)


def test_detector_times_invert_the_delay():
    ssb = numpy.linspace(1e6, 5e7, 50)
    t = detector_times(ORBIT, ssb, *SKY)
    assert numpy.allclose(t + ssb_delay(ORBIT, t, *SKY), ssb, rtol=0,
                          atol=1e-6)


@pytest.mark.parametrize('need', [7.0, 1000.3, 1105919.3, 2 ** 20 + 1])
def test_smooth_length_is_the_smallest_5_smooth_cover(need):
    brute = min(n for n in (2 ** a * 3 ** b * 5 ** c for a in range(40)
                            for b in range(14) for c in range(10))
                if n >= need)
    assert _smooth_length(need) == brute


def test_grid_spans_whole_periods():
    grid = SSBGrid(0.0, 730 * DAY, 0.0135, 189216, 1 / (730 * DAY),
                   periods=5)
    assert grid.delta_f == pytest.approx(1 / (5 * ORBITAL_PERIOD))
    assert grid.size * grid.delta_t == pytest.approx(5 * ORBITAL_PERIOD)
    assert grid.size >= 189216 * 5 * ORBITAL_PERIOD / (730 * DAY)
    assert grid.harmonic_shift == 5
    assert grid.in_data.sum() == math.ceil(730 * DAY / grid.delta_t)


def test_resampling_evaluates_the_signal_at_detector_time():
    pytest.importorskip('finufft')
    start, days, nbins = 2e6, 30.0, 40
    delta_f = 1 / (days * DAY)
    f0 = 0.0151
    grid = SSBGrid(start, start + days * DAY, f0, nbins, delta_f, periods=1)
    rng = numpy.random.default_rng(1)
    bins = rng.normal(size=(2, nbins)) + 1j * rng.normal(size=(2, nbins))
    series = resample_to_ssb(grid, bins, ORBIT, *SKY)
    n_in = int(grid.in_data.sum())
    ssb = grid.times[:n_in]
    t = detector_times(ORBIT, ssb, *SKY)
    freqs = f0 + numpy.arange(nbins) * delta_f
    modes = numpy.exp(2j * numpy.pi * freqs[None, :]
                      * (t - start)[:, None])
    expected = (bins @ modes.T) \
        * numpy.exp(-2j * numpy.pi * f0 * (ssb - start))[None, :]
    assert numpy.allclose(series[:, :n_in], expected, rtol=0, atol=1e-7)
    assert numpy.all(series[:, n_in:] == 0)
