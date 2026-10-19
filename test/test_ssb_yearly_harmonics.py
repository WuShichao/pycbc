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


def _terms():
    from pycbc.tdi.inference import channel_terms
    return channel_terms(('A', 'E'), generation=2, delta_t=10.0)


def test_low_harmonics_reproduce_the_ssb_frame_response():
    from pycbc.tdi.ssb_yearly_harmonics import (harmonic_coefficients,
                                                ssb_frame_response,
                                                yearly_response_samples)
    from pycbc.tdi.ssb_yearly_harmonics import ORBITAL_ANGULAR_FREQUENCY
    terms = _terms()
    freqs = numpy.linspace(0.0135, 0.0205, 40)
    samples = yearly_response_samples(ORBIT, terms, freqs, *SKY, *SKY)
    rng = numpy.random.default_rng(4)
    times = rng.uniform(0, ORBITAL_PERIOD, 3)
    direct = [ssb_frame_response(ORBIT, terms,
                                 t + 1e-3 * numpy.arange(len(freqs)), freqs,
                                 *SKY, *SKY) for t in times]
    worst = []
    for kmax in (2, 4, 6):
        orders, rho = harmonic_coefficients(samples, kmax)
        loss = []
        for t, d in zip(times, direct):
            phase = numpy.exp(1j * orders * ORBITAL_ANGULAR_FREQUENCY * t)
            m = numpy.tensordot(phase, rho, axes=1)
            match = abs(numpy.vdot(m, d)) / (numpy.linalg.norm(m)
                                             * numpy.linalg.norm(d))
            loss.append(1 - match)
        worst.append(max(loss))
    # the K = 4 expansion of the SSB-search cards loses under 1e-6
    assert worst[1] < 1e-6
    assert worst[2] < worst[1] < worst[0]


def test_sky_log_derivative_predicts_a_small_offset():
    from pycbc.tdi.ssb_yearly_harmonics import (sky_log_derivatives,
                                                yearly_response_samples)
    terms = _terms()
    freqs = numpy.linspace(0.0135, 0.0205, 20)
    base = yearly_response_samples(ORBIT, terms, freqs, *SKY, *SKY,
                                   samples=8)
    L = sky_log_derivatives(ORBIT, terms, freqs, *SKY, scales=(2.0, 0.5),
                            samples=8)
    delta = numpy.array([1e-3, -2e-3])        # in the scaled units
    moved = yearly_response_samples(ORBIT, terms, freqs,
                                    SKY[0] + 2.0 * delta[0],
                                    SKY[1] + 0.5 * delta[1], *SKY, samples=8)
    model = base * numpy.exp(numpy.tensordot(delta, L, axes=1))
    first_order = numpy.abs(moved - base).max()
    assert numpy.abs(model - moved).max() < 0.02 * first_order


def test_harmonics_need_enough_samples():
    from pycbc.tdi.ssb_yearly_harmonics import harmonic_coefficients
    with pytest.raises(ValueError):
        harmonic_coefficients(numpy.zeros((8, 2)), 4)


def test_frequency_interpolation_is_exact_for_linear_values():
    from pycbc.tdi.ssb_yearly_harmonics import interpolate_in_frequency
    coarse = numpy.linspace(0.01, 0.02, 11)
    values = (1 + 2j) * coarse - 3j
    fine = numpy.linspace(0.01, 0.02, 97)
    assert numpy.allclose(interpolate_in_frequency(values, coarse, fine),
                          (1 + 2j) * fine - 3j)
