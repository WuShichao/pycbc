"""Arrival times between grid samples, pycbc.filter.subsample_arrival_time."""
import numpy
import pytest

from pycbc.filter.fstatistic import twof
from pycbc.filter.subsample_arrival_time import (correlation_at,
                                                 fractional_correlations,
                                                 oversampling_for_loss,
                                                 refine_peak, rms_bandwidth)
from pycbc.tdi.linear_exponent_surrogate import LinearExponentSurrogate
from pycbc.tdi.ssb_yearly_harmonics import SSBGrid

DAY = 86400.0


def family(days=60.0, seed=2):
    rng = numpy.random.default_rng(seed)
    grid = SSBGrid(0.0, days * DAY, 0.015, int(20 * days),
                   1 / (days * DAY), periods=1)
    f = grid.frequencies
    u = (f - f[0]) / (f[-1] - f[0])
    t0 = (0.2 + 0.6 * u) * days * DAY
    phi = numpy.concatenate([[0], numpy.cumsum(
        0.5 * (t0[1:] + t0[:-1]) * numpy.diff(f))])
    # a narrow whitened band, as in the stBBH search, so the lag peak is wide
    amp = numpy.exp(-0.5 * ((u - 0.4) / 0.08) ** 2)
    coarse = numpy.linspace(f[0], f[-1], 10)
    shape = (16, 2, 2, 10)
    R0 = 1 + 0.3 * (rng.normal(size=shape) + 1j * rng.normal(size=shape))
    zero = numpy.zeros((1, grid.size))
    return LinearExponentSurrogate(
        grid, coarse, R0, numpy.zeros((2,) + shape),
        amp * numpy.exp(-2j * numpy.pi * phi), t0, zero, zero, kmax=3)


def test_interpolants_agree_with_the_fft_samples():
    fam = family()
    rng = numpy.random.default_rng(1)
    data = rng.normal(size=(2, fam.size)) + 1j * rng.normal(size=(2, fam.size))
    b, t, rho = fam.pieces(numpy.zeros(3))
    spectrum = fam.correlation_spectrum(data, b, t, rho)
    z = fam.correlate(data, b, t, rho)
    duration = fam.grid.duration
    lags = numpy.array([0, 5, 400, fam.size - 2])
    assert numpy.allclose(correlation_at(spectrum, lags * fam.grid.delta_t,
                                         duration), z[:, lags])
    fine = fractional_correlations(spectrum, 4)
    assert numpy.allclose(fine[0], z)
    delays = (lags + 0.75) * fam.grid.delta_t
    assert numpy.allclose(correlation_at(spectrum, delays, duration),
                          fine[3][:, lags])


def test_a_signal_between_samples_is_recovered_without_loss():
    fam = family()
    grid = fam.grid
    b, t, rho = fam.pieces(numpy.zeros(3))
    truth = (37 + 0.5) * grid.delta_t          # half way between samples
    weights = numpy.array([0.8 + 0.3j, -0.2 + 0.6j])
    signal = numpy.tensordot(weights, fam.template(b, t, rho, truth), axes=1)
    power = numpy.sum(numpy.abs(signal) ** 2)
    spectrum = fam.correlation_spectrum(signal, b, t, rho)

    def statistic(delay):
        # the exact Gram matrix of the template, which lies inside the data
        y = fam.template(b, t, rho, delay)
        g = numpy.einsum('acf,bcf->ab', numpy.conj(y), y)
        z = correlation_at(spectrum, [delay], grid.duration)
        return float(twof(z, *[numpy.array([x]) for x in
                               (g[0, 0].real, g[1, 1].real, g[0, 1])])[0][0])
    on_grid = max(statistic(j * grid.delta_t) for j in (36, 37, 38, 39))
    delay, best = refine_peak(statistic, 36 * grid.delta_t,
                              39 * grid.delta_t, tolerance=1e-3)
    assert abs(delay - truth) < 1e-2
    assert best >= on_grid
    # the peak lost (2 pi sigma_f dt / 2)^2 on the grid, nothing refined
    sigma_f = rms_bandwidth(grid.frequencies,
                            numpy.sum(numpy.abs(signal) ** 2, axis=0))
    predicted = (numpy.pi * sigma_f * grid.delta_t) ** 2
    assert 1 - on_grid / best == pytest.approx(predicted, rel=0.15)
    assert statistic(truth) == pytest.approx(power, rel=1e-9)
    assert best == pytest.approx(power, rel=1e-9)


def test_oversampling_bounds_the_loss():
    # the stBBH search: sigma_f 0.640 mHz, a 142.678 s grid, 8.0 per cent
    # lost half a step from the peak
    assert (numpy.pi * 0.640e-3 * 142.678) ** 2 == pytest.approx(0.080,
                                                                  abs=3e-3)
    assert oversampling_for_loss(0.640e-3, 142.678, 1e-3) == 10
    assert oversampling_for_loss(1e-6, 1.0, 1e-3) == 1
