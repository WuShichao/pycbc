"""Exact harmonic-shift correlations against independent physical projection."""
import numpy as np
import pytest

pytest.importorskip('finufft')

from pycbc.filter.detector_harmonic import DetectorHarmonicLikelihood
from pycbc.tdi.detector_projection import DetectorBandProjector
from pycbc.tdi.harmonic_waveform import harmonic_correlation_spectrum, harmonic_spectrum
from pycbc.tdi.ssb_yearly_harmonics import SSBGrid
from test_ssb_yearly_harmonics import _StillOrbit, ORBIT, SKY, DAY


@pytest.mark.parametrize('periods', [1, 2, 4])
def test_exact_harmonic_correlation_spectrum_at_every_periodic_lag(periods):
    rng = np.random.default_rng(20267244)
    n, orders, first = 48, np.array([-3, 0, 2]), .0173
    source = rng.normal(size=n) + 1j * rng.normal(size=n)
    rho = rng.normal(size=(3, 2, 2, n)) + 1j * rng.normal(size=(3, 2, 2, n))
    data = rng.normal(size=(2, n)) + 1j * rng.normal(size=(2, n))
    spectrum = harmonic_correlation_spectrum(source, rho, orders, data, periods)
    times = np.arange(n) / n
    fast = n * np.fft.ifft(spectrum, axis=-1) * np.exp(2j * np.pi * first * times)
    f = first + np.arange(n)
    direct = []
    for delay in times:
        y = harmonic_spectrum(source * np.exp(-2j * np.pi * f * delay), rho, orders, periods)
        direct.append(np.einsum('acf,cf->a', y.conj(), data))
    np.testing.assert_allclose(fast, np.array(direct).T, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize('orbit', [_StillOrbit(), ORBIT])
@pytest.mark.parametrize('oversampling,signed', [(1, False), (3, True)])
def test_detector_adjoint_fft_equals_whitened_original_band_correlations(orbit, oversampling, signed):
    rng = np.random.default_rng(20267245)
    duration = 2 * DAY
    grid = SSBGrid(1e6, 1e6 + duration, .015, 12, 1 / duration,
                   periods=2, period=duration, guard=4)
    projection = DetectorBandProjector(grid, orbit, *SKY, oversampling=4)
    source = rng.normal(size=grid.size) + 1j * rng.normal(size=grid.size)
    orders = np.array([-2, 0, 1])
    rho = rng.normal(size=(3, 2, 2, grid.size)) + 1j * rng.normal(size=(3, 2, 2, grid.size))
    data = rng.normal(size=(2, grid.data_nbins)) + 1j * rng.normal(size=(2, grid.data_nbins))
    weight = rng.uniform(.3, 2., size=data.shape)
    likelihood = DetectorHarmonicLikelihood(projection, data, weight)
    fast = likelihood.correlations(source, rho, orders, periods=2,
                                    oversampling=oversampling, signed=signed)
    delays = likelihood.delays(oversampling, signed)
    direct, gram = likelihood.statistics(source, rho, orders, delays, periods=2)
    np.testing.assert_allclose(fast, direct.T, rtol=1e-9, atol=1e-9)
    assert np.linalg.eigvalsh(gram).min() > 0
    # Moving the PSD weights to the source before retiming is not the same.
    wrong = projection.adjoint(data)
    assert np.linalg.norm(likelihood.adjoint_data - wrong) > 1e-3


def test_invalid_original_band_and_delay_inputs_are_rejected():
    grid = SSBGrid(0., DAY, .015, 8, 1 / DAY, periods=1, period=DAY)
    projection = DetectorBandProjector(grid, _StillOrbit(), *SKY)
    for weight in (np.ones((2, 7)), np.zeros((2, 8)), np.full((2, 8), np.nan)):
        with pytest.raises(ValueError):
            DetectorHarmonicLikelihood(projection, np.ones((2, 8)), weight)
    likelihood = DetectorHarmonicLikelihood(projection, np.ones((2, 8)), np.ones((2, 8)))
    with pytest.raises(ValueError):
        likelihood.statistics(np.ones(grid.size), np.ones((1, 2, 2, grid.size)), [0], [np.nan])
    with pytest.raises(ValueError):
        likelihood.delays(0)
    with pytest.raises(MemoryError):
        likelihood.correlations(np.ones(grid.size), np.ones((1, 2, 2, grid.size)), [0], max_output_bytes=1)


def test_cached_adjoint_is_consistent_with_owned_data_and_grid_periods():
    grid = SSBGrid(0., DAY, .015, 8, 1 / DAY, periods=2, period=DAY)
    projection = DetectorBandProjector(grid, _StillOrbit(), *SKY)
    data, weight = np.ones((2, 8), complex), np.ones((2, 8))
    likelihood = DetectorHarmonicLikelihood(projection, data, weight)
    data[:] = 0.
    weight[:] = 0.
    assert np.all(likelihood.data == 1) and np.all(likelihood.weight == 1)
    with pytest.raises(ValueError):
        likelihood.data[0, 0] = 0.
    source, rho = np.ones(grid.size), np.ones((1, 2, 2, grid.size))
    np.testing.assert_allclose(likelihood.correlations(source, rho, [0]),
                               likelihood.correlations(source, rho, [0], periods=2))
    with pytest.raises(ValueError):
        likelihood.correlations(source, rho, [0], periods=1)
