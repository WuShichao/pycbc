"""Dense Fourier controls for the time-periodic response formula."""
import numpy as np
import pytest

from pycbc.tdi.harmonic_waveform import harmonic_spectrum


@pytest.mark.parametrize('periods', [1, 2, 4])
def test_frequency_shifts_equal_direct_time_periodic_response(periods):
    n, orders = 128, np.array([-3, 0, 2])
    rng = np.random.default_rng(20267240)
    source = rng.normal(size=n) + 1j * rng.normal(size=n)
    source[:16] = source[-16:] = 0.
    responses = rng.normal(size=(3, 2, 2, n)) + 1j * rng.normal(size=(3, 2, 2, n))
    t = np.arange(n) / n
    direct_time = sum(np.fft.ifft(r * source, axis=-1) *
                      np.exp(2j * np.pi * k * periods * t)
                      for k, r in zip(orders, responses))
    expected = np.fft.fft(direct_time, axis=-1)
    np.testing.assert_allclose(harmonic_spectrum(source, responses, orders, periods),
                               expected, rtol=1e-10, atol=1e-10)


def test_band_edges_do_not_wrap_and_invalid_shapes_fail():
    source = np.zeros(10, complex)
    source[-1] = 1.
    assert not np.any(harmonic_spectrum(source, np.ones((1, 10)), [1]))
    for orders in ([1., 2.], [1, 1], []):
        with pytest.raises(ValueError):
            harmonic_spectrum(source, np.ones((len(orders), 10)), orders)
    with pytest.raises(ValueError):
        harmonic_spectrum(source, np.ones((1, 11)), [0])
    with pytest.raises(ValueError):
        harmonic_spectrum(source, np.ones((1, 10)), [0], periods=0)
    with pytest.raises(ValueError):
        harmonic_spectrum(source * np.nan, np.ones((1, 10)), [0])


def test_stationary_time_linearization_loses_chirp_phase_curvature():
    n, shift, curvature = 128, 3, .01
    f = np.arange(n)
    source = np.exp(-.5j * curvature * f**2)
    exact = harmonic_spectrum(source, np.ones((1, n)), [shift])
    linearized = source * np.exp(1j * curvature * f * shift)
    # The missing term is a predictable curvature phase, not FFT roundoff.
    np.testing.assert_allclose(exact[shift:] / linearized[shift:],
                               np.exp(-.5j * curvature * shift**2), atol=1e-10)
