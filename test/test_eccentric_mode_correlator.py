"""Direct mode-clock and Gram references for streamed arrival-time evaluation."""
import numpy as np
import pytest

from pycbc.filter.eccentric_mode_correlator import EccentricModeCorrelator
from pycbc.filter.mean_anomaly import mean_anomaly_twof
from pycbc.tdi.ssb_yearly_harmonics import ORBITAL_ANGULAR_FREQUENCY as OMEGA


def inputs():
    rng = np.random.default_rng(821)
    f = np.linspace(.01, .03, 73)
    b = rng.normal(size=(3, len(f))) + 1j*rng.normal(size=(3, len(f)))
    t = np.array([np.linspace(-5e6, 2e7, len(f)) + 2e6*k for k in range(3)])
    rho = rng.normal(size=(3, 2, 2, len(f))) + 1j*rng.normal(size=(3, 2, 2, len(f)))
    orders = np.array([-1, 0, 1])
    data = rng.normal(size=(2, len(f))) + 1j*rng.normal(size=(2, len(f)))
    delays = np.array([-2e6, 0., 1e6, 7e6])
    return f, b, t, rho, orders, data, delays


def direct(f, b, t, rho, orders, delays, window):
    values = []
    for delta in delays:
        h = np.zeros((len(b), 2, 2, len(f)), complex)
        for mode in range(len(b)):
            for i, order in enumerate(orders):
                h[mode] += rho[i]*np.exp(1j*order*OMEGA*(t[mode]+delta))
            h[mode] *= b[mode]*np.exp((-2j*np.pi*delta)*f)
            if window:
                h[mode] *= (t[mode]+delta >= window[0]) & (t[mode]+delta < window[1])
        values.append(h)
    return np.array(values)


@pytest.mark.parametrize('window', [None, (0., 2e7)])
def test_streaming_preserves_separate_clocks_cross_modes_and_coherence(window):
    f, b, t, rho, orders, data, delays = inputs()
    engine = EccentricModeCorrelator(f, b, t, rho, orders, time_window=window)
    z, g = engine.at(data, delays, lag_chunk=2, frequency_chunk=11)
    h = direct(f, b, t, rho, orders, delays, window)
    want_z = np.einsum('lkacf,cf->kal', h.conj(), data)
    want_g = np.einsum('lkacf,lmbcf->kambl', h.conj(), h)
    np.testing.assert_allclose(z, want_z, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(g, want_g, rtol=2e-12, atol=2e-12)
    whole_z, whole_g = engine.at(data, delays, lag_chunk=4, frequency_chunk=73)
    np.testing.assert_allclose(z, whole_z, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(g, whole_g, rtol=2e-12, atol=2e-12)
    c = np.array([-1, 0, 2])
    stat, angles = mean_anomaly_twof(z, g, c, 16)
    for j, angle in enumerate(angles):
        coherent = np.einsum('k,lkacf->lacf', np.exp(1j*c*angle), h)
        for l, template in enumerate(coherent):
            zz = np.einsum('acf,cf->a', template.conj(), data)
            gg = np.einsum('acf,bcf->ab', template.conj(), template)
            want = np.vdot(zz, np.linalg.solve(gg, zz)).real
            np.testing.assert_allclose(stat[j, l], want, atol=2e-12)


def test_zero_observation_coverage_is_zero_and_output_memory_is_bounded():
    f, b, t, rho, orders, data, delays = inputs()
    engine = EccentricModeCorrelator(f, b, t, rho, orders, time_window=(1e10, 2e10))
    z, g = engine.at(data, delays)
    np.testing.assert_array_equal(z, 0)
    np.testing.assert_array_equal(g, 0)
    with pytest.raises(MemoryError):
        engine.at(data, delays, max_output_bytes=1)


@pytest.mark.parametrize('bad', [0, -1, 1.5, True])
def test_invalid_chunk_sizes_are_refused(bad):
    f, b, t, rho, orders, data, delays = inputs()
    engine = EccentricModeCorrelator(f, b, t, rho, orders)
    with pytest.raises(ValueError):
        engine.at(data, delays, frequency_chunk=bad)



def test_cupy_streaming_matches_cpu():
    cp = pytest.importorskip('cupy')
    try:
        available = cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        available = False
    if not available:
        pytest.skip('CUDA device unavailable')
    f, b, t, rho, orders, data, delays = inputs()
    cpu = EccentricModeCorrelator(f, b, t, rho, orders, time_window=(0., 2e7))
    gpu = EccentricModeCorrelator(f, b, t, rho, orders, time_window=(0., 2e7), xp=cp)
    cz, cg = cpu.at(data, delays, lag_chunk=2, frequency_chunk=11)
    gz, gg = gpu.at(cp.asarray(data), delays, lag_chunk=3, frequency_chunk=17)
    assert isinstance(gz, cp.ndarray)
    np.testing.assert_allclose(gz.get(), cz, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(gg.get(), cg, atol=1e-10, rtol=1e-10)
