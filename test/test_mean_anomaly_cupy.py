"""Optional GPU regressions for coherent two-polarisation mode statistics."""
import numpy as np
import pytest

from pycbc.filter.mean_anomaly import mean_anomaly_components, mean_anomaly_twof

cp = pytest.importorskip('cupy')
try:
    available = cp.cuda.runtime.getDeviceCount() > 0
except cp.cuda.runtime.CUDARuntimeError:
    available = False
pytestmark = pytest.mark.skipif(not available, reason='CUDA device unavailable')


@pytest.mark.parametrize('shared', [True, False])
@pytest.mark.parametrize('rank', [0, 1, 2])
def test_gpu_matches_cpu_with_full_cross_mode_gram(shared, rank):
    rng = np.random.default_rng(81)
    h = rng.normal(size=(4, 2, 51)) + 1j*rng.normal(size=(4, 2, 51))
    if rank == 1:
        h[:, 1] = 1j*h[:, 0]
    elif rank == 0:
        h *= 0
    data = rng.normal(size=(51, 2, 3)) + 1j*rng.normal(size=(51, 2, 3))
    weights = np.ones((51, 3)) if shared else rng.uniform(.1, 2, (51, 3))
    z = np.einsum('kaf,fs,frs->kars', h.conj(), weights, data)
    g = np.einsum('kaf,lbf,fs->kalbs', h.conj(), h, weights)
    if shared:
        g = g[..., 0]
    c = np.array([-2, 0, 0, 3])
    gpu_z, gpu_g = cp.asarray(z), cp.asarray(g)
    want = mean_anomaly_components(z, g, c, 32)
    got = mean_anomaly_components(gpu_z, gpu_g, c, 32, xp=cp)
    for a, b in zip(want, got):
        assert isinstance(b, cp.ndarray)
        np.testing.assert_allclose(b.get(), a, atol=2e-12, rtol=2e-12)
    want, angles = mean_anomaly_twof(z, g, c, 32)
    got, gpu_angles = mean_anomaly_twof(gpu_z, gpu_g, c, 32, xp=cp)
    assert isinstance(got, cp.ndarray)
    np.testing.assert_allclose(got.get(), want, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(gpu_angles.get(), angles)


@pytest.mark.parametrize('case', ['nan', 'nonhermitian', 'indefinite'])
def test_gpu_rejects_invalid_inputs(case):
    z = cp.ones((1, 2), dtype=complex)
    g = cp.eye(2, dtype=complex).reshape(1, 2, 1, 2)
    if case == 'nan':
        z[0, 0] = cp.nan
    elif case == 'nonhermitian':
        g[0, 0, 0, 1] = 1j
    else:
        g[0, 0, 0, 0] = -1
    with pytest.raises(ValueError):
        mean_anomaly_twof(z, g, [0], xp=cp)
