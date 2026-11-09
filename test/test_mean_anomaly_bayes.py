"""Physical orientation/amplitude marginal from mode FFTs equals direct sums."""
import numpy as np
import pytest

from pycbc.filter.bstatistic import DistancePowerLaw, OrientationNodes, log_orientation_marginal
from pycbc.filter.mean_anomaly import mean_anomaly_log_bayes


@pytest.mark.parametrize('shared', [True, False])
def test_tied_mode_bayes_matches_direct_coherent_templates(shared):
    rng = np.random.default_rng(20261109)
    h = (rng.normal(size=(3, 2, 31))+1j*rng.normal(size=(3, 2, 31)))/7
    data = rng.normal(size=(31, 4))+1j*rng.normal(size=(31, 4))
    weight = np.ones((31, 4)) if shared else rng.uniform(.3, 2, (31, 4))
    z = np.einsum('kaf,fs,fs->kas', h.conj(), data, weight)
    g = np.einsum('kaf,lbf,fs->kalbs', h.conj(), h, weight)
    if shared:g = g[..., 0]
    c = np.array([-1, 0, 2])
    prior, nodes = DistancePowerLaw(.5, 10), OrientationNodes(4, 4)
    got, by_angle, angles = mean_anomaly_log_bayes(z, g, c, prior, nodes,
        n_l0=16, noise_variance=1.7, order=48)
    direct = []
    for angle in angles:
        ht = np.einsum('k,kaf->af', np.exp(1j*c*angle), h)
        zz = np.einsum('af,fs,fs->as', ht.conj(), data, weight)/1.7
        gg = np.einsum('af,bf,fs->abs', ht.conj(), ht, weight)/1.7
        direct.append(log_orientation_marginal(zz, gg[0, 0].real,
            gg[1, 1].real, gg[0, 1], prior, nodes, order=48))
    direct = np.array(direct)
    np.testing.assert_allclose(by_angle, direct, atol=2e-12, rtol=2e-12)
    top = direct.max(axis=0)
    want = top+np.log(np.exp(direct-top).mean(axis=0))
    np.testing.assert_allclose(got, want, atol=2e-12)
    shifted, _, _ = mean_anomaly_log_bayes(z, g, c+3, prior, nodes,
        n_l0=16, noise_variance=1.7, order=48)
    np.testing.assert_allclose(shifted, got, atol=2e-12)


def test_zero_coverage_bayes_factor_is_one_and_scalar_shape_is_preserved():
    prior, nodes = DistancePowerLaw(.5, 10), OrientationNodes(4, 4)
    value, at, _ = mean_anomaly_log_bayes(np.zeros((1, 2), complex),
        np.zeros((1, 2, 1, 2), complex), [0], prior, nodes)
    assert value.shape == ()
    np.testing.assert_array_equal(at, 0)
    assert value == 0


@pytest.mark.parametrize('variance', [0., -1., np.nan])
def test_invalid_noise_variance_is_refused(variance):
    with pytest.raises(ValueError, match='noise_variance'):
        mean_anomaly_log_bayes(np.ones((1, 2), complex),
            np.eye(2).reshape(1, 2, 1, 2), [0], DistancePowerLaw(.5, 10),
            OrientationNodes(4, 4), noise_variance=variance)


def test_cupy_bayes_matches_cpu():
    cp = pytest.importorskip('cupy')
    try:
        available = cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        available = False
    if not available:pytest.skip('CUDA device unavailable')
    z = np.array([[1+2j, 3-.5j], [.2-.1j, 2+1j]])
    g = np.eye(4, dtype=complex).reshape(2, 2, 2, 2)
    prior, nodes = DistancePowerLaw(.5, 10), OrientationNodes(4, 4)
    cpu = mean_anomaly_log_bayes(z, g, [-1, 0], prior, nodes, 8, order=48)
    gpu = mean_anomaly_log_bayes(cp.asarray(z), cp.asarray(g), [-1, 0],
                                prior, nodes, 8, order=48, xp=cp)
    for a, b in zip(cpu, gpu):
        np.testing.assert_allclose(a, b.get(), atol=2e-11, rtol=2e-11)
