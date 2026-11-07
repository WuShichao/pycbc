"""pycbc.filter.mean_anomaly: the statistic at every initial mean anomaly from
one transform equals direct evaluation."""
import numpy
import pytest

from pycbc.filter.mean_anomaly import mean_anomaly_statistic, mode_offsets
from pycbc.filter.mean_anomaly import (mean_anomaly_components,
                                       mean_anomaly_twof)


def _modes(rng, K=5, n=400):
    h = rng.normal(size=(K, n)) + 1j * rng.normal(size=(K, n))
    h *= numpy.linspace(1.0, 0.1, K)[:, None]
    return h


def _direct(h, d, c, l0):
    out = []
    for t in l0:
        ht = numpy.sum(numpy.exp(1j * c * t)[:, None] * h, axis=0)
        out.append(abs(numpy.vdot(ht, d)) ** 2 / numpy.vdot(ht, ht).real)
    return numpy.array(out)


@pytest.mark.parametrize('c', [[0, 1, 2, -1, 1], [0, 4, 1, 2, 3], [-2, 0, 0, 1, 5]])
def test_transform_equals_direct(c):
    rng = numpy.random.default_rng(3)
    c = numpy.array(c)
    h = _modes(rng)
    d = rng.normal(size=h.shape[1]) + 1j * rng.normal(size=h.shape[1])
    z = numpy.array([numpy.vdot(hk, d) for hk in h])
    gram = numpy.array([[numpy.vdot(a, b) for b in h] for a in h])
    stat, l0 = mean_anomaly_statistic(z, gram, c)
    numpy.testing.assert_allclose(stat, _direct(h, d, c, l0), rtol=1e-10)


def test_many_samples_share_the_gram():
    rng = numpy.random.default_rng(4)
    c = numpy.array([0, 1, 2, -1])
    h = _modes(rng, K=4)
    gram = numpy.array([[numpy.vdot(a, b) for b in h] for a in h])
    ds = [rng.normal(size=h.shape[1]) + 1j * rng.normal(size=h.shape[1]) for _ in range(3)]
    z = numpy.array([[numpy.vdot(hk, d) for d in ds] for hk in h])
    stat, l0 = mean_anomaly_statistic(z, gram, c, n_l0=16)
    for j, d in enumerate(ds):
        numpy.testing.assert_allclose(stat[:, j], _direct(h, d, c, l0), rtol=1e-10)


def test_mode_offsets_of_pyefpehm_keys():
    assert list(mode_offsets([(2, 2, 2), (2, 2, 3), (2, 0, 1), (2, -2, 2)])) == [0, 1, 1, 4]


def test_too_few_l0_values_are_refused():
    with pytest.raises(ValueError):
        mean_anomaly_statistic(numpy.ones(2), numpy.eye(2), numpy.array([0, 3]), n_l0=4)


@pytest.mark.parametrize('shared', [True, False])
def test_two_polarisations_match_direct_coherent_templates(shared):
    rng = numpy.random.default_rng(19)
    c = numpy.array([-2, 0, 0, 3])
    h = rng.normal(size=(4, 2, 43)) + 1j * rng.normal(size=(4, 2, 43))
    data = rng.normal(size=(43, 3)) + 1j * rng.normal(size=(43, 3))
    # The imaginary off-diagonal Gram and overlapping modes expose conjugation
    # errors that orthogonal modes and diagonal Grams would hide.
    weights = numpy.ones((43, 3)) if shared else rng.uniform(.1, 2, (43, 3))
    z = numpy.einsum('kaf,fs,fs->kas', h.conj(), weights, data)
    g = numpy.einsum('kaf,lbf,fs->kalbs', h.conj(), h, weights)
    if shared:
        g = g[..., 0]
    zl, gl, l0 = mean_anomaly_components(z, g, c, 32)
    stat, angles = mean_anomaly_twof(z, g, c, 32)
    numpy.testing.assert_array_equal(angles, l0)
    for i, angle in enumerate(l0):
        ht = numpy.einsum('k,kaf->af', numpy.exp(1j * c * angle), h)
        for j in range(3):
            gj = numpy.einsum('af,bf,f->ab', ht.conj(), ht, weights[:, j])
            zj = ht.conj() @ (weights[:, j] * data[:, j])
            numpy.testing.assert_allclose(zl[i, :, j], zj, atol=1e-12)
            numpy.testing.assert_allclose(gl[i] if shared else gl[i, :, :, j],
                                          gj, atol=1e-12)
            want = numpy.vdot(zj, numpy.linalg.solve(gj, zj)).real
            numpy.testing.assert_allclose(stat[i, j], want, rtol=1e-12)
    single, _ = mean_anomaly_twof(z[..., 0], g if shared else g[..., 0], c, 32)
    numpy.testing.assert_allclose(single, stat[:, 0])


def test_rank_deficiency_and_zero_template_are_finite():
    rng = numpy.random.default_rng(20)
    h = _modes(rng, K=3, n=50)
    h = numpy.stack([h, 1j * h], axis=1)
    d = rng.normal(size=50) + 1j * rng.normal(size=50)
    z = numpy.einsum('kaf,f->ka', h.conj(), d)
    g = numpy.einsum('kaf,lbf->kalb', h.conj(), h)
    c = numpy.array([-1, 0, 2])
    got, l0 = mean_anomaly_twof(z, g, c, 16)
    want = _direct(h[:, 0], d, c, l0)
    numpy.testing.assert_allclose(got, want, rtol=1e-12)
    zero, _ = mean_anomaly_twof(z * 0, g * 0, c)
    numpy.testing.assert_array_equal(zero, 0)


def test_multiple_sample_axes_and_broadcast_gram():
    z = numpy.ones((1, 2, 3, 4), complex)
    g = numpy.zeros((1, 2, 1, 2, 4), complex)
    g[0, 0, 0, 0] = numpy.arange(1, 5)
    g[0, 1, 0, 1] = 2
    got, _ = mean_anomaly_twof(z, g, [0])
    numpy.testing.assert_allclose(got, numpy.broadcast_to(
        1 / numpy.arange(1, 5) + .5, (2, 3, 4)))


@pytest.mark.parametrize('case', ['fractional', 'empty', 'shape', 'samples',
                                  'nan', 'grid', 'fractional_grid', 'rcond',
                                  'indefinite', 'nonhermitian'])
def test_twof_invalid_inputs(case):
    z = numpy.ones((1, 2), complex)
    g = numpy.eye(2, dtype=complex).reshape(1, 2, 1, 2)
    c, kwargs = [0], {}
    if case == 'fractional':
        c = [0.5]
    elif case == 'empty':
        z, g, c = z[:0], g[:0], []
    elif case == 'shape':
        g = numpy.eye(2)
    elif case == 'samples':
        g = numpy.repeat(g[..., None], 3, axis=-1)
    elif case == 'nan':
        z[0, 0] = numpy.nan
    elif case == 'grid':
        kwargs['n_l0'] = 0
    elif case == 'fractional_grid':
        kwargs['n_l0'] = 4.5
    elif case == 'rcond':
        kwargs['rcond'] = -1
    elif case == 'indefinite':
        g[0, 0, 0, 0] = -1
    elif case == 'nonhermitian':
        g[0, 0, 0, 1] = 1j
    with pytest.raises(ValueError):
        mean_anomaly_twof(z, g, c, **kwargs)


def test_coherent_modes_do_not_gain_free_amplitudes():
    # Opposite sidebands cancel at l0=0. Adding individual mode powers would
    # report a signal even though the physical template vanishes there.
    h = numpy.array([[[1, 0], [0, 1]], [[-1, 0], [0, -1]]], complex)
    d = numpy.array([1, 1j])
    z = numpy.einsum('kaf,f->ka', h.conj(), d)
    g = numpy.einsum('kaf,lbf->kalb', h.conj(), h)
    got, _ = mean_anomaly_twof(z, g, [0, 1], 16)
    assert got[0] == 0
    numpy.testing.assert_allclose(got[1:], 2, rtol=1e-12)
