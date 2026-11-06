"""pycbc.filter.mean_anomaly: the statistic at every initial mean anomaly from
one transform equals direct evaluation."""
import numpy
import pytest

from pycbc.filter.mean_anomaly import mean_anomaly_statistic, mode_offsets


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
