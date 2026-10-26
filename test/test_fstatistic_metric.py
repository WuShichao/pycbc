"""The F-statistic metric against the mismatch it approximates."""
import numpy
import pytest
from scipy.optimize import minimize_scalar

from pycbc.tmpltbank.fstatistic_metric import (MetricChart,
                                               fstatistic_metric,
                                               orientation_weights)

F = numpy.linspace(0.010, 0.012, 4000)
U = (F - F.mean()) / (F[-1] - F[0])
RNG = numpy.random.default_rng(7)
# a smooth complex response per basis and channel
RESPONSE = numpy.array([[c[0] + c[1] * U + 1j * (c[2] + c[3] * U ** 2)
                         for c in RNG.normal(size=(2, 4))]
                        for _ in range(2)])
POWERS = numpy.array([2, 3, 4])


def bases(theta):
    """Two bases, two channels; phase and log amplitude linear in theta."""
    phase = 2e3 * (theta @ U[None, :] ** POWERS[:, None])
    logamp = 0.3 * theta[0] * U
    return RESPONSE * numpy.exp(logamp + 1j * phase)


def derivatives(theta):
    y = bases(theta)
    out = []
    for i, p in enumerate(POWERS):
        d = 2e3j * U ** p + (0.3 * U if i == 0 else 0)
        out.append(y * d)
    return numpy.array(out)


def captured(signal, templates):
    """Largest fraction of |s|^2 in the real span of the templates."""
    a = numpy.array([numpy.concatenate([t.real.ravel(), t.imag.ravel()])
                     for t in templates]).T
    s = numpy.concatenate([signal.real.ravel(), signal.imag.ravel()])
    q, _ = numpy.linalg.qr(a)
    return numpy.sum((q.T @ s) ** 2) / numpy.sum(s ** 2)


def mismatch(theta, delta, weights, project):
    signal = numpy.tensordot(weights, bases(theta), axes=1)
    shifted = bases(theta + delta)

    def loss(t):
        y = shifted * numpy.exp(-2j * numpy.pi * F * t)
        if project == 'fstat':
            templates = [y[0], 1j * y[0], y[1], 1j * y[1]]
        else:
            s = numpy.tensordot(weights, y, axes=1)
            templates = [s, 1j * s]
        return 1 - numpy.sqrt(captured(signal, templates))
    return minimize_scalar(loss, bounds=(-200, 200), method='bounded',
                           options={'xatol': 1e-6}).fun


@pytest.mark.parametrize('project', ['fstat', 'orientation'])
def test_metric_predicts_the_maximised_mismatch(project):
    theta = numpy.array([0.01, -0.02, 0.005])
    weights = orientation_weights(0.7, 0.4)
    metric = fstatistic_metric(bases(theta), derivatives(theta), weights, F,
                               project=project)
    chart = MetricChart(metric)
    for direction in RNG.normal(size=(5, 3)):
        # a displacement of mismatch 1e-3 under the metric
        eta = direction / numpy.linalg.norm(direction) * numpy.sqrt(1e-3)
        delta = chart.from_chart(eta)
        assert chart.mismatch(delta) == pytest.approx(1e-3)
        assert mismatch(theta, delta, weights, project) \
            == pytest.approx(1e-3, rel=0.03)


def test_fstat_metric_is_below_the_fixed_orientation_metric():
    theta = numpy.zeros(3)
    weights = orientation_weights(1.1, 0.2)
    free = fstatistic_metric(bases(theta), derivatives(theta), weights, F)
    fixed = fstatistic_metric(bases(theta), derivatives(theta), weights, F,
                              project='orientation')
    assert numpy.all(numpy.linalg.eigvalsh(fixed - free) > -1e-12)


def test_chart_round_trip_and_ordering():
    a = RNG.normal(size=(4, 4))
    chart = MetricChart(a @ a.T + 0.1 * numpy.eye(4))
    dxi = RNG.normal(size=(10, 4))
    assert numpy.allclose(chart.from_chart(chart.to_chart(dxi)), dxi)
    assert numpy.allclose(numpy.sum(chart.to_chart(dxi) ** 2, axis=1),
                          chart.mismatch(dxi))
    assert numpy.all(numpy.diff(chart.eigenvalues) <= 0)


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError):
        MetricChart([[1.0, 0.0], [0.0, -1.0]])
    with pytest.raises(ValueError):
        fstatistic_metric(bases(numpy.zeros(3)), derivatives(numpy.zeros(3)),
                          orientation_weights(0, 0), F, project='sky')


def test_a_chart_takes_a_direction_the_metric_does_not_measure():
    """An eigenvalue within roundoff of zero, as a degenerate direction gives,
    is floored; a negative one beyond roundoff is refused."""
    import numpy
    import pytest
    from pycbc.tmpltbank.fstatistic_metric import MetricChart
    V = numpy.linalg.qr(numpy.random.default_rng(0).normal(size=(3, 3)))[0]
    chart = MetricChart(V @ numpy.diag([4.0, 1.0, -2e-14]) @ V.T)
    assert chart.eigenvalues.min() > 0
    with pytest.raises(ValueError):
        MetricChart(V @ numpy.diag([4.0, 1.0, -1e-6]) @ V.T)
