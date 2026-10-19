"""Amplitude and phase marginalisation of pycbc.filter.bstatistic."""
import numpy
import pytest
from scipy import integrate, stats
from scipy.special import i0, i0e, i1

from pycbc.filter.bstatistic import (DistancePowerLaw, draw_amplitude,
                                     draw_phase, log_amplitude_integrand,
                                     log_amplitude_marginal, log_i0)

PRIOR = DistancePowerLaw(0.5, 5000.0, power=2.0)


def reference(x, g, prior):
    """The marginal by adaptive quadrature in ln h, about its maximum."""
    v = numpy.linspace(numpy.log(prior.hmin), numpy.log(prior.hmax), 20001)
    f = log_amplitude_integrand(numpy.exp(v), x, g, prior) + v
    top = f.max()
    breaks = sorted(set([v[0], v[-1], v[numpy.argmax(f)]]))
    total = 0.0
    for a, b in zip(breaks[:-1], breaks[1:]):
        total += integrate.quad(
            lambda u: numpy.exp(log_amplitude_integrand(
                numpy.exp(u), x, g, prior) + u - top),
            a, b, limit=500, epsabs=0, epsrel=1e-11)[0]
    return top + numpy.log(total)


def test_prior_is_normalised():
    total = integrate.quad(lambda h: numpy.exp(PRIOR.log_pdf(h)), PRIOR.hmin,
                           PRIOR.hmax, limit=500, points=[1e-3, 1e-2, 0.1])
    assert total[0] == pytest.approx(1.0, rel=1e-8)


@pytest.mark.parametrize('snr', [0.0, 0.5, 2.0, 8.0, 30.0])
@pytest.mark.parametrize('g', [1e-2, 1.0, 1e3])
def test_marginal_matches_quadrature(snr, g):
    # x = snr sqrt(g) puts the likelihood peak at h = snr / sqrt(g)
    x = snr * numpy.sqrt(g)
    assert log_amplitude_marginal(x, g, PRIOR) == pytest.approx(
        reference(x, g, PRIOR), abs=1e-7)


def test_a_peak_beyond_the_closest_distance_is_integrated_not_clipped():
    # the peak at h = 40 / sqrt(g) = 4 lies above hmax = 2
    x, g = 40.0 * 10, 100.0
    assert log_amplitude_marginal(x, g, PRIOR) == pytest.approx(
        reference(x, g, PRIOR), abs=1e-7)


def test_vectorised_over_lags_and_orientations():
    x = numpy.array([[0.3, 5.0], [20.0, 1.0]])
    g = numpy.array([[1.0, 2.0], [0.5, 1e-3]])
    out = log_amplitude_marginal(x, g, PRIOR)
    assert out.shape == (2, 2)
    for i in range(2):
        for j in range(2):
            assert out[i, j] == pytest.approx(
                log_amplitude_marginal(x[i, j], g[i, j], PRIOR), abs=1e-12)


def test_phase_integral_is_the_bessel_function():
    X, h = 3.0 - 1.5j, 0.8
    numeric = integrate.quad(lambda p: numpy.exp(
        h * numpy.real(numpy.exp(-1j * p) * X)), 0, 2 * numpy.pi)[0]
    assert numpy.log(numeric / (2 * numpy.pi)) == pytest.approx(
        log_i0(h * abs(X)), rel=1e-12)


@pytest.mark.parametrize('x,g', [
    (150.0, 100.0),     # SNR 15, likelihood peak at h = 1.5 inside the prior
    (0.5, 1.0),         # SNR 0.5, prior-dominated
])
def test_amplitude_draws_follow_the_conditional_posterior(x, g):
    rng = numpy.random.default_rng(11)
    draws = numpy.array([draw_amplitude(x, g, PRIOR, rng)
                         for _ in range(3000)])
    grid = numpy.geomspace(PRIOR.hmin, PRIOR.hmax, 400001)
    density = numpy.exp(log_amplitude_integrand(grid, x, g, PRIOR)
                        - log_amplitude_marginal(x, g, PRIOR))
    cdf = numpy.concatenate([[0], numpy.cumsum(
        0.5 * (density[1:] + density[:-1]) * numpy.diff(grid))])
    assert stats.kstest(draws, lambda q: numpy.interp(q, grid, cdf)).pvalue \
        > 1e-3


def test_phase_draws_are_von_mises_about_arg_x():
    rng = numpy.random.default_rng(12)
    X, h = 2.0 * numpy.exp(1.1j), 1.5
    phi = numpy.array([draw_phase(h, X, rng) for _ in range(20000)])
    assert numpy.all((phi >= 0) & (phi < 2 * numpy.pi))
    mean = numpy.mean(numpy.exp(1j * phi))
    assert numpy.angle(mean) == pytest.approx(1.1, abs=0.02)
    # the mean resultant length of von Mises(kappa) is I1 / I0
    kappa = h * abs(X)
    assert abs(mean) == pytest.approx(i1(kappa) / i0(kappa), abs=0.01)


def test_gpu_marginal_matches_the_cpu():
    cupy = pytest.importorskip('cupy')
    try:
        cupy.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip('no usable GPU')
    rng = numpy.random.default_rng(13)
    g = 10 ** rng.uniform(-2, 3, 500)
    x = rng.uniform(0, 30, 500) * numpy.sqrt(g)
    cpu = log_amplitude_marginal(x, g, PRIOR)
    gpu = log_amplitude_marginal(cupy.asarray(x), cupy.asarray(g), PRIOR,
                                 xp=cupy)
    assert numpy.allclose(cupy.asnumpy(gpu), cpu, rtol=0, atol=1e-10)


def test_log_i0_is_continuous_at_the_switch_to_the_series():
    x = numpy.array([699.999999, 700.0, 700.000001, 1e4, 1e8])
    exact = x + numpy.log(i0e(x))
    assert numpy.allclose(log_i0(x), exact, rtol=0, atol=1e-11)


def loud_case(snr=12.0, cos_iota=0.4, psi=0.9, seed=3):
    """z and G of a noise-free signal of the given orientation."""
    from pycbc.tmpltbank.fstatistic_metric import orientation_weights
    rng = numpy.random.default_rng(seed)
    a = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    gram = a @ a.conj().T + numpy.eye(2)
    w = orientation_weights(numpy.arccos(cos_iota), psi)
    h = snr / numpy.sqrt(numpy.real(w.conj() @ gram @ w))
    z = (gram @ w) * h * numpy.exp(0.7j)
    return z.reshape(2, 1), (numpy.array([gram[0, 0].real]),
                             numpy.array([gram[1, 1].real]),
                             numpy.array([gram[0, 1]]))


def brute_orientation(z, gram, prior, n=400):
    from pycbc.filter.bstatistic import OrientationNodes, _project
    from pycbc.tmpltbank.fstatistic_metric import orientation_weights
    c = numpy.linspace(-1, 1, n + 1)
    c = 0.5 * (c[1:] + c[:-1])
    p = (numpy.arange(n) + 0.5) * numpy.pi / n        # over a full pi
    cc, pp = numpy.meshgrid(c, p, indexing='ij')
    v = orientation_weights(numpy.arccos(cc.ravel()), pp.ravel()).T
    x, g = _project(z, *gram, v, numpy)
    lnb = log_amplitude_marginal(x[0], g[0], prior)
    return numpy.log(numpy.mean(numpy.exp(lnb - lnb.max()))) + lnb.max()


def test_orientation_weights_sum_to_one_and_psi_has_period_quarter_turn():
    from pycbc.filter.bstatistic import OrientationNodes, _project
    from pycbc.tmpltbank.fstatistic_metric import orientation_weights
    nodes = OrientationNodes(8, 4)
    assert nodes.weights.sum() == pytest.approx(1.0)
    z, gram = loud_case()
    w0 = orientation_weights(1.0, 0.3)
    w1 = orientation_weights(1.0, 0.3 + numpy.pi / 2)
    assert numpy.allclose(w1, -w0)


@pytest.mark.parametrize('snr', [0.0, 4.0, 12.0])
def test_orientation_marginal_converges_to_the_sphere_integral(snr):
    from pycbc.filter.bstatistic import (OrientationNodes,
                                         log_orientation_marginal)
    z, gram = loud_case(snr)
    exact = brute_orientation(z, gram, PRIOR)
    coarse = log_orientation_marginal(z, *gram, PRIOR,
                                      OrientationNodes(16, 8))[0]
    fine = log_orientation_marginal(z, *gram, PRIOR,
                                    OrientationNodes(64, 32))[0]
    assert fine == pytest.approx(exact, abs=1e-3)
    assert abs(fine - exact) <= abs(coarse - exact) + 1e-6


def test_orientation_draws_follow_the_marginal_posterior():
    from pycbc.filter.bstatistic import (OrientationNodes, _project,
                                         draw_orientation)
    from pycbc.tmpltbank.fstatistic_metric import orientation_weights
    z, gram = loud_case(snr=8.0, cos_iota=0.4)
    nodes = OrientationNodes(16, 8)
    rng = numpy.random.default_rng(21)
    draws = numpy.array([draw_orientation(z[:, 0], gram, PRIOR, nodes, rng)
                         for _ in range(1500)])
    # the cos(iota) marginal on a fine grid
    c = numpy.linspace(-1, 1, 801)
    p = (numpy.arange(200) + 0.5) * (numpy.pi / 2) / 200
    cc, pp = numpy.meshgrid(c, p, indexing='ij')
    v = orientation_weights(numpy.arccos(cc.ravel()), pp.ravel()).T
    x, g = _project(z, *gram, v, numpy)
    lnb = log_amplitude_marginal(x[0], g[0], PRIOR).reshape(cc.shape)
    density = numpy.exp(lnb - lnb.max()).mean(axis=1)
    cdf = numpy.concatenate([[0], numpy.cumsum(
        0.5 * (density[1:] + density[:-1]) * numpy.diff(c))])
    cdf /= cdf[-1]
    assert stats.kstest(draws[:, 0], lambda q: numpy.interp(q, c, cdf)) \
        .pvalue > 1e-3
    assert numpy.all((draws[:, 1] >= 0) & (draws[:, 1] < numpy.pi / 2))


@pytest.mark.parametrize('snr', [4.0, 12.0, 30.0])
def test_doubling_converges_where_a_fixed_grid_does_not(snr):
    from pycbc.filter.bstatistic import (OrientationNodes,
                                         converged_orientation_marginal,
                                         log_orientation_marginal)
    z, gram = loud_case(snr)
    value, nodes = converged_orientation_marginal(z, *gram, PRIOR,
                                                  tolerance=1e-4)
    finest = log_orientation_marginal(z, *gram, PRIOR,
                                      OrientationNodes(128, 64))[0]
    assert value[0] == pytest.approx(finest, abs=2e-4)
    fixed = log_orientation_marginal(z, *gram, PRIOR,
                                     OrientationNodes(16, 8))[0]
    if snr >= 30:
        # the search's 16 x 8 grid loses a third of a nat at SNR 30
        assert finest - fixed > 0.1


def test_pruning_keeps_the_total_to_the_margin():
    from pycbc.filter.bstatistic import (OrientationNodes,
                                         log_orientation_marginal)
    rng = numpy.random.default_rng(31)
    # a loud peak among many arrival times of noise
    n = 300
    z = rng.normal(size=(2, n)) + 1j * rng.normal(size=(2, n))
    zl, gram = loud_case(snr=15.0)
    z[:, 140] = zl[:, 0]
    g00 = numpy.full(n, gram[0][0])
    g11 = numpy.full(n, gram[1][0])
    g01 = numpy.full(n, gram[2][0])
    nodes = OrientationNodes(16, 8)
    full = log_orientation_marginal(z, g00, g11, g01, PRIOR, nodes)
    pruned = log_orientation_marginal(z, g00, g11, g01, PRIOR, nodes,
                                      margin=20.0)
    total = lambda v: numpy.log(numpy.sum(numpy.exp(v - v.max()))) + v.max()
    assert abs(total(pruned) - total(full)) < numpy.exp(-20.0)
    assert numpy.isneginf(pruned).sum() > n // 2
    assert pruned[140] == pytest.approx(full[140], abs=1e-12)
