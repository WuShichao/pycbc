"""The bstatistic_surrogate inference model on a small synthetic family."""
import warnings

import numpy
import pytest

from pycbc.filter.bstatistic import (DistancePowerLaw, OrientationNodes,
                                     log_orientation_marginal)
from pycbc.inference.models.bstatistic_surrogate import BStatisticSurrogate
from pycbc.tdi.linear_exponent_surrogate import (DailyGram,
                                                 LinearExponentSurrogate)
from pycbc.tdi.ssb_yearly_harmonics import SSBGrid
from pycbc.tmpltbank.fstatistic_metric import orientation_weights

DAY = 86400.0
PRIOR = DistancePowerLaw(0.05, 50.0, power=2.0)
TRUTH = dict(delta_tc=3.3e4, inclination=1.0, polarization=0.4,
             distance=0.5, signal_phase=2.0)


def family(days=60.0, seed=2):
    rng = numpy.random.default_rng(seed)
    grid = SSBGrid(0.0, days * DAY, 0.015, int(20 * days),
                   1 / (days * DAY), periods=1)
    f = grid.frequencies
    u = (f - f[0]) / (f[-1] - f[0])
    t0 = (0.2 + 0.6 * u) * days * DAY
    phi = numpy.concatenate([[0], numpy.cumsum(
        0.5 * (t0[1:] + t0[:-1]) * numpy.diff(f))])
    amp = numpy.exp(-0.5 * ((u - 0.4) / 0.08) ** 2)
    coarse = numpy.linspace(f[0], f[-1], 10)
    shape = (16, 2, 2, 10)
    # independent bases, so the orientation is measurable
    R0 = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    L = 0.05 * (rng.normal(size=(2,) + shape) + 0j)
    # one intrinsic coordinate: a chirp-like phase and a time slope
    D = 2j * numpy.pi * 30 * (u - 0.4) ** 2
    T = 3e3 * (u - 0.4)
    fam = LinearExponentSurrogate(grid, coarse, R0, L,
                                  amp * numpy.exp(-2j * numpy.pi * phi), t0,
                                  D[None, :], T[None, :], kmax=3)
    return fam, DailyGram(fam)


def inject(fam, xi, noise=1.0, seed=4):
    b, t, rho = fam.pieces(xi)
    w = orientation_weights(TRUTH['inclination'], TRUTH['polarization'])
    y = fam.template(b, t, rho, TRUTH['delta_tc'])
    h = PRIOR.scale / TRUTH['distance']
    signal = h * numpy.exp(1j * TRUTH['signal_phase']) \
        * numpy.tensordot(w, y, axes=1)
    rng = numpy.random.default_rng(seed)
    n = numpy.sqrt(noise) * (rng.normal(size=signal.shape)
                             + 1j * rng.normal(size=signal.shape))
    return signal + n, signal


def model(fam, daily, data, noise=1.0, window=(-2e5, 2e5), oversampling=4):
    reference = dict(x=0.0, eclipticlongitude=1.0, eclipticlatitude=0.2)
    scales = dict(x=1.0, eclipticlongitude=1e-3, eclipticlatitude=1e-3)
    return BStatisticSurrogate(['x'], fam, daily, data, noise, reference,
                               scales, PRIOR, window,
                               oversampling=oversampling,
                               orientation_nodes=(16, 8))


def test_likelihood_peaks_at_the_injected_intrinsic_point():
    fam, daily = family()
    data, signal = inject(fam, numpy.array([0.3, 0.0, 0.0]))
    m = model(fam, daily, data)
    values = {}
    for x in (0.0, 0.3, 0.6):
        m.update(x=x)
        values[x] = m.loglikelihood
    assert values[0.3] > values[0.0] and values[0.3] > values[0.6]
    # a loud signal: ln B is SNR^2 / 2, give or take the noise's O(SNR)
    # fluctuation and the Occam factors of the marginalised parameters
    snr2 = numpy.sum(numpy.abs(signal) ** 2)
    assert abs(values[0.3] - 0.5 * snr2) < 5 * numpy.sqrt(snr2) + 50


def test_time_samples_match_exact_inner_products():
    fam, daily = family()
    data, _ = inject(fam, numpy.zeros(3))
    m = model(fam, daily, data, noise=2.0, window=(2.8e4, 3.8e4))
    z, g, lnb, _ = m._time_samples(dict(x=0.0))
    b, t, rho = fam.pieces(numpy.zeros(3))
    nodes = OrientationNodes(16, 8)
    for k in (0, len(m.delays) // 2, len(m.delays) - 1):
        y = fam.template(b, t, rho, m.delays[k])
        zk = numpy.einsum('acf,cf->a', numpy.conj(y), data) / 2.0
        assert numpy.allclose(z[:, k], zk, rtol=1e-7)
        gk = numpy.einsum('acf,bcf->ab', numpy.conj(y), y) / 2.0
        exact = log_orientation_marginal(
            zk.reshape(2, 1), numpy.array([gk[0, 0].real]),
            numpy.array([gk[1, 1].real]), numpy.array([gk[0, 1]]),
            PRIOR, nodes)[0]
        # the daily Gram matrix is a stationary-phase approximation
        assert lnb[k] == pytest.approx(exact, abs=0.05 * abs(exact) + 0.05)


def test_reconstruction_recovers_a_loud_injection():
    fam, daily = family()
    data, _ = inject(fam, numpy.zeros(3))
    m = model(fam, daily, data)
    m.update(x=0.0)
    with warnings.catch_warnings():
        # a discarded imaginary part of the Gram matrix once biased this
        warnings.simplefilter('error')
        draws = [m.reconstruct(seed=s) for s in range(60)]
    for name in ('delta_tc', 'distance', 'inclination'):
        values = numpy.array([d[name] for d in draws])
        low, high = numpy.percentile(values, [5, 95])
        assert low < TRUTH[name] < high, name
    for d in draws:
        assert 0 <= d['polarization'] < 2 * numpy.pi
        assert 0 <= d['signal_phase'] < 2 * numpy.pi
        assert 0 <= d['inclination'] <= numpy.pi


def test_reference_must_match_the_family():
    fam, daily = family()
    with pytest.raises(ValueError):
        BStatisticSurrogate(['x'], fam, daily, numpy.zeros((2, fam.size)),
                            1.0, dict(x=0.0), dict(x=1.0), PRIOR, (0, 1e5))
