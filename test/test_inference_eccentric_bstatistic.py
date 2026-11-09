"""Eccentric inference likelihood uses physical tied modes and arrival prior."""
import numpy as np
import pytest

from pycbc.filter.bstatistic import DistancePowerLaw, OrientationNodes
from pycbc.filter.mean_anomaly import mean_anomaly_log_bayes, mean_anomaly_twof
from pycbc.inference.models.eccentric_bstatistic import EccentricBStatistic


def components():
    rng = np.random.default_rng(923)
    h = (rng.normal(size=(3, 2, 31))+1j*rng.normal(size=(3, 2, 31)))/5
    d = rng.normal(size=(31, 3))+1j*rng.normal(size=(31, 3))
    return np.einsum('kaf,fl->kal', h.conj(), d), np.einsum('kaf,mbf->kamb', h.conj(), h)


def test_likelihood_update_and_nonuniform_arrival_prior_match_explicit_integral():
    z, gram = components()
    c = [-1, 0, 1]
    weights = np.array([.2, .3, .5])
    prior = DistancePowerLaw(.5, 10)
    calls = []
    def provider(params, delays):
        calls.append(dict(params))
        assert len(delays) == 3
        return z*params['scale'], gram
    model = EccentricBStatistic(['scale'], provider, c, [-2., 0., 3.], prior,
        delay_weights=weights, n_l0=16, orientation_nodes=(4, 4), amplitude_order=48,
        noise_variance=1.3)
    for scale in (1., 1.7):
        model.update(scale=scale)
        direct, _, _ = mean_anomaly_log_bayes(z*scale, gram, c, prior,
            OrientationNodes(4, 4), 16, noise_variance=1.3, order=48)
        want = np.log(np.sum(weights*np.exp(direct)))
        np.testing.assert_allclose(model.loglikelihood, want, atol=2e-12)
    assert len(calls) == 2
    stat, _ = model.anomaly_statistics()
    want, _ = mean_anomaly_twof(z*1.7/1.3, gram/1.3, c, 16)
    np.testing.assert_allclose(stat, want, atol=2e-12)


@pytest.mark.parametrize('weights', [[0, 0, 0], [-1, 1, 1], [1, 2]])
def test_bad_arrival_prior_is_refused(weights):
    with pytest.raises(ValueError, match='delay_weights'):
        EccentricBStatistic([], lambda p, d: None, [0], [-1., 0., 1.],
                            DistancePowerLaw(.5, 10), delay_weights=weights)



def test_cell_provider_uses_surrogate_only_and_matches_direct_delayed_modes():
    from pycbc.inference.models.eccentric_bstatistic import EccentricCellComponents
    from pycbc.tdi.eccentric_mode_source import EccentricModeFamilySource
    from pycbc.waveform.eccentric_mode_family import EccentricModeFamily
    f = np.linspace(.01, .02, 61)
    def generator(params):
        e = params['eccentricity']
        return {(2, 2, n): (np.ones((2, len(f)), complex)*(1+n*e),
            2*np.pi*(n*1e5*f+2e7*f*f)+e*f,
            np.ones(len(f), bool)) for n in (1, 2, 3)}
    ref = dict(mass1=36., mass2=31., eccentricity=.05)
    family = EccentricModeFamily(ref, f, .015, generator=generator,
        step_mass=0., step_e=.002, step_q=0., circular_phase_basis='taylorf2')
    def forbidden(*args, **kwargs):
        raise AssertionError('online component provider called a native generator')
    family.generator = family._circ_generator = forbidden
    rho = np.array([[[1., .3j], [.2+.4j, 1.]]], complex)[..., None]*np.ones(len(f))
    data = np.random.default_rng(1).normal(size=(2, len(f))).astype(complex)
    provider = EccentricCellComponents(family, f, data, rho, [0])
    delays = np.array([-50., 0., 70.])
    z, g = provider({'eccentricity': .051}, delays)
    h = np.zeros((3, 2, 2, len(f)), complex)
    source = EccentricModeFamilySource(family, de=.001)
    for j, key in enumerate(provider.keys):
        rec = source.frequency_harmonics(f)[key]
        h[j] = rho[0]*rec['plus']
    templates = np.array([h*np.exp((-2j*np.pi*lag)*f) for lag in delays])
    want_z = np.einsum('lkacf,cf->kal', templates.conj(), data)
    want_g = np.einsum('lkacf,lmbcf->kambl', templates.conj(), templates)
    np.testing.assert_allclose(z, want_z, atol=2e-8, rtol=2e-8)
    np.testing.assert_allclose(g, want_g, atol=2e-8, rtol=2e-8)
