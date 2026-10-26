"""Physical prior mass in non-affine AMR posterior draws."""
import math

import numpy
import pytest
from scipy.special import logsumexp
from scipy.stats import nbinom

from pycbc.distributions import JointDistribution, Uniform
from pycbc.inference.io import loadfile
from pycbc.inference.models.base import BaseModel
from pycbc.inference.sampler.amr import AMRSampler
from pycbc.inference.sampler.base import setup_output


class FlatPriorModel(BaseModel):
    name = 'test_curved_prior'

    def __init__(self, logl=lambda x: 0.):
        self.value = logl
        super().__init__(['x'], prior=JointDistribution(
            ['x'], Uniform(x=(0., 1.))))

    def _loglikelihood(self):
        return self.value(self.current_params['x'])


def curved_sampler(scale=1., offset=0., divisions=1,
                   logl=lambda x: 0., size=65536):
    sampler = AMRSampler(FlatPriorModel(logl), initial_divisions=divisions,
                         max_evaluations=divisions, whitening_rounds=0,
                         posterior_samples=size, seed=20267511)

    def curve(s):
        return scale*s*s + offset, numpy.log(2*scale*s[:, 0])

    sampler._curve = curve
    sampler._uncurve = lambda u: numpy.sqrt((u-offset)/scale)
    sampler.run()
    return sampler


def posterior_weights(sampler):
    weights = numpy.exp(sampler.samples['logwt'] - sampler.logz)
    assert weights.sum() == pytest.approx(1., abs=1e-12)
    return weights


@pytest.mark.parametrize('fraction', [.02, .2, .8, 1.])
@pytest.mark.parametrize('requested', [1, 3, 20])
def test_extra_acceptance_estimates_the_admitted_fraction(fraction, requested):
    # T is the total number of trials for requested + 1 successes.
    successes = requested + 1
    failures = numpy.arange(int(nbinom.ppf(1-1e-14, successes, fraction))+1)
    trials = failures + successes
    probability = nbinom.pmf(failures, successes, fraction)
    estimate = requested / (trials-1)
    assert probability.sum() == pytest.approx(1., abs=2e-13)
    assert numpy.sum(probability*estimate) == pytest.approx(
        fraction, abs=2e-13)


def test_a_constant_likelihood_recovers_the_physical_prior():
    sampler = curved_sampler()
    weights = posterior_weights(sampler)
    x = sampler.samples['x']
    # Raw proposals are uniform in s; logwt changes their law to uniform x.
    assert x.mean() == pytest.approx(1/3, abs=.01)
    assert weights @ x == pytest.approx(.5, abs=.01)
    assert weights @ (x <= .25) == pytest.approx(.25, abs=.01)
    assert sampler.nevaluations == 1
    assert sampler.diagnostics['posterior_extra_points'] == 1
    assert sampler.diagnostics['posterior_trials'] == len(x)+1
    assert sampler.diagnostics['posterior_inverse_failures'] == 0
    assert sampler.diagnostics['posterior_geometry_logz'] == pytest.approx(
        0., abs=.02)


def test_a_clipped_curved_box_retains_its_original_prior_mass():
    sampler = curved_sampler(scale=1.4, offset=-.2)
    weights = posterior_weights(sampler)
    x = sampler.samples['x']
    assert numpy.all((x >= 0) & (x <= 1))
    assert weights @ x == pytest.approx(.5, abs=.01)
    assert weights @ (x <= .25) == pytest.approx(.25, abs=.01)
    assert sampler.diagnostics['posterior_trials'] > len(x)+1
    assert sampler.diagnostics['posterior_geometry_logz'] == pytest.approx(
        0., abs=.02)


def test_different_box_likelihoods_and_admitted_fractions_are_both_weighted():
    sampler = curved_sampler(scale=1.4, offset=-.2, divisions=2,
                             logl=lambda x: 0. if x < .15 else math.log(4.))
    weights = posterior_weights(sampler)
    exact = .15+4*.85
    assert weights @ (sampler.samples['x'] <= .15) == pytest.approx(
        .15/exact, abs=.008)
    assert sampler.diagnostics['posterior_extra_points'] == 2
    assert sampler.diagnostics['posterior_geometry_logz'] == pytest.approx(
        math.log(exact), abs=.025)
    assert sampler.nevaluations == 2


def test_file_reader_uses_the_curved_posterior_weights(tmp_path):
    sampler = curved_sampler(size=32768)
    setup_output(sampler, str(tmp_path/'curved.hdf'), check_nsamples=False)
    sampler.finalize()
    with loadfile(sampler.checkpoint_file, 'r') as fp:
        raw = fp.read_samples(['x'], raw_samples=True)
        posterior = fp.read_samples(['x'], seed=31)
        diagnostics = fp.read_diagnostics()
    assert raw['x'].mean() == pytest.approx(1/3, abs=.01)
    assert posterior['x'].mean() == pytest.approx(.5, abs=.01)
    assert numpy.mean(posterior['x'] <= .25) == pytest.approx(.25, abs=.01)
    assert diagnostics['posterior_weighting'] == 'curved prior geometry'


def test_a_nonfinite_jacobian_inside_the_prior_is_not_silently_rejected():
    sampler = AMRSampler(FlatPriorModel(), initial_divisions=1,
                         max_evaluations=1, whitening_rounds=0,
                         posterior_samples=128, seed=3)
    sampler._grid()
    sampler._logz = 0.
    sampler._curve = lambda s: (s, numpy.full(len(s), numpy.nan))
    with pytest.raises(ArithmeticError, match='Jacobian unresolved'):
        sampler._draw_posterior(sampler._leaves(),
                                 sampler._log_mass(sampler._leaves()))


def test_a_global_check_correction_is_not_counted_twice_in_geometry_weights():
    sampler = curved_sampler(size=32768)
    leaf = sampler._leaves()
    sampler._logz = 12.
    sampler._draw_posterior(leaf, sampler._log_mass(leaf))
    assert logsumexp(sampler.samples['logwt']) == pytest.approx(12., abs=1e-12)
    assert sampler.diagnostics['posterior_geometry_logz'] == pytest.approx(
        0., abs=.02)
    assert posterior_weights(sampler) @ sampler.samples['x'] == pytest.approx(
        .5, abs=.01)


def test_failed_inverse_trials_are_counted_in_the_admitted_fraction():
    sampler = AMRSampler(FlatPriorModel(), initial_divisions=1,
                         max_evaluations=1, whitening_rounds=0,
                         posterior_samples=32768, seed=20267512)

    def curve(s):
        u = 2*s-.5
        outside = numpy.any((u < 0) | (u > 1), axis=1)
        u[outside] = numpy.nan
        return u, numpy.full(len(s), math.log(2.))

    sampler._curve = curve
    sampler.run()
    weights = posterior_weights(sampler)
    assert weights @ sampler.samples['x'] == pytest.approx(.5, abs=.01)
    diag = sampler.diagnostics
    assert diag['posterior_inverse_failures'] > 0
    assert diag['posterior_trials'] == (
        32768+1+diag['posterior_inverse_failures'])
    assert diag['posterior_geometry_logz'] == pytest.approx(0., abs=.02)
