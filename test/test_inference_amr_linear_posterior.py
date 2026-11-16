"""AMR posterior draws linear in ln L across a box (linear_posterior): exact once importance-weighted against the
points' own ln L, with smaller weight spread than the piecewise-constant draws."""
import numpy
import pytest

from pycbc.distributions import JointDistribution, Uniform
from pycbc.inference.models.base import BaseModel
from pycbc.inference.sampler.amr import AMRSampler

MEAN = numpy.array([0.43, 0.61])
COV = numpy.array([[0.004, 0.0025], [0.0025, 0.003]])
PREC = numpy.linalg.inv(COV)


def logl(x):
    d = numpy.atleast_2d(x) - MEAN
    return -0.5 * numpy.einsum('ij,jk,ik->i', d, PREC, d)


class GaussModel(BaseModel):
    name = 'test_linear_posterior'

    def __init__(self):
        super().__init__(['x', 'y'], prior=JointDistribution(['x', 'y'], Uniform(x=(0., 1.)), Uniform(y=(0., 1.))))

    def _loglikelihood(self):
        return float(logl([self.current_params['x'], self.current_params['y']])[0])


def run(linear):
    sampler = AMRSampler(GaussModel(), initial_divisions=4, max_evaluations=120, mass_fraction=1e-3,
                         whitening_rounds=0, posterior_samples=40000, seed=3, max_level=40,
                         linear_posterior=linear)
    sampler._curve = lambda s: (s, numpy.zeros(len(s)))
    sampler._uncurve = lambda u: u
    sampler.run()
    smp = sampler.samples
    x = numpy.column_stack([smp['x'], smp['y']])
    lw = smp['logwt'] + logl(x) - smp['loglikelihood']
    w = numpy.exp(lw - lw.max())
    w /= w.sum()
    return x, w, 1 / numpy.sum(w ** 2)


def test_linear_draws_are_exact_and_steadier():
    x0, w0, ess0 = run(False)
    x1, w1, ess1 = run(True)
    for x, w in ((x0, w0), (x1, w1)):
        mu = w @ x
        cov = (x - mu).T @ ((x - mu) * w[:, None])
        assert mu == pytest.approx(MEAN, abs=2e-3)
        assert cov == pytest.approx(COV, rel=0.05)
    # the weights' variance (N / ESS - 1), mostly from the first-order variation across a box
    n = len(w0)
    print(ess0, ess1)
    assert n / ess1 - 1 < 0.3 * (n / ess0 - 1)


def test_linear_factor_and_draw():
    a = numpy.array([[-3.0, 0.0, 1e-9, 2.5]])
    t = numpy.linspace(0, 1, 200001)
    expect = sum(numpy.log(numpy.trapz(numpy.exp(ai * (t - 0.5)), t)) for ai in a[0])
    assert AMRSampler._linear_log_factor(a)[0] == pytest.approx(expect, abs=1e-8)
    u = numpy.random.default_rng(1).random(200000)
    for ai in (-3.0, 2.5):
        tt = AMRSampler._linear_unit_draw(numpy.full(len(u), ai), u)
        mean = numpy.trapz(t * numpy.exp(ai * t), t) / numpy.trapz(numpy.exp(ai * t), t)
        assert tt.mean() == pytest.approx(mean, abs=3e-3)
