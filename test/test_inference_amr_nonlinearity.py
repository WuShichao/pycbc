"""AMR refinement by the likelihood's curvature (box_nonlinearity, card 2026-11-22-amr-coverage-noise-bound): a broad
Gaussian with fringes along one axis, as the weak eccentric harmonics make against the noise, and narrow Gaussians
along three more. Without the hook the boxes, sized by the posterior's mass, span several fringes and the linear
draws' importance weights spread; with it every box that may hold mass is resolved to its fringes along that axis,
and with parabolic slopes the linear surrogate follows a resolved box to second order."""
import numpy
import pytest

from pycbc.distributions import JointDistribution, Uniform
from pycbc.inference.models.base import BaseModel
from pycbc.inference.sampler.amr import AMRSampler

SX, SY, AMP, PERIOD, ND = 0.15, 0.03, 2.5, 0.05, 4
K = 2 * numpy.pi / PERIOD
NAMES = ['x'] + ['y%d' % k for k in range(ND - 1)]


def logl(X):
    X = numpy.atleast_2d(X)
    return (-0.5 * ((X[:, 0] - 0.5) / SX) ** 2 - 0.5 * numpy.sum(((X[:, 1:] - 0.5) / SY) ** 2, axis=1)
            + AMP * numpy.cos(K * (X[:, 0] - 0.5)))


class FringeModel(BaseModel):
    name = 'test_fringes'

    def __init__(self):
        super().__init__(NAMES, prior=JointDistribution(NAMES, *[Uniform(**{n: (0., 1.)}) for n in NAMES]))

    def _loglikelihood(self):
        return float(logl([[self.current_params[n] for n in NAMES]])[0])


def nonlinearity(lower, width, loglikelihood, loglikelihood_max):
    # how far ln L departs from a linear function across the box, per axis: the fringe's second-order term (at most
    # its full swing 2 AMP) and the Gaussians'
    h = 0.5 * width
    out = numpy.empty_like(width)
    out[:, 0] = AMP * numpy.minimum(2.0, 0.5 * (K * h[:, 0]) ** 2) + 0.5 * (h[:, 0] / SX) ** 2
    out[:, 1:] = 0.5 * (h[:, 1:] / SY) ** 2
    return out


def exact_logz():
    x = numpy.linspace(0, 1, 20001)
    y = numpy.linspace(0, 1, 4001)
    fx = numpy.trapezoid(numpy.exp(-0.5 * ((x - 0.5) / SX) ** 2 + AMP * numpy.cos(K * (x - 0.5))), x)
    fy = numpy.trapezoid(numpy.exp(-0.5 * ((y - 0.5) / SY) ** 2), y)
    return float(numpy.log(fx) + (ND - 1) * numpy.log(fy))


def run(hook):
    s = AMRSampler(FringeModel(), initial_divisions=4, mass_fraction=2e-4, max_evaluations=400000,
                   posterior_samples=20000, linear_posterior=True, seed=2, max_level=80, whitening_rounds=0,
                   box_nonlinearity=nonlinearity if hook else None, parabolic_slopes=hook)
    # the linear draws are those of a curved frame: the identity
    s._curve = lambda q: (q, numpy.zeros(len(q)))
    s._uncurve = lambda u: u
    s.run()
    X = numpy.column_stack([s.samples[n] for n in NAMES])
    lw = logl(X) - numpy.asarray(s.samples['loglikelihood'])
    w = numpy.exp(lw - lw.max())
    return s, w, w.sum() ** 2 / numpy.sum(w ** 2) / len(w)


def test_the_curvature_hook_resolves_fringes_and_evens_the_weights():
    s0, w0, eff0 = run(False)
    s1, w1, eff1 = run(True)
    print('without: ESS/N %.3f, %d evaluations; with: ESS/N %.3f, %d evaluations' % (
        eff0, s0.nevaluations, eff1, s1.nevaluations))
    assert eff0 < 0.75
    assert eff1 > 0.93
    assert s1.nevaluations < 1.2 * s0.nevaluations
    assert s1._diagnostics['nonlinearity_computed'] > 0
    assert s1.logz == pytest.approx(exact_logz(), abs=0.03)
