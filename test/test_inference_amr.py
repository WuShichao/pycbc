"""The ``amr`` sampler on a model whose evidence and posterior are known."""
import math

import numpy
import pytest

from pycbc.distributions import JointDistribution, Uniform
from pycbc.inference.io import loadfile
from pycbc.inference import models
from pycbc.inference.sampler import samplers
from pycbc.inference.sampler.amr import AMRSampler
from pycbc.workflow import WorkflowConfigParser

HALF = 5.0


def normal_model(mean=(0.0, 0.0), cov=(1.0, 1.0)):
    prior = JointDistribution(
        ['x', 'y'], Uniform(x=(-HALF, HALF)), Uniform(y=(-HALF, HALF)))
    return models.TestNormal(['x', 'y'], mean=list(mean), cov=list(cov),
                             prior=prior)


def test_registered():
    assert samplers['amr'] is AMRSampler


def test_evidence_of_a_normalised_likelihood_is_one_over_the_prior_volume():
    # the likelihood integrates to one inside the box, so Z = 1 / 100
    sampler = AMRSampler(normal_model(), mass_fraction=1e-6,
                         max_evaluations=40000, posterior_samples=4000)
    sampler.run()
    error = sampler.logz + math.log((2 * HALF) ** 2)
    assert abs(error) < 2e-3
    assert abs(error) < 3 * sampler.logz_err < 1e-2


def test_evidence_does_not_depend_on_the_starting_grid():
    values = []
    for divisions in (2, 5):
        sampler = AMRSampler(normal_model(), initial_divisions=divisions,
                             mass_fraction=1e-5, max_evaluations=40000)
        sampler.run()
        values.append(sampler.logz)
    assert values[0] == pytest.approx(values[1], abs=3e-3)


def test_posterior_samples_recover_the_mean_and_width():
    sampler = AMRSampler(normal_model(mean=(1.0, -0.5), cov=(0.25, 1.0)),
                         mass_fraction=1e-5, max_evaluations=40000,
                         posterior_samples=20000, seed=3)
    sampler.run()
    samples = sampler.samples
    assert numpy.mean(samples['x']) == pytest.approx(1.0, abs=0.02)
    assert numpy.mean(samples['y']) == pytest.approx(-0.5, abs=0.03)
    assert numpy.std(samples['x']) == pytest.approx(0.5, rel=0.03)
    assert numpy.std(samples['y']) == pytest.approx(1.0, rel=0.03)
    # samples are drawn inside boxes, not at their centres
    assert len(numpy.unique(samples['x'])) > 0.9 * len(samples['x'])


def test_box_masses_cover_the_prior_exactly():
    sampler = AMRSampler(normal_model(), max_evaluations=5000)
    sampler.run()
    cells = sampler.cells
    volume = numpy.prod(cells['width'], axis=1).sum()
    assert volume == pytest.approx(1.0, abs=1e-12)


def test_evaluation_budget_is_respected():
    sampler = AMRSampler(normal_model(), mass_fraction=1e-9,
                         max_evaluations=300)
    sampler.run()
    assert sampler.nevaluations <= 300 + 2


def test_config_runs_and_writes_a_readable_file(tmp_path):
    cp = WorkflowConfigParser()
    cp.read_string("""
[model]
name = test_normal
[variable_params]
x =
y =
[prior-x]
name = uniform
min-x = -5
max-x = 5
[prior-y]
name = uniform
min-y = -5
max-y = 5
[sampler]
name = amr
initial-divisions = 3
mass-fraction = 1e-4
max-evaluations = 4000
posterior-samples = 1000
seed = 1
""")
    model = models.read_from_config(cp)
    output = str(tmp_path / 'amr.hdf')
    sampler = AMRSampler.from_config(cp, model, output_file=output)
    sampler.run()
    sampler.finalize()
    with loadfile(sampler.checkpoint_file, 'r') as fp:
        assert fp.attrs['sampler'] == 'amr'
        assert fp.attrs['log_evidence'] == pytest.approx(sampler.logz)
        posterior = fp.read_samples(['x', 'y'])
        assert len(posterior['x']) == 1000
        assert 'cells' in fp[fp.sampler_group]


def test_unknown_options_are_refused(tmp_path):
    cp = WorkflowConfigParser()
    cp.read_string("""
[model]
name = test_normal
[variable_params]
x =
[prior-x]
name = uniform
min-x = -1
max-x = 1
[sampler]
name = amr
nlive = 100
""")
    model = models.read_from_config(cp)
    with pytest.raises(ValueError, match="unknown option"):
        AMRSampler.from_config(cp, model,
                               output_file=str(tmp_path / 'out.hdf'))
