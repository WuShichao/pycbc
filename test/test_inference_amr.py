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


def ridge_model(width):
    # a unit-length ridge of the given width, at 30 degrees to the axes
    angle = math.radians(30)
    rotation = numpy.array([[math.cos(angle), -math.sin(angle)],
                            [math.sin(angle), math.cos(angle)]])
    cov = rotation @ numpy.diag([1.0, width ** 2]) @ rotation.T
    prior = JointDistribution(
        ['x', 'y'], Uniform(x=(-HALF, HALF)), Uniform(y=(-HALF, HALF)))
    return models.TestNormal(['x', 'y'], mean=[0.3, -0.2], cov=cov,
                             prior=prior)


@pytest.mark.parametrize('width', [0.05, 0.01])
def test_a_thin_oblique_ridge_is_followed(width):
    sampler = AMRSampler(ridge_model(width), mass_fraction=1e-5,
                         max_evaluations=100000, posterior_samples=20000,
                         seed=1)
    sampler.run()
    error = sampler.logz + math.log((2 * HALF) ** 2)
    assert abs(error) < 1e-2
    assert abs(error) < 3 * sampler.logz_err
    x = sampler.samples['x']
    assert numpy.mean(x) == pytest.approx(0.3, abs=0.02)
    assert numpy.std(x) == pytest.approx(math.sqrt(0.75 + 0.25 * width ** 2),
                                         rel=0.03)


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


def test_a_whitened_frame_follows_a_ridge_with_fewer_evaluations():
    plain = AMRSampler(ridge_model(0.01), mass_fraction=1e-5,
                       max_evaluations=400000, posterior_samples=20000,
                       seed=1)
    plain.run()
    framed = AMRSampler(ridge_model(0.01), mass_fraction=1e-5,
                        pilot_evaluations=2000, max_evaluations=400000,
                        posterior_samples=20000, seed=1)
    framed.run()
    error = framed.logz + math.log((2 * HALF) ** 2)
    assert abs(error) < 3 * framed.logz_err < 0.05
    assert framed.nevaluations < plain.nevaluations / 3
    x = framed.samples['x']
    assert numpy.mean(x) == pytest.approx(0.3, abs=0.02)
    assert numpy.std(x) == pytest.approx(math.sqrt(0.75), rel=0.03)
    # the frame's boxes map into the cube
    cells = framed.cells
    centre = cells['frame_origin'] + (cells['lower'] + 0.5 * cells['width']) \
        @ cells['frame_matrix'].T
    inside = numpy.all((centre >= 0) & (centre <= 1), axis=1)
    assert inside.any()


def lattice_partition(path, radius=2.0, width=0.5):
    """Cells of an oblique chart about the normal model's mean, with the
    parameters in the other order than the model's."""
    from pycbc.tmpltbank.adaptive_cell_partition import write_partition
    from pycbc.tmpltbank.fstatistic_metric import MetricChart
    metric = numpy.array([[1.0, 0.6], [0.6, 1.0]])
    reference, scales = numpy.array([-0.5, 1.0]), numpy.array([1.0, 0.5])
    axis = (numpy.arange(-12, 12) + 0.5) * width
    centres = numpy.array(numpy.meshgrid(axis, axis)).reshape(2, -1).T
    candidate = numpy.linalg.norm(centres, axis=1) <= radius
    write_partition(path, ['y', 'x'], reference, scales, metric, centres,
                    numpy.full(centres.shape, width),
                    -numpy.sum(centres ** 2, axis=1), candidate)
    chart = MetricChart(metric)

    def inside(points):
        """Whether each (x, y) lies in a candidate cell."""
        xi = (points[:, ::-1] - reference) / scales
        eta = chart.to_chart(xi)
        cell = numpy.floor(eta / width + 12).astype(int)
        ok = numpy.all((cell >= 0) & (cell < 24), axis=1)
        flat = numpy.zeros(len(points), dtype=bool)
        index = cell[ok, 1] * 24 + cell[ok, 0]
        flat[ok] = candidate[index]
        return flat
    return inside


def test_a_search_partition_restricts_the_quadrature_to_its_cells(tmp_path):
    path = str(tmp_path / 'partition.hdf')
    inside = lattice_partition(path)
    model = normal_model(mean=(1.0, -0.5), cov=(0.25, 1.0))
    sampler = AMRSampler(model, partition_file=path, mass_fraction=1e-5,
                         max_evaluations=40000, posterior_samples=4000,
                         seed=4)
    sampler.run()
    # the evidence is the normal's mass inside the cells over the prior's
    # area
    rng = numpy.random.default_rng(9)
    draws = rng.normal([1.0, -0.5], [0.5, 1.0], size=(400000, 2))
    mass = inside(draws).mean()
    assert 0.5 < mass < 0.99
    assert sampler.logz == pytest.approx(
        math.log(mass / (2 * HALF) ** 2), abs=5e-3)
    samples = numpy.array([sampler.samples['x'], sampler.samples['y']]).T
    assert inside(samples).all()


def test_a_partition_takes_no_pilot(tmp_path):
    path = str(tmp_path / 'partition.hdf')
    lattice_partition(path)
    with pytest.raises(ValueError):
        AMRSampler(normal_model(), partition_file=path,
                   pilot_evaluations=100)


def test_a_laplace_frame_integrates_a_thin_oblique_ridge():
    """A ridge 400 times thinner than the prior, oblique to every axis: the
    pilot's samples whiten it badly, its curvature does not."""
    from scipy.stats import multivariate_normal
    names = ['a', 'b', 'c', 'd']
    rotation, _ = numpy.linalg.qr(numpy.random.default_rng(0).normal(
        size=(4, 4)))
    cov = rotation @ numpy.diag([1.5, 0.05, 0.03, 0.02]) ** 2 @ rotation.T

    class Ridge(models.base.BaseModel):
        name = 'test_ridge'

        def _loglikelihood(self):
            x = [self.current_params[n] for n in names]
            return multivariate_normal.logpdf(x, numpy.zeros(4), cov)
    prior = JointDistribution(names, *[Uniform(**{n: (-4.0, 4.0)})
                                       for n in names])
    draws = numpy.random.default_rng(1).multivariate_normal(
        numpy.zeros(4), cov, 200000)
    truth = math.log(numpy.all(numpy.abs(draws) < 4, axis=1).mean()
                     / 8.0 ** 4)
    sampler = AMRSampler(Ridge(names, prior=prior), initial_divisions=2,
                         pilot_evaluations=1000, laplace_frame=True,
                         mass_fraction=1e-4, max_evaluations=30000,
                         posterior_samples=10)
    sampler.run()
    assert sampler.logz == pytest.approx(truth, abs=0.02)
