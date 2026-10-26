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
    # and the run says it stopped short
    assert sampler.diagnostics['stopped_by'] == 'max-evaluations'
    assert sampler.diagnostics['boxes_to_refine'] > 0


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
        diagnostics = fp.read_diagnostics()
    assert diagnostics['evaluations'] == sampler.nevaluations
    assert diagnostics['stopped_by'] == (
        'max-evaluations' if diagnostics['boxes_to_refine'] else
        'convergence')
    assert diagnostics['logz'] == pytest.approx(sampler.logz)


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


def test_exploration_finds_the_higher_of_two_peaks():
    """A low broad peak and a high narrow one whose ln L rises over a basin
    of about 1e-3 of the prior, as a match does: the pilot finds the broad
    one; with the exploration the frame is about the narrow one."""
    names = ['a', 'b', 'c']
    narrow = numpy.array([2.5, -3.0, 1.0])
    broad = numpy.array([-2.0, 2.0, -1.0])

    class TwoPeaks(models.base.BaseModel):
        name = 'test_two_peaks'

        def _loglikelihood(self):
            x = numpy.array([self.current_params[n] for n in names])
            r2n = numpy.sum((x - narrow) ** 2)
            r2b = numpy.sum((x - broad) ** 2)
            return max(30.0 * numpy.exp(-r2n / (2 * 0.25 ** 2)),
                       20.0 * numpy.exp(-r2b / 2.0))
    prior = JointDistribution(names, *[Uniform(**{n: (-4.0, 4.0)})
                                       for n in names])
    peaks = {}
    for explore in (0.0, 1e-3):
        sampler = AMRSampler(TwoPeaks(names, prior=prior),
                             initial_divisions=2, pilot_evaluations=500,
                             laplace_frame=True, mass_fraction=1e-4,
                             max_evaluations=30000, posterior_samples=2000,
                             seed=3, explore_basin=explore)
        sampler.run()
        peaks[explore] = (sampler.diagnostics['frame_peak'],
                          numpy.mean(sampler.samples['a']))
    # without the exploration the frame is about the broad peak
    assert peaks[0.0][0] < 21
    assert peaks[1e-3][0] > 29.9
    assert peaks[1e-3][1] == pytest.approx(narrow[0], abs=0.05)


def test_a_start_point_leads_the_frame_to_the_higher_peak():
    """The two-peak likelihood without exploration, but with a start point
    near the narrow peak, as a search would give: the frame is about it."""
    names = ['a', 'b', 'c']
    narrow = numpy.array([2.5, -3.0, 1.0])
    broad = numpy.array([-2.0, 2.0, -1.0])

    class TwoPeaks(models.base.BaseModel):
        name = 'test_two_peaks_start'

        def _loglikelihood(self):
            x = numpy.array([self.current_params[n] for n in names])
            return max(30.0 * numpy.exp(-numpy.sum((x - narrow) ** 2)
                                        / (2 * 0.25 ** 2)),
                       20.0 * numpy.exp(-numpy.sum((x - broad) ** 2) / 2.0))
    prior = JointDistribution(names, *[Uniform(**{n: (-4.0, 4.0)})
                                       for n in names])
    start = ((narrow + 0.1) + 4.0) / 8.0
    sampler = AMRSampler(TwoPeaks(names, prior=prior), initial_divisions=2,
                         pilot_evaluations=500, laplace_frame=True,
                         mass_fraction=1e-4, max_evaluations=30000,
                         posterior_samples=2000, seed=3, start_cube=[start])
    sampler.run()
    assert sampler.diagnostics['frame_peak'] > 29.9
    with pytest.raises(ValueError):
        AMRSampler(TwoPeaks(names, prior=prior), start_cube=[[1.5, 0, 0]])


def test_an_exploration_beyond_the_budget_is_refused():
    with pytest.raises(ValueError):
        AMRSampler(normal_model(), max_evaluations=1000, explore_basin=1e-3)


def test_a_partition_takes_no_exploration(tmp_path):
    path = str(tmp_path / 'cells.hdf')
    lattice_partition(path)
    with pytest.raises(ValueError):
        AMRSampler(normal_model(), partition_file=path, explore_basin=1e-3,
                   max_evaluations=100000)


def test_differences_at_the_edge_of_the_support_are_one_sided():
    """A quadratic whose point sits on two bounds of its support: the
    gradient and Hessian come out as inside, not zero."""
    from pycbc.inference.sampler.amr import _finite_differences
    A = numpy.array([[4.0, 1.0, 0.3], [1.0, 2.0, -0.5], [0.3, -0.5, 3.0]])
    b = numpy.array([0.2, -0.1, 0.5])

    def f(X):
        X = numpy.atleast_2d(X)
        v = -0.5 * numpy.einsum('ki,ij,kj->k', X - b, A, X - b)
        v[(X[:, 0] > 0.2 + 1e-12) | (X[:, 2] < 0.5 - 1e-12)] = -numpy.inf
        return v
    x = numpy.array([0.2, 0.0, 0.5])
    grad, hess = _finite_differences(f, x, numpy.full(3, 1e-3))
    assert numpy.allclose(grad, -A @ (x - b), atol=1e-6)
    assert numpy.allclose(hess, -A, atol=1e-6)


class HiddenBump(models.base.BaseModel):
    """A flat likelihood with a bump the starting boxes' midpoints miss:
    ``L = 1 + B exp(-r^2 / 2 s^2)``, the bump holding four times the flat
    part's mass, so that ``Z = 5``."""
    name = 'test_hidden_bump'
    centre, width = numpy.array([0.1, -0.2]), 0.15
    height = 400 / (2 * numpy.pi * 0.15 ** 2)

    def _loglikelihood(self):
        x = numpy.array([self.current_params[n] for n in ('x', 'y')])
        r2 = numpy.sum((x - self.centre) ** 2)
        return numpy.log1p(self.height * numpy.exp(-r2 / (2 * self.width
                                                           ** 2)))


def hidden_bump():
    prior = JointDistribution(
        ['x', 'y'], Uniform(x=(-HALF, HALF)), Uniform(y=(-HALF, HALF)))
    return HiddenBump(['x', 'y'], prior=prior)


def test_the_check_finds_mass_the_midpoints_miss():
    """Without the check the refinement stops on the flat part and claims
    a small error; with it the bump is found and refined, and the evidence
    is right within the error the check gives."""
    plain = AMRSampler(hidden_bump(), mass_fraction=1e-4,
                       max_evaluations=20000)
    plain.run()
    assert plain.logz < 0.1 and plain.logz_err < 0.05
    for seed in range(3):
        checked = AMRSampler(hidden_bump(), mass_fraction=1e-4,
                             max_evaluations=20000, check_points=1000,
                             check_tolerance=0.05, seed=seed)
        checked.run()
        diag = checked.diagnostics
        assert diag['stopped_by'] == 'check'
        assert checked.logz == pytest.approx(math.log(5), abs=max(
            3 * checked.logz_err, 1e-3))
        assert checked.logz_err <= 0.05
        # the posterior follows the bump: four fifths of its mass
        near = numpy.hypot(checked.samples['x'] - 0.1,
                           checked.samples['y'] + 0.2) < 1.0
        assert numpy.mean(near) == pytest.approx(0.806, abs=0.03)


def test_the_check_is_unbiased_where_the_midpoints_are_exact_enough():
    """On a resolved Gaussian the check's correction is within its error
    of zero and the evidence keeps its accuracy."""
    sampler = AMRSampler(normal_model(), mass_fraction=1e-6,
                         max_evaluations=40000, check_points=2000,
                         check_tolerance=0.01)
    sampler.run()
    error = sampler.logz + math.log((2 * HALF) ** 2)
    assert abs(error) < max(3 * sampler.logz_err, 2e-3)
    assert abs(sampler.diagnostics['check_correction']) < max(
        3 * sampler.logz_err, 2e-3)


def test_a_single_check_point_is_refused():
    with pytest.raises(ValueError):
        AMRSampler(normal_model(), check_points=1)


class CubedCoordinates(object):
    """``u = x^3``: a curvilinear frame whose Jacobian changes 400 times
    over the prior."""
    names = ['x']

    def forward(self, theta):
        return theta ** 3

    def inverse(self, y):
        return numpy.cbrt(y)

    def log_jacobian(self, theta):
        return -numpy.log(3 * theta[:, 0] ** 2)


class SkewedNormal(models.base.BaseModel):
    """A normal likelihood wide in ``x``, where the cubed frame is most
    curved, with ``Z`` in closed form."""
    name = 'test_skewed_normal'
    curvilinear_coordinates = CubedCoordinates()

    def _loglikelihood(self):
        x, y = self.current_params['x'], self.current_params['y']
        return -0.5 * ((x - 1.2) / 0.6) ** 2 - 0.5 * (y / 0.4) ** 2


def test_the_curvilinear_box_masses_take_the_jacobian_across_the_box():
    from scipy.special import erf, logsumexp
    prior = JointDistribution(['x', 'y'], Uniform(x=(0.2, 4.0)),
                              Uniform(y=(-3.0, 3.0)))

    def gauss(lo, hi, mu, sd):
        return sd * math.sqrt(math.pi / 2) * (
            erf((hi - mu) / (sd * math.sqrt(2)))
            - erf((lo - mu) / (sd * math.sqrt(2))))
    exact = math.log(gauss(0.2, 4.0, 1.2, 0.6) * gauss(-3.0, 3.0, 0.0, 0.4)
                     / (3.8 * 6.0))
    sampler = AMRSampler(SkewedNormal(['x', 'y'], prior=prior),
                         initial_divisions=2, pilot_evaluations=300,
                         curvilinear_frame=True, mass_fraction=1e-4,
                         max_evaluations=20000, check_points=2000,
                         check_tolerance=0.01, seed=2)
    sampler.run()
    assert sampler.logz == pytest.approx(exact, abs=max(
        3 * sampler.logz_err, 2e-3))
    assert abs(sampler.diagnostics['midpoint_logz'] - exact) < 0.01
    assert sampler.diagnostics['check_uncovered'] < 1e-3
    x = sampler.samples['x']
    weight = numpy.exp(sampler.samples['logwt']
                       - logsumexp(sampler.samples['logwt']))
    normal = gauss(0.2, 4.0, 1.2, 0.6)
    for edge in (0.6, 1.2, 1.8, 2.4):
        assert weight @ (x <= edge) == pytest.approx(
            gauss(0.2, edge, 1.2, 0.6) / normal, abs=0.02)


def test_a_peak_on_the_prior_edge_keeps_its_mass_in_a_tilted_frame():
    """A normal likelihood centred on a face of the prior, in a whitened
    frame whose boxes straddle that face: the boxes whose centres lie
    beyond it hold the mass of their part inside."""
    from scipy.special import erf
    model = normal_model(mean=(HALF, 0.0), cov=(1.0, 0.3))
    exact = math.log(0.5 * erf(2 * HALF / math.sqrt(2))
                     * erf(HALF / math.sqrt(0.6)) / (2 * HALF) ** 2)
    sampler = AMRSampler(model, initial_divisions=2, pilot_evaluations=400,
                         laplace_frame=True, mass_fraction=1e-5,
                         max_evaluations=40000, seed=1)
    sampler.run()
    assert sampler.logz == pytest.approx(exact, abs=0.01)
