"""The ``amr`` sampler's coverage bound: a box is refined while the most
posterior mass its template distance allows exceeds ``mass_fraction`` of the
evidence, so a peak narrower than the starting grid, unseen by every box
centre, is still found."""
import numpy
import pytest
from scipy.special import logsumexp

from pycbc.distributions import JointDistribution, Uniform
from pycbc.inference.models.base import BaseModel
from pycbc.inference.sampler.amr import AMRSampler

HALF = 5.0
RHO = 40.0          # SNR per unit of x: the peak's width is 1 / RHO
PEAK = 12.0         # the SNR at the peak
MEAN = (0.613, -1.377)


class FilterPeak(BaseModel):
    """``ln L = rho(x)^2 / 2`` with ``rho(x) = max(PEAK - RHO |x - MEAN|,
    0)``: an SNR that changes by at most the template distance ``RHO |x -
    c|``, as a matched filter's does."""
    name = 'test_filter_peak'

    def _loglikelihood(self):
        x = numpy.array([self.current_params[p] for p in ('x', 'y')])
        rho = max(PEAK - RHO * numpy.linalg.norm(x - MEAN), 0.0)
        return 0.5 * rho ** 2


def model():
    prior = JointDistribution(
        ['x', 'y'], Uniform(x=(-HALF, HALF)), Uniform(y=(-HALF, HALF)))
    return FilterPeak(['x', 'y'], prior=prior)


def distance(centre, probes):
    return RHO * numpy.linalg.norm(probes - centre[:, None, :], axis=-1)


def exact_logz():
    # the integral of exp(rho^2 / 2) over the cone's disc, plus the flat
    # rest, over the prior's area
    r = numpy.linspace(0, PEAK / RHO, 200001)
    inner = numpy.trapz(2 * numpy.pi * r * numpy.exp(0.5 * (PEAK - RHO * r)
                                                     ** 2), r)
    area = (2 * HALF) ** 2
    return numpy.log((inner + area - numpy.pi * (PEAK / RHO) ** 2) / area)


def run(**kw):
    sampler = AMRSampler(model(), initial_divisions=4, mass_fraction=1e-4,
                         max_evaluations=20000, posterior_samples=2000,
                         seed=2, **kw)
    sampler.run()
    return sampler


def test_without_the_bound_the_peak_is_missed():
    # every centre of the starting grid lies beyond the peak's support: flat
    # ln L = 0, nothing to refine
    assert run().logz == pytest.approx(0.0, abs=1e-9)


def test_the_bound_finds_the_peak_and_its_evidence():
    sampler = run(box_distance=distance)
    assert exact_logz() > 50
    assert sampler.logz == pytest.approx(exact_logz(), abs=0.05)
    diag = sampler.diagnostics
    assert diag['coverage_flagged'] == 0
    assert diag['coverage_log_bound_fraction'] < numpy.log(0.05)
    x = numpy.column_stack([sampler.samples['x'], sampler.samples['y']])
    assert numpy.all(numpy.linalg.norm(x - MEAN, axis=1) < PEAK / RHO)


def test_the_bound_is_the_triangle_inequality():
    sampler = AMRSampler(model(), box_distance=distance, coverage_kappa=0.0)
    logl = numpy.array([0.5 * 3.0 ** 2, 0.0, -numpy.inf])
    bound = sampler._coverage_bound(logl, numpy.array([2.0, 5.0, 1.0]))
    assert bound == pytest.approx([0.5 * 25.0, 12.5, -numpy.inf])


def test_an_snr_bound_gives_the_distance_bound_it_is_built_from():
    def snr_bound(centre, probes, rho):
        d = distance(centre, probes).max(axis=1)
        return rho + d, d
    a = run(box_distance=distance)
    b = run(box_snr_bound=snr_bound)
    assert b.logz == pytest.approx(a.logz, abs=1e-9)
    assert b.diagnostics['evaluations'] == a.diagnostics['evaluations']


def test_a_global_bound_certifies_the_far_prior_and_the_start_is_probed():
    # the signal estimated at the start (the peak itself): far from it the
    # SNR is at most its match with the signal, nothing of the noise here
    def snr_bound(centre, probes, rho):
        pts = numpy.concatenate([centre[:, None, :], probes], axis=1)
        far = numpy.linalg.norm(pts - MEAN, axis=-1).min(axis=1)
        top = numpy.maximum(PEAK - RHO * far, 0.0)
        d = distance(centre, probes).max(axis=1)
        return numpy.minimum(rho + d, top), d
    start = (numpy.array(MEAN) + HALF) / (2 * HALF)
    sampler = run(box_snr_bound=snr_bound, start_cube=start)
    assert sampler.logz == pytest.approx(exact_logz(), abs=0.05)
    full = run(box_distance=distance)
    assert sampler.diagnostics['evaluations'] < full.diagnostics['evaluations']



# a thin ridge across the prior, at 30 degrees to the axes: the SNR falls by
# RHO per unit across it and by EPS RHO along it
EPS = 0.02
ANGLE = numpy.radians(30)
NORMAL = numpy.array([-numpy.sin(ANGLE), numpy.cos(ANGLE)])
ALONG = numpy.array([numpy.cos(ANGLE), numpy.sin(ANGLE)])


def ridge_metric(z):
    return numpy.sqrt((z @ NORMAL) ** 2 + (EPS * (z @ ALONG)) ** 2)


class FilterRidge(BaseModel):
    name = 'test_filter_ridge'

    def _loglikelihood(self):
        x = numpy.array([self.current_params[p] for p in ('x', 'y')])
        return 0.5 * max(PEAK - RHO * ridge_metric(x - MEAN), 0.0) ** 2


def ridge_bound(centre, probes, rho):
    """The SNR bound with each probe's distance: the local one, and the
    global one, the match with the signal at the probes (no noise)."""
    pts = numpy.concatenate([centre[:, None, :], probes], axis=1)
    top = numpy.maximum(PEAK - RHO * ridge_metric(pts - MEAN), 0).max(axis=1)
    d = RHO * ridge_metric(probes - centre[:, None, :])
    return numpy.minimum(rho + d.max(axis=1), top), d.max(axis=1), d


def ridge_exact():
    """ln Z and the posterior's sd along the ridge, on a fine grid."""
    g = numpy.linspace(-HALF, HALF, 4001)
    x, y = numpy.meshgrid(g, g)
    z = numpy.stack([x - MEAN[0], y - MEAN[1]], axis=-1)
    logl = 0.5 * numpy.maximum(PEAK - RHO * ridge_metric(z), 0) ** 2
    top = logl.max()
    w = numpy.exp(logl - top)
    mean = numpy.trapz(numpy.trapz(w, g), g)
    t = z @ ALONG
    m1 = numpy.trapz(numpy.trapz(w * t, g), g) / mean
    m2 = numpy.trapz(numpy.trapz(w * t ** 2, g), g) / mean
    return top + numpy.log(mean / (2 * HALF) ** 2), numpy.sqrt(m2 - m1 ** 2)


def test_the_bound_follows_a_thin_oblique_ridge_from_its_start():
    prior = JointDistribution(
        ['x', 'y'], Uniform(x=(-HALF, HALF)), Uniform(y=(-HALF, HALF)))
    start = (numpy.array(MEAN) + 3 * ALONG + HALF) / (2 * HALF)
    sampler = AMRSampler(FilterRidge(['x', 'y'], prior=prior),
                         initial_divisions=1, mass_fraction=1e-4,
                         max_evaluations=40000, posterior_samples=2000,
                         seed=2, box_snr_bound=ridge_bound, start_cube=start)
    sampler.run()
    assert sampler.diagnostics['stopped_by'] == 'convergence'
    assert sampler.diagnostics['coverage_peak_moves'] >= 1
    logz, sd = ridge_exact()
    assert sampler.logz == pytest.approx(logz, abs=0.05)
    x = numpy.column_stack([sampler.samples['x'], sampler.samples['y']])
    assert ((x - MEAN) @ ALONG).std() == pytest.approx(sd, rel=0.1)


def test_the_coverage_tree_refuses_a_frame():
    with pytest.raises(ValueError):
        AMRSampler(model(), box_snr_bound=lambda c, p, r: (r, r),
                   curvilinear_frame=True, pilot_evaluations=100)



def test_the_tree_proposal_gives_the_evidence_by_importance_sampling():
    prior = JointDistribution(
        ['x', 'y'], Uniform(x=(-HALF, HALF)), Uniform(y=(-HALF, HALF)))
    m = FilterRidge(['x', 'y'], prior=prior)
    start = (numpy.array(MEAN) + 3 * ALONG + HALF) / (2 * HALF)
    sampler = AMRSampler(m, initial_divisions=1, mass_fraction=1e-4,
                         max_evaluations=40000, posterior_samples=2000,
                         seed=2, box_snr_bound=ridge_bound, start_cube=start)
    sampler.run()
    cube, logq = sampler.tree_proposal(20000, numpy.random.default_rng(5))
    x = -HALF + 2 * HALF * cube
    logl = 0.5 * numpy.maximum(PEAK - RHO * ridge_metric(x - MEAN), 0) ** 2
    # the cube's density, the prior uniform on it: Z = mean of L / q
    logw = logl - logq
    logz = logsumexp(logw) - numpy.log(len(logw))
    w = numpy.exp(logw - logw.max())
    ess = w.sum() ** 2 / numpy.sum(w ** 2)
    assert logz == pytest.approx(ridge_exact()[0], abs=0.03)
    assert ess > 0.5 * len(logw)
    assert numpy.all((cube >= 0) & (cube <= 1))
    assert sampler.tree_log_density(cube) == pytest.approx(logq)



class Mixed(object):
    """Curvilinear coordinates in which a sheet oblique in the parameters is
    along an axis: y1 = x + 3 y + 0.02 y^2 (thin), y2 = y."""
    names = ('x', 'y')

    def forward(self, theta):
        theta = numpy.atleast_2d(theta)
        return numpy.column_stack([theta[:, 0] + 3 * theta[:, 1]
                                   + 0.02 * theta[:, 1] ** 2, theta[:, 1]])

    def inverse(self, y):
        y = numpy.atleast_2d(y)
        return numpy.column_stack([y[:, 0] - 3 * y[:, 1] - 0.02 * y[:, 1] ** 2,
                                   y[:, 1]])

    def log_jacobian(self, theta):
        return numpy.zeros(len(numpy.atleast_2d(theta)))


SHEET = numpy.array([400.0, 4.0])      # SNR per unit of y1 and y2


def sheet_metric(theta, at):
    c = Mixed()
    d = c.forward(theta.reshape(-1, 2)) - c.forward(at.reshape(-1, 2))
    return numpy.sqrt(numpy.sum((d * SHEET) ** 2, axis=1)).reshape(
        theta.shape[:-1])


class FilterSheet(BaseModel):
    name = 'test_filter_sheet'

    def _loglikelihood(self):
        x = numpy.array([self.current_params[p] for p in ('x', 'y')])
        return 0.5 * max(PEAK - sheet_metric(x, numpy.array(MEAN)), 0.0) ** 2


def sheet_bound(centre, probes, rho):
    pts = numpy.concatenate([centre[:, None, :], probes], axis=1)
    top = numpy.maximum(PEAK - sheet_metric(pts, numpy.array(MEAN)[None, None]
                                            .repeat(pts.shape[0], 0)
                                            .repeat(pts.shape[1], 1)), 0)
    d = sheet_metric(probes, centre[:, None, :].repeat(probes.shape[1], 1))
    return numpy.minimum(rho + d.max(axis=1), top.max(axis=1)), d.max(axis=1), d


def sheet_logz(n=6001):
    g = numpy.linspace(-HALF, HALF, n)
    x, y = numpy.meshgrid(g, g)
    pts = numpy.stack([x, y], axis=-1)
    logl = 0.5 * numpy.maximum(PEAK - sheet_metric(
        pts, numpy.broadcast_to(numpy.array(MEAN), pts.shape)), 0) ** 2
    top = logl.max()
    return top + numpy.log(numpy.trapz(numpy.trapz(numpy.exp(logl - top), g), g)
                           / (2 * HALF) ** 2)


def sheet_run(chart):
    prior = JointDistribution(
        ['x', 'y'], Uniform(x=(-HALF, HALF)), Uniform(y=(-HALF, HALF)))
    m = FilterSheet(['x', 'y'], prior=prior)
    m.curvilinear_coordinates = Mixed()
    start = (numpy.array(MEAN) + HALF) / (2 * HALF)
    sampler = AMRSampler(m, initial_divisions=1, mass_fraction=1e-4,
                         max_evaluations=40000, posterior_samples=2000,
                         seed=2, box_snr_bound=sheet_bound, start_cube=start,
                         coverage_chart=chart, coverage_resolution=3.0)
    sampler.run()
    return sampler


def test_the_chart_tree_resolves_an_oblique_sheet_with_fewer_boxes():
    cube, chart = sheet_run(False), sheet_run(True)
    exact = sheet_logz()
    assert chart.logz == pytest.approx(exact, abs=0.1)
    assert chart.diagnostics['evaluations'] < cube.diagnostics['evaluations']
    # the chart tree's proposal: exact density (with the map's Jacobian)
    u, logq = chart.tree_proposal(20000, numpy.random.default_rng(4))
    ok = numpy.all((u >= 0) & (u <= 1), axis=1)
    assert chart.tree_log_density(u[ok]) == pytest.approx(logq[ok])
    x = -HALF + 2 * HALF * u[ok]
    logl = 0.5 * numpy.maximum(PEAK - sheet_metric(
        x, numpy.broadcast_to(numpy.array(MEAN), x.shape)), 0) ** 2
    logz = logsumexp(logl - logq[ok]) - numpy.log(len(u))
    assert logz == pytest.approx(exact, abs=0.05)
