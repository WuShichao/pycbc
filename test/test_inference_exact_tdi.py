"""Tests for the exact-TDI inference model and its amplitude prior.

The heavy tests need the companion campaign repository and, for the family
response, CUDA.  They skip when those are absent, so the module is safe in a
plain PyCBC checkout.
"""
import configparser
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest

import pycbc
from pycbc.distributions.power_law import UniformAmplitudePowerLaw
from pycbc.inference.models import ExactTDIModel, read_from_config
from pycbc.workflow import WorkflowConfigParser

AMR_ROOT = os.environ.get('PYCBC_EXACT_TDI_AMR_ROOT',
                          '/mnt/d/lisa_tests_with_cc')
AMR_FILE = os.path.join(
    AMR_ROOT, 'results/2026-10-19/orientation_interpolated',
    'amr_pycbc_distance_flat_orientation_interp.hdf')
SUFFICIENT = os.path.join(
    AMR_ROOT, 'results/2026-10-18/demarginalisation/sufficient_full.npz')
EXAMPLE_CONFIG = os.path.join(
    os.path.dirname(os.path.abspath(pycbc.__file__)), '..', 'examples',
    'inference', 'amr-partition', 'amr-nessai.ini')
HAVE_AMR = os.path.isdir(AMR_ROOT) and os.path.isfile(AMR_FILE)


def _cuda_available():
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


requires_amr = pytest.mark.skipif(
    not HAVE_AMR, reason='the companion campaign repository is not available')
requires_family = pytest.mark.skipif(
    not (HAVE_AMR and _cuda_available()),
    reason='the campaign repository or CUDA is not available')


@pytest.fixture(scope='module')
def family_model():
    config = WorkflowConfigParser([EXAMPLE_CONFIG])
    model = read_from_config(config)
    return model


@pytest.fixture(scope='module')
def exact_model():
    names = ['eta_0', 'eta_1', 'eta_2', 'eta_3', 'inclination',
             'polarization', 'coa_phase', 'amplitude_u', 'delta_tc']
    return ExactTDIModel(names, amr_root=AMR_ROOT, response='exact')


@pytest.mark.parametrize('power,lower,upper', [
    (4.0, 1.0, 5.0),
    (4.0, 1.0, 20.0),
    (0.0, 1.0, 20.0),
])
def test_amplitude_power_law(power, lower, upper):
    """The three project priors normalise, integrate and invert correctly."""
    dist = UniformAmplitudePowerLaw(power=power, amplitude_u=(lower, upper))
    x = np.linspace(lower, upper, 400001)
    pdf = np.array([dist.pdf(amplitude_u=value) for value in x])
    total = np.trapezoid(pdf, x)
    assert total == pytest.approx(1.0, rel=2e-5)
    # the pdf and logpdf agree, and both vanish outside the support
    for value in (lower, 0.5 * (lower + upper), upper):
        assert np.exp(dist.logpdf(amplitude_u=value)) == pytest.approx(
            dist.pdf(amplitude_u=value), rel=1e-12)
    assert dist.pdf(amplitude_u=0.5 * lower) == 0.0
    assert dist.logpdf(amplitude_u=2.0 * upper) == -np.inf
    # the inverse cdf really is the inverse of the cdf
    q = np.linspace(0.01, 0.99, 51)
    mapped = np.array([dist.cdfinv(amplitude_u=value)['amplitude_u']
                       for value in q])
    assert np.all(mapped >= lower) and np.all(mapped <= upper)
    if power == 0.0:
        assert np.allclose(mapped, lower + q * (upper - lower), atol=1e-10)
    # a large draw has the right first moment
    rng = np.random.default_rng(20261019)
    draw = dist.rvs(size=200000)['amplitude_u']
    if power != 1.0:
        mean = ((1.0 - power) / (2.0 - power)
                * (upper ** (2.0 - power) - lower ** (2.0 - power))
                / (upper ** (1.0 - power) - lower ** (1.0 - power)))
    else:
        mean = (upper - lower) / np.log(upper / lower)
    assert draw.mean() == pytest.approx(mean, rel=0.01)


@requires_amr
def test_prior_volume_and_support(family_model):
    """The ellipsoid volume and the log prior behave on and off the support."""
    volume = family_model.prior_volume_ellipsoid
    assert volume == pytest.approx(0.48482888181455547, rel=1e-8)
    truth = dict(zip(family_model.variable_params,
                     list(family_model.truth_eta[:4])
                     + [1.0, 0.7, 6.6728030534, 0.0]))
    family_model.update(**truth)
    logprior = family_model.logprior
    assert np.isfinite(logprior)
    outside = dict(truth)
    outside['eta_0'] = 2.9
    outside['eta_1'] = 0.0
    outside['eta_2'] = 0.0
    outside['eta_3'] = 0.0
    family_model.update(**outside)
    assert family_model.logprior == -np.inf


@requires_amr
def test_family_likelihood_matches_campaign(family_model):
    """The family likelihood equals the campaign's own Run computation."""
    R = family_model.R
    truth = dict(zip(family_model.variable_params,
                     list(family_model.truth_eta[:4])
                     + [1.0, 0.7, 6.6728030534, 0.0]))
    family_model.update(**truth)
    value = family_model.loglikelihood

    eta6 = np.zeros(6)
    eta6[:4] = family_model.truth_eta[:4]
    for _, _, gd, z in R.evaluate(family_model.X, [eta6]):
        G = family_model.cp.asnumpy(
            R.gram_of(gd, family_model.cp.asarray([0])))[0]
        zz = family_model.cp.asnumpy(z[:, 0])
    c = R.OP.coefficients(np.cos(1.0), 0.7)
    z_c = np.conj(c) @ zz
    nn = float(np.real(np.conj(c) @ G @ c))
    s = R.A_min * np.sqrt(nn)
    u = 6.6728030534
    arg = u * R.A_min * abs(z_c)
    from scipy.special import i0e
    direct = float(np.log(i0e(arg)) + arg - 0.5 * (u * s) ** 2)
    assert value == pytest.approx(direct, rel=1e-10)

    # the recorded eta is the archived one
    import h5py
    with h5py.File(AMR_FILE, 'r') as fh:
        archived = np.asarray(fh.attrs['amr_injection_eta'], float)
    assert np.array_equal(family_model.truth_eta, archived)

    # the archived sufficient statistics is reproduced to the known pycbc
    # data-path difference (the archive was built on the older package)
    if os.path.isfile(SUFFICIENT):
        stored = np.load(SUFFICIENT)
        row = int(np.argmax(stored['twoF']))
        cell = int(stored['cell'][row])
        lag = int(stored['lag'][row])
        eta6 = np.zeros(6)
        eta6[:4] = stored['eta'][cell][:4]
        for _, twoF, _, _ in R.evaluate(family_model.X, [eta6]):
            two_f = family_model.cp.asnumpy(twoF)
        ratio = two_f[lag] / float(stored['twoF'][row])
        assert ratio == pytest.approx(1.0, abs=0.02)


@requires_amr
def test_exact_likelihood_matches_campaign(exact_model):
    """The pointwise exact likelihood equals the whitened inner product."""
    names = exact_model.variable_params
    truth = dict(zip(names, list(exact_model.truth_eta[:4])
                     + [1.0, 0.7, 0.3, 542.1005802811611 / 100.0, 0.0]))
    for scale in (1.0, 1.05, 0.95):
        point = dict(truth)
        point['eta_0'] = truth['eta_0'] + 0.0
        point['amplitude_u'] = truth['amplitude_u'] * scale
        exact_model.update(**point)
        value = exact_model.loglikelihood
        source = exact_model.ex.spectrum(exact_model._point(
            exact_model.current_params))
        weight = exact_model.ex.weight
        whitened_data = exact_model.data * np.sqrt(weight)
        whitened_signal = source * np.sqrt(weight)
        brute = float(
            np.sum(np.real(np.conj(whitened_data) * whitened_signal))
            - 0.5 * np.sum(np.abs(whitened_signal) ** 2))
        assert value == pytest.approx(brute, rel=1e-10)


@requires_family
def test_campaign_mode_matches_posterior_amr():
    """The campaign likelihood mode is the campaign's own PosteriorAMR.lnB."""
    names = ['eta_0', 'eta_1', 'eta_2', 'eta_3']
    model = ExactTDIModel(names, amr_root=AMR_ROOT, likelihood='campaign')
    post = model.PS.PosteriorAMR(
        model.R, model.X, model.seg, model.truth_eta)
    eta4 = np.asarray(model.truth_eta[:4], float)
    model.update(**dict(zip(names, eta4)))
    value = model.loglikelihood
    direct, _ = post.lnB([np.concatenate([eta4, [0.0, 0.0]])])
    assert np.isfinite(value)
    assert value == pytest.approx(float(direct[0]), rel=1e-6)


def _config_with_sampler(tmpdir, name, sampler_section):
    with open(EXAMPLE_CONFIG) as fh:
        text = fh.read()
    head = text.split('[sampler]')[0]
    path = os.path.join(tmpdir, 'amr-%s.ini' % name)
    with open(path, 'w') as fh:
        fh.write(head)
        fh.write('\n[sampler]\n' + sampler_section)
    return path


@requires_family
def test_model_runs_under_two_samplers(tmpdir):
    """The same model runs under nessai and under emcee."""
    cli = os.path.join(os.path.dirname(os.path.abspath(pycbc.__file__)),
                       '..', 'bin', 'inference', 'pycbc_inference')
    configs = {
        'emcee': 'name = emcee\nnwalkers = 20\nniterations = 20\n'
                 'checkpoint-interval = 120\n',
        'nessai': 'name = nessai\nnlive = 50\nmax_iteration = 20\n'
                  'stopping = 0.2\nplot = False\n',
    }
    for name, section in configs.items():
        config = _config_with_sampler(str(tmpdir), name, section)
        output = os.path.join(str(tmpdir), '%s.hdf' % name)
        env = dict(os.environ)
        env['PYTHONPATH'] = os.path.dirname(
            os.path.dirname(os.path.abspath(pycbc.__file__)))
        subprocess.run(
            [sys.executable, cli, '--config-file', config,
             '--output-file', output, '--force'],
            check=True, env=env, capture_output=True, timeout=1800)
        assert os.path.isfile(output)
        import h5py
        with h5py.File(output, 'r') as fh:
            assert 'samples' in fh
            assert len(fh['samples/loglikelihood']) > 0
