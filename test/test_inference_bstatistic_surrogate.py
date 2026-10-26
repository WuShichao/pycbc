"""The bstatistic_surrogate inference model on a small synthetic family."""
import functools
import os
import tempfile
import warnings

import numpy
import pytest

from pycbc.filter.bstatistic import (AmplitudeTable, DistancePowerLaw,
                                     OrientationNodes,
                                     log_orientation_marginal)
from pycbc.inference.models.bstatistic_surrogate import BStatisticSurrogate
from pycbc.tdi.linear_exponent_surrogate import (DailyGram,
                                                 LinearExponentSurrogate)
from pycbc.tdi.ssb_yearly_harmonics import SSBGrid
from pycbc.tmpltbank.fstatistic_metric import orientation_weights

DAY = 86400.0
PRIOR = DistancePowerLaw(0.05, 50.0, power=2.0)
TRUTH = dict(delta_tc=3.3e4, inclination=1.0, polarization=0.4,
             distance=0.5, coa_phase=1.0)


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
    signal = h * numpy.exp(2j * TRUTH['coa_phase']) \
        * numpy.tensordot(w, y, axes=1)
    rng = numpy.random.default_rng(seed)
    n = numpy.sqrt(noise) * (rng.normal(size=signal.shape)
                             + 1j * rng.normal(size=signal.shape))
    return signal + n, signal


@functools.lru_cache()
def table_file():
    """One amplitude table for every test: it depends only on the prior's
    power."""
    path = os.path.join(tempfile.mkdtemp(), 'amplitude_table.hdf')
    AmplitudeTable(PRIOR, y_max=20.0).write(path)
    return path


def model(fam, daily, data, noise=1.0, window=(-2e5, 2e5), oversampling=4):
    reference = dict(x=0.0, eclipticlongitude=1.0, eclipticlatitude=0.2)
    scales = dict(x=1.0, eclipticlongitude=1e-3, eclipticlatitude=1e-3)
    return BStatisticSurrogate(['x'], fam, daily, data, noise, reference,
                               scales, PRIOR, window,
                               oversampling=oversampling,
                               orientation_nodes=(16, 8),
                               amplitude_table=table_file())


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
    kept = numpy.nonzero(numpy.isfinite(lnb[:len(m.delays)]))[0]
    assert len(kept) > 0
    for k in (kept[0], kept[len(kept) // 2], kept[-1]):
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
        draws = [m.reconstruct(seed=s) for s in range(200)]
    # 99 per cent intervals: three 90 per cent ones fail together a quarter
    # of the time by construction
    for name in ('delta_tc', 'distance', 'inclination'):
        values = numpy.array([d[name] for d in draws])
        low, high = numpy.percentile(values, [0.5, 99.5])
        assert low < TRUTH[name] < high, name
    for d in draws:
        assert 0 <= d['polarization'] < 2 * numpy.pi
        assert 0 <= d['coa_phase'] < numpy.pi
        assert 0 <= d['inclination'] <= numpy.pi


def test_reference_must_match_the_family():
    fam, daily = family()
    with pytest.raises(ValueError):
        BStatisticSurrogate(['x'], fam, daily, numpy.zeros((2, fam.size)),
                            1.0, dict(x=0.0), dict(x=1.0), PRIOR, (0, 1e5),
                            amplitude_table=False)


def test_the_model_runs_on_the_gpu():
    cupy = pytest.importorskip('cupy')
    from pycbc.scheme import CUPYScheme
    try:
        scheme = CUPYScheme()
    except Exception:
        pytest.skip('no usable GPU')
    fam, daily = family()
    data, _ = inject(fam, numpy.zeros(3))
    m = model(fam, daily, data)
    m.update(x=0.1)
    expected = m.loglikelihood
    with scheme:
        gpu = LinearExponentSurrogate(
            fam.grid, fam.coarse_frequencies, fam.R0, fam.L, fam.b0, fam.t0,
            fam.D, fam.T, kmax=3)
        mg = model(gpu, DailyGram(gpu), cupy.asarray(data))
        mg.update(x=0.1)
        assert mg.loglikelihood == pytest.approx(expected, abs=1e-8)
        rec = mg.reconstruct(seed=1)
    assert 0 <= rec['polarization'] < 2 * numpy.pi


CONFIG = """
[model]
name = bstatistic_surrogate
low-frequency-cutoff = 0.0150
high-frequency-cutoff = 0.0153
intrinsic-parameters = mchirp q spin1z spin2z
time-window = -259200 259200
distance-min = 1
distance-max = 1000
grid-periods = 1
coarse-bins = 200
orientation-nodes = 16 8
reference-mchirp = 31.831389387890596

[variable_params]
mchirp =

[static_params]
q = 2.0711690321562553
spin1z = -0.1493668476836033
spin2z = -0.7820976081717529
eclipticlongitude = 3.5763564997428436
eclipticlatitude = 0.3429283438977949
f_lower = 0.015
tdi_f_upper = 0.0155
tdi_t_start = 864000

[prior-mchirp]
name = uniform
min-mchirp = 31.8
max-mchirp = 31.86

[waveform_transforms-mass1+mass2]
name = mchirp_q_to_mass1_mass2
"""


def stbbh_data(days=120.0, distance=20.0, seed=3):
    """An exact sparse-TDI stellar-mass binary in LISA A and E noise."""
    from pycbc.conversions import mass1_from_mchirp_q, mass2_from_mchirp_q
    from pycbc.psd.analytical_space import analytical_psd_lisa_tdi_AE
    from pycbc.tdi.inference import sparse_tdi_fd_det_sequence
    from pycbc.types import FrequencySeries
    df = 1 / (days * DAY)
    q, mc = 2.0711690321562553, 31.831389387890596
    source = dict(mass1=mass1_from_mchirp_q(mc, q),
                  mass2=mass2_from_mchirp_q(mc, q), spin1z=-0.1493668476836033,
                  spin2z=-0.7820976081717529,
                  eclipticlongitude=3.5763564997428436,
                  eclipticlatitude=0.3429283438977949,
                  inclination=1.2870182448496255,
                  polarization=0.5011808775099882,
                  coa_phase=2.2688697243830247, distance=distance,
                  f_lower=0.015, tdi_f_upper=0.0155, tdi_t_start=10 * DAY)
    kmin, kmax = int(numpy.ceil(0.0135 / df)), int(numpy.floor(0.0168 / df))
    freqs = numpy.arange(kmin, kmax + 1) * df
    signal = sparse_tdi_fd_det_sequence(
        ifos=('A', 'E'), sample_points=freqs, approximant='TDISparse',
        tdi_source='lal', tdi_approximant='IMRPhenomD',
        tdi_band_edges=[0.0135, 0.0168], tdi_delta_f=df, t_obs_start=0.0,
        t_obs_end=days * DAY, tdi_epoch=0.0, tdi_channels=('A', 'E'),
        **source)
    n = int(round(days * DAY * 0.2)) // 2 + 1
    psd = numpy.asarray(analytical_psd_lisa_tdi_AE(n, df, df, tdi='2.0'))
    rng = numpy.random.default_rng(seed)
    data, psds, snr2 = {}, {}, 0.0
    for c in ('A', 'E'):
        d = numpy.zeros(n, complex)
        s = numpy.asarray(signal[c])
        sd = numpy.sqrt(psd[kmin:kmax + 1] / (4 * df))
        d[kmin:kmax + 1] = s + sd * (rng.normal(size=len(s))
                                     + 1j * rng.normal(size=len(s)))
        band = (freqs >= 0.015) & (freqs <= 0.0153)
        snr2 += numpy.sum((4 * df / psd[kmin:kmax + 1]
                           * numpy.abs(s) ** 2)[band])
        data['LISA_' + c] = FrequencySeries(d, delta_f=df, epoch=0.0)
        psds['LISA_' + c] = FrequencySeries(psd, delta_f=df, epoch=0.0)
    return data, psds, snr2, mc


def test_from_config_on_an_exact_tdi_injection():
    """Built from a configuration and exact sparse-TDI data, the model
    peaks at the injected chirp mass with ln B near SNR^2 / 2 in the
    band, and recovers the injected distance."""
    from pycbc.inference import models
    from pycbc.workflow import WorkflowConfigParser
    cp = WorkflowConfigParser()
    cp.read_string(CONFIG.replace(
        'grid-periods = 1',
        'grid-periods = 1\namplitude-table-file = ' + table_file()))
    data, psds, snr2, mc = stbbh_data()
    m = models.read_from_config(cp, data=data, psds=psds)
    unit = m.scales['mchirp']
    values = {}
    for k in (-1, 0, 1):
        m.update(mchirp=mc + k * unit)
        values[k] = m.loglikelihood
    assert values[0] > values[-1] and values[0] > values[1]
    assert abs(values[0] - 0.5 * snr2) < 5 * numpy.sqrt(snr2) + 50
    m.update(mchirp=mc)
    distance = [m.reconstruct(seed=s)['distance'] for s in range(30)]
    low, high = numpy.percentile(distance, [5, 95])
    assert low < 20.0 < high


def test_from_config_refuses_unknown_options():
    from pycbc.inference import models
    from pycbc.workflow import WorkflowConfigParser
    cp = WorkflowConfigParser()
    cp.read_string(CONFIG.replace(
        'grid-periods = 1', 'grid-periods = 1\nnlive = 10'))
    with pytest.raises(ValueError, match='unknown option'):
        models.read_from_config(cp, data={}, psds={})


def test_amr_sampler_integrates_the_model():
    """The amr sampler's evidence and posterior of the intrinsic coordinate
    agree with a direct integral of the model over its prior."""
    from pycbc.distributions import JointDistribution, Uniform
    from pycbc.inference.sampler.amr import AMRSampler
    fam, daily = family()
    data, _ = inject(fam, numpy.array([0.3, 0.0, 0.0]))
    prior = JointDistribution(['x'], Uniform(x=(-1.0, 1.0)))
    reference = dict(x=0.0, eclipticlongitude=1.0, eclipticlatitude=0.2)
    scales = dict(x=1.0, eclipticlongitude=1e-3, eclipticlatitude=1e-3)
    m = BStatisticSurrogate(['x'], fam, daily, data, 1.0, reference, scales,
                            PRIOR, (-2e5, 2e5), oversampling=4,
                            orientation_nodes=(16, 8), prior=prior,
                            amplitude_table=table_file())
    grid = numpy.linspace(-1, 1, 4001)
    values = []
    for x in grid:
        m.update(x=x)
        values.append(m.loglikelihood)
    values = numpy.array(values)
    top = values.max()
    density = numpy.exp(values - top)
    logz = top + numpy.log(numpy.trapz(density, grid) / 2.0)
    sampler = AMRSampler(m, mass_fraction=1e-5, max_evaluations=4000,
                         posterior_samples=4000, seed=2)
    sampler.run()
    assert sampler.logz == pytest.approx(logz, abs=max(
        3 * sampler.logz_err, 1e-3))
    mean = numpy.trapz(grid * density, grid) / numpy.trapz(density, grid)
    assert numpy.mean(sampler.samples['x']) == pytest.approx(mean, abs=2e-3)


def test_shared_computation_matches_one_point_at_a_time():
    """On the exact sparse-TDI injection, the log likelihoods of a batch of
    points from one shared NUFFT grid agree with the model's per-point
    ones, inside the region the grid covers; points outside it fall back
    to the per-point path."""
    from pycbc.inference import models
    from pycbc.workflow import WorkflowConfigParser
    cp = WorkflowConfigParser()
    cp.read_string(CONFIG.replace(
        'grid-periods = 1',
        'grid-periods = 1\namplitude-table-file = ' + table_file()))
    data, psds, snr2, mc = stbbh_data()
    m = models.read_from_config(cp, data=data, psds=psds)
    unit = m.scales['mchirp']
    region = [dict(mchirp=mc + u * unit) for u in numpy.linspace(-1, 1, 41)]
    full = []
    for p in region:
        q = dict(m.static_params)
        q.update(p)
        full.append(q)
    m.share(full)
    points = [dict(mchirp=mc + u * unit)
              for u in (-0.8, -0.3, 0.0, 0.45, 0.9, 3.0)]
    batch = m.loglikelihoods(points)
    single = []
    for p in points:
        m.update(**p)
        single.append(m.loglikelihood)
    single = numpy.array(single)
    near = single > single.max() - 20
    assert numpy.abs(batch - single)[near].max() < 0.02
    # the region's points come from the shared grid itself
    shared = m._shared_regions[-1]
    inner = []
    for p in points[:5]:
        q = dict(m.static_params)
        q.update(p)
        inner.append(q)
    assert shared.inside(inner).all()
    own = shared(inner)
    assert numpy.isfinite(own).all()
    stats = m.shared_statistics
    assert stats['regions'] == 1 and stats['regions_refused'] == 0
    assert stats['shared_points'] >= 5 and stats['single_points'] >= 1
    assert numpy.abs(own - single[:5])[near[:5]].max() < 0.02
    # the point beyond the region is computed one at a time, exactly
    assert batch[-1] == pytest.approx(single[-1], abs=1e-9)
    # the grid holds no time the region's shifted delays never reach
    _, shift, _ = shared.correlator.project(shared._thetas(full),
                                            shared.grid.frame)
    lo, hi = shared.grid.time_limits()
    pad = 0.1 * (m.delays[-1] - m.delays[0] + numpy.ptp(shift)) \
        + 4 * m.grid_step
    assert lo >= m.delays[0] + shift.min() - pad
    assert hi <= m.delays[-1] + shift.max() + pad
    # a grid beyond the memory allowed is refused, and its size kept
    with pytest.raises(MemoryError):
        m.share(full, grid_options=dict(memory_limit=1.0))
    assert m.shared_statistics['regions_refused'] == 1
    assert m.shared_statistics['largest_refused_gb'] > 0


def test_a_sky_block_gives_the_model_the_exact_response():
    """With a sky block the model's response at a sky offset is the exact
    one to the block's tolerance, and the shared computation agrees with
    the per-point one there."""
    from pycbc.inference import models
    from pycbc.workflow import WorkflowConfigParser
    lon = 3.5763564997428436
    config = (CONFIG.replace('grid-periods = 1', 'grid-periods = 1\n'
                             'harmonics = 8\nsky-block-radius = 0.02\n'
                             'sky-block-points = 200\n'
                             'reference-eclipticlongitude = %r\n'
                             'amplitude-table-file = ' % lon + table_file())
              .replace('[variable_params]\nmchirp =',
                       '[variable_params]\nmchirp =\neclipticlongitude =')
              .replace('\neclipticlongitude = %r\n' % lon, '\n')
              .replace('[waveform_transforms', '[prior-eclipticlongitude]\n'
                       'name = uniform\nmin-eclipticlongitude = %r\n'
                       'max-eclipticlongitude = %r\n\n[waveform_transforms'
                       % (lon - 0.02, lon + 0.02)))
    cp = WorkflowConfigParser()
    cp.read_string(config)
    data, psds, snr2, mc = stbbh_data()
    m = models.read_from_config(cp, data=data, psds=psds)
    block = m.surrogate.sky_block
    assert block is not None and block.error <= block.tolerance
    off = numpy.array([[0.01 / m.scales['eclipticlongitude'], 0.0]])
    exact = block.exact(off) if len(block.frequencies) == len(
        block.output_frequencies) else None
    if exact is not None:
        got = m.surrogate.responses(off)
        got = numpy.asarray(got.get() if hasattr(got, 'get') else got)
        assert numpy.linalg.norm(got - exact) <= 3 * block.tolerance \
            * numpy.linalg.norm(exact)
    unit = m.scales['mchirp']
    points = [dict(mchirp=mc + u * unit, eclipticlongitude=lon + d)
              for u, d in zip(numpy.linspace(-1, 1, 41),
                              numpy.linspace(-0.015, 0.015, 41)[::-1])]
    full = []
    for p in points:
        q = dict(m.static_params)
        q.update(p)
        full.append(q)
    m.share(full)
    batch = m.loglikelihoods(points[::8])
    single = []
    for p in points[::8]:
        m.update(**p)
        single.append(m.loglikelihood)
    single = numpy.array(single)
    near = single > single.max() - 20
    assert numpy.abs(batch - single)[near].max() < 0.02


def test_a_recentred_model_is_the_model_built_there():
    """A model recentred on another sky, sharing the family's intrinsic
    part, gives the log likelihoods of a model built with that reference
    sky, at points of it."""
    from pycbc.inference import models
    from pycbc.workflow import WorkflowConfigParser
    lon, lat = 3.5763564997428436, 0.3429283438977949
    shifted = lon + 0.03

    def build(longitude):
        cp = WorkflowConfigParser()
        cp.read_string(CONFIG.replace(
            'grid-periods = 1', 'grid-periods = 1\nharmonics = 8\n'
            'amplitude-table-file = ' + table_file()).replace(
            'eclipticlongitude = %r' % lon,
            'eclipticlongitude = %r' % longitude))
        return models.read_from_config(cp, data=data, psds=psds)
    data, psds, snr2, mc = stbbh_data()
    there = build(shifted)
    moved = build(lon).recentred(shifted, lat, 0.02, sky_block_points=200)
    assert moved.reference['eclipticlongitude'] == shifted
    unit = moved.scales['mchirp']
    points = [dict(mchirp=mc + u * unit, eclipticlongitude=shifted)
              for u in (-0.5, 0.0, 0.7)]
    numpy.testing.assert_allclose(moved.loglikelihoods(points),
                                  there.loglikelihoods(points), atol=0.02)


def test_a_sky_block_search_finds_the_source_in_its_block():
    """Two blocks of the sky, the injection inside the second: the merged
    candidates put the source there, once."""
    from pycbc.filter.sky_block_search import (sky_block_centres,
                                               sky_block_search)
    from pycbc.inference import models
    from pycbc.workflow import WorkflowConfigParser
    lon, lat = 3.5763564997428436, 0.3429283438977949
    cp = WorkflowConfigParser()
    cp.read_string(CONFIG.replace(
        'grid-periods = 1', 'grid-periods = 1\nharmonics = 8\n'
        'amplitude-table-file = ' + table_file()))
    data, psds, snr2, mc = stbbh_data()
    m = models.read_from_config(cp, data=data, psds=psds)
    radius = 0.02
    centres = sky_block_centres((lon - 0.05, lon + 0.03),
                                (lat - 0.01, lat + 0.01), radius)
    assert len(centres) == 2
    inside = [abs(numpy.angle(numpy.exp(1j * (c[0] - lon)))) * numpy.cos(
        c[1]) <= radius for c in centres]
    assert inside == [False, True]
    s = m.static_params
    unit = m.scales['mchirp']
    region = numpy.array([[mc + u * unit, s['q'], s['spin1z'], s['spin2z']]
                          for u in numpy.linspace(-2, 2, 21)])
    found, blocks = sky_block_search(m, centres, radius, region, sky_step=1.0,
                                     top=5, sky_block_points=200)
    assert len(blocks) == 2
    best = found[0]
    assert best['block'] == 1
    assert abs(best['longitude'] - lon) < radius
    # the source once: no other candidate this close to it
    assert all(abs(c['longitude'] - best['longitude']) > 2 * radius
               or abs(c['time'] - best['time']) > 600 for c in found[1:])


def test_a_sky_block_mixture_takes_each_point_from_its_block():
    from pycbc.inference import models
    from pycbc.inference.models.bstatistic_surrogate import SkyBlockMixture
    from pycbc.workflow import WorkflowConfigParser
    lon, lat = 3.5763564997428436, 0.3429283438977949
    cp = WorkflowConfigParser()
    cp.read_string(CONFIG.replace(
        'grid-periods = 1', 'grid-periods = 1\nharmonics = 8\n'
        'amplitude-table-file = ' + table_file()))
    data, psds, snr2, mc = stbbh_data()
    m = models.read_from_config(cp, data=data, psds=psds)
    blocks = [m.recentred(lon - 0.04, lat, 0.02, sky_block_points=200),
              m.recentred(lon, lat, 0.02, sky_block_points=200)]
    mix = SkyBlockMixture(blocks)
    points = [dict(mchirp=mc, eclipticlongitude=lon + d)
              for d in (-0.05, -0.035, -0.01, 0.005)]
    numpy.testing.assert_array_equal(mix.block_of(points), [0, 0, 1, 1])
    got = mix.loglikelihoods(points)
    own = [blocks[j].loglikelihoods([p])[0]
           for j, p in zip((0, 0, 1, 1), points)]
    numpy.testing.assert_allclose(got, own)
    mix.update(**points[2])
    assert mix.loglikelihood == pytest.approx(own[2])
    # the source's own block gives it the larger likelihood
    assert got[3] > got[0]
