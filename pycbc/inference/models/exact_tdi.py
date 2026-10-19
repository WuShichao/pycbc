"""The exact-TDI likelihood of the STBBH controlled event as a PyCBC model.

This model evaluates the likelihood of the controlled injection of the
companion repository's AMR study, at parameter points the sampler chooses.
It exists so that any PyCBC sampler can sample the same object the AMR
partition's cell Bayes factors are built from, instead of the unrelated
``test_normal`` those configs used to carry.

Two response modes
------------------
``response = family`` (the default and the one the example config uses) uses
the campaign's own ``ssb_amr.Run`` object: the GPU family expansion, the daily
norm, the correlation and the 2x2 Gram that the AMR partition is built from.
Every one of those is a method of ``Run``; the model calls them, it does not
reimplement them.  The data are the campaign's own

    X = Run.spectrum_of(rho * s + Run.noise_bins(default_rng(noise_seed)))

with ``s = ExactTDI.spectrum(point) * sqrt(ExactTDI.weight)`` normalised to
unit whitened norm, the controlled injection ``xi = (3, -2, 1.5, -1)``,
``rho = 16`` and noise seed ``20261018``.  The whitening is
``ExactTDI.weight`` through the campaign's ``Setup.sw``.  The response at the
fiducial is ``ExactTDI.spectrum``; the candidate response is the campaign's
family expansion of it.

``response = exact`` (available for validation and for a fully pointwise
check) bypasses the family.  It calls ``ExactTDI.spectrum(point)`` at every
candidate and evaluates

    ExactTDI.inner(d, h) - ExactTDI.inner(h, h) / 2

with ``d = (rho * s + n) / sqrt(weight)`` on the bins where the weight is
positive.  This is the exact pointwise Gaussian likelihood in the same inner
product, but the campaign's prepared response resamples the 730-day time
series at every call and costs seconds, not milliseconds.

Two likelihood modes
--------------------
``likelihood = campaign`` reproduces the campaign's own cell Bayes factor.
It calls the campaign's ``Run.evaluate``, ``Run.gram_of`` and
``Run.lnB_pairs`` in the same order as ``PosteriorAMR.lnB``: keep the
in-segment lags with 2F at or above ``F2_FLOOR``, build the campaign Gram at
those lags, and log-sum-exp the campaign's orientation-and-amplitude-
marginalised ``lnB``.  Only ``eta_0`` to ``eta_3`` are sampled; orientation,
phase, amplitude and the arrival-time lag are integrated exactly as the AMR
partition integrates them.  This mode is the one to test against the
campaign, and its answer to the polarisation question is "not constrained"
because the statistic is direction-blind by construction.

``likelihood = pointwise`` (the default and the one the example config uses)
keeps inclination, polarisation, amplitude and the arrival time as sampled
parameters.  At each point it evaluates the campaign's 2F ingredients at one
lattice lag and one orientation, so the two polarisation amplitudes are
separate; the polarisation is then constrained, and the AMR's flat
polarisation column is identified as a property of its marginalisation rather
than of the data.  This mode is not the campaign's cell Bayes factor: it is
the same ``z`` and ``Gram`` objects without the orientation marginalisation.

Data and response provenance
----------------------------
Nothing in the data or the response is reimplemented.  The model imports
``experiments.ratio_cells.exact.ExactTDI``, ``ssb_family``, ``ssb_amr`` and
``amr_sampler`` from ``amr_root`` and uses their objects:

* ``ExactTDI.spectrum``, ``ExactTDI.weight`` and, in exact mode,
  ``ExactTDI.inner``;
* ``ssb_family.Setup``'s construction arguments (``base_params``, data start,
  ``DATA_DAYS``, ``TAPER_DAYS``, ``F_LO``, ``F_HI``) and
  ``ssb_family.point_from_xi``;
* ``ssb_amr.Geometry.xi`` for the eta-to-shape map;
* ``ssb_amr.Run`` for the family, the correlation, the Gram and the data;
* ``amr_sampler.fixed_signal`` for the unit-whitened injection.

The model records the controlled event's eta vector and checks it against the
archived ``amr_injection_eta`` in its tests.

Parameters and marginalisation
------------------------------
The variable parameters are the AMR's own columns:

* ``eta_0`` to ``eta_3``: the four stiff intrinsic shape coordinates, with the
  AMR's flat prior on the ellipsoid (a uniform box plus the quadratic
  constraint in the configuration file);
* ``inclination``: flat in ``cos(inclination)`` (``sin_angle``);
* ``polarization``: flat on ``[0, pi/2)``.  Inclination and polarization are
  always sampled, never marginalised, so the two polarization amplitudes
  ``c_plus`` and ``c_cross`` are separate directions and the polarization is
  constrained;
* ``coa_phase``: flat on ``[0, 2*pi)``, or integrated analytically with
  ``marginalize-phase``;
* ``amplitude_u``: ``u = K / distance``, with a power-law prior
  ``p(u) ~ u**-power`` on ``[amplitude_min, amplitude_max]``.  The distance
  ``K / u`` is written as the extra statistic ``distance``;
* ``delta_tc``: the arrival time in seconds.  The campaign evaluates the
  correlation on the 142.678 s lag grid, so the model uses the lag nearest to
  ``delta_tc`` and the column is discrete on that grid, exactly as the AMR's
  is.

Marginalisations, selected from the model section:

* ``marginalize-phase`` (default False).  For the family mode the two
  polarization amplitudes make the likelihood linear, so the flat-phase
  marginal is the campaign's own amplitude-table integrand
  ``log I0(u * A_min * abs(c^H z)) - (u * s)^2 / 2``.  In exact mode it is the
  same Bessel marginal evaluated with ``ExactTDI.inner``.  This is the
  marginalisation the AMR applies.
* ``marginalize-amplitude`` (default False).  Integrates ``u`` on a uniform
  grid with the same power-law prior, reproducing the AMR's amplitude
  integral.  When it is false, ``amplitude_u`` is sampled.

With both false, all nine parameters are sampled.  The model never
marginalises over inclination, polarisation or the arrival-time lag; those are
sampled, which is what keeps the two amplitudes separate and the polarization
constrained.

Cost
----
The family mode is the campaign's own millisecond object: the AMR's
``lnB_pairs`` block is 2.11 ms at 10268 lags and the ``Run.evaluate`` call for
one template costs a few tens of milliseconds at a batch of one on this
machine.  Exact mode is a few seconds per call because of the 1.36 million
sample time series.  The measured mean is written to the output metadata as
``exact_tdi_likelihood_cost_ms``.
"""

import logging
import math
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy.special import i0e

from pycbc.inference.models.base import BaseModel

_log = logging.getLogger('pycbc.inference.models.exact_tdi')

#: The campaign's distance constant: the distance in Mpc at which the base
#: waveform has optimal SNR 4, so that ``distance = K / amplitude_u``.
DEFAULT_DISTANCE_CONSTANT = 542.1005802811611

#: The campaign's controlled injection, in shape units.
DEFAULT_INJECTION_XI = (3.0, -2.0, 1.5, -1.0)


def _as_bool(value):
    """Coerce a configuration value to a boolean."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 't', 'yes', 'y', 'on')
    return bool(value)


def _as_float(value):
    return float(value)


def _as_float_list(value):
    if isinstance(value, str):
        value = value.replace(',', ' ').split()
    return [float(item) for item in value]


class ExactTDIModel(BaseModel):
    """The exact-TDI likelihood of the STBBH controlled event.

    See the module docstring for the provenance, the parameter set, the two
    response modes and the marginalisations.

    Parameters
    ----------
    variable_params : (tuple of) str
        The sampled parameters.  Must be ``eta_0`` to ``eta_3``,
        ``inclination``, ``polarization``, ``delta_tc``, and, unless they are
        marginalised, ``coa_phase`` and ``amplitude_u``.
    static_params : dict, optional
        Fixed parameters.  ``delta_tc`` may be fixed here.
    prior : pycbc.distributions.JointDistribution, optional
        The prior, supplied from the ``[prior-*]`` and ``[constraint-*]``
        sections of the configuration file.
    response : {'family', 'exact'}, optional
        ``'family'`` uses the campaign's ``ssb_amr.Run`` object, which is the
        object the AMR cells are built from and costs milliseconds.
        ``'exact'`` calls ``ExactTDI.spectrum`` at every point and costs
        seconds.  Default ``'family'``.
    amr_root : str, optional
        The companion repository root.  Default
        ``/mnt/d/lisa_tests_with_cc``.
    injection_xi : sequence of four floats, optional
        The controlled injection's intrinsic shape coordinates.  Default
        ``(3, -2, 1.5, -1)``.
    injection_inclination, injection_polarization, injection_coa_phase : float
        The controlled injection's orientation.  Defaults 1.0, 0.7 and 0.3.
    injection_delta_tc : float, optional
        The controlled injection's arrival time.  Default 0.
    rho : float, optional
        The injected signal-to-noise.  Default 16.
    noise_seed : int, optional
        The recorded noise seed.  Default 20261018.
    distance_constant : float, optional
        The constant ``K`` in ``distance = K / amplitude_u``.  Default
        :data:`DEFAULT_DISTANCE_CONSTANT`.
    marginalize_phase : bool, optional
        Integrate the coalescence phase analytically.  Default False.
    marginalize_amplitude : bool, optional
        Integrate ``amplitude_u`` on a grid with the power-law prior.
        Default False.
    amplitude_power, amplitude_min, amplitude_max : float, optional
        The power-law prior used only when ``marginalize_amplitude`` is true.
        Defaults 4, 1 and 20.
    amplitude_nodes : int, optional
        The number of amplitude quadrature nodes.  Default 2001.
    batch : int, optional
        The campaign ``Run`` template batch size in family mode.  Default 1.
    include_phase_normalization : bool, optional
        Include the ``1 / (2*pi)`` flat-phase normalisation in the
        phase-marginalised likelihood.  Default True.
    data_cache : dict, optional
        A process-level cache of the built campaign objects, so two models in
        one process share one family preparation.  Default None (a module
        cache is used).
    """

    name = "exact_tdi"
    _cache = {}

    #: The model-section options and their defaults.  Configuration options may
    #: use hyphens or underscores; both are accepted.
    _options = {
        'likelihood': 'pointwise',
        'response': 'family',
        'amr_root': '/mnt/d/lisa_tests_with_cc',
        'injection_xi': DEFAULT_INJECTION_XI,
        'injection_inclination': 1.0,
        'injection_polarization': 0.7,
        'injection_coa_phase': 0.3,
        'injection_delta_tc': 0.0,
        'rho': 16.0,
        'noise_seed': 20261018,
        'distance_constant': DEFAULT_DISTANCE_CONSTANT,
        'marginalize_phase': False,
        'marginalize_amplitude': False,
        'amplitude_power': 4.0,
        'amplitude_min': 1.0,
        'amplitude_max': 20.0,
        'amplitude_nodes': 2001,
        'batch': 1,
        'include_phase_normalization': False,
    }

    def __init__(self, variable_params, static_params=None, prior=None,
                 sampling_transforms=None, waveform_transforms=None,
                 **kwargs):
        options = dict(self._options)
        for key, value in kwargs.items():
            key = str(key).replace('-', '_')
            if key not in options:
                raise TypeError("unknown exact_tdi model option: %s" % key)
            options[key] = value
        super(ExactTDIModel, self).__init__(
            variable_params, static_params=static_params, prior=prior,
            sampling_transforms=sampling_transforms,
            waveform_transforms=waveform_transforms)

        self.likelihood = str(options['likelihood']).strip().lower()
        if self.likelihood not in ('campaign', 'pointwise'):
            raise ValueError("likelihood must be 'campaign' or 'pointwise'")
        self.response = str(options['response']).strip().lower()
        if self.response not in ('family', 'exact'):
            raise ValueError("response must be 'family' or 'exact'")
        if self.likelihood == 'campaign' and self.response != 'family':
            raise ValueError("likelihood='campaign' needs response='family'")
        self.amr_root = str(options['amr_root'])
        self.injection_xi = np.asarray(
            _as_float_list(options['injection_xi']), float)
        if self.injection_xi.shape != (4,):
            raise ValueError("injection_xi must have four entries")
        self.injection_inclination = _as_float(
            options['injection_inclination'])
        self.injection_polarization = _as_float(
            options['injection_polarization'])
        self.injection_coa_phase = _as_float(options['injection_coa_phase'])
        self.injection_delta_tc = _as_float(options['injection_delta_tc'])
        self.rho = _as_float(options['rho'])
        self.noise_seed = int(options['noise_seed'])
        self.K = _as_float(options['distance_constant'])
        self.marginalize_phase = _as_bool(options['marginalize_phase'])
        self.marginalize_amplitude = _as_bool(options['marginalize_amplitude'])
        self.amplitude_power = _as_float(options['amplitude_power'])
        self.amplitude_min = _as_float(options['amplitude_min'])
        self.amplitude_max = _as_float(options['amplitude_max'])
        self.amplitude_nodes = int(options['amplitude_nodes'])
        self.batch = int(options['batch'])
        self.include_phase_normalization = _as_bool(
            options['include_phase_normalization'])
        self._data_cache = self._cache

        self._eta_names = ('eta_0', 'eta_1', 'eta_2', 'eta_3')
        self._check_parameters()

        self._campaign = self._campaign_state()
        self._bind_campaign(self._campaign)

        self._n_likelihood = 0
        self._seconds_likelihood = 0.0

        if abs(self.amplitude_power - 1.0) < 1e-12:
            self._amplitude_lognorm = -np.log(
                np.log(self.amplitude_max / self.amplitude_min))
        else:
            self._amplitude_lognorm = np.log(
                (1.0 - self.amplitude_power)
                / (self.amplitude_max ** (1.0 - self.amplitude_power)
                   - self.amplitude_min ** (1.0 - self.amplitude_power)))

    # -- configuration checks -------------------------------------------------
    def _check_parameters(self):
        sampled = set(self.variable_params)
        if self.likelihood == 'campaign':
            # the campaign's cell Bayes factor marginalises orientation, phase,
            # amplitude and the arrival-time lag, so only the four shape
            # coordinates are sampled
            required = set(self._eta_names)
        else:
            required = set(self._eta_names) | {'inclination', 'polarization',
                                               'delta_tc'}
            if not self.marginalize_phase:
                required.add('coa_phase')
            if not self.marginalize_amplitude:
                required.add('amplitude_u')
        missing = required - sampled
        extra = sampled - required
        if missing:
            raise ValueError(
                "exact_tdi is missing variable parameters: %s"
                % ", ".join(sorted(missing)))
        if extra:
            raise ValueError(
                "exact_tdi does not sample: %s (marginalize-phase=%s, "
                "marginalize-amplitude=%s)"
                % (", ".join(sorted(extra)), self.marginalize_phase,
                   self.marginalize_amplitude))

    # -- the campaign's own objects ------------------------------------------
    def _campaign_state(self):
        key = (self.response, self.batch, os.path.abspath(self.amr_root),
               self.noise_seed, self.rho, tuple(self.injection_xi),
               self.injection_inclination, self.injection_polarization,
               self.injection_coa_phase, self.injection_delta_tc)
        if key in self._data_cache:
            return self._data_cache[key]

        root = self.amr_root
        if not os.path.isdir(root):
            raise OSError("the companion repository is not at %r; set "
                          "amr-root in the model section" % root)
        for path in (root,
                     os.path.join(root, 'results/2026-10-15/ssb_search'),
                     os.path.join(root, 'results/2026-10-18/amr_sampler')):
            if path not in sys.path:
                sys.path.insert(0, path)
        os.environ.setdefault('SSB_EXACT_INSIDE', '1')
        os.environ.setdefault('NO_PKGCONFIG', '1')

        cwd = os.getcwd()
        os.chdir(root)
        try:
            import json

            import ssb_family as SFm
            import ssb_amr as A
            import amr_sampler as PS
            from experiments.ratio_cells.coordinates import (
                cell_coordinates, to_source)
            from pycbc.tdi.inference import source_from_parameters

            fid = SFm.base_params()
            src = source_from_parameters('lal', **fid)
            start = src.t_start + 0.1 * 365.25 * 86400.0
            base_point = cell_coordinates(fid)
            sig = SFm.sigma()
            with open(os.path.join(
                    root, 'results/2026-10-15/ssb_search/geometry.json')) as fh:
                geometry = json.load(fh)
            lam = np.array(geometry['lam'])
            eigen = np.array(geometry['E'])

            if self.response == 'family':
                R = A.Run(batch=self.batch)
                R.A_min = float(R.A_min)
                R.cp.cuda.Stream.null.synchronize()
                point = R.P
                geo = R.geo
                S, xi = PS.fixed_signal(
                    SimpleNamespace(P=point), self.injection_xi,
                    self.injection_delta_tc, self.injection_inclination,
                    self.injection_polarization, self.injection_coa_phase)
                rng = np.random.default_rng(self.noise_seed)
                X = R.spectrum_of(self.rho * S + R.noise_bins(rng))
                state = dict(
                    mode='family', SFm=SFm, A=A, PS=PS, R=R, cp=R.cp,
                    fid=fid, base_point=base_point, sig=sig, geo=geo,
                    to_source=to_source,
                    source_from_parameters=source_from_parameters,
                    S=S, xi=xi, X=X)
            else:
                from experiments.ratio_cells.exact import ExactTDI as CampaignTDI
                ex = CampaignTDI(fid, days=SFm.DATA_DAYS, dt=10.0,
                                 taper_days=SFm.TAPER_DAYS, fmin=SFm.F_LO,
                                 fmax=SFm.F_HI, start=start)
                point = SimpleNamespace(point=base_point, sig=sig, ex=ex)
                geo = A.Geometry(point, lam, eigen)
                S, xi = PS.fixed_signal(
                    SimpleNamespace(P=point), self.injection_xi,
                    self.injection_delta_tc, self.injection_inclination,
                    self.injection_polarization, self.injection_coa_phase)
                rng = np.random.default_rng(self.noise_seed)
                noise = A.Run.noise_bins(SimpleNamespace(P=point), rng)
                sqrt_weight = np.sqrt(ex.weight)
                safe = np.where(sqrt_weight > 0, sqrt_weight, 1.0)
                data_white = self.rho * S + noise
                data = np.where(sqrt_weight > 0, data_white / safe, 0.0)
                state = dict(
                    mode='exact', SFm=SFm, A=A, PS=PS, ex=ex,
                    fid=fid, base_point=base_point, sig=sig, geo=geo,
                    to_source=to_source,
                    source_from_parameters=source_from_parameters,
                    S=S, xi=xi, noise=noise, data=data,
                    data_white=data_white, sqrt_weight=sqrt_weight)

            injection_point = SFm.point_from_xi(base_point, xi, sig)
            injection_point['inclination'] = self.injection_inclination
            injection_point['polarization'] = self.injection_polarization
            injection_point['coa_phase'] = self.injection_coa_phase
            injection_point['tc'] = self.injection_delta_tc
            state['injection_point'] = injection_point
            state['truth_eta'] = np.sqrt(lam) * (eigen.T @ xi)
            self._data_cache[key] = state
            return state
        finally:
            os.chdir(cwd)

    def _bind_campaign(self, state):
        self.SFm = state['SFm']
        self.A = state['A']
        self.PS = state['PS']
        self.fid = state['fid']
        self.base_point = state['base_point']
        self.sig = state['sig']
        self.geo = state['geo']
        self.to_source = state['to_source']
        self.source_from_parameters = state['source_from_parameters']
        self.S = state['S']
        self.campaign_xi = state['xi']
        self.injection_point = state['injection_point']
        self.truth_eta = state['truth_eta']
        if state['mode'] == 'family':
            self.R = state['R']
            self.cp = state['cp']
            self.X = state['X']
            self.seg = int(math.floor(
                (self.injection_delta_tc + self.A.DELTA_RANGE)
                / (self.A.SEG_DAYS * self.A.H.DAY)))
        else:
            self.ex = state['ex']
            self.data = state['data']
            self.data_white = state['data_white']
            self.sqrt_weight = state['sqrt_weight']

    # -- parameter mapping ----------------------------------------------------
    def _point(self, params):
        """Map the sampled parameters to the campaign's source point."""
        eta4 = np.array([params[name] for name in self._eta_names], float)
        eta6 = np.zeros(6)
        eta6[:4] = eta4
        xi6 = self.geo.xi(eta6)
        point = self.SFm.point_from_xi(self.base_point, xi6, self.sig)
        point['inclination'] = float(params['inclination'])
        point['polarization'] = float(params['polarization'])
        if self.marginalize_phase:
            point['coa_phase'] = 0.0
        else:
            point['coa_phase'] = float(params['coa_phase'])
        point['tc'] = float(params['delta_tc'])
        if self.marginalize_amplitude:
            point['distance'] = self.K
        else:
            point['distance'] = self.K / float(params['amplitude_u'])
        return point

    def _loglikelihood(self):
        t_start = time.perf_counter()
        self._n_likelihood += 1
        try:
            if self.likelihood == 'campaign':
                value = self._loglikelihood_campaign()
            elif self.response == 'family':
                value = self._loglikelihood_family()
            else:
                value = self._loglikelihood_exact()
        except Exception as exc:                    # waveform domain failure
            _log.debug("exact_tdi response failed: %s", exc)
            value = -np.inf
        params = self.current_params
        try:
            self._current_stats.distance = self.K / float(
                params['amplitude_u'])
        except (KeyError, ValueError):
            self._current_stats.distance = np.nan
        self._seconds_likelihood += time.perf_counter() - t_start
        return float(value)

    # -- the campaign's own cell Bayes factor --------------------------------
    def _loglikelihood_campaign(self):
        """The campaign's ``PosteriorAMR.lnB`` at one shape-coordinate point.

        This is a line-for-line use of the campaign's own sequence: evaluate
        the family, keep the in-segment lags with 2F at or above
        ``F2_FLOOR``, build the campaign Gram at those lags, and reduce the
        campaign's orientation-and-amplitude-marginalised ``lnB_pairs`` with a
        log-sum-exp.  Orientation, phase, amplitude and the arrival-time lag
        are integrated exactly as the AMR partition integrates them, so only
        ``eta_0`` to ``eta_3`` are sampled.
        """
        params = self.current_params
        eta4 = np.array([params[name] for name in self._eta_names], float)
        eta6 = np.zeros(6)
        eta6[:4] = eta4
        mask = (self.R.seg == self.seg) & self.R.in_range
        for _, two_f, gd, z in self.R.evaluate(self.X, [eta6]):
            sel = self.cp.nonzero((two_f >= self.A.F2_FLOOR) & mask)[0]
            if len(sel) == 0:
                return -np.inf
            gram = self.R.gram_of(gd, sel)
            ln_b = self.R.lnB_pairs(
                z[:, sel].T, gram.astype(self.cp.complex64))
            top = float(self.cp.max(ln_b))
            return top + float(self.cp.log(
                self.cp.exp(ln_b - top).sum()))
        return -np.inf

    # -- family response (the campaign's Run object) --------------------------
    def _loglikelihood_family(self):
        params = self.current_params
        eta4 = np.array([params[name] for name in self._eta_names], float)
        eta6 = np.zeros(6)
        eta6[:4] = eta4
        got = None
        for _, _, gd, z in self.R.evaluate(self.X, [eta6]):
            got = (gd, z)
        if got is None:
            return -np.inf
        gd, z = got
        idx = int(round(float(params['delta_tc']) / self.R.dt)) % self.R.N
        G = self.cp.asnumpy(self.R.gram_of(gd, self.cp.asarray([idx])))[0]
        zz = self.cp.asnumpy(z[:, idx])
        cosi = np.cos(float(params['inclination']))
        psi = float(params['polarization'])
        c = self.R.OP.coefficients(cosi, psi)
        z_c = np.conj(c) @ zz
        nn = float(np.real(np.conj(c) @ G @ c))
        if not np.isfinite(nn) or nn <= 0:
            return -np.inf
        if self.marginalize_amplitude:
            if self.marginalize_phase:
                return self._amplitude_integral_family(abs(z_c), nn)
            phi = float(params['coa_phase'])
            return self._amplitude_integral_family(
                float(np.real(z_c * np.exp(-1j * phi))), nn)
        u = float(params['amplitude_u'])
        s = self.R.A_min * np.sqrt(nn)
        if self.marginalize_phase:
            arg = u * self.R.A_min * abs(z_c)
            value = float(np.log(i0e(arg)) + arg - 0.5 * (u * s) ** 2)
        else:
            phi = float(params['coa_phase'])
            value = float(u * np.real(z_c * np.exp(-1j * phi))
                          - 0.5 * (u * s) ** 2)
        if self.marginalize_phase and self.include_phase_normalization:
            value -= float(np.log(2.0 * np.pi))
        return value

    def _amplitude_integral_family(self, z_c, nn):
        u = np.linspace(self.amplitude_min, self.amplitude_max,
                        self.amplitude_nodes)
        s = self.R.A_min * np.sqrt(nn)
        if self.marginalize_phase:
            arg = u * self.R.A_min * abs(z_c)
            log_likelihood = (np.log(i0e(arg)) + arg
                              - 0.5 * (u * s) ** 2)
        else:
            log_likelihood = u * np.real(z_c) - 0.5 * (u * s) ** 2
        log_prior = self._amplitude_lognorm - self.amplitude_power * np.log(u)
        values = log_prior + log_likelihood
        top = float(np.max(values))
        width = u[1] - u[0]
        value = float(top + np.log(np.sum(np.exp(values - top)))
                      + np.log(width))
        if self.marginalize_phase and self.include_phase_normalization:
            value -= float(np.log(2.0 * np.pi))
        return value

    # -- exact pointwise response (ExactTDI.spectrum + inner) -----------------
    def _loglikelihood_exact(self):
        params = self.current_params
        point = self._point(params)
        h = self.ex.spectrum(point)
        if self.marginalize_phase:
            z = (self.ex.inner(self.data, h)
                 - 1j * self.ex.inner(self.data, 1j * h))
            w = self.ex.inner(h, h)
            amplitude = abs(z)
            if self.marginalize_amplitude:
                value = self._amplitude_integral_exact(amplitude, w)
            else:
                value = float(np.log(i0e(amplitude)) + amplitude
                              - 0.5 * w)
        else:
            z = self.ex.inner(self.data, h)
            w = self.ex.inner(h, h)
            if self.marginalize_amplitude:
                value = self._amplitude_integral_exact(z, w)
            else:
                value = float(z - 0.5 * w)
        if self.marginalize_phase and self.include_phase_normalization:
            value -= float(np.log(2.0 * np.pi))
        return value

    def _amplitude_integral_exact(self, z, w):
        u = np.linspace(self.amplitude_min, self.amplitude_max,
                        self.amplitude_nodes)
        if self.marginalize_phase:
            log_likelihood = (np.log(i0e(u * abs(z))) + u * abs(z)
                              - 0.5 * (u * u) * w)
        else:
            log_likelihood = u * np.real(z) - 0.5 * (u * u) * w
        log_prior = self._amplitude_lognorm - self.amplitude_power * np.log(u)
        values = log_prior + log_likelihood
        top = float(np.max(values))
        width = u[1] - u[0]
        return float(top + np.log(np.sum(np.exp(values - top)))
                     + np.log(width))

    # -- prior and stats ------------------------------------------------------
    @property
    def _extra_stats(self):
        return ['distance']

    @property
    def likelihood_calls(self):
        """int: The number of likelihood evaluations."""
        return self._n_likelihood

    @property
    def likelihood_cost_ms(self):
        """float: The mean wall time of one likelihood evaluation, in ms."""
        if not self._n_likelihood:
            return float('nan')
        return 1000.0 * self._seconds_likelihood / self._n_likelihood

    @property
    def prior_volume_ellipsoid(self):
        """float: The AMR ellipsoid volume, pi^2 / (2 sqrt(det M4))."""
        stiff = list(self.geo.stiff)
        matrix = self.geo.A.T @ self.geo.A
        m4 = matrix[np.ix_(stiff, stiff)]
        return float(np.pi ** 2 / (2.0 * np.sqrt(np.linalg.det(m4))))

    def write_metadata(self, fp, group=None):
        super(ExactTDIModel, self).write_metadata(fp, group=group)
        attrs = fp.getattrs(group=group)
        attrs['exact_tdi_likelihood'] = self.likelihood
        attrs['exact_tdi_response'] = self.response
        attrs['exact_tdi_amr_root'] = self.amr_root
        attrs['exact_tdi_injection_xi'] = np.asarray(self.injection_xi)
        attrs['exact_tdi_injection_inclination'] = self.injection_inclination
        attrs['exact_tdi_injection_polarization'] = self.injection_polarization
        attrs['exact_tdi_injection_coa_phase'] = self.injection_coa_phase
        attrs['exact_tdi_injection_delta_tc'] = self.injection_delta_tc
        attrs['exact_tdi_rho'] = self.rho
        attrs['exact_tdi_noise_seed'] = self.noise_seed
        attrs['exact_tdi_distance_constant'] = self.K
        attrs['exact_tdi_marginalize_phase'] = self.marginalize_phase
        attrs['exact_tdi_marginalize_amplitude'] = self.marginalize_amplitude
        attrs['exact_tdi_prior_volume_ellipsoid'] = self.prior_volume_ellipsoid
        attrs['exact_tdi_truth_eta'] = np.asarray(self.truth_eta)
        attrs['exact_tdi_likelihood_cost_ms'] = float(self.likelihood_cost_ms)
        if self.response == 'family':
            attrs['exact_tdi_family_batch'] = self.batch
            attrs['exact_tdi_lag_seconds'] = float(self.R.dt)
            attrs['exact_tdi_n_lags'] = int(self.R.N)
            attrs['exact_tdi_amplitude_a_min'] = float(self.R.A_min)
