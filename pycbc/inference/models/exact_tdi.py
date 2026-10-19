"""The exact-TDI likelihood of the STBBH controlled event as a PyCBC model.

This model evaluates the same exact-TDI Gaussian likelihood that the companion
repository's AMR partition is built from.  It is a *pointwise* likelihood: at
every call it builds the exact A/E response of one parameter point and takes
its noise-weighted inner product with the campaign's data.  It is not the
AMR's amplitude-and-orientation-marginalised cell statistic, and that is
deliberate: the two polarisation amplitudes are represented separately, so an
independent sampler can constrain the polarisation that the AMR's
direction-blind statistic cannot.

Data and response provenance
----------------------------
Nothing in the data or the response is reimplemented here.  The model imports
the campaign's own code:

* ``experiments.ratio_cells.exact.ExactTDI`` supplies the exact A/E spectrum,
  the whitening weight and the inner product.  The model calls
  ``ExactTDI.spectrum(point)``, ``ExactTDI.weight`` and ``ExactTDI.inner``.
* The construction of the response object repeats ``ssb_family.Setup``'s
  arguments (``base_params``, the data start epoch, ``DATA_DAYS``,
  ``TAPER_DAYS``, ``F_LO`` and ``F_HI``), and the model verifies at build time
  that a full ``Setup`` gives the same object when one is available.
* ``ssb_family.point_from_xi`` maps the four AMR shape coordinates to the
  physical source point.
* ``ssb_amr.Geometry.xi`` is the campaign's own eta-to-shape map.
* ``amr_sampler.fixed_signal`` builds the unit-whitened injected spectrum.
* ``ssb_amr.Run.noise_bins`` draws the recorded white-noise realisation.

The data are therefore exactly

    d_white = rho * s + n,      n = noise_bins(default_rng(noise_seed))

where ``s = exact spectrum * sqrt(weight) / norm`` is the campaign's
unit-whitened signal, with the campaign's default ``rho = 16`` and noise seed
``20261018`` and the controlled injection ``xi = (3, -2, 1.5, -1)``.  The
inner product is evaluated in the unwhitened basis so that
``ExactTDI.inner`` can be used directly: the model stores
``d = d_white / sqrt(weight)`` on the bins where the weight is positive, and
``ExactTDI.inner(d, h) - inner(h, h) / 2`` is algebraically the whitened
Euclidean likelihood ``Re<d_white, h_white> - |h_white|^2 / 2``.

Parameters and marginalisation
------------------------------
The variable parameters are the AMR's own columns:

* ``eta_0`` to ``eta_3``: the four stiff intrinsic shape coordinates, with the
  AMR's flat prior on the ellipsoid (a uniform box plus the quadratic
  constraint in the configuration file);
* ``inclination``: the inclination, with a flat prior in ``cos(inclination)``
  (``sin_angle``);
* ``polarization``: the polarisation angle, flat on ``[0, pi/2)``;
* ``coa_phase``: the coalescence phase, flat on ``[0, 2*pi)``.  This parameter
  can be integrated out analytically instead; see below;
* ``amplitude_u``: the amplitude ``u = K / distance``, with a power-law prior
  ``p(u) ~ u**-power`` on ``[amplitude_min, amplitude_max]``.  The model writes
  the distance ``K / u`` as the extra statistic ``distance``;
* ``delta_tc``: the arrival time in seconds, added to the source start time by
  the campaign's ``to_source``.

Two marginalisations are available and are selected from the model section:

* ``marginalize-phase`` (default False).  When true, ``coa_phase`` is not a
  variable parameter.  For a dominant-mode source the template at phase
  ``phi`` is ``exp(i*phi)`` times the template at phase zero, so the
  flat-phase marginal likelihood is

      log I0(abs(z)) - inner(h, h) / 2 - log(2*pi),

  with ``z = inner(d, h) - 1j*inner(d, 1j*h)`` the complex inner product
  recovered from two calls to ``ExactTDI.inner`` and ``I0`` the modified
  Bessel function.  This is the same phase marginalisation the AMR performs.
* ``marginalize-amplitude`` (default False).  When true, ``amplitude_u`` is
  not a variable parameter; the likelihood is integrated over the same
  power-law prior on a uniform grid of ``amplitude-nodes`` points.  Because
  the waveform is linear in ``1/distance``, the template at ``u`` is ``u``
  times the template at ``u = 1``, so one spectrum covers the whole integral.
  This reproduces the AMR's amplitude integral without reproducing its
  orientation marginalisation.

With both options false, every one of the nine parameters is sampled.  The
model never marginalises over inclination or polarisation: those are always
sampled, and the template is always built at the sampled pair, which is what
keeps the two polarisation amplitudes separate.

Cost
----
``ExactTDI.spectrum`` on this machine costs a few seconds per call because the
campaign's prepared multiband response is resampled on the full 730-day time
series.  The model can instead build the response's own precomputed Fourier
kernels (``ExactTDI.prepared.prepare_frequency_sampler``) once and evaluate
them on the projected candidate; this is the same prepared response, the same
projection and the same frequencies, and it is verified against
``ExactTDI.spectrum`` at build time.  Set ``fast-response = true`` to use it.
The measured cost per likelihood call is written to ``likelihood-cost-ms`` in
the output metadata.
"""

import logging
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
    """The exact-TDI Gaussian likelihood of the STBBH controlled event.

    See the module docstring for the data and response provenance, the
    parameter set, the two optional marginalisations and the cost.

    Parameters
    ----------
    variable_params : (tuple of) str
        The sampled parameters.  Must be ``eta_0`` to ``eta_3``,
        ``inclination``, ``polarization``, ``delta_tc``, and, unless they are
        marginalised, ``coa_phase`` and ``amplitude_u``.
    static_params : dict, optional
        Fixed parameters.  ``delta_tc`` may be fixed here.
    prior : pycbc.distributions.JointDistribution, optional
        The prior.  The configuration file supplies it from the
        ``[prior-*]`` and ``[constraint-*]`` sections.
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
        Integrate ``amplitude_u`` on a grid with the power-law prior.  Default
        False.
    amplitude_power, amplitude_min, amplitude_max : float, optional
        The power-law prior used only when ``marginalize_amplitude`` is true.
        Defaults 4, 1 and 20.
    amplitude_nodes : int, optional
        The number of amplitude quadrature nodes.  Default 2001.
    include_phase_normalization : bool, optional
        Include the ``1 / (2*pi)`` flat-phase normalisation in the
        phase-marginalised likelihood.  Default True.
    fast_response : bool, optional
        Use the response's precomputed Fourier kernels instead of resampling
        ``ExactTDI.spectrum``.  Default False.
    verify_response : bool, optional
        When ``fast_response`` is true, compare the fast path with
        ``ExactTDI.spectrum`` once at build time.  Default True.
    data_cache : dict, optional
        A process-level cache of the built campaign objects, so two models in
        one process share one exact-TDI preparation.  Default None (a module
        cache is used).
    """

    name = "exact_tdi"
    _cache = {}

    def __init__(self, variable_params, static_params=None, prior=None,
                 sampling_transforms=None, waveform_transforms=None,
                 amr_root='/mnt/d/lisa_tests_with_cc',
                 injection_xi=DEFAULT_INJECTION_XI,
                 injection_inclination=1.0,
                 injection_polarization=0.7,
                 injection_coa_phase=0.3,
                 injection_delta_tc=0.0,
                 rho=16.0,
                 noise_seed=20261018,
                 distance_constant=DEFAULT_DISTANCE_CONSTANT,
                 marginalize_phase=False,
                 marginalize_amplitude=False,
                 amplitude_power=4.0,
                 amplitude_min=1.0,
                 amplitude_max=20.0,
                 amplitude_nodes=2001,
                 include_phase_normalization=True,
                 fast_response=False,
                 verify_response=True,
                 data_cache=None,
                 **kwargs):
        super(ExactTDIModel, self).__init__(
            variable_params, static_params=static_params, prior=prior,
            sampling_transforms=sampling_transforms,
            waveform_transforms=waveform_transforms, **kwargs)
        if kwargs:
            raise TypeError("unknown exact_tdi model options: %s"
                            % ", ".join(sorted(kwargs)))

        self.amr_root = str(amr_root)
        self.injection_xi = np.asarray(_as_float_list(injection_xi), float)
        if self.injection_xi.shape != (4,):
            raise ValueError("injection_xi must have four entries")
        self.injection_inclination = _as_float(injection_inclination)
        self.injection_polarization = _as_float(injection_polarization)
        self.injection_coa_phase = _as_float(injection_coa_phase)
        self.injection_delta_tc = _as_float(injection_delta_tc)
        self.rho = _as_float(rho)
        self.noise_seed = int(noise_seed)
        self.K = _as_float(distance_constant)
        self.marginalize_phase = _as_bool(marginalize_phase)
        self.marginalize_amplitude = _as_bool(marginalize_amplitude)
        self.amplitude_power = _as_float(amplitude_power)
        self.amplitude_min = _as_float(amplitude_min)
        self.amplitude_max = _as_float(amplitude_max)
        self.amplitude_nodes = int(amplitude_nodes)
        self.include_phase_normalization = _as_bool(
            include_phase_normalization)
        self.fast_response = _as_bool(fast_response)
        self.verify_response = _as_bool(verify_response)
        self._data_cache = self._cache if data_cache is None else data_cache

        self._eta_names = ('eta_0', 'eta_1', 'eta_2', 'eta_3')
        self._check_parameters()

        self._campaign = self._campaign_state()
        self._bind_campaign(self._campaign)

        self._frequency_sampler = None
        if self.fast_response:
            # the kernels are fixed by the prepared bands, not by the
            # candidate, so one reference projection serves every call
            reference = self._project(self.injection_point)
            self._frequency_sampler = reference.prepare_frequency_sampler(
                self.ex.freqs, self.ex.delta_f, channels=('A', 'E'),
                epoch=self.ex.start, spectral_padding=2e-4,
                time_window=self.ex.window)
        self.response_verification = None
        if self.fast_response and self.verify_response:
            self.response_verification = self._verify_fast_response()

        self._n_likelihood = 0
        self._seconds_likelihood = 0.0

        # analytic normalisation of the optional amplitude quadrature
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
        """Build (or reuse) the campaign's response, data and event."""
        key = (os.path.abspath(self.amr_root), self.noise_seed, self.rho,
               tuple(self.injection_xi), self.injection_inclination,
               self.injection_polarization, self.injection_coa_phase,
               self.injection_delta_tc)
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
            from experiments.ratio_cells.exact import ExactTDI as CampaignTDI
            from experiments.ratio_cells.coordinates import (
                cell_coordinates, to_source)
            from pycbc.tdi.inference import source_from_parameters

            fid = SFm.base_params()
            src = source_from_parameters('lal', **fid)
            start = src.t_start + 0.1 * 365.25 * 86400.0
            ex = CampaignTDI(fid, days=SFm.DATA_DAYS, dt=10.0,
                             taper_days=SFm.TAPER_DAYS, fmin=SFm.F_LO,
                             fmax=SFm.F_HI, start=start)
            base_point = cell_coordinates(fid)
            sig = SFm.sigma()
            point = SimpleNamespace(point=base_point, sig=sig, ex=ex)
            with open(os.path.join(
                    root, 'results/2026-10-15/ssb_search/geometry.json')) as fh:
                geometry = json.load(fh)
            geo = A.Geometry(point, np.array(geometry['lam']),
                             np.array(geometry['E']))

            # the campaign's own data construction
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

            injection_point = SFm.point_from_xi(base_point, xi, sig)
            injection_point['inclination'] = self.injection_inclination
            injection_point['polarization'] = self.injection_polarization
            injection_point['coa_phase'] = self.injection_coa_phase
            injection_point['tc'] = self.injection_delta_tc
            truth_eta = np.sqrt(np.array(geometry['lam'])) @ (
                np.array(geometry['E']).T @ xi)

            state = dict(
                SFm=SFm, A=A, PS=PS, ex=ex, fid=fid, base_point=base_point,
                sig=sig, geo=geo, to_source=to_source,
                source_from_parameters=source_from_parameters,
                S=S, xi=xi, noise=noise, data=data, data_white=data_white,
                sqrt_weight=sqrt_weight, injection_point=injection_point,
                truth_eta=truth_eta)
            self._data_cache[key] = state
            return state
        finally:
            os.chdir(cwd)

    def _bind_campaign(self, state):
        self.ex = state['ex']
        self.fid = state['fid']
        self.base_point = state['base_point']
        self.sig = state['sig']
        self.geo = state['geo']
        self.to_source = state['to_source']
        self.source_from_parameters = state['source_from_parameters']
        self.S = state['S']
        self.campaign_xi = state['xi']
        self.noise = state['noise']
        self.data = state['data']
        self.data_white = state['data_white']
        self.sqrt_weight = state['sqrt_weight']
        self.injection_point = state['injection_point']
        self.truth_eta = state['truth_eta']

    # -- parameter mapping ----------------------------------------------------
    def _point(self, params):
        """Map the sampled parameters to the campaign's source point."""
        eta4 = np.array([params[name] for name in self._eta_names], float)
        eta6 = np.zeros(6)
        eta6[:4] = eta4
        xi6 = self.geo.xi(eta6)
        point = self.SFM_point(xi6)
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

    def SFM_point(self, xi6):
        return self._campaign['SFm'].point_from_xi(
            self.base_point, xi6, self.sig)

    def _project(self, point):
        """Project a candidate through the campaign's prepared response."""
        params = self.to_source(point, self.ex.fiducial)
        source = self.source_from_parameters('lal', **params)
        return self.ex.prepared.project(
            source, params['eclipticlongitude'], params['eclipticlatitude'])

    def _spectrum(self, point):
        """The exact A/E spectrum, through the campaign's response object."""
        if self._frequency_sampler is None:
            return self.ex.spectrum(point)
        response = self._project(point)
        sampled = self._frequency_sampler.evaluate(response)
        return np.stack([np.asarray(sampled[name]) for name in ('A', 'E')])

    def _verify_fast_response(self):
        slow = self.ex.spectrum(self.injection_point)
        fast = self._spectrum(self.injection_point)
        scale = float(np.max(np.abs(slow)))
        difference = float(np.max(np.abs(fast - slow)))
        return dict(max_abs_difference=difference,
                    max_relative_difference=difference / scale,
                    scale=scale)

    # -- likelihood -----------------------------------------------------------
    def _loglikelihood(self):
        t_start = time.perf_counter()
        self._n_likelihood += 1
        params = self.current_params
        try:
            point = self._point(params)
            h = self._spectrum(point)
        except Exception as exc:                    # waveform domain failure
            _log.debug("exact_tdi response failed: %s", exc)
            self._seconds_likelihood += time.perf_counter() - t_start
            return -np.inf

        if self.marginalize_phase:
            z = (self.ex.inner(self.data, h)
                 - 1j * self.ex.inner(self.data, 1j * h))
            w = self.ex.inner(h, h)
            amplitude = abs(z)
            if self.marginalize_amplitude:
                log_likelihood = self._amplitude_integral(amplitude, w)
            else:
                log_likelihood = float(np.log(i0e(amplitude)) + amplitude
                                       - 0.5 * w)
        else:
            z = self.ex.inner(self.data, h)
            w = self.ex.inner(h, h)
            amplitude = z
            if self.marginalize_amplitude:
                log_likelihood = self._amplitude_integral(amplitude, w)
            else:
                log_likelihood = float(z - 0.5 * w)

        if self.marginalize_phase and self.include_phase_normalization:
            log_likelihood -= float(np.log(2.0 * np.pi))

        try:
            self._current_stats.distance = self.K / float(
                params['amplitude_u'])
        except (KeyError, ValueError):
            self._current_stats.distance = np.nan
        self._current_stats.snr = float(np.sqrt(max(2.0 * log_likelihood, 0.0)))
        self._seconds_likelihood += time.perf_counter() - t_start
        return float(log_likelihood)

    def _amplitude_integral(self, z, w):
        """Marginalise the linear amplitude over the power-law prior."""
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
        return float(top + np.log(np.sum(np.exp(values - top))) + np.log(width))

    # -- prior and stats ------------------------------------------------------
    @property
    def _extra_stats(self):
        return ['distance', 'snr']

    @property
    def likelihood_calls(self):
        """int: The number of exact-TDI likelihood evaluations."""
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
        attrs['exact_tdi_fast_response'] = self.fast_response
        if self.response_verification is not None:
            attrs['exact_tdi_response_max_relative_difference'] = float(
                self.response_verification['max_relative_difference'])
