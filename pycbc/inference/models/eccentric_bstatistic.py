# Copyright (C) 2026 Shichao Wu
# This program is free software under the GNU General Public License,
# version 3 or (at your option) any later version.
"""Physical coherent Bayes likelihood of an eccentric surrogate cell.

A component provider retains the source's per-mode response clocks and
supplies raw correlations and all mode cross-Grams on an arrival-time
quadrature. This model ties the modes through the same physical amplitude
prior used by BStatisticSurrogate, and marginalizes mean anomaly as well.
"""
import numpy

from pycbc.filter.bstatistic import OrientationNodes
from pycbc.filter.mean_anomaly import mean_anomaly_log_bayes, mean_anomaly_twof
from .base import BaseModel


class EccentricBStatistic(BaseModel):
    """An eccentric-cell likelihood with explicit component-provider protocol.

    component_provider(params,delays) must return z of shape (K,2,L) and
    gram of shape (K,2,K,2) or (K,2,K,2,L), in matching template/data units.
    offsets are the K integer n-m mean-anomaly offsets. The provider owns
    source interpolation, sky response, windowing and frequency convergence.
    It may use EccentricModeCorrelator; this model calls no waveform generator.

    The result is ln B against noise. Inclination, polarization, distance,
    phase, anomaly and arrival time are integrated coherently, with tied
    modes. delay_weights define normalized discrete arrival-time prior
    masses (equal by default); a continuous integral requires converged
    quadrature. Orientation/anomaly/amplitude convergence belongs to callers.
    This programmatic interface is not the optimized sky-block FFT engine.
    """
    name = 'eccentric_bstatistic'

    def __init__(self, variable_params, component_provider, offsets, delays,
                 distance_prior, delay_weights=None, n_l0=64,
                 orientation_nodes=(16, 8), amplitude_order=64,
                 noise_variance=1.0, xp=numpy, **kwargs):
        super().__init__(variable_params, **kwargs)
        self.xp = xp
        if not callable(component_provider):
            raise ValueError('component_provider must be callable')
        self.component_provider = component_provider
        self.offsets = numpy.asarray(offsets)
        self.delays = numpy.asarray(delays, dtype=float)
        if (self.delays.ndim != 1 or not len(self.delays) or
                not numpy.all(numpy.isfinite(self.delays)) or
                numpy.any(numpy.diff(self.delays) <= 0)):
            raise ValueError('delays must be finite, increasing and nonempty')
        weights = (numpy.ones(len(self.delays)) if delay_weights is None
                   else numpy.asarray(delay_weights, dtype=float))
        if (weights.shape != self.delays.shape or
                not numpy.all(numpy.isfinite(weights)) or
                numpy.any(weights < 0) or not weights.sum() > 0):
            raise ValueError('delay_weights need finite nonnegative masses with positive sum')
        self.delay_weights = weights / weights.sum()
        if not numpy.isfinite(noise_variance) or noise_variance <= 0:
            raise ValueError('noise_variance must be positive and finite')
        self.noise_variance = float(noise_variance)
        self.distance_prior = distance_prior
        self.n_l0, self.amplitude_order = n_l0, amplitude_order
        self.nodes = OrientationNodes(*orientation_nodes)

    def mode_statistics(self, params=None):
        """Raw provider correlations/Gram at one intrinsic and sky point."""
        params = self.current_params if params is None else params
        z, gram = self.component_provider(params, self.delays)
        if tuple(z.shape) != (len(self.offsets), 2, len(self.delays)):
            raise ValueError('provider correlations must have shape (K,2,L)')
        return z, gram

    def anomaly_statistics(self, params=None):
        """Coherent 2F on (anomaly,arrival-time), for search/follow-up."""
        z, gram = self.mode_statistics(params)
        var = self.noise_variance
        return mean_anomaly_twof(z/var, gram/var, self.offsets,
                                 self.n_l0, xp=self.xp)

    def log_bayes_by_delay(self, params=None):
        """ln B after the physical amplitude/orientation/anomaly integrals."""
        z, gram = self.mode_statistics(params)
        return mean_anomaly_log_bayes(z, gram, self.offsets,
            self.distance_prior, self.nodes, n_l0=self.n_l0,
            noise_variance=self.noise_variance, order=self.amplitude_order,
            xp=self.xp)[0]

    def _loglikelihood(self):
        xp = self.xp
        values = self.log_bayes_by_delay()
        active = numpy.flatnonzero(self.delay_weights > 0)
        selected = values[xp.asarray(active)]
        terms = selected + xp.log(xp.asarray(self.delay_weights[active]))
        top = xp.max(terms)
        result = top + xp.log(xp.sum(xp.exp(terms-top)))
        return float(result)



class EccentricCellComponents:
    """Bind a repaired PN-basis cell to the mode likelihood provider protocol.

    The family must be face-on, use circular_phase_basis='taylorf2', and
    supply (2,2,n) modes with their shared quadrupole angular pattern.
    All cell coefficients are prepared offline. Candidate evaluation uses
    only analytic PN ratios, interpolated cell pieces and the supplied
    yearly-response expansion. Source weights (e.g. sqrt(4 df/PSD)) are
    supplied explicitly; data must use the matching whitened convention.

    response_provider(params) may supply a sky-dependent response array
    with the same fixed yearly orders; otherwise response is fixed.
    Accuracy/coverage of the cell, response and delay quadrature still need
    independent validation. This direct implementation is an oracle for the
    optimized sky-block search, not a claimed production speedup.
    """
    def __init__(self, family, frequencies, data, response, orders,
                 source_weight=1., clock_time=0., time_window=None,
                 response_provider=None, xp=numpy):
        from pycbc.filter.mean_anomaly import mode_offsets
        if family.circular_phase_basis != 'taylorf2':
            raise ValueError('online component provider needs a taylorf2-basis cell')
        if family.reference.get('inclination', 0.) != 0.:
            raise ValueError('the shared physical orientation basis needs face-on source modes')
        self.family, self.xp = family, xp
        self.frequencies = numpy.asarray(frequencies, dtype=float)
        self.data = xp.asarray(data, dtype=xp.complex128)
        self.response, self.orders = response, numpy.asarray(orders)
        self.response_provider = response_provider
        self.weight = numpy.broadcast_to(numpy.asarray(source_weight, dtype=float),
                                          self.frequencies.shape)
        if not numpy.all(numpy.isfinite(self.weight)) or numpy.any(self.weight < 0):
            raise ValueError('source_weight must be finite and nonnegative')
        self.clock_time, self.time_window = float(clock_time), time_window
        self.keys = tuple(sorted(k for k in family.keys if k[:2] == (2, 2)))
        if not self.keys:
            raise ValueError('the cell has no shared-pattern (2,2,n) modes')
        self.offsets = mode_offsets(self.keys)
        r = family.reference
        self.mc0 = (r['mass1']*r['mass2'])**.6/(r['mass1']+r['mass2'])**.2

    def __call__(self, params, delays):
        from pycbc.filter.eccentric_mode_correlator import EccentricModeCorrelator
        from pycbc.tdi.eccentric_mode_source import EccentricModeFamilySource
        r, xp = self.family.reference, self.xp
        allowed = {'mchirp', 'q', 'eccentricity', 'spin1z', 'spin2z',
                   'eclipticlongitude', 'eclipticlatitude'}
        unsupported = set(params) - allowed
        if unsupported:
            raise ValueError('unsupported or marginalized provider parameters: '+
                             ', '.join(sorted(unsupported)))
        if self.response_provider is None and set(params) & {
                'eclipticlongitude', 'eclipticlatitude'}:
            raise ValueError('sky parameters require a response_provider')
        for key in ('spin1z', 'spin2z'):
            if key in params and params[key] != r.get(key, 0.):
                raise ValueError('this cell provider keeps spins fixed')
        mc = float(params.get('mchirp', self.mc0))
        q = float(params.get('q', self.family.q0))
        e = float(params.get('eccentricity', r.get('eccentricity', 0.)))
        if not numpy.all(numpy.isfinite([mc, q, e])) or mc <= 0 or q <= 0 or not 0 <= e < 1:
            raise ValueError('source mass, ratio and eccentricity are invalid')
        source = EccentricModeFamilySource(self.family,
            d=numpy.log(mc/self.mc0), dq=numpy.log(q/self.family.q0),
            de=e-r.get('eccentricity', 0.), clock_time=self.clock_time)
        records = source.frequency_harmonics(self.frequencies)
        b = numpy.zeros((len(self.keys), len(self.frequencies)), complex)
        t = numpy.zeros(b.shape, float)
        for j, key in enumerate(self.keys):
            if key in records:
                record = records[key]
                idx = record['indices']
                b[j, idx] = record['plus']*self.weight[idx]
                t[j, idx] = record['time']
        response = (self.response if self.response_provider is None
                    else self.response_provider(params))
        engine = EccentricModeCorrelator(self.frequencies, b, t, response,
            self.orders, time_window=self.time_window, xp=xp)
        return engine.at(self.data, delays)


class EccentricCellPE(BaseModel):
    """PE likelihood of an aligned-spin eccentric cell: ln B against noise,
    marginalised over mean anomaly, arrival time, distance, inclination,
    polarization and phase, at sampled ``mchirp, q, eccentricity, chi_eff,
    eclipticlongitude, eclipticlatitude``.

    ``block_model`` is an :class:`pycbc.filter.eccentric_block_search.
    EccentricFFTBlockModel` (or its fast subclass) whose ``source_builder``
    takes ``theta = (Mc, e, chi, q)`` and whose ``response_builder`` takes any
    sky offset in its shape units (e.g. a
    :class:`pycbc.tdi.ssb_yearly_harmonics.GridSkyResponse`); its complete
    Grams should be anchored (ln Mc, chi, q, sky) so a template costs no Gram.
    ``sky_reference`` is ``(lon0, lat0, scale_lon, scale_lat)``: the sky
    offset is ``((lon - lon0) / scale_lon, (lat - lat0) / scale_lat)``.

    Arrival time has a uniform prior over ``delay_window`` (seconds, the
    model's lag convention), integrated on samples ``dt / oversampling``
    apart where it matters (see :meth:`evaluate`); check convergence in
    ``oversampling``. Mean anomaly is uniform on ``n_l0`` nodes, the
    orientation on ``OrientationNodes(*orientation_nodes)``.
    """
    name = 'eccentric_cell_pe'
    intrinsic = ('mchirp', 'eccentricity', 'chi_eff', 'q')

    def __init__(self, variable_params, block_model, sky_reference, delay_window, distance_prior,
                 n_l0=64, orientation_nodes=(16, 8), amplitude_order=64, oversampling=4,
                 screen_margin=40.0, bayes_margin=30.0, noise_variance=1.0, amplitude_table=None,
                 screen_anomalies=None, shared_options=None, **kwargs):
        super().__init__(variable_params, **kwargs)
        self.block = block_model
        self.xp = block_model.xp
        self.sky_reference = tuple(float(v) for v in sky_reference)
        lo, hi = (float(v) for v in delay_window)
        if not hi > lo:
            raise ValueError('delay_window must be increasing')
        self.delay_window = (lo, hi)
        self.distance_prior = distance_prior
        self.n_l0, self.amplitude_order = int(n_l0), int(amplitude_order)
        self.nodes = OrientationNodes(*orientation_nodes)
        # amplitude_table: a pycbc.filter.bstatistic.AmplitudeTable of distance_prior (its interpolated amplitude
        # integrals replace the quadrature); screen_anomalies: anomaly nodes of the whole-lag 2F screen (default
        # n_l0; fewer only lowers F there, and screen_margin absorbs it)
        self.table = amplitude_table
        self.screen_anomalies = int(screen_anomalies or n_l0)
        self.oversampling = int(oversampling)
        self.screen_margin, self.bayes_margin = float(screen_margin), float(bayes_margin)
        self.noise_variance = float(noise_variance)
        # shared_options: keyword arguments of pycbc.filter.eccentric_shared.EccentricSharedRegion (data_window is
        # required) for share(); without it the model evaluates every point directly
        self.shared_options = shared_options
        self._last = None

    #: the shared regions kept at once, the oldest dropped first
    max_shared_regions = 2
    #: points outside every shared region that one batch must have before a new region is built for them
    min_new_region = 3000

    @property
    def batches_points(self):
        """Whether :meth:`loglikelihoods` should get whole batches: once shared regions are possible."""
        return self.shared_options is not None

    @property
    def shared_statistics(self):
        """Counts of the shared computation (as BStatisticSurrogate's)."""
        if getattr(self, '_shared_statistics', None) is None:
            self._shared_statistics = dict(shared_points=0, single_points=0, shared_seconds=0.0, single_seconds=0.0,
                                           regions=0, region_seconds=0.0, regions_refused=0,
                                           largest_refused_gb=0.0)
        return self._shared_statistics

    def share(self, region, **kwargs):
        """Set up an :class:`pycbc.filter.eccentric_shared.EccentricSharedRegion` over ``region`` (parameter
        dicts); :meth:`loglikelihoods` then uses it for the points it covers. Raises ``MemoryError`` when a mode's
        grid does not fit."""
        import time
        from pycbc.filter.eccentric_shared import EccentricSharedRegion
        if self.shared_options is None:
            raise ValueError('share needs shared_options')
        regions = getattr(self, '_shared_regions', [])
        while len(regions) >= self.max_shared_regions:
            regions.pop(0)
        self._shared_regions = regions
        stats = self.shared_statistics
        tick = time.time()
        options = dict(self.shared_options)
        options.update(kwargs)
        try:
            shared = EccentricSharedRegion(self, region, **options)
        except MemoryError as err:
            stats['regions_refused'] += 1
            stats['largest_refused_gb'] = max(stats['largest_refused_gb'], getattr(err, 'need', 0) / 1e9)
            err.__traceback__ = None
            import gc
            gc.collect()
            if self.xp is not numpy:
                self.xp.get_default_memory_pool().free_all_blocks()
            raise
        stats['regions'] += 1
        stats['region_seconds'] += time.time() - tick
        regions.append(shared)
        self._shared_regions = regions[-self.max_shared_regions:]
        return shared

    def _share_split(self, points, depth=0):
        try:
            self.share(points)
            return True
        except MemoryError:
            if depth >= 3 or len(points) < 2 * self.min_new_region // 4:
                return False
            key = numpy.array([p['mchirp'] for p in points])
            half = numpy.median(key)
            done = self._share_split([p for p, k in zip(points, key) if k <= half], depth + 1)
            return self._share_split([p for p, k in zip(points, key) if k > half], depth + 1) or done

    def loglikelihoods(self, points):
        """The log likelihood of each of ``points`` (dicts of the variable parameters): from the shared regions
        where they cover the point, directly otherwise; a batch with many points outside every region gets a new
        one for them."""
        import logging
        import time
        full = []
        for p in points:
            q = dict(self.static_params)
            q.update(p)
            full.append(q)
        out = numpy.full(len(full), numpy.nan)
        todo = numpy.ones(len(full), dtype=bool)
        tick = time.time()
        regions = getattr(self, '_shared_regions', [])
        if regions and len(full) >= self.min_new_region:
            covered = numpy.zeros(len(full), dtype=bool)
            for shared in regions:
                covered |= shared.inside(full)
            outside = numpy.nonzero(~covered)[0]
            if len(outside) >= self.min_new_region:
                self._share_split([full[i] for i in outside])
                regions = self._shared_regions
        for shared in regions:
            idx = numpy.nonzero(todo)[0]
            if len(idx) == 0:
                break
            hit = shared.inside([full[i] for i in idx])
            if hit.any():
                sel = idx[hit]
                out[sel] = shared([full[i] for i in sel])
                todo[sel] = ~numpy.isfinite(out[sel])
        middle = time.time()
        nshared = int((~todo).sum())
        for i in numpy.nonzero(todo)[0]:
            out[i] = self.evaluate(full[i])['loglikelihood']
        stats = self.shared_statistics
        stats['shared_points'] += nshared
        stats['single_points'] += int(todo.sum())
        stats['shared_seconds'] += middle - tick
        stats['single_seconds'] += time.time() - middle
        logging.info("batch of %d: %d shared in %.2f s, %d directly in %.2f s", len(full), nshared, middle - tick,
                     int(todo.sum()), time.time() - middle)
        return out

    def _theta_sky(self, params):
        theta = numpy.array([params['mchirp'], params['eccentricity'], params['chi_eff'], params['q']], float)
        lon0, lat0, slon, slat = self.sky_reference
        dlon = numpy.mod(params['eclipticlongitude'] - lon0 + numpy.pi, 2 * numpy.pi) - numpy.pi
        sky = numpy.array([dlon / slon, (params['eclipticlatitude'] - lat0) / slat])
        return theta, sky

    def _log_bayes(self, z, g):
        return mean_anomaly_log_bayes(z, g, self.block.offsets, self.distance_prior, self.nodes, n_l0=self.n_l0,
                                      noise_variance=self.noise_variance, order=self.amplitude_order,
                                      table=self.table, xp=self.xp)

    def evaluate(self, params=None):
        """ln B and the time quadrature at ``params`` (the current point by
        default): a dict with ``loglikelihood``, the quadrature's delays,
        widths, ln B and the correlations and Grams of its active samples.

        The integrand is evaluated in three steps, each exact to its margin:
        2F (maximised over amplitude, phase and anomaly) at every whole lag;
        lags with F = 2F/2 within ``screen_margin`` of the largest are
        oversampled ``oversampling`` times by exact FFT shifts; ln B, which
        never exceeds F there, is computed only at fine samples with F above
        the best ln B less ``bayes_margin`` (each dropped sample contributes
        less than exp(-bayes_margin) of the peak)."""
        from pycbc.filter.eccentric_mode_fft import WindowedModeFFT
        from pycbc.filter.eccentric_fast_search import mean_anomaly_twof_analytic
        params = self.current_params if params is None else params
        key = tuple(float(params[k]) for k in self.intrinsic + ('eclipticlongitude', 'eclipticlatitude'))
        if self._last is not None and self._last[0] == key:
            return self._last[1]
        # release the previous point's arrays before preparing this one (GPU memory)
        self._last = None
        xp = self.xp
        host = lambda a: a.get() if hasattr(a, 'get') else numpy.asarray(a)
        theta, sky = self._theta_sky(params)
        engine, atimes, ganchors = self.block._prepare(theta, sky)
        dt = self.block.grid.delta_t
        lo, hi = self.delay_window
        lags = numpy.arange(int(numpy.ceil(lo / dt)), int(numpy.floor(hi / dt)))
        var = self.noise_variance
        z = engine.correlations(lags)
        g = WindowedModeFFT.interpolated_gram(engine, lags * dt, atimes, ganchors)
        twof = host(mean_anomaly_twof_analytic(z / var, g / var, self.block.offsets, self.screen_anomalies,
                                               xp=xp)[0].max(axis=0))
        keep = numpy.flatnonzero(twof / 2 >= twof.max() / 2 - self.screen_margin)
        # the neighbours too: F between whole lags can exceed both ends
        keep = numpy.unique(numpy.clip(numpy.concatenate([keep - 1, keep, keep + 1]), 0, len(lags) - 1))
        P = self.oversampling
        zf, fdelays = engine.oversampled_correlations(lags[keep], P)
        zf = zf.reshape(zf.shape[0], 2, -1)
        fdelays = fdelays.ravel()
        gf = WindowedModeFFT.interpolated_gram(engine, fdelays, atimes, ganchors)
        ftwof = host(mean_anomaly_twof_analytic(zf / var, gf / var, self.block.offsets, self.n_l0, xp=xp)[0].max(axis=0))
        order = numpy.argsort(-ftwof)
        best = float(host(self._log_bayes(zf[..., order[:1]], gf[..., order[:1]])[0])[0])
        active = numpy.flatnonzero(ftwof / 2 >= best - self.bayes_margin)
        lnb = host(self._log_bayes(zf[..., active], gf[..., active])[0])
        width = dt / P
        terms = lnb + numpy.log(width)
        top = terms.max()
        logl = float(top + numpy.log(numpy.sum(numpy.exp(terms - top))) - numpy.log(hi - lo))
        out = dict(loglikelihood=logl, delays=fdelays[active], widths=numpy.full(len(active), width), log_bayes=lnb,
                   z=zf[..., active], gram=gf[..., active], anchor_times=atimes, anchor_grams=ganchors,
                   theta=theta, sky=sky, lags_screened=int(len(keep)), samples_active=int(len(active)))
        # the engine (all mode spectra on the full grid) is not kept: neither here (unless keep_engine) nor in the
        # block model's last-prepared slot
        if getattr(self, 'keep_engine', False):
            out['engine'] = engine
        self.block._prepared = None
        del engine, z, g, zf, gf
        self._last = (key, out)
        return out

    def _loglikelihood(self):
        return self.evaluate()['loglikelihood']

    def reconstruct(self, rec=None, seed=None):
        """Draw the marginalised parameters at the current point: ``delta_tc``
        (the arrival-time delay, the model's lag convention), ``mean_anomaly``,
        ``inclination``, ``polarization``, ``distance``, ``coa_phase`` (half
        the dominant (2,2,2) mode's phase, modulo pi, as BStatisticSurrogate),
        ``loglr``, ``snr`` and ``loglikelihood``."""
        from pycbc.filter.bstatistic import draw_orientation, draw_amplitude, draw_phase, _project
        from pycbc.filter.mean_anomaly import mean_anomaly_components
        from pycbc.tmpltbank.fstatistic_metric import orientation_weights
        rng = numpy.random.default_rng(seed)
        rec = {} if rec is None else rec
        ev = self.evaluate()
        terms = ev['log_bayes'] + numpy.log(ev['widths'])
        weight = numpy.exp(terms - terms.max())
        k = rng.choice(len(weight), p=weight / weight.sum())
        # the drawn sample's own correlations; the delay is jittered within its width
        delay = float(ev['delays'][k] + rng.random() * ev['widths'][k])
        z = ev['z'][..., k:k + 1]
        g = ev['gram'][..., k:k + 1]
        _, by_angle, angles = self._log_bayes(z, g)
        by_angle = by_angle.get() if hasattr(by_angle, 'get') else numpy.asarray(by_angle)
        by_angle = by_angle.reshape(len(angles))
        w = numpy.exp(by_angle - by_angle.max())
        a = rng.choice(len(w), p=w / w.sum())
        step = 2 * numpy.pi / len(angles)
        anomaly = float(numpy.mod(angles[a] + (rng.random() - 0.5) * step, 2 * numpy.pi))
        zl, gl, _ = mean_anomaly_components(z, g, self.block.offsets, self.n_l0, xp=self.xp)
        zl = zl.get() if hasattr(zl, 'get') else numpy.asarray(zl)
        gl = gl.get() if hasattr(gl, 'get') else numpy.asarray(gl)
        zz = zl[a, :, 0] / self.noise_variance
        gg = gl[a][..., 0] if gl.ndim == 4 else gl[a]
        gram = (float(gg[0, 0].real) / self.noise_variance, float(gg[1, 1].real) / self.noise_variance,
                complex(gg[0, 1]) / self.noise_variance)
        cos_iota, psi = draw_orientation(zz, gram, self.distance_prior, self.nodes, rng)
        wv = orientation_weights(numpy.arccos(cos_iota), psi)
        x, gx = _project(zz.reshape(2, 1), *(numpy.array([v]) for v in gram), wv.reshape(1, 2), numpy)
        h = draw_amplitude(float(x[0, 0]), float(gx[0, 0]), self.distance_prior, rng)
        X = complex(numpy.conj(wv) @ zz)
        phase = draw_phase(h, X, rng)
        loglr = h * numpy.real(numpy.exp(-1j * phase) * X) - 0.5 * h * h * float(gx[0, 0])
        quarter = rng.integers(4)
        rec.update(delta_tc=float(delay), mean_anomaly=anomaly, loglr=float(loglr),
                   snr=float(numpy.sqrt(max(2 * loglr, 0.0))), inclination=float(numpy.arccos(cos_iota)),
                   polarization=float(psi + quarter * numpy.pi / 2), distance=float(self.distance_prior.scale / h),
                   coa_phase=float(numpy.mod(0.5 * (phase + quarter * numpy.pi), numpy.pi)),
                   loglikelihood=float(ev['loglikelihood']))
        return rec
