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

    ``template_convention='hlm'`` (with ``mode_keys``, the block's modes
    ``(l, m, n)``): the block's templates are multipole moments without
    angular factors (:func:`pycbc.waveform.eccentric_mode_family.efpe_modes`
    with ``hlm=True``), possibly with modes of other patterns than (2,2). The
    sampler's likelihood is then the (2,2,n) modes' (their face-on
    normalisation restored by one constant), exactly as before; the
    reconstruction draws one arrival-time sample per point from it and
    evaluates there the likelihood of every mode
    (:func:`pycbc.filter.hm_marginal.hm_draw_batch`): ``marginal_loglr`` is
    the (2,2) model's plus that sample's difference, an unbiased estimate of
    the full model's whose importance weights (the driver's own ln B against
    the box) give the full model's posterior; the angles are drawn from the
    full model, ``coa_phase`` over ``[0, 2 pi)``.
    """
    name = 'eccentric_cell_pe'
    intrinsic = ('mchirp', 'eccentricity', 'chi_eff', 'q')

    def __init__(self, variable_params, block_model, sky_reference, delay_window, distance_prior,
                 n_l0=64, orientation_nodes=(16, 8), amplitude_order=64, oversampling=4,
                 screen_margin=40.0, bayes_margin=30.0, noise_variance=1.0, amplitude_table=None,
                 screen_anomalies=None, shared_options=None, template_convention='face-on', mode_keys=None,
                 **kwargs):
        super().__init__(variable_params, **kwargs)
        self.block = block_model
        self.patterns = None
        if template_convention == 'hlm':
            from pycbc.filter.hm_marginal import PatternSet, projector
            if mode_keys is None:
                raise ValueError('template_convention hlm needs mode_keys')
            self.patterns = PatternSet(mode_keys)
            self._k22 = numpy.array([i for i, k in enumerate(self.patterns.keys) if k[:2] == (2, 2)])
            self._off22 = numpy.asarray(block_model.offsets)[self._k22]
            # the face-on plus polarisation of a (2,2) mode is its hlm times conj(Ap22(0))
            self._a22 = complex(numpy.conj(projector(2, 2, 0.)[0]))
        elif template_convention != 'face-on':
            raise ValueError('template_convention: face-on or hlm')
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
    #: points a region is built from at most (a uniform subsample of a larger batch)
    max_region_points = 4000
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
            # the region that has evaluated the fewest points goes: the oldest is often the one the sampler is
            # working in (a frame's region, made before the refinement's small ones), and a count that decays by
            # batch forgot it after a dozen small batches. The newest region is kept until it has been tried.
            uses = [getattr(r, 'points_used', 0) for r in regions]
            if len(regions) > 1:
                uses[-1] = numpy.inf
            regions.pop(int(numpy.argmin(uses)))
        self._shared_regions = regions
        stats = self.shared_statistics
        tick = time.time()
        options = dict(self.shared_options)
        options.update(kwargs)
        if len(region) > self.max_region_points:
            # the region's extent from a uniform subsample: its sky basis and frames grow with the points (a
            # curvilinear frame's first batch of 13385 boxes overran 10 GB of host memory)
            pick = numpy.random.default_rng(len(region)).choice(len(region), self.max_region_points, replace=False)
            region = [region[i] for i in numpy.sort(pick)]
        try:
            shared = EccentricSharedRegion(self, region, **options)
        except MemoryError as err:
            import logging
            stats['regions_refused'] += 1
            stats['largest_refused_gb'] = max(stats['largest_refused_gb'], getattr(err, 'need', 0) / 1e9)
            used = self.xp.get_default_memory_pool().used_bytes() / 1e9 if self.xp is not numpy else 0.
            logging.info("share refused for %d points: %s (device pool in use %.2f GB)", len(region),
                         str(err)[:300], used)
            err.__traceback__ = None
            import gc
            gc.collect()
            if self.xp is not numpy:
                self.xp.get_default_memory_pool().free_all_blocks()
            raise
        stats['regions'] += 1
        stats['region_seconds'] += time.time() - tick
        shared.points_used = 0
        regions.append(shared)
        self._shared_regions = regions
        return shared

    def _share_split(self, points, depth=0):
        try:
            self.share(points)
            return True
        except MemoryError:
            if depth >= 3 or len(points) < 2 * self.min_new_region // 4:
                return False
            # halve along the points' widest direction: the leading principal axis of their standardised
            # coordinates (the posterior is widest along q, which ln Mc alone does not separate)
            X = numpy.array([[numpy.log(p['mchirp']), p['eccentricity'], p['chi_eff'], numpy.log(p['q']),
                              p['eclipticlongitude'], p['eclipticlatitude']] for p in points])
            X = (X - X.mean(axis=0)) / numpy.maximum(X.std(axis=0), 1e-300)
            axis = numpy.linalg.svd(X, full_matrices=False)[2][0]
            key = X @ axis
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
        # each region's inside() once per batch (it forms every point's sky response: it was called twice, for
        # the coverage and again for the evaluation)
        hits = {}
        inside_seconds = 0.

        def hit(shared):
            nonlocal inside_seconds
            if id(shared) not in hits:
                t0 = time.time()
                hits[id(shared)] = shared.inside(full)
                inside_seconds += time.time() - t0
            return hits[id(shared)]
        if not regions and self.shared_options is not None and len(full) >= self.min_new_region:
            # the first large batch (as a sampler's whitened frame) is its own region
            self._share_split(full)
            regions = getattr(self, '_shared_regions', [])
        elif regions:
            covered = numpy.zeros(len(full), dtype=bool)
            for shared in regions:
                covered |= hit(shared)
            outside = numpy.nonzero(~covered)[0]
            # the points outside every region, pooled over batches: a refinement's batches each hold a few dozen of
            # them, too few for a region of their own, which then went directly batch after batch
            pool = getattr(self, '_outside_pool', [])
            if len(full) >= 50:
                # only a refinement's batches (a frame's climb asks a dozen points at a time, along its path)
                pool.extend(full[i] for i in outside)
            if len(pool) >= self.min_new_region:
                t0 = time.time()
                self._share_split(pool)
                self.shared_statistics['pool_region_seconds'] = \
                    self.shared_statistics.get('pool_region_seconds', 0.) + time.time() - t0
                regions = self._shared_regions
                pool = []
            self._outside_pool = pool
        used = {}
        for shared in regions:
            idx = numpy.nonzero(todo)[0]
            if len(idx) == 0:
                break
            sel = idx[hit(shared)[idx]]
            if len(sel):
                out[sel] = shared([full[i] for i in sel])
                used[id(shared)] = len(sel)
                todo[sel] = ~numpy.isfinite(out[sel])
        self.shared_statistics['inside_seconds'] = self.shared_statistics.get('inside_seconds', 0.) + inside_seconds
        for r in regions:
            r.points_used = getattr(r, 'points_used', 0) + used.get(id(r), 0)
        middle = time.time()
        nshared = int((~todo).sum())
        if todo.any() and self.xp is not numpy:
            # the direct engine's workspace: the shared evaluation's freed blocks back to the device
            self.xp.get_default_memory_pool().free_all_blocks()
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

    def _sub22(self, z, g):
        """The (2,2,n) modes' correlations and Grams in the face-on normalisation, and their offsets."""
        if self.patterns is None:
            return z, g, self.block.offsets
        k = self._k22
        a = self._a22
        return numpy.conj(a) * z[k], abs(a) ** 2 * g[k][:, :, k], self._off22

    def _twof(self, z, g, n_l0):
        """2F of the (2,2) model per anomaly node and sample (the screens)."""
        from pycbc.filter.eccentric_fast_search import mean_anomaly_twof_analytic
        z, g, off = self._sub22(z, g)
        return mean_anomaly_twof_analytic(z, g, off, n_l0, xp=self.xp)

    def _log_bayes(self, z, g, validate=True):
        z, g, offsets = self._sub22(z, g)
        return mean_anomaly_log_bayes(z, g, offsets, self.distance_prior, self.nodes, n_l0=self.n_l0,
                                      noise_variance=self.noise_variance, order=self.amplitude_order,
                                      table=self.table, xp=self.xp, validate=validate)

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
        twof = host(self._twof(z / var, g / var, self.screen_anomalies)[0].max(axis=0))
        keep = numpy.flatnonzero(twof / 2 >= twof.max() / 2 - self.screen_margin)
        # the neighbours too: F between whole lags can exceed both ends
        keep = numpy.unique(numpy.clip(numpy.concatenate([keep - 1, keep, keep + 1]), 0, len(lags) - 1))
        P = self.oversampling
        zf, fdelays = engine.oversampled_correlations(lags[keep], P)
        zf = zf.reshape(zf.shape[0], 2, -1)
        fdelays = fdelays.ravel()
        gf = WindowedModeFFT.interpolated_gram(engine, fdelays, atimes, ganchors)
        ftwof = host(self._twof(zf / var, gf / var, self.n_l0)[0].max(axis=0))
        order = numpy.argsort(-ftwof)
        best = float(host(self._log_bayes(zf[..., order[:1]], gf[..., order[:1]])[0])[0])
        active = numpy.flatnonzero(ftwof / 2 >= best - self.bayes_margin)
        # in chunks: the orientation quadrature holds n_l0 x nodes values per sample, and a quiet point keeps
        # most of its samples
        lnb = numpy.concatenate([host(self._log_bayes(zf[..., active[c:c + 256]], gf[..., active[c:c + 256]])[0])
                                 for c in range(0, len(active), 256)])
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

    def _shared_evaluate(self, params=None):
        """The current point's quadrature from a shared region that covers it, or ``None``."""
        params = self.current_params if params is None else params
        for shared in getattr(self, '_shared_regions', []):
            if shared.inside([params])[0]:
                ev = shared.evaluate(params)
                if ev is not None:
                    return ev
        return None

    @property
    def lognl(self):
        """The noise model's log likelihood of the data, ``-<d, d> / 2``: ``-sum |d|^2 / (2 noise_variance)``
        over the whitened data spectrum (``noise_variance`` per real component), without the normalisation
        constant, as PyCBC's Gaussian-noise models define it."""
        if getattr(self, '_lognl', None) is None:
            d = self.block.data
            total = (abs(d) ** 2).sum()
            total = float(total.get() if hasattr(total, 'get') else total)
            self._lognl = -0.5 * total / self.noise_variance
        return self._lognl

    def _params_of(self, theta, sky):
        """The parameter dict of ``theta = (Mc, e, chi, q)`` and a sky offset (the inverse of _theta_sky)."""
        lon0, lat0, slon, slat = self.sky_reference
        q = dict(self.static_params)
        q.update(mchirp=float(theta[0]), eccentricity=float(theta[1]), chi_eff=float(theta[2]), q=float(theta[3]),
                 eclipticlongitude=float(numpy.mod(lon0 + sky[0] * slon, 2 * numpy.pi)),
                 eclipticlatitude=float(lat0 + sky[1] * slat))
        return q

    def _full_at(self, params, delays):
        """The full model's (``self.full_model``) correlations and Grams of every mode at one delay per point:
        through its shared regions, built over the points first (the sampler's (2,2) regions released before), and
        directly (the nearest of the point's own samples) where none covers a point."""
        import logging
        fm, xp = self.full_model, self.xp
        M = len(params)
        covered = numpy.zeros(M, dtype=bool)
        for shared in getattr(fm, '_shared_regions', []):
            covered |= shared.inside(params)
        outside = numpy.nonzero(~covered)[0]
        if fm.shared_options is not None and len(outside) >= min(fm.min_new_region, M):
            logging.info("full model: sharing %d of %d points", len(outside), M)
            fm._share_split([params[i] for i in outside])
        K = len(fm.patterns.keys)
        z = xp.zeros((K, 2, M), dtype=xp.complex128)
        g = xp.zeros((K, 2, K, 2, M), dtype=xp.complex128)
        todo = numpy.arange(M)
        for shared in getattr(fm, '_shared_regions', []):
            if not len(todo):
                break
            sel = todo[shared.inside([params[i] for i in todo])]
            if len(sel):
                zz, gg = shared.at_delays([params[i] for i in sel], delays[sel])
                z[..., xp.asarray(sel)] = zz
                g[..., xp.asarray(sel)] = gg
                todo = numpy.setdiff1d(todo, sel)
        if len(todo):
            logging.info("full model: %d of %d points directly", len(todo), M)
        # diagnostics: three points' correlations, region against direct, in both models (phase of each mode)
        host = lambda a: a.get() if hasattr(a, 'get') else numpy.asarray(a)
        for i in range(min(3, M)):
            if i in todo:
                continue
            fm.update(**{k: params[i][k] for k in fm.variable_params})
            ev = fm.evaluate()
            k = int(numpy.argmin(numpy.abs(ev['delays'] - delays[i])))
            zd = host(ev['z'][..., k])
            zr = host(z[..., i])
            self.update(**{k_: params[i][k_] for k_ in self.variable_params})
            e2 = self.evaluate()
            k2 = int(numpy.argmin(numpy.abs(e2['delays'] - delays[i])))
            z2 = host(e2['z'][..., k2])
            k222 = fm.patterns.keys.index((2, 2, 2))
            logging.info("full model check %d: delay %.1f / direct %.1f / sampler direct %.1f; arg region/direct per "
                         "mode %s; |region/direct| %s; arg full direct/sampler direct (2,2,2) %.3f", i, delays[i],
                         ev['delays'][k], e2['delays'][k2],
                         numpy.round(numpy.angle(zr[:, 0] / zd[:, 0]), 3).tolist(),
                         numpy.round(numpy.abs(zr[:, 0] / zd[:, 0]), 3).tolist(),
                         float(numpy.angle(zd[k222, 0] / z2[self.patterns.keys.index((2, 2, 2)), 0])))
        for i in todo:
            fm.update(**{k: params[i][k] for k in fm.variable_params})
            ev = fm.evaluate()
            k = int(numpy.argmin(numpy.abs(ev['delays'] - delays[i])))
            z[..., i] = xp.asarray(ev['z'][..., k])
            gi = ev['gram']
            g[..., i] = xp.asarray(gi if gi.ndim == 4 else gi[..., k])
        return z, g

    def _reconstruct_many_full(self, points, seeds, chunk):
        """:meth:`reconstruct_many` with a full model (every mode; ``self.full_model``) beside this (2,2) one: the
        time samples drawn from this model's quadratures through its regions, which are then released; the full
        model's correlations at those samples, and its ln B and angles (:func:`pycbc.filter.hm_marginal.
        hm_draw_batch`); ``marginal_loglr`` this model's marginal plus the full model's difference at the sample."""
        from pycbc.filter.bstatistic import AmplitudeTable
        from pycbc.filter.hm_marginal import hm_draw_batch
        xp = self.xp
        host = lambda a: a.get() if hasattr(a, 'get') else numpy.asarray(a)
        import time as _time
        import logging as _logging
        t_pass1 = _time.time()
        rngs, delays, sample_delays, lnb22, marg, params = [], [], [], [], [], []
        z22s, g22s = [], []
        for c in range(0, len(points), chunk):
            part = points[c:c + chunk]
            part_rngs = [numpy.random.default_rng(seed) for seed in seeds[c:c + len(part)]]
            recs = self._sample_times(part, part_rngs)
            zc = numpy.stack([r['z'] for r in recs], axis=-1)
            gc_ = numpy.stack([r['gram'] for r in recs], axis=-1)
            for r, rng in zip(recs, part_rngs):
                delays.append(r['delay'])
                sample_delays.append(r['sample_delay'])
                rngs.append(rng)
                marg.append(r['loglikelihood'])
                params.append(self._params_of(r['theta'], r['sky']))
            v = self._log_bayes(xp.asarray(zc), xp.asarray(gc_), validate=False)[0]
            lnb22.extend(numpy.ravel(host(v)).tolist())
            if getattr(self, 'hm_draws', 'importance') == 'importance':
                z22s.append(zc)
                g22s.append(gc_)
            del recs, zc, gc_
        _logging.info("reconstruction: (2,2) quadratures and time samples of %d points in %.1f s", len(points),
                      _time.time() - t_pass1)
        # the sampler's regions released: the full model's grids (every mode) need the room
        import gc
        import logging
        self._shared_regions = []
        self._outside_pool = []
        self._last = None
        self.block.release_search_cache()
        fm = self.full_model
        fm._last = None
        fm.block.release_search_cache()
        gc.collect()
        if xp is not numpy:
            xp.get_default_memory_pool().free_all_blocks()
            logging.info("full model: device pool in use %.2f GB before its regions",
                         xp.get_default_memory_pool().used_bytes() / 1e9)
        if getattr(fm, '_hm_table', None) is None:
            fm._hm_table = AmplitudeTable(self.distance_prior, kernel='cosh', xp=xp)
        import time
        t0 = time.time()
        z, g = self._full_at(params, numpy.array(sample_delays))
        logging.info("full model: correlations of %d points in %.1f s", len(params), time.time() - t0)
        t0 = time.time()
        var = self.noise_variance
        if getattr(self, 'hm_draws', 'importance') == 'importance':
            out = self._importance_draws(numpy.concatenate(z22s, axis=-1), numpy.concatenate(g22s, axis=-1),
                                         host(z), host(g), rngs, delays, marg, sample_delays=sample_delays)
            logging.info("full model: draws and weights of %d points in %.1f s", len(params), time.time() - t0)
            return out
        out = []
        for c in range(0, len(points), chunk):
            sl = slice(c, min(len(points), c + chunk))
            lnb, draws = hm_draw_batch(z[..., sl] / var, g[..., sl] / var, fm.patterns.keys, self.distance_prior,
                                       self.nodes, rngs[sl], n_l0=self.n_l0, table=fm._hm_table, xp=xp,
                                       prune_margin=getattr(self, 'hm_prune_margin', 40.), table22=self.table)
            for i, d in zip(range(sl.start, sl.stop), draws):
                delta = float(lnb[i - sl.start] - lnb22[i])
                h = d['amplitude']
                out.append(dict(delta_tc=delays[i], mean_anomaly=d['mean_anomaly'], loglr=d['loglr'],
                                snr=float(numpy.sqrt(max(2 * d['loglr'], 0.0))), inclination=d['inclination'],
                                polarization=d['polarization'], distance=float(self.distance_prior.scale / h),
                                coa_phase=d['coa_phase'], loglikelihood=float(self.lognl + d['loglr']),
                                marginal_loglr=marg[i] + delta, hm_delta=delta))
        logging.info("full model: ln B and draws of %d points in %.1f s", len(params), time.time() - t0)
        return out

    def _importance_draws(self, z22, g22, z, g, rngs, delays, marg, sample_delays=None):
        """The extrinsic draws of the (2,2) model at each point's time sample (``z22 (K22, 2, M)``, ``g22``, the
        sampler's own), spread over the full model's ranges (``phi`` over ``[0, 2 pi)``, ``psi`` over ``[0, pi)``:
        the (2,2) model's equivalent branches taken at random), and the full model's log likelihood there (``z``,
        ``g`` every mode's, from its regions): ``hm_delta = ln L_full - ln L_22`` at the drawn values, the importance
        weight that turns the (2,2) draws into the full model's (exact; efficient while the other modes move ln L
        little across the posterior, as for stBBHs). ``marginal_loglr`` is the (2,2) marginal plus it. All points
        at once (pycbc.filter.bstatistic's batched draws; one point at a time took 36 ms a point)."""
        import logging
        from pycbc.filter.bstatistic import (draw_orientation_batch, draw_amplitude_batch, flat_choice,
                                             _orientation_vectors)
        from pycbc.filter.hm_marginal import log_likelihood_batch
        from pycbc.filter.mean_anomaly import mean_anomaly_components
        xp = self.xp
        host = lambda a: a.get() if hasattr(a, 'get') else numpy.asarray(a)
        M = z.shape[-1]
        var = self.noise_variance
        rng = rngs[0] if len(rngs) else numpy.random.default_rng()
        # the anomaly node of each point by its (2,2) ln B per node (ln B divides by the noise variance itself)
        # in chunks: all 30000 points of a 10000-sample run at once asked 16 GB of the device (the Grams' eigen
        # decomposition); the Grams were checked when these samples were first evaluated
        parts = []
        for c in range(0, M, 2048):
            _, ba, angles = self._log_bayes(xp.asarray(z22[..., c:c + 2048]), xp.asarray(g22[..., c:c + 2048]),
                                            validate=False)
            parts.append(host(ba).reshape(len(angles), -1))
        angles = numpy.asarray(host(angles), dtype=float)
        by_angle = numpy.concatenate(parts, axis=1)
        a = flat_choice(by_angle.T, rng.random((M, 1)))[:, 0]
        step = 2 * numpy.pi / len(angles)
        anomaly = numpy.mod(angles[a] + (rng.random(M) - 0.5) * step, 2 * numpy.pi)
        zs, gs, off = self._sub22(z22 / var, g22 / var)
        zl, gl, _ = mean_anomaly_components(zs, gs, off, self.n_l0)
        r = numpy.arange(M)
        zz = zl[a, :, r]                                                  # (M, 2)
        gg = gl[a, :, :, r]                                               # (M, 2, 2)
        g00, g11, g01 = gg[:, 0, 0].real, gg[:, 1, 1].real, gg[:, 0, 1]
        ci, ps = draw_orientation_batch(zz.T, g00, g11, g01, self.distance_prior, self.nodes, rng,
                                        table=self.table, xp=xp)
        ci, ps = host(ci), host(ps)
        wv = host(_orientation_vectors(xp.asarray(ci), xp.asarray(ps), xp))   # (M, 2)
        X = numpy.sum(numpy.conj(wv) * zz, axis=1)
        gx = (numpy.abs(wv[:, 0]) ** 2 * g00 + numpy.abs(wv[:, 1]) ** 2 * g11
              + 2 * numpy.real(numpy.conj(wv[:, 0]) * wv[:, 1] * g01))
        h = numpy.concatenate([host(draw_amplitude_batch(xp.asarray(numpy.abs(X[c:c + 4096])),
                                                         xp.asarray(gx[c:c + 4096]), self.distance_prior, rng,
                                                         xp=xp)) for c in range(0, M, 4096)])
        phase = numpy.mod(rng.vonmises(numpy.angle(X), h * numpy.abs(X)), 2 * numpy.pi)
        quarter = rng.integers(4, size=M)
        branch = rng.integers(2, size=M)
        pol = numpy.mod(ps + quarter * numpy.pi / 2, numpy.pi)
        phi = numpy.mod(0.5 * (phase + quarter * numpy.pi) + branch * numpy.pi, 2 * numpy.pi)
        iota = numpy.arccos(ci)
        lnl = log_likelihood_batch(z / var, g / var, self.full_model.patterns.keys, h, anomaly, iota, pol, phi)
        lnl22 = log_likelihood_batch(z22 / var, g22 / var, self.patterns.keys, h, anomaly, iota, pol, phi)
        delta = lnl - lnl22
        # the delay is drawn within its sample's width (delays) while the angles belong to the sample's own time:
        # moved by dt, the orbital phase and the mean anomaly advance by pi f dt at the frequency the data fix
        # best (phase_reference_frequency; every harmonic's phase at its own frequency is then kept, 2 dphi + (n - 2)
        # dl0 = n pi f dt). Without it the phase at a fixed epoch was biased by 2 pi f <dt>, 0.5 rad
        f_ph = getattr(self, 'phase_reference_frequency', None)
        if f_ph is not None and sample_delays is not None:
            jit = numpy.asarray(delays, dtype=float) - numpy.asarray(sample_delays, dtype=float)
            phi = numpy.mod(phi + numpy.pi * f_ph * jit, 2 * numpy.pi)
            anomaly = numpy.mod(anomaly + numpy.pi * f_ph * jit, 2 * numpy.pi)
        logging.info("importance draws: hm_delta median %.3f, 5-95%% %s", numpy.median(delta),
                     numpy.percentile(delta, [5, 95]).round(3).tolist())
        return [dict(delta_tc=delays[i], mean_anomaly=float(anomaly[i]), loglr=float(lnl[i]),
                     snr=float(numpy.sqrt(max(2 * lnl[i], 0.0))), inclination=float(iota[i]),
                     polarization=float(pol[i]), distance=float(self.distance_prior.scale / h[i]),
                     coa_phase=float(phi[i]), loglikelihood=float(self.lognl + lnl[i]),
                     marginal_loglr=marg[i] + float(delta[i]), hm_delta=float(delta[i])) for i in range(M)]

    def _sample_times(self, part, rngs):
        """For each of ``part`` (dicts of the variable parameters), one arrival-time sample drawn by its ln B (with
        its generator in ``rngs``): the point's marginal ``loglikelihood``, the sample's delay and its jittered
        ``delay``, its correlations and Grams (host arrays), ``theta`` and ``sky``. Through the shared regions'
        batched draw where they cover the point, directly otherwise."""
        full = []
        for p in part:
            q = dict(self.static_params)
            q.update(p)
            full.append(q)
        out = [None] * len(full)
        todo = numpy.arange(len(full))
        for shared in getattr(self, '_shared_regions', []):
            if not len(todo):
                break
            sel = todo[shared.inside([full[i] for i in todo])]
            if len(sel):
                recs = shared._evaluate([full[i] for i in sel], keep='sample', rngs=[rngs[i] for i in sel])
                for i, r in zip(sel, recs):
                    out[i] = r
            todo = numpy.array([i for i in todo if out[i] is None], dtype=int)
        host = lambda a: a.get() if hasattr(a, 'get') else numpy.asarray(a)
        for i in todo:
            self.update(**part[i])
            ev = self._shared_evaluate()
            if ev is None:
                ev = self.evaluate()
            rng = rngs[i]
            terms = ev['log_bayes'] + numpy.log(ev['widths'])
            weight = numpy.exp(terms - terms.max())
            k = rng.choice(len(weight), p=weight / weight.sum())
            gm = ev['gram']
            out[i] = dict(loglikelihood=float(ev['loglikelihood']), sample_delay=float(ev['delays'][k]),
                          delay=float(ev['delays'][k] + rng.random() * ev['widths'][k]),
                          z=host(ev['z'][..., k]), gram=host(gm if gm.ndim == 4 else gm[..., k]),
                          theta=ev['theta'], sky=ev['sky'])
        return out

    def _quadratures(self, part):
        """The time quadratures of ``part`` (dicts of the variable parameters): through the shared regions that
        cover them, the rest directly."""
        full = []
        for p in part:
            q = dict(self.static_params)
            q.update(p)
            full.append(q)
        evs = [None] * len(full)
        todo = numpy.arange(len(full))
        for shared in getattr(self, '_shared_regions', []):
            if not len(todo):
                break
            sel = todo[shared.inside([full[i] for i in todo])]
            if len(sel):
                for i, ev in zip(sel, shared._evaluate([full[i] for i in sel], keep=True)):
                    evs[i] = ev
            todo = numpy.array([i for i in todo if evs[i] is None], dtype=int)
        for i, p in enumerate(part):
            if evs[i] is None:
                self.update(**p)
                evs[i] = self._shared_evaluate()
                if evs[i] is None:
                    evs[i] = self.evaluate()
        return evs

    def reconstruct_many(self, points, seeds, chunk=512):
        """:meth:`reconstruct` at each of ``points`` (dicts of the variable parameters) with its seed, ``chunk``
        points at a time: the time quadratures of the points a shared region covers computed together, the rest
        directly; then the draws per point (the quadratures' device arrays are released chunk by chunk)."""
        # the posterior's points outside every region (as many as a third: the regions were built for the
        # sampler's boxes) get a region of their own first, as a large batch does in loglikelihoods
        if getattr(self, 'shared_options', None) is not None and len(points) >= self.min_new_region:
            full = []
            for p in points:
                q = dict(self.static_params)
                q.update(p)
                full.append(q)
            covered = numpy.zeros(len(full), dtype=bool)
            for shared in getattr(self, '_shared_regions', []):
                covered |= shared.inside(full)
            outside = numpy.nonzero(~covered)[0]
            if len(outside) >= self.min_new_region:
                import logging
                logging.info("reconstruct_many: %d of %d points outside every region: sharing them", len(outside),
                             len(full))
                self._share_split([full[i] for i in outside])
        if getattr(self, 'full_model', None) is not None:
            return self._reconstruct_many_full(points, seeds, chunk)
        out = []
        for c in range(0, len(points), chunk):
            part = points[c:c + chunk]
            full = []
            for p in part:
                q = dict(self.static_params)
                q.update(p)
                full.append(q)
            evs = [None] * len(full)
            todo = numpy.arange(len(full))
            for shared in getattr(self, '_shared_regions', []):
                if not len(todo):
                    break
                sel = todo[shared.inside([full[i] for i in todo])]
                if len(sel):
                    for i, ev in zip(sel, shared._evaluate([full[i] for i in sel], keep=True)):
                        evs[i] = ev
                todo = numpy.array([i for i in todo if evs[i] is None], dtype=int)
            for i, p in enumerate(part):
                if evs[i] is None:
                    self.update(**p)
                    evs[i] = self._shared_evaluate()
                    if evs[i] is None:
                        evs[i] = self.evaluate()
            out.extend(self._draw_many(evs, seeds[c:c + len(part)]))
            del evs
        return out

    def _draw_many(self, evs, seeds):
        """The draws of :meth:`reconstruct` for many points' quadratures ``evs`` at once: each point's own
        generator, used in the same order as :meth:`reconstruct` uses it (so the draws are the same), with the
        anomaly-resolved ln B and the mean-anomaly components of all the drawn time samples computed together."""
        from pycbc.filter.bstatistic import draw_orientation, draw_amplitude, draw_phase, _project
        from pycbc.filter.mean_anomaly import mean_anomaly_components
        from pycbc.tmpltbank.fstatistic_metric import orientation_weights
        xp = self.xp

        def host(a):
            return a.get() if hasattr(a, 'get') else numpy.asarray(a)
        rngs, delays, zs, gs = [], [], [], []
        for ev, seed in zip(evs, seeds):
            rng = numpy.random.default_rng(seed)
            terms = ev['log_bayes'] + numpy.log(ev['widths'])
            weight = numpy.exp(terms - terms.max())
            k = rng.choice(len(weight), p=weight / weight.sum())
            delays.append(float(ev['delays'][k] + rng.random() * ev['widths'][k]))
            rngs.append(rng)
            zs.append(xp.asarray(ev['z'][..., k:k + 1]))
            g = ev['gram']
            # a Gram shared by the samples has no sample axis
            gs.append(xp.asarray(g[..., None] if g.ndim == 4 else g[..., k:k + 1]))
        z = xp.concatenate(zs, axis=-1)
        g = xp.concatenate(gs, axis=-1)
        if self.patterns is not None:
            return self._draw_many_hm(evs, rngs, delays, z, g)
        _, by_angle, angles = self._log_bayes(z, g)
        by_angle = host(by_angle).reshape(len(angles), len(evs))
        zl, gl, _ = mean_anomaly_components(z, g, self.block.offsets, self.n_l0, xp=xp)
        zl, gl = host(zl), host(gl)
        step = 2 * numpy.pi / len(angles)
        out = []
        for i, (ev, rng) in enumerate(zip(evs, rngs)):
            w = numpy.exp(by_angle[:, i] - by_angle[:, i].max())
            a = rng.choice(len(w), p=w / w.sum())
            anomaly = float(numpy.mod(angles[a] + (rng.random() - 0.5) * step, 2 * numpy.pi))
            zz = zl[a, :, i] / self.noise_variance
            gg = gl[a][..., i]
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
            out.append(dict(delta_tc=delays[i], mean_anomaly=anomaly, loglr=float(loglr),
                            snr=float(numpy.sqrt(max(2 * loglr, 0.0))), inclination=float(numpy.arccos(cos_iota)),
                            polarization=float(psi + quarter * numpy.pi / 2),
                            distance=float(self.distance_prior.scale / h),
                            coa_phase=float(numpy.mod(0.5 * (phase + quarter * numpy.pi), numpy.pi)),
                            loglikelihood=float(self.lognl + loglr), marginal_loglr=float(ev['loglikelihood'])))
        return out

    def _draw_many_hm(self, evs, rngs, delays, z, g):
        """The draws of :meth:`_draw_many` for an ``hlm`` model: at each point's drawn arrival-time sample, the full
        model's ln B and angles (:func:`pycbc.filter.hm_marginal.hm_draw_batch`) and the (2,2) model's ln B there;
        ``marginal_loglr`` is the point's (2,2) marginal plus their difference (``hm_delta``)."""
        from pycbc.filter.bstatistic import AmplitudeTable
        from pycbc.filter.hm_marginal import hm_draw_batch
        xp = self.xp
        if getattr(self, '_hm_table', None) is None:
            self._hm_table = AmplitudeTable(self.distance_prior, kernel='cosh', xp=xp)
        var = self.noise_variance
        lnb22 = self._log_bayes(z, g)[0]
        lnb22 = lnb22.get() if hasattr(lnb22, 'get') else numpy.asarray(lnb22)
        lnb22 = numpy.ravel(lnb22)
        lnb, draws = hm_draw_batch(z / var, g / var, self.patterns.keys, self.distance_prior, self.nodes, rngs,
                                   n_l0=self.n_l0, table=self._hm_table, xp=xp)
        out = []
        for i, (ev, d) in enumerate(zip(evs, draws)):
            delta = float(lnb[i] - lnb22[i])
            h = d['amplitude']
            out.append(dict(delta_tc=delays[i], mean_anomaly=d['mean_anomaly'], loglr=d['loglr'],
                            snr=float(numpy.sqrt(max(2 * d['loglr'], 0.0))), inclination=d['inclination'],
                            polarization=d['polarization'], distance=float(self.distance_prior.scale / h),
                            coa_phase=d['coa_phase'], loglikelihood=float(self.lognl + d['loglr']),
                            marginal_loglr=float(ev['loglikelihood']) + delta, hm_delta=delta))
        return out

    def reconstruct(self, rec=None, seed=None, ev=None):
        """Draw the marginalised parameters at the current point: ``delta_tc``
        (the arrival-time delay, the model's lag convention), ``mean_anomaly``,
        ``inclination``, ``polarization``, ``distance``, ``coa_phase`` (half
        the dominant (2,2,2) mode's phase, modulo pi, as BStatisticSurrogate).
        As PyCBC's marginalised models reconstruct
        (:meth:`pycbc.inference.models.tools.DistMarg.reconstruct`): ``loglr``
        is the log likelihood ratio at the drawn point, not marginalised,
        ``loglikelihood = lognl + loglr``, and ``snr = sqrt(2 loglr)``;
        ``marginal_loglr`` is the model's own ln B, marginalised over all of
        them, the quantity the sampler sampled."""
        from pycbc.filter.bstatistic import draw_orientation, draw_amplitude, draw_phase, _project
        from pycbc.filter.mean_anomaly import mean_anomaly_components
        from pycbc.tmpltbank.fstatistic_metric import orientation_weights
        rec = {} if rec is None else rec
        if ev is None:
            # ev: the current point's quadrature when the caller has it (reconstruct_many)
            ev = self._shared_evaluate()
        if ev is None:
            ev = self.evaluate()
        if self.patterns is not None:
            rec.update(self._draw_many([ev], [seed])[0])
            return rec
        rng = numpy.random.default_rng(seed)
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
                   loglikelihood=float(self.lognl + loglr), marginal_loglr=float(ev['loglikelihood']))
        return rec
