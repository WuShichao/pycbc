# Copyright (C) 2026  Shichao Wu
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""The Bayes factor of a linear-exponent template family as a likelihood.

The intrinsic parameters and the sky are sampled; the arrival time, the
orientation, the amplitude (distance) and the phase are marginalised, and
:meth:`BStatisticSurrogate.reconstruct` draws them back, sample by sample,
for ``pycbc_inference_model_stats --reconstruct-parameters``.

The likelihood is the Bayes factor against noise,

    B = (1/W) int dDelta int dOmega int dh p(h) I0(h |X|) exp(-h^2 g / 2),

over an arrival-time window of width ``W``, with ``X = w^H z / s^2`` and
``g = w^H G w / s^2`` from the family's correlation ``z`` and Gram matrix
``G`` and the per-component noise variance ``s^2`` of the data spectrum.
"""

import logging
import os

import numpy

from pycbc.filter.bstatistic import (AmplitudeTable, OrientationNodes,
                                     _project,
                                     draw_amplitude, draw_orientation,
                                     draw_phase, log_orientation_marginal)
from pycbc.filter.subsample_arrival_time import (
    correlation_at, interpolate_twice_sampled, rms_bandwidth)
from pycbc.tmpltbank.fstatistic_metric import orientation_weights

from .base import BaseModel


def _measured_coordinates(surrogate, reference, coarse=4000):
    """:class:`pycbc.waveform.taylorf2_jets.MeasuredPhaseCoordinates` of
    the family, weighted by the reference's power at the frequencies it
    sweeps while the data last: the phase elsewhere is not measured."""
    from pycbc.waveform.taylorf2_jets import (MeasuredPhaseCoordinates,
                                              PhaseCoordinates)
    grid = surrogate.grid
    step = max(1, grid.size // coarse)
    f = numpy.asarray(grid.frequencies, dtype=float)[::step]
    b0 = _to_numpy(surrogate.b0[::step])
    t0 = _to_numpy(surrogate.t0[::step])
    seen = (t0 >= grid.start) & (t0 < grid.stop)
    return MeasuredPhaseCoordinates(
        [reference[n] for n in PhaseCoordinates.names], f,
        numpy.abs(b0) ** 2 * seen, getattr(surrogate, 'f_lower', f[0]))


def _boolean(value):
    return value.strip().lower() in ('1', 'true', 'yes')


def _to_numpy(array):
    return array.get() if hasattr(array, 'get') else numpy.asarray(array)


class BStatisticSurrogate(BaseModel):
    """Bayes factor of a linear-exponent family, marginalised over arrival
    time, orientation, distance and phase.

    Parameters
    ----------
    variable_params : list of str
        The sampled parameters: some of ``reference``'s keys.
    surrogate : pycbc.tdi.linear_exponent_surrogate.LinearExponentSurrogate
        The family about the reference source.
    daily_gram : pycbc.tdi.linear_exponent_surrogate.DailyGram
        Its Gram matrix over the data window.
    data_spectrum : array, shape ``(nchannels, N)``
        The SSB-time data spectrum.
    noise_variance : float
        Variance of each real component of the data spectrum's noise.
    reference : dict
        The reference point: the intrinsic coordinates in the family's
        order, then ``eclipticlongitude`` and ``eclipticlatitude``.
    scales : dict
        The shape unit of each reference coordinate.
    distance_prior : pycbc.filter.bstatistic.DistancePowerLaw
        The distance prior.
    time_window : tuple
        The arrival-time offsets marginalised over, s.
    oversampling : int
        Fine samples per grid step of the arrival time.
    orientation_nodes : tuple
        The largest ``(n_cos, n_psi)`` of the orientation quadrature; each
        likelihood uses the coarsest one that has converged, see
        :meth:`_orientation_nodes`.
    margin : float
        Terms of the marginal below ``exp(-margin)`` of its total are
        skipped, see
        :func:`pycbc.filter.bstatistic.log_orientation_marginal`; only
        without a table.
    metric : array, optional
        The orientation-averaged F-statistic metric in the shape units of
        ``reference``'s coordinates, for laying cells over the family.
    amplitude_table : bool, AmplitudeTable or str
        Read the amplitude marginal from a
        :class:`pycbc.filter.bstatistic.AmplitudeTable`: built here (True),
        the one given, or the one in an HDF file, built and written there
        if the file does not exist; False integrates every value.
    """
    name = 'bstatistic_surrogate'

    def __init__(self, variable_params, surrogate, daily_gram,
                 data_spectrum, noise_variance, reference, scales,
                 distance_prior, time_window, oversampling=10,
                 orientation_nodes=(32, 16), margin=40.0,
                 amplitude_table=True, metric=None, **kwargs):
        super().__init__(variable_params, **kwargs)
        self.metric = None if metric is None else numpy.asarray(metric)
        # coordinates in which the degenerate valley is straight, for
        # samplers
        if sorted(set(variable_params) & {'mchirp', 'q', 'spin1z',
                                          'spin2z'}) == sorted(
                ('mchirp', 'q', 'spin1z', 'spin2z')):
            self.curvilinear_coordinates = _measured_coordinates(
                surrogate, reference)
        self.surrogate, self.daily_gram = surrogate, daily_gram
        # in the surrogate's precision once, not at every correlation
        self.data_spectrum = data_spectrum.astype(surrogate.dtype,
                                                  copy=False)
        self.noise_variance = float(noise_variance)
        self.reference = dict(reference)
        self.scales = dict(scales)
        if len(self.reference) != surrogate.nintrinsic + 2:
            raise ValueError("reference needs the family's {} intrinsic "
                             "coordinates and the sky".format(
                                 surrogate.nintrinsic))
        self.distance_prior = distance_prior
        self.oversampling = int(oversampling)
        self.margin = float(margin)
        self.nodes = OrientationNodes(*orientation_nodes)
        grid = surrogate.grid
        step = grid.delta_t / self.oversampling
        lags = numpy.arange(grid.size)
        delays = numpy.where(lags < grid.size // 2, lags, lags - grid.size) \
            * grid.delta_t
        lo, hi = time_window
        inside = (delays >= lo) & (delays < hi)
        if not inside.any():
            raise ValueError("the time window holds no arrival times")
        order = numpy.argsort(delays[inside])
        self._lags = numpy.nonzero(inside)[0][order]
        self.delays = delays[self._lags]
        xp = self.xp = surrogate.xp
        self._lags_xp = xp.asarray(self._lags)
        self.time_step = step
        self.grid_step = grid.delta_t
        self.window_width = float(hi - lo)
        # a lag is sampled finely where its peak is narrower than two grid
        # steps, 1 / (2 pi sigma_f sqrt(2 ln B)) < 2 dt, and where it adds
        # to the integral more than 1e-4 / (number of lags) of the largest
        b0 = surrogate.b0.get() if hasattr(surrogate.b0, 'get') \
            else numpy.asarray(surrogate.b0)
        sigma_f = rms_bandwidth(grid.frequencies, numpy.abs(b0) ** 2)
        self._narrow = 0.5 / (numpy.pi * sigma_f * grid.delta_t) ** 2
        self._sigma_f = sigma_f
        self._tau = numpy.log(1e4 * len(self._lags))
        if isinstance(amplitude_table, AmplitudeTable):
            self.table = amplitude_table
        elif isinstance(amplitude_table, str):
            if os.path.exists(amplitude_table):
                self.table = AmplitudeTable.read(amplitude_table,
                                                 distance_prior)
            else:
                self.table = AmplitudeTable(distance_prior, xp=xp)
                self.table.write(amplitude_table)
        else:
            self.table = (AmplitudeTable(distance_prior, xp=xp)
                          if amplitude_table else None)

    def _shape_coordinates(self, params):
        point = dict(self.reference)
        point.update({k: params[k] for k in point if k in params})
        return numpy.array([(point[k] - self.reference[k]) / self.scales[k]
                            for k in self.reference])

    def _time_samples(self, params):
        """``z``, the Gram matrix and ``ln B`` at the arrival times of the
        window, scaled by the noise variance: every grid lag, and finer
        samples where the grid does not resolve the peak."""
        s = self.surrogate
        b, t, rho = s.pieces(self._shape_coordinates(params))
        spectrum = s.correlation_spectrum(self.data_spectrum, b, t, rho)
        xp = self.xp
        n = spectrum.shape[-1]
        q = self.daily_gram.prepare(b, t, rho)
        var = self.noise_variance
        table = self.table
        margin = None if table is not None else self.margin

        def samples(correlation, lags, delays):
            z = correlation[:, lags].astype(xp.complex128) / var
            g = tuple(v / var for v in self.daily_gram.at_prepared(q, delays))
            return z, g
        coarse = xp.fft.ifft(spectrum, axis=-1, norm='forward')
        z, g = samples(coarse, self._lags_xp, xp.asarray(self.delays))
        nodes = self._orientation_nodes(z, g)

        def marginal(z, g):
            return log_orientation_marginal(
                z, *g, self.distance_prior, nodes, margin=margin,
                table=table, xp=xp)
        lnb = marginal(z, g)
        width = numpy.full(len(self.delays), self.grid_step)
        delays = self.delays
        values = _to_numpy(lnb)
        top = values.max()
        need = (values >= top - self._tau) & (values >= self._narrow)
        need = need | numpy.roll(need, 1) | numpy.roll(need, -1)
        if need.any():
            # a peak of width sigma_t = 1 / (2 pi sigma_f sqrt(2 ln B)) is
            # integrated to e^-2pi^2 by samples sigma_t apart
            peak = max(float(values[need].max()), 0.5)
            sigma_t = 1 / (2 * numpy.pi * self._sigma_f * numpy.sqrt(
                2 * peak))
            P = int(numpy.ceil(self.grid_step / sigma_t))
        if need.any() and P > 1:
            k = numpy.nonzero(need)[0]
            step = self.grid_step / P
            width[k] = step
            # the correlation at half steps from one more transform, and
            # any finer delay by band-limited interpolation between them
            if getattr(self, '_half_step', None) is None:
                m = xp.arange(n, dtype=xp.float64)
                self._half_step = xp.exp(1j * numpy.pi * m / n).astype(
                    spectrum.dtype)
            half = xp.fft.ifft(spectrum * self._half_step, axis=-1,
                               norm='forward')
            df = numpy.concatenate([self.delays[k] + p * step
                                    for p in range(1, P)])
            duration = n * self.grid_step
            zf = interpolate_twice_sampled(
                coarse, half, self.grid_step / 2, numpy.mod(df, duration),
                xp=xp) / var
            gf = tuple(v / var for v in self.daily_gram.at_prepared(q, df))
            lf = marginal(zf, gf)
            z = xp.concatenate([z, zf], axis=1)
            g = tuple(xp.concatenate([a, c]) for a, c in zip(g, gf))
            lnb = xp.concatenate([lnb, lf])
            delays = numpy.concatenate([delays, df])
            width = numpy.concatenate([width, numpy.full(len(df), step)])
        return z, g, lnb, (spectrum, q, delays, width)

    def _orientation_nodes(self, z, g):
        """The coarsest orientation quadrature, doubling from 8 x 4 up to
        ``orientation_nodes``, whose values at the ``samples`` arrival
        times of largest ``|z_0|^2 / g_00 + |z_1|^2 / g_11`` all change by
        less than 1e-3 on the next doubling, all candidates evaluated at
        once. A weak signal's broad orientation posterior needs few nodes,
        a loud one's many."""
        sets = self._candidate_node_sets()
        choice = self._node_choice(z[:, None], tuple(v[None] for v in g),
                                   sets)
        return sets[int(choice[0])]

    #: arrival times at which the orientation quadrature must converge
    node_samples = 5

    def _candidate_node_sets(self):
        if getattr(self, '_node_sets', None) is None:
            sets = []
            n_cos, n_psi = 8, 4
            while True:
                sets.append(OrientationNodes(n_cos, n_psi))
                if n_cos >= self.nodes.n_cos:
                    break
                n_cos, n_psi = 2 * n_cos, 2 * n_psi
            self._node_sets = sets
        return self._node_sets

    def _node_choice(self, z, g, sets):
        """The index of the quadrature :meth:`_orientation_nodes` chooses
        for each row of ``z`` ``(2, M, L)`` and ``g`` (three ``(M, L)``),
        ``-inf`` entries of no weight."""
        xp = self.xp
        M, L = z.shape[1:]
        x = _to_numpy(xp.abs(z[0]) ** 2 / xp.maximum(g[0], 1e-300)
                      + xp.abs(z[1]) ** 2 / xp.maximum(g[1], 1e-300))
        x = numpy.where(numpy.isfinite(x), x, -numpy.inf)
        n = min(self.node_samples, L)
        top = numpy.argsort(x, axis=1)[:, ::-1][:, :n]
        at = xp.asarray((top + numpy.arange(M)[:, None] * L).ravel())
        flat_z = z.reshape(2, -1)[:, at]
        flat_g = tuple(v.reshape(-1)[at] for v in g)
        if self.table is not None:
            values = _to_numpy(self.table.log_orientation_marginals(
                flat_z, *flat_g, sets, xp=xp))
        else:
            values = numpy.array([_to_numpy(log_orientation_marginal(
                flat_z, *flat_g, self.distance_prior, v, xp=xp))
                for v in sets])
        values = values.reshape(len(sets), M, n)
        choice = numpy.full(M, len(sets) - 1)
        for i in range(len(sets) - 2, -1, -1):
            done = numpy.all(numpy.abs(values[i + 1] - values[i]) < 1e-3,
                             axis=1)
            choice[done] = i
        return choice

    def share(self, region, **kwargs):
        """Evaluate points of ``region`` (a list of parameter dicts spanning
        it) from computations they share, through :class:`SharedLikelihood`;
        :meth:`loglikelihoods` then uses it for the points it covers.
        Raises ``MemoryError`` when the region is too large for one grid."""
        import time
        regions = getattr(self, '_shared_regions', [])
        # make room on the device first: the oldest region goes
        while len(regions) >= self.max_shared_regions:
            regions.pop(0)
        self._shared_regions = regions
        stats = self.shared_statistics
        tick = time.time()
        try:
            shared = SharedLikelihood(self, region, **kwargs)
        except MemoryError as err:
            stats['regions_refused'] += 1
            stats['largest_refused_gb'] = max(stats['largest_refused_gb'],
                                              getattr(err, 'need', 0) / 1e9)
            raise
        stats['regions'] += 1
        stats['region_seconds'] += time.time() - tick
        regions.append(shared)
        self._shared_regions = regions[-self.max_shared_regions:]
        self._share_options = kwargs
        return shared

    @property
    def shared_statistics(self):
        """Counts of the shared computation, for diagnostics: points from
        shared regions and one at a time in :meth:`loglikelihoods`, and the
        seconds each took; regions set up, their seconds, regions refused
        as too large and the largest grid refused, GB."""
        if getattr(self, '_shared_statistics', None) is None:
            self._shared_statistics = dict(
                shared_points=0, single_points=0, shared_seconds=0.0,
                single_seconds=0.0, regions=0, region_seconds=0.0,
                regions_refused=0, largest_refused_gb=0.0)
        return self._shared_statistics

    #: the shared regions kept at once, the oldest dropped first
    max_shared_regions = 2
    #: points outside every shared region that one batch must have before a
    #: new region is built for them
    min_new_region = 3000

    def _share_split(self, points, depth=0):
        """Share ``points``, halving them along their first phase
        coordinate while one grid cannot hold them."""
        try:
            self.share(points, **getattr(self, '_share_options', {}))
            return True
        except MemoryError:
            if depth >= 3 or len(points) < 2 * self.min_new_region // 4:
                return False
            key = numpy.array([p['mchirp'] for p in points])
            half = numpy.median(key)
            low = [p for p, k in zip(points, key) if k <= half]
            high = [p for p, k in zip(points, key) if k > half]
            done = self._share_split(low, depth + 1)
            return self._share_split(high, depth + 1) or done

    def loglikelihoods(self, points):
        """The log likelihood of each of ``points`` (dicts of the variable
        parameters, the static ones added): from the shared computations
        where :meth:`share` set them up and they cover the point, one at a
        time otherwise. A batch with many points outside every region gets a
        new region for them."""
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
        if regions and todo.sum() >= self.min_new_region:
            # new region for the points no region covers yet
            pending = numpy.nonzero(todo)[0]
            covered = numpy.zeros(len(pending), dtype=bool)
            for shared in regions:
                covered |= shared.inside([full[i] for i in pending])
            outside = pending[~covered]
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
                # a point whose arrival times reach past the grid's goes
                # one at a time
                todo[sel] = ~numpy.isfinite(out[sel])
        middle = time.time()
        nshared = int((~todo).sum())
        for i in numpy.nonzero(todo)[0]:
            self.update(**points[i])
            out[i] = self.loglikelihood
        stats = self.shared_statistics
        stats['shared_points'] += nshared
        stats['single_points'] += int(todo.sum())
        stats['shared_seconds'] += middle - tick
        stats['single_seconds'] += time.time() - middle
        logging.info("batch of %d: %d shared in %.2f s, %d one at a time in "
                     "%.2f s", len(full), nshared, middle - tick,
                     int(todo.sum()), time.time() - middle)
        return out

    def _loglikelihood(self):
        _, _, lnb, (_, _, _, width) = self._time_samples(self.current_params)
        terms = _to_numpy(lnb) + numpy.log(width)
        top = terms.max()
        return float(top + numpy.log(numpy.sum(numpy.exp(terms - top))
                                     / self.window_width))

    def reconstruct(self, rec=None, seed=None):
        """Draw the marginalised parameters at the current point.

        Returns ``rec`` (a new dict by default) with ``delta_tc``,
        ``inclination``, ``polarization`` in ``[0, 2 pi)``, ``distance``,
        ``coa_phase`` in ``[0, pi)``, ``loglr`` and
        ``snr``, the matched-filter log likelihood ratio of the drawn signal
        and ``sqrt(2 loglr)``, and ``loglikelihood``, the marginal one.

        ``coa_phase`` is the waveform generator's: a quadrupole signal's
        phase ``phi`` is ``2 coa_phase`` at the reference source's reference
        frequency, so ``coa_phase`` is defined modulo ``pi``.
        """
        rng = numpy.random.default_rng(seed)
        rec = {} if rec is None else rec
        params = self.current_params
        _, _, lnb, (spectrum, gram, delays, width) = \
            self._time_samples(params)
        xp = self.xp
        terms = _to_numpy(lnb) + numpy.log(width)
        weight = numpy.exp(terms - terms.max())
        k = rng.choice(len(weight), p=weight / weight.sum())
        # each sample stands for [delay, delay + width)
        delay = delays[k] + rng.random() * width[k]
        z = _to_numpy(correlation_at(spectrum, [delay],
                                     self.surrogate.grid.duration,
                                     xp=xp))[:, 0]
        z = z / self.noise_variance
        g00, g11, g01 = (_to_numpy(x)[0] / self.noise_variance for x in
                         self.daily_gram.at_prepared(gram, [delay]))
        g = (float(numpy.real(g00)), float(numpy.real(g11)), complex(g01))
        cos_iota, psi = draw_orientation(z, g, self.distance_prior,
                                         self.nodes, rng)
        w = orientation_weights(numpy.arccos(cos_iota), psi)
        x, gg = _project(z.reshape(2, 1), *(numpy.array([v]) for v in g),
                         w.reshape(1, 2), numpy)
        h = draw_amplitude(float(x[0, 0]), float(gg[0, 0]),
                           self.distance_prior, rng)
        X = complex(numpy.conj(w) @ z)
        phase = draw_phase(h, X, rng)
        # the matched-filter log likelihood ratio of the drawn signal
        loglr = h * numpy.real(numpy.exp(-1j * phase) * X) \
            - 0.5 * h * h * float(gg[0, 0])
        # psi + k pi / 2 with phase + k pi is the same signal
        quarter = rng.integers(4)
        rec.update(delta_tc=float(delay), loglr=float(loglr),
                   snr=float(numpy.sqrt(max(2 * loglr, 0.0))),
                   inclination=float(numpy.arccos(cos_iota)),
                   polarization=float(psi + quarter * numpy.pi / 2),
                   distance=float(self.distance_prior.scale / h),
                   coa_phase=float(numpy.mod(
                       0.5 * (phase + quarter * numpy.pi), numpy.pi)),
                   loglikelihood=self.loglikelihood)
        return rec

    @classmethod
    def from_data(cls, variable_params, data, psds, reference, intrinsic,
                  low_frequency_cutoff, high_frequency_cutoff, time_window,
                  distance_prior, orbit=None, to_source=None, periods=None,
                  coarse_bins=1500, harmonics=4, response_samples=32,
                  derivative_step=0.05, max_time_loss=1e-3, precision='double',
                  expansion_order=2, source_derivatives=None,
                  exact_source_phase=None, sky_block_radius=None,
                  sky_block_points=600, **kwargs):
        """Build the model from frequency-domain TDI data.

        Parameters
        ----------
        data, psds : dict
            Channel name (``LISA_A`` or ``A``) to FrequencySeries, all with
            the same epoch and resolution.
        reference : dict
            The reference source: every parameter ``to_source`` needs, the
            sky as ``eclipticlongitude`` and ``eclipticlatitude``, the
            source clock as ``f_lower`` and ``tdi_t_start``.
        intrinsic : list of str
            The intrinsic coordinates of the family, keys of ``reference``.
        low_frequency_cutoff, high_frequency_cutoff : float
            The analysis band, Hz.
        time_window, distance_prior
            As in the class.
        orbit : optional
            The orbit; default the equal-arm LISA orbit with ``t0 = 0`` of
            :func:`pycbc.tdi.inference.sparse_tdi_fd_det_sequence`.
        to_source : callable, optional
            Point to source parameters for
            :func:`pycbc.tdi.linear_exponent_surrogate.lal_plus_source`.
        periods : int, optional
            The SSB grid's length in orbits; default the fewest that keep
            the correlations within ``time_window`` from wrapping round, see
            :func:`grid_periods_needed`.
        coarse_bins, harmonics, response_samples
            The response's coarse frequencies, its yearly harmonics and its
            samples per orbit.
        derivative_step : float
            Step of the family's derivative columns, in shape units. The
            shape unit of a coordinate is ``1 / sqrt(M_ii)`` of the
            F-statistic metric ``M`` at the reference, so the units follow
            the source and the data rather than being set by hand. Along a
            nearly degenerate combination the unit is wide in each of its
            coordinates alone (for a two-year stellar-mass binary, mass
            ratio 4 per unit), so a step of half a unit is a secant across
            the waveform's curvature rather than its tangent; the default
            0.05 keeps it local.
        exact_source_phase : bool, optional
            Use the family of :mod:`pycbc.tdi.pn_coefficient_surrogate`,
            exact in the TaylorF2 phase over the whole parameter range, for
            the intrinsic part; default when the intrinsic coordinates are
            mchirp, q, spin1z and spin2z. ``expansion_order`` and
            ``derivative_step`` then concern only the tangents.
        sky_block_radius : float, optional
            Take the response harmonics over the sky within this half-width
            of the reference, rad, from the exact response (see
            :class:`pycbc.tdi.sky_block_response.SkyBlockResponse`), in
            place of the first-order sky model, which holds to about 1%
            only within 0.01 rad. ``harmonics`` must then cover the Doppler
            phase across the block: 8 for 0.07 rad.
        sky_block_points : int, optional
            Sky points the block's basis is fitted to. Default 600.
        source_derivatives : {'taylorf2', 'stencil'}, optional
            How the family's derivatives are taken, see
            ``surrogate_from_source`` in
            :mod:`pycbc.tdi.linear_exponent_surrogate`;
            default exact TaylorF2 jets when the intrinsic coordinates are
            mchirp, q, spin1z and spin2z.
        expansion_order : {1, 2}
            Order of the family's expansion in the intrinsic coordinates,
            see ``surrogate_from_source`` in
            :mod:`pycbc.tdi.linear_exponent_surrogate`.
        max_time_loss : float
            Largest fraction of the statistic lost between fine arrival-time
            samples; sets the oversampling from the rms bandwidth.
        precision : {'double', 'single'}
            Precision of the family's correlation and transforms; the
            marginals are always in double precision.
        """
        from pycbc.coordinates.space_orbit import LisaEqualArmOrbit
        from pycbc.filter.subsample_arrival_time import (
            oversampling_for_loss, rms_bandwidth)
        from pycbc.tdi.inference import DEFAULT_ARM, channel_terms
        from pycbc.tdi.linear_exponent_surrogate import (
            DailyGram, surrogate_from_source)
        from pycbc.tdi.ssb_yearly_harmonics import (
            SSBGrid, guard_bins, ssb_data_spectrum)
        from pycbc.tmpltbank.fstatistic_metric import (
            metric_from_gram, tangent_gram)
        # the orbit of pycbc.tdi's sparse response, unless told otherwise
        dtype = {'double': numpy.complex128,
                 'single': numpy.complex64}[precision]
        orbit = (LisaEqualArmOrbit(armlength=DEFAULT_ARM, t0=0.0)
                 if orbit is None else orbit)
        names = sorted(data)
        channels = tuple(n.split('_')[-1] for n in names)
        first = data[names[0]]
        df = float(first.delta_f)
        kmin = int(numpy.ceil(low_frequency_cutoff / df))
        kmax = int(numpy.floor(high_frequency_cutoff / df))
        freqs = numpy.arange(kmin, kmax + 1) * df
        weight = {n: 4 * df / numpy.asarray(psds[n])[kmin:kmax + 1]
                  for n in names}
        bins = numpy.array([numpy.asarray(data[n])[kmin:kmax + 1]
                            * numpy.sqrt(weight[n]) for n in names])
        start = float(first.start_time)
        if periods is None:
            periods = grid_periods_needed(
                reference if to_source is None else to_source(
                    dict(reference)), start, 1 / df, freqs[0], freqs[-1],
                max(abs(v) for v in time_window))
            logging.info("SSB grid of %d orbital periods", periods)
        # guard bins either side of the band, so that neither the
        # resampled data's Doppler spread nor the harmonics' shifts wrap
        # round to the band's other end
        grid = SSBGrid(start, start + 1 / df, freqs[0], len(freqs), df,
                       periods=periods, guard=guard_bins(
                           freqs[-1], periods, harmonics))
        # the source carries the first channel's whitening, the response
        # the ratio of each channel's to it
        sw = numpy.sqrt(numpy.interp(grid.frequencies, freqs,
                                     weight[names[0]], left=0, right=0))
        coarse = numpy.linspace(grid.frequencies[0],
                                grid.frequencies[-1], coarse_bins)
        ratio = numpy.array([numpy.sqrt(numpy.interp(
            coarse, freqs, weight[n] / weight[names[0]])) for n in names])
        terms = channel_terms(channels, generation=2, delta_t=5.0)
        lon = reference['eclipticlongitude']
        lat = reference['eclipticlatitude']

        taylorf2 = sorted(intrinsic) == sorted(
            ('mchirp', 'q', 'spin1z', 'spin2z'))
        if source_derivatives is None:
            source_derivatives = 'taylorf2' if taylorf2 else 'stencil'
        if exact_source_phase is None:
            exact_source_phase = taylorf2

        def build(scales, step, dtype=numpy.complex128, order=1,
                  directions=None, derivatives='stencil'):
            fam = surrogate_from_source(
                grid, sw, orbit, terms, reference,
                {k: scales[k] for k in intrinsic},
                (scales['eclipticlongitude'], scales['eclipticlatitude']),
                coarse, to_source=to_source, step=step,
                kmax=harmonics, samples=response_samples, dtype=dtype,
                order=order, directions=directions, derivatives=derivatives)
            # the response ratio in the family's own precision
            fam.R0 = (fam.R0 * fam.xp.asarray(ratio)[None, None]).astype(
                fam.R0.dtype)
            return fam

        keys = list(intrinsic) + ['eclipticlongitude', 'eclipticlatitude']
        nodes = OrientationNodes(4, 2)

        def unit_lengths(fam):
            """The metric in the family's shape units, averaged over
            orientations, and the whitened power at each frequency."""
            y, dy = fam.tangents()
            window = fam.xp.asarray(grid.in_data)

            def windowed(s):
                return _to_numpy(fam.xp.fft.fft(
                    fam.xp.fft.ifft(s, axis=-1) * window, axis=-1))
            # one derivative at a time to the host, which sums the Gram
            y = windowed(y)
            host = numpy.empty(dy.shape, dtype=numpy.complex64)
            for i in range(len(dy)):
                host[i] = windowed(dy[i])
            del dy
            gram = tangent_gram(y, host, grid.frequencies)
            metric = sum(w * metric_from_gram(gram, v)
                         for w, v in zip(nodes.weights, nodes.vectors))
            return metric, numpy.sum(numpy.abs(y) ** 2, axis=(0, 1))

        # the metric from derivatives over a part in 1e6 of each value and
        # 1e-6 rad of sky, checked against steps ten times smaller
        probe = {k: 1e-6 * max(abs(reference[k]), 1e-3) for k in intrinsic}
        probe.update(eclipticlongitude=1e-6, eclipticlatitude=1e-6)
        metric, y = unit_lengths(build(probe, 1.0,
                                       derivatives=source_derivatives))
        finer, _ = unit_lengths(build(probe, 0.1,
                                      derivatives=source_derivatives))
        units = numpy.sqrt(numpy.diag(metric))
        if not numpy.allclose(numpy.sqrt(numpy.diag(finer)), units,
                              rtol=1e-2):
            raise RuntimeError("the metric's derivatives have not converged: "
                               "{} against {}".format(
                                   units, numpy.sqrt(numpy.diag(finer))))
        scales = {k: probe[k] / u for k, u in zip(keys, units)}
        # in shape units the metric has a unit diagonal
        kwargs.setdefault('metric', metric / numpy.outer(units, units))
        logging.info("shape units %s", scales)
        # the stencil steps along the metric's eigenvectors in the intrinsic
        # shape units, so that a nearly degenerate combination's curvature
        # is measured, not left over from the axes'
        nint = len(intrinsic)
        intrinsic_metric = kwargs['metric'][:nint, :nint]
        _, directions = numpy.linalg.eigh(intrinsic_metric)
        if exact_source_phase:
            from pycbc.tdi.pn_coefficient_surrogate import (
                PNCoefficientSurrogate)
            fam = PNCoefficientSurrogate(
                build(scales, derivative_step, dtype, 1,
                      derivatives=source_derivatives),
                reference, scales, intrinsic, reference['f_lower'])
        else:
            fam = build(scales, derivative_step, dtype, expansion_order,
                        directions, source_derivatives)
        if sky_block_radius:
            # the response harmonics over a block of the sky from the exact
            # response, in place of the first-order sky model
            from pycbc.filter.nufft_correlation import source_power_weights
            from pycbc.tdi.sky_block_response import SkyBlockResponse
            fam.sky_block = SkyBlockResponse(
                orbit, terms, coarse, (lon, lat),
                (scales['eclipticlongitude'], scales['eclipticlatitude']),
                sky_block_radius, kmax=harmonics, samples=response_samples,
                ratio=ratio, weight=source_power_weights(fam),
                points=sky_block_points, test_points=100,
                exact_frequencies=min(coarse_bins, 160))
        daily = DailyGram(fam)
        spectrum, noise = ssb_data_spectrum(grid, bins, orbit, lon, lat,
                                            xp=fam.xp)
        sigma_f = rms_bandwidth(grid.frequencies, y)
        oversampling = oversampling_for_loss(sigma_f, grid.delta_t,
                                             max_time_loss)
        point = {k: reference[k] for k in keys}
        model = cls(variable_params, fam, daily, spectrum, noise, point,
                    scales, distance_prior, time_window,
                    oversampling=oversampling, **kwargs)
        model._recentre_args = dict(
            grid=grid, bins=bins, orbit=orbit, terms=terms, coarse=coarse,
            ratio=ratio, harmonics=harmonics, samples=response_samples,
            time_window=time_window, kwargs=kwargs)
        return model

    def recentred(self, longitude, latitude, sky_block_radius,
                  sky_block_points=600):
        """The model of a block of the sky centred at ``longitude`` and
        ``latitude``: the same family's intrinsic part, a sky block about the
        new centre (see :class:`pycbc.tdi.sky_block_response.SkyBlockResponse`)
        and the data resampled to SSB time there, the block's phase
        centre."""
        import copy
        from pycbc.filter.nufft_correlation import source_power_weights
        from pycbc.tdi.linear_exponent_surrogate import DailyGram
        from pycbc.tdi.ssb_yearly_harmonics import ssb_data_spectrum
        from pycbc.tdi.sky_block_response import SkyBlockResponse
        a = getattr(self, '_recentre_args', None)
        if a is None:
            raise ValueError("only a model from from_data can be recentred")
        fam = copy.copy(self.surrogate)
        fam.sky_block = SkyBlockResponse(
            a['orbit'], a['terms'], a['coarse'], (longitude, latitude),
            (self.scales['eclipticlongitude'],
             self.scales['eclipticlatitude']), sky_block_radius,
            kmax=a['harmonics'], samples=a['samples'], ratio=a['ratio'],
            weight=source_power_weights(fam), points=sky_block_points,
            test_points=100, exact_frequencies=min(len(a['coarse']), 160))
        spectrum, noise = ssb_data_spectrum(a['grid'], a['bins'], a['orbit'],
                                            longitude, latitude, xp=fam.xp)
        point = dict(self.reference)
        point.update(eclipticlongitude=float(longitude),
                     eclipticlatitude=float(latitude))
        kwargs = dict(a['kwargs'])
        if self.table is not None:
            kwargs['amplitude_table'] = self.table
        model = type(self)(list(self.variable_params), fam, DailyGram(fam),
                           spectrum, noise, point, self.scales,
                           self.distance_prior, a['time_window'],
                           oversampling=self.oversampling, **kwargs)
        model._recentre_args = a
        return model

    _CONFIG_OPTIONS = {
        'low-frequency-cutoff': float, 'high-frequency-cutoff': float,
        'intrinsic-parameters': str, 'time-window': str,
        'distance-min': float, 'distance-max': float,
        'distance-power': float, 'grid-periods': int, 'coarse-bins': int,
        'harmonics': int, 'response-samples': int,
        'derivative-step': float, 'max-time-loss': float,
        'expansion-order': int, 'source-derivatives': str,
        'exact-source-phase': _boolean,
        'orientation-nodes': str, 'margin': float,
        'amplitude-table-file': str, 'sky-block-radius': float,
        'sky-block-points': int}

    @classmethod
    def from_config(cls, cp, data_section='data', data=None, psds=None,
                    **kwargs):
        """Build the model from a configuration file.

        The ``[data]`` section is read as for the Gaussian-noise models,
        unless ``data`` and ``psds`` are given. ``[model]`` takes:

        * ``name = bstatistic_surrogate``
        * ``low-frequency-cutoff``, ``high-frequency-cutoff``: the band, Hz.
        * ``intrinsic-parameters``: the family's intrinsic coordinates.
        * ``reference-{param}``: the reference value of each intrinsic
          coordinate and of ``eclipticlongitude`` and ``eclipticlatitude``;
          a static parameter of that name serves too.
        * ``time-window = LOW HIGH``: arrival-time offsets marginalised
          over, s.
        * ``distance-min``, ``distance-max``, ``distance-power`` (default
          2): the distance prior, in the units of the waveform's distance.
        * ``grid-periods`` (the fewest that hold the data and the
          templates, see :func:`grid_periods_needed`), ``coarse-bins``
          (1500), ``harmonics`` (4),
          ``response-samples`` (32), ``derivative-step`` (0.05),
          ``max-time-loss`` (1e-3), ``orientation-nodes`` (``32 16``),
          ``margin`` (40): see :meth:`from_data` and the class.
        * ``amplitude-table-file``: an HDF file of the amplitude table,
          written on first use; it depends only on ``distance-power``.

        The source parameters are the static parameters and the reference
        point passed through the ``[waveform_transforms]``.
        """
        from pycbc import transforms
        from pycbc.filter.bstatistic import DistancePowerLaw
        from .data_utils import (data_opts_from_config, data_from_cli,
                                 fd_data_from_strain_dict)
        section = 'model'
        options = set(cp.options(section)) - {'name'}
        refs = {o for o in options if o.startswith('reference-')}
        unknown = options - refs - set(cls._CONFIG_OPTIONS)
        if unknown:
            raise ValueError("unknown option(s) in [model] for {}: {}".format(
                cls.name, ', '.join(sorted(unknown))))
        get = {k: conv(cp.get(section, k))
               for k, conv in cls._CONFIG_OPTIONS.items()
               if cp.has_option(section, k)}
        for required in ('low-frequency-cutoff', 'high-frequency-cutoff',
                         'intrinsic-parameters', 'time-window',
                         'distance-min', 'distance-max'):
            if required not in get:
                raise ValueError("[model] needs " + required)
        args = cls._init_args_from_config(cp)
        flow, fhigh = get['low-frequency-cutoff'], get['high-frequency-cutoff']
        if data is None or psds is None:
            detectors = cp.get(data_section, 'instruments').split()
            opts = data_opts_from_config(cp, data_section,
                                         {d: flow for d in detectors})
            strain, psd_strain = data_from_cli(opts)
            data, psds = fd_data_from_strain_dict(opts, strain, psd_strain)
        intrinsic = get['intrinsic-parameters'].split()
        reference = dict(args.get('static_params') or {})
        for name in intrinsic + ['eclipticlongitude', 'eclipticlatitude']:
            option = 'reference-' + name
            if cp.has_option(section, option):
                reference[name] = float(cp.get(section, option))
            elif name not in reference:
                raise ValueError("no reference value for " + name)
        waveform_transforms = args.get('waveform_transforms')
        to_source = None
        if waveform_transforms:
            def to_source(point):
                return transforms.apply_transforms(dict(point),
                                                   waveform_transforms)
        prior = DistancePowerLaw(get['distance-min'], get['distance-max'],
                                 get.get('distance-power', 2.0))
        window = tuple(float(v) for v in get['time-window'].split())
        nodes = tuple(int(v) for v in
                      get.get('orientation-nodes', '32 16').split())
        args.update(kwargs)
        return cls.from_data(
            args.pop('variable_params'), data, psds, reference, intrinsic,
            flow, fhigh, window, prior, to_source=to_source,
            periods=get.get('grid-periods'),
            coarse_bins=get.get('coarse-bins', 1500),
            harmonics=get.get('harmonics', 4),
            response_samples=get.get('response-samples', 32),
            derivative_step=get.get('derivative-step', 0.05),
            expansion_order=get.get('expansion-order', 2),
            source_derivatives=get.get('source-derivatives'),
            exact_source_phase=get.get('exact-source-phase'),
            sky_block_radius=get.get('sky-block-radius'),
            sky_block_points=get.get('sky-block-points', 600),
            max_time_loss=get.get('max-time-loss', 1e-3),
            orientation_nodes=nodes, margin=get.get('margin', 40.0),
            amplitude_table=get.get('amplitude-table-file', True), **args)


def grid_periods_needed(source, start, duration, f_low, f_high, window,
                        margin=1e-3):
    """The fewest whole orbital periods ``T`` of an SSB grid on which the
    correlations at arrival times within ``window`` (s) of the reference do
    not wrap round: with the reference's in-band support ``[t_a, t_b]``
    relative to the data start and data of ``duration``, ``T > max(duration -
    t_a, t_b) + window``. A grid fixed at five years would be too short for
    data of more than about four years and wasteful for shorter data."""
    from pycbc.tdi.linear_exponent_surrogate import lal_plus_source
    from pycbc.tdi.ssb_yearly_harmonics import ORBITAL_PERIOD
    freqs = numpy.linspace(f_low, f_high, 64)
    _, times = lal_plus_source(source, freqs, start)
    t_a, t_b = times.min() - start, times.max() - start
    need = (max(duration - t_a, t_b) + window) * (1 + margin)
    return max(1, int(numpy.ceil(need / ORBITAL_PERIOD)))


class SkyBlockMixture(object):
    """Blocks of the sky as one model: each point's log likelihood comes from
    the model of the block holding its sky position (or the nearest block's
    centre when none does), each block a
    :meth:`BStatisticSurrogate.recentred` model with its own sky block.

    It offers what :class:`pycbc.inference.sampler.amr.AMRSampler` asks of
    a model, and :meth:`reconstruct` from the point's block; batches and
    shared regions are split by block."""
    names = ('eclipticlongitude', 'eclipticlatitude')

    def __init__(self, blocks, prior_distribution=None):
        if not blocks:
            raise ValueError("need at least one block")
        self.blocks = list(blocks)
        first = self.blocks[0]
        self.variable_params = first.variable_params
        self.static_params = first.static_params
        self.prior_distribution = (first.prior_distribution
                                   if prior_distribution is None
                                   else prior_distribution)
        self.sampling_transforms = None
        self.sampling_params = self.variable_params
        self.curvilinear_coordinates = getattr(
            first, 'curvilinear_coordinates', None)
        self.current_params = {}

    def _full(self, point):
        q = dict(self.static_params)
        q.update(point)
        return q

    def block_of(self, points):
        """The index of the block of each point (parameter dicts)."""
        out = numpy.empty(len(points), dtype=int)
        for i, p in enumerate(points):
            q = self._full(p)
            n = _sky_unit(q[self.names[0]], q[self.names[1]])
            best, chosen = -2.0, 0
            for j, b in enumerate(self.blocks):
                off = numpy.array([[(q[k] - b.reference[k]) / b.scales[k]
                                    for k in self.names]])
                if b.surrogate.sky_block.inside(off)[0]:
                    chosen = j
                    break
                c = n @ _sky_unit(b.reference[self.names[0]],
                                  b.reference[self.names[1]])
                if c > best:
                    best, chosen = c, j
            out[i] = chosen
        return out

    def update(self, **params):
        self.current_params = dict(params)

    @property
    def logprior(self):
        return self.prior_distribution(**self.current_params)

    @property
    def loglikelihood(self):
        block = self.blocks[self.block_of([self.current_params])[0]]
        block.update(**self.current_params)
        return block.loglikelihood

    def loglikelihoods(self, points):
        index = self.block_of(points)
        out = numpy.empty(len(points))
        for j in numpy.unique(index):
            sel = numpy.nonzero(index == j)[0]
            out[sel] = self.blocks[j].loglikelihoods([points[i]
                                                      for i in sel])
        return out

    def share(self, region, **kwargs):
        """Share each block's part of ``region``; a part too large for one
        grid is left to the point-by-point path. Raises ``MemoryError`` only
        when no block could share its part."""
        index = self.block_of(region)
        shared = 0
        for j in numpy.unique(index):
            part = [region[i] for i in numpy.nonzero(index == j)[0]]
            try:
                self.blocks[j].share(part, **kwargs)
                shared += 1
            except MemoryError:
                pass
        if not shared:
            raise MemoryError("no block could share its part of the region")

    @property
    def shared_statistics(self):
        total = {}
        for b in self.blocks:
            for k, v in b.shared_statistics.items():
                total[k] = (max(total.get(k, 0), v) if k.startswith('largest')
                            else total.get(k, 0) + v)
        return total

    def reconstruct(self, rec=None, seed=None):
        block = self.blocks[self.block_of([self.current_params])[0]]
        block.update(**self.current_params)
        return block.reconstruct(rec=rec, seed=seed)


def _sky_unit(lon, lat):
    return numpy.array([numpy.cos(lat) * numpy.cos(lon),
                        numpy.cos(lat) * numpy.sin(lon), numpy.sin(lat)])


class SharedLikelihood(object):
    """The marginal log likelihood of :class:`BStatisticSurrogate` for many
    points of a region at once, from computations they share.

    The correlations of every point come from one
    :class:`pycbc.filter.nufft_correlation.PhaseGrid` over the region; the
    Gram matrix of a point is its own, from day sums of the reference's
    power times ``(Mc / Mc0)^(5/3)`` at the point's closed-form stationary
    times, a batch of points per kernel call; the orientation marginals of
    all points and arrival times go through one kernel call per
    quadrature.

    Parameters
    ----------
    model : BStatisticSurrogate
        A model whose family is a
        :class:`pycbc.tdi.pn_coefficient_surrogate.PNCoefficientSurrogate`.
    region : list of dict
        Points spanning the region, with their sky positions.
    sky_tolerance : float
        The relative error, in the template's norm, allowed the response
        on the region's sky basis (see
        :class:`pycbc.filter.nufft_correlation.SkyBasis`); a point beyond
        it, or keeping more phase outside the grid's directions than the
        grid allows, is not :meth:`inside` the region.
    edge_margin : float
        A point's ``ln B`` must fall this far below its largest at both
        ends of the grid's arrival times, or its log likelihood is not
        given (``nan``) and the model computes it one point at a time.
    probes : int
        Points of the region whose arrival-time support, from the model's
        own likelihood, sets the grid's arrival times.
    grid_options : dict
        Passed to :class:`pycbc.filter.nufft_correlation.PhaseGrid`.

    In grid time ``T = delay + shift``, with ``shift`` a template's mean
    arrival-time offset from the reference, the points of a region peak
    together; the grid covers the ``T`` where the probes' ``ln B`` is
    within ``edge_margin`` of their largest, twice over, for the probes
    within it of the best. Every point checks that its ``ln B`` has fallen
    by ``edge_margin`` at the ends.
    """
    def __init__(self, model, region, sky_tolerance=1e-4, edge_margin=30.0,
                 probes=16, grid_options=None):
        from pycbc.filter.nufft_correlation import (PhaseGrid,
                                                    PhaseNUFFTCorrelation)
        self.model = model
        fam = model.surrogate
        self.names = ['mchirp', 'q', 'spin1z', 'spin2z']
        offsets = self._sky(region)
        middle = 0.5 * (offsets.max(axis=0) + offsets.min(axis=0))
        self.sky_tolerance = float(sky_tolerance)
        self.correlator = PhaseNUFFTCorrelation(
            fam, model.data_spectrum, sky=middle, sky_points=offsets,
            sky_tolerance=self.sky_tolerance)
        thetas = self._thetas(region)
        self.edge_margin = float(edge_margin)
        window = self._time_window(region, thetas, probes)
        options = dict(time_range=window)
        options.update(grid_options or {})
        self.grid = PhaseGrid(self.correlator, thetas, window, **options)

    def _time_window(self, region, thetas, probes):
        """The grid times the region's points need: the span where the
        probes' ``ln B`` lies within ``edge_margin`` of each one's largest,
        for the probes within it of the best, widened by itself and eight
        lags either side."""
        m = self.model
        frame = self.correlator.frame(thetas, 3)
        pick = numpy.random.default_rng(0).choice(
            len(region), min(probes, len(region)), replace=False)
        _, shift, _ = self.correlator.project(thetas[pick], frame)
        L = len(m.delays)
        curves = [_to_numpy(m._time_samples(region[j])[2])[:L]
                  for j in pick]
        tops = numpy.array([numpy.nanmax(v) for v in curves])
        lo, hi = numpy.inf, -numpy.inf
        # the probes that matter: within the margin of the best; a point
        # whose peak lies elsewhere fails its edge check
        for i in numpy.nonzero(tops >= tops.max() - self.edge_margin)[0]:
            keep = curves[i] >= tops[i] - self.edge_margin
            T = m.delays[keep] + shift[i]
            lo, hi = min(lo, T.min()), max(hi, T.max())
        pad = (hi - lo) + 8 * m.grid_step
        # no point of the region reaches grid times beyond its delays
        # shifted: a wide likelihood leaves the window there
        _, shifts, _ = self.correlator.project(thetas, frame)
        reach = (m.delays[0] + shifts.min() - m.grid_step,
                 m.delays[-1] + shifts.max() + 2 * m.grid_step)
        return (max(lo - pad, reach[0]), min(hi + pad, reach[1]))

    def _sky(self, points):
        m = self.model
        names = ('eclipticlongitude', 'eclipticlatitude')
        return numpy.array([[(p[n] - m.reference[n]) / m.scales[n]
                             for n in names] for p in points])

    def _thetas(self, points):
        return numpy.array([[p[n] for n in self.names] for p in points])

    def _gram_at(self, xis, thetas, delays, chunk=64):
        """``(g00, g11, g01)`` of shape ``(M, L)`` at ``delays`` ``(M, L)``,
        each template's own: its day sums and response as the model forms
        them, batched on cupy (stationary times in closed form, no
        waveform), one template at a time on the CPU."""
        m = self.model
        xp, dg, fam = m.xp, m.daily_gram, m.surrogate
        var = m.noise_variance
        if xp is numpy:
            out = [numpy.empty(delays.shape), numpy.empty(delays.shape),
                   numpy.empty(delays.shape, dtype=complex)]
            for i, xi in enumerate(xis):
                q = dg.prepare(*fam.pieces(xi))
                for c, v in enumerate(dg.at_prepared(q, delays[i])):
                    out[c][i] = v / var
            return tuple(out)
        parts = [[], [], []]
        for j0 in range(0, len(xis), chunk):
            js = slice(j0, j0 + chunk)
            pw, fw = fam.day_sums(thetas[js], dg, reach=float(
                numpy.abs(delays).max()) + dg.day)
            rho = fam.responses(xis[js, -2:])
            for c, v in enumerate(dg.at_batch(pw, fw, rho, delays[js])):
                parts[c].append(v / var)
        return tuple(xp.concatenate(v) for v in parts)

    def __call__(self, points):
        """The log likelihood of each point; ``nan`` where its ``ln B``
        has not fallen by ``edge_margin`` at the ends of the grid's arrival
        times."""
        m = self.model
        xp = m.xp
        thetas = self._thetas(points)
        sky = self._sky(points)
        xis = numpy.array([m._shape_coordinates(p) for p in points])
        M = len(thetas)
        var = m.noise_variance
        gs = m.grid_step
        # each point's lags whose grid time, and the refinement's up to a
        # lag later, the grid holds
        _, shift, _ = self.correlator.project(thetas, self.grid.frame)
        t_lo, t_hi = self.grid.time_limits()
        d0, L = m.delays[0], len(m.delays)
        first = numpy.clip(numpy.ceil((t_lo - shift - d0) / gs), 0,
                           L - 1).astype(int)
        last = numpy.clip(numpy.floor((t_hi - gs - shift - d0) / gs), 0,
                          L - 1).astype(int)
        count = numpy.maximum(last - first + 1, 0)
        width_l = max(int(count.max()), 1)
        K = first[:, None] + numpy.arange(width_l)[None]
        valid = K <= last[:, None]
        K = numpy.minimum(K, L - 1)
        delays = m.delays[K]
        z = self.grid.evaluate(thetas, delays, sky).astype(
            xp.complex128) / var
        g = self._gram_at(xis, thetas, delays)
        lnb, choice = self._marginal(z, g)
        values = numpy.where(valid, _to_numpy(lnb), -numpy.inf)
        top = values.max(axis=1, keepdims=True)
        # the integrand has fallen off at both ends, or the window ends
        # where the model's does
        rows = numpy.arange(M)
        cut = top[:, 0] - self.edge_margin
        ok = ((values[rows, 0] <= cut) | (first == 0)) & (
            (values[rows, numpy.maximum(count - 1, 0)] <= cut)
            | (last == L - 1)) & (count > 0)
        width = numpy.full((M, width_l), gs)
        terms = [values + numpy.log(width)]
        # finer arrival times where a point's peak is narrower than the
        # grid, as the model does
        need = (values >= top - m._tau) & (values >= m._narrow)
        need = need | numpy.roll(need, 1, axis=1) | numpy.roll(need, -1,
                                                               axis=1)
        need &= valid
        peak = numpy.maximum(numpy.where(need, values, -numpy.inf).max(
            axis=1), 0.5)
        sigma_t = 1 / (2 * numpy.pi * m._sigma_f * numpy.sqrt(2 * peak))
        P = numpy.ceil(gs / sigma_t).astype(int)
        refine = need.any(axis=1) & (P > 1)
        if refine.any():
            rows = numpy.nonzero(refine)[0]
            fine, widths = [], []
            for j in rows:
                k = numpy.nonzero(need[j])[0]
                step = gs / P[j]
                fine.append(numpy.concatenate([delays[j, k] + p * step
                                               for p in range(1, P[j])]))
                widths.append(step)
                width[j, k] = step
            terms[0] = values + numpy.log(width)
            longest = max(len(f) for f in fine)
            pad = numpy.array([numpy.concatenate(
                [f, numpy.full(longest - len(f), f[-1])]) for f in fine])
            fvalid = numpy.array([numpy.arange(longest) < len(f)
                                  for f in fine])
            zf = self.grid.evaluate(thetas[rows], pad, sky[rows]).astype(
                xp.complex128) / var
            gf = self._gram_at(xis[rows], thetas[rows], pad)
            lf = _to_numpy(self._marginal(zf, gf, choice=choice[rows])[0])
            extra = numpy.full((M, longest), -numpy.inf)
            extra[rows] = numpy.where(fvalid, lf + numpy.log(
                numpy.array(widths))[:, None], -numpy.inf)
            terms.append(extra)
        allterms = numpy.concatenate(terms, axis=1)
        top = allterms.max(axis=1, keepdims=True)
        out = (top[:, 0] + numpy.log(numpy.exp(allterms - top).sum(axis=1))
               - numpy.log(m.window_width))
        return numpy.where(ok, out, numpy.nan)

    def inside(self, points):
        """Whether each point lies in the region the shared computation
        covers: on the grid, within the Taylor reach of its third phase
        coordinate, with its phase in the grid's directions and its
        response on the sky basis, each to its tolerance."""
        thetas = self._thetas(points)
        ok = self.grid.inside(thetas)
        if ok.any():
            sky = self._sky([p for p, k in zip(points, ok) if k])
            ok[ok] = self.correlator.sky_error(sky) <= self.sky_tolerance
        return ok

    def _marginal(self, z, g, margin=25.0, choice=None):
        """``ln B`` of shape ``(M, L)``, each point with the coarsest
        orientation quadrature that has converged at its largest
        F-statistic, as the model chooses it.

        ``F = z^H G^-1 z / 2`` bounds ``ln B`` from above at each arrival
        time (the marginal is below the maximum over amplitude, phase and
        any complex orientation). With ``ln B*`` computed at a point's
        largest ``F``, arrival times whose bound is ``margin`` plus the log
        of their number below it add less than ``exp(-margin)`` to the
        point's arrival-time sum; they are given ``-inf`` without the
        quadrature.

        Returns ``ln B`` and the quadrature chosen for each point, which a
        call for the same points' finer arrival times takes as
        ``choice``."""
        m = self.model
        xp = m.xp
        M, L = z.shape[1:]
        sets = m._candidate_node_sets()
        g00, g11, g01 = g
        det = xp.maximum(g00 * g11 - xp.abs(g01) ** 2, 1e-300)
        F = 0.5 * (g11 * xp.abs(z[0]) ** 2 + g00 * xp.abs(z[1]) ** 2
                   - 2 * xp.real(xp.conj(z[0]) * g01 * z[1])) / det
        flat_z = z.reshape(2, -1)
        flat_g = tuple(v.reshape(-1) for v in g)
        # the quadrature the model would choose for each point
        if choice is None:
            choice = m._node_choice(z, g, sets)
        best = xp.asarray(_to_numpy(xp.argmax(F, axis=1))
                          + numpy.arange(M) * L)
        at_best = _to_numpy(m.table.log_orientation_marginals(
            flat_z[:, best], *(v[best] for v in flat_g), sets, xp=xp))
        # the bound's cut, from the exact ln B at the largest F
        anchor = at_best[choice, numpy.arange(M)]
        keep = F >= xp.asarray(anchor - margin - numpy.log(L))[:, None]
        out = xp.full((M, L), -numpy.inf)
        keep_flat = keep.reshape(-1)
        for i in numpy.unique(choice):
            rows = xp.asarray(numpy.nonzero(choice == i)[0])
            mask = xp.zeros(M, dtype=bool)
            mask[rows] = True
            sel = xp.nonzero(keep_flat & xp.repeat(mask, L))[0]
            if len(sel) == 0:
                continue
            vals = m.table.log_orientation_marginals(
                flat_z[:, sel], *(v[sel] for v in flat_g), [sets[i]],
                xp=xp)[0]
            out.reshape(-1)[sel] = vals
        return out, choice
