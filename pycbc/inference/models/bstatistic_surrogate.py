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

import numpy

from pycbc.filter.bstatistic import (AmplitudeTable, OrientationNodes,
                                     _project,
                                     draw_amplitude, draw_orientation,
                                     draw_phase, log_orientation_marginal)
from pycbc.filter.subsample_arrival_time import (correlation_at,
                                                 rms_bandwidth)
from pycbc.tmpltbank.fstatistic_metric import orientation_weights

from .base import BaseModel


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
        :func:`pycbc.filter.bstatistic.log_orientation_marginal`.
    """
    name = 'bstatistic_surrogate'

    def __init__(self, variable_params, surrogate, daily_gram,
                 data_spectrum, noise_variance, reference, scales,
                 distance_prior, time_window, oversampling=10,
                 orientation_nodes=(32, 16), margin=40.0,
                 amplitude_table=True, **kwargs):
        super().__init__(variable_params, **kwargs)
        self.surrogate, self.daily_gram = surrogate, daily_gram
        self.data_spectrum = data_spectrum
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
        self._tau = numpy.log(1e4 * len(self._lags))
        self.table = (AmplitudeTable(distance_prior) if amplitude_table
                      else None)

    def _shape_coordinates(self, params):
        point = dict(self.reference)
        point.update({k: params[k] for k in point if k in params})
        return numpy.array([(point[k] - self.reference[k]) / self.scales[k]
                            for k in self.reference])

    def _time_samples(self, params):
        """``z``, the Gram matrix and ``ln B`` at every fine arrival time
        of the window, scaled by the noise variance."""
        s = self.surrogate
        b, t, rho = s.pieces(self._shape_coordinates(params))
        spectrum = s.correlation_spectrum(self.data_spectrum, b, t, rho)
        xp = self.xp
        P = self.oversampling
        n = spectrum.shape[-1]
        ramp = 2j * numpy.pi * xp.arange(n, dtype=xp.float64) / (P * n)
        fine = {}

        def correlation(p):
            # the p-th fraction of a step, transformed only when asked for
            if p not in fine:
                shifted = spectrum if p == 0 else spectrum * xp.exp(
                    ramp * p).astype(spectrum.dtype)
                fine[p] = xp.fft.ifft(shifted, axis=-1) * n
            return fine[p]
        gram = self.daily_gram(b, t, rho)
        var = self.noise_variance

        def samples(p, lags, delays):
            z = correlation(p)[:, lags].astype(xp.complex128) / var
            g = tuple(v / var for v in self.daily_gram.at_delays(gram, delays))
            return z, g
        # every lag of the window on the grid
        z, g = samples(0, self._lags_xp, xp.asarray(self.delays))
        nodes = self._orientation_nodes(z, g)

        def marginal(z, g):
            return log_orientation_marginal(
                z, *g, self.distance_prior, nodes, margin=self.margin,
                table=self.table, xp=xp)
        lnb = marginal(z, g)
        width = numpy.full(len(self.delays), self.grid_step)
        delays = self.delays
        # the lags whose peaks the grid does not resolve, and neighbours
        values = _to_numpy(lnb)
        top = values.max()
        need = (values >= top - self._tau) & (values >= self._narrow)
        need = need | numpy.roll(need, 1) | numpy.roll(need, -1)
        if P > 1 and need.any():
            k = numpy.nonzero(need)[0]
            width[k] = self.time_step
            zs, gs, ls, ds = [z], [g], [lnb], [delays]
            for p in range(1, P):
                d = self.delays[k] + p * self.time_step
                zp, gp = samples(p, self._lags_xp[xp.asarray(k)],
                                 xp.asarray(d))
                zs.append(zp)
                gs.append(gp)
                ls.append(marginal(zp, gp))
                ds.append(d)
            z = xp.concatenate(zs, axis=1)
            g = tuple(xp.concatenate([v[i] for v in gs]) for i in range(3))
            lnb = xp.concatenate(ls)
            delays = numpy.concatenate(ds)
            width = numpy.concatenate(
                [width] + [numpy.full(len(k), self.time_step)] * (P - 1))
        return z, g, lnb, (spectrum, gram, delays, width)

    def _orientation_nodes(self, z, g):
        """The coarsest orientation quadrature, doubling from 8 x 4 up to
        ``orientation_nodes``, whose value at the sample with the largest
        F-statistic changes by less than 1e-3 on the next doubling. A weak
        signal's broad orientation posterior needs few nodes, a loud one's
        many."""
        xp = self.xp
        x = xp.abs(z[0]) ** 2 / xp.maximum(g[0], 1e-300) \
            + xp.abs(z[1]) ** 2 / xp.maximum(g[1], 1e-300)
        k = int(xp.argmax(x))
        one = (z[:, k:k + 1],) + tuple(v[k:k + 1] for v in g)
        n_cos, n_psi = 8, 4
        nodes = OrientationNodes(n_cos, n_psi)
        value = float(log_orientation_marginal(*one, self.distance_prior,
                                               nodes, table=self.table,
                                               xp=xp)[0])
        while n_cos < self.nodes.n_cos:
            finer = OrientationNodes(2 * n_cos, 2 * n_psi)
            new = float(log_orientation_marginal(*one, self.distance_prior,
                                                 finer, table=self.table,
                                                 xp=xp)[0])
            if abs(new - value) < 1e-3:
                break
            n_cos, n_psi, nodes, value = 2 * n_cos, 2 * n_psi, finer, new
        return nodes

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
        ``signal_phase`` (the overall phase of the signal), ``loglr`` and
        ``snr``, the matched-filter log likelihood ratio of the drawn signal
        and ``sqrt(2 loglr)``, and ``loglikelihood``, the marginal one.
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
                         self.daily_gram.at_delays(gram, xp.asarray([delay])))
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
                   signal_phase=float(numpy.mod(phase + quarter * numpy.pi,
                                                2 * numpy.pi)),
                   loglikelihood=self.loglikelihood)
        return rec

    @classmethod
    def from_data(cls, variable_params, data, psds, reference, intrinsic,
                  low_frequency_cutoff, high_frequency_cutoff, time_window,
                  distance_prior, orbit=None, to_source=None, periods=5,
                  coarse_bins=1500, harmonics=4, response_samples=32,
                  derivative_step=0.5, max_time_loss=1e-3, precision='double',
                  **kwargs):
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
        periods, coarse_bins, harmonics, response_samples
            The SSB grid's length in orbits, the response's coarse
            frequencies, its yearly harmonics and its samples per orbit.
        derivative_step : float
            Step of the family's derivative columns, in shape units. The
            shape unit of a coordinate is ``1 / sqrt(M_ii)`` of the
            F-statistic metric ``M`` at the reference, so the units follow
            the source and the data rather than being set by hand.
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
        from pycbc.tdi.ssb_yearly_harmonics import (SSBGrid,
                                                    ssb_data_spectrum)
        from pycbc.tmpltbank.fstatistic_metric import fstatistic_metric
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
        grid = SSBGrid(start, start + 1 / df, freqs[0], len(freqs), df,
                       periods=periods)
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

        def build(scales, step, dtype=numpy.complex128):
            fam = surrogate_from_source(
                grid, sw, orbit, terms, reference,
                {k: scales[k] for k in intrinsic},
                (scales['eclipticlongitude'], scales['eclipticlatitude']),
                coarse, to_source=to_source, step=step,
                kmax=harmonics, samples=response_samples, dtype=dtype)
            # the response ratio in the family's own precision
            fam.R0 = (fam.R0 * fam.xp.asarray(ratio)[None, None]).astype(
                fam.R0.dtype)
            return fam

        keys = list(intrinsic) + ['eclipticlongitude', 'eclipticlatitude']
        nodes = OrientationNodes(4, 2)

        def unit_lengths(fam):
            """sqrt of the metric's diagonal in the family's shape units,
            averaged over orientations."""
            y, dy = fam.tangents()
            window = fam.xp.asarray(grid.in_data)

            def windowed(s):
                return _to_numpy(fam.xp.fft.fft(
                    fam.xp.fft.ifft(s, axis=-1) * window, axis=-1))
            y, dy = windowed(y), windowed(dy)
            metric = sum(w * fstatistic_metric(y, dy, v, grid.frequencies)
                         for w, v in zip(nodes.weights, nodes.vectors))
            return numpy.sqrt(numpy.diag(metric)), y

        # the metric from derivatives over a part in 1e6 of each value and
        # 1e-6 rad of sky, checked against steps ten times smaller
        probe = {k: 1e-6 * max(abs(reference[k]), 1e-3) for k in intrinsic}
        probe.update(eclipticlongitude=1e-6, eclipticlatitude=1e-6)
        units, y = unit_lengths(build(probe, 1.0))
        finer, _ = unit_lengths(build(probe, 0.1))
        if not numpy.allclose(finer, units, rtol=1e-2):
            raise RuntimeError("the metric's derivatives have not converged: "
                               "{} against {}".format(units, finer))
        scales = {k: probe[k] / u for k, u in zip(keys, units)}
        logging.info("shape units %s", scales)
        fam = build(scales, derivative_step, dtype)
        daily = DailyGram(fam)
        spectrum, noise = ssb_data_spectrum(grid, bins, orbit, lon, lat,
                                            xp=fam.xp)
        sigma_f = rms_bandwidth(grid.frequencies, numpy.sum(
            numpy.abs(y) ** 2, axis=(0, 1)))
        oversampling = oversampling_for_loss(sigma_f, grid.delta_t,
                                             max_time_loss)
        point = {k: reference[k] for k in keys}
        return cls(variable_params, fam, daily, spectrum, noise, point,
                   scales, distance_prior, time_window,
                   oversampling=oversampling, **kwargs)

    _CONFIG_OPTIONS = {
        'low-frequency-cutoff': float, 'high-frequency-cutoff': float,
        'intrinsic-parameters': str, 'time-window': str,
        'distance-min': float, 'distance-max': float,
        'distance-power': float, 'grid-periods': int, 'coarse-bins': int,
        'harmonics': int, 'response-samples': int,
        'derivative-step': float, 'max-time-loss': float,
        'orientation-nodes': str, 'margin': float}

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
        * ``grid-periods`` (5), ``coarse-bins`` (1500), ``harmonics`` (4),
          ``response-samples`` (32), ``derivative-step`` (0.5),
          ``max-time-loss`` (1e-3), ``orientation-nodes`` (``32 16``),
          ``margin`` (40): see :meth:`from_data` and the class.

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
            periods=get.get('grid-periods', 5),
            coarse_bins=get.get('coarse-bins', 1500),
            harmonics=get.get('harmonics', 4),
            response_samples=get.get('response-samples', 32),
            derivative_step=get.get('derivative-step', 0.5),
            max_time_loss=get.get('max-time-loss', 1e-3),
            orientation_nodes=nodes, margin=get.get('margin', 40.0), **args)
