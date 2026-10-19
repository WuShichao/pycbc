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

import numpy

from pycbc.filter.bstatistic import (OrientationNodes, _project,
                                     draw_amplitude, draw_orientation,
                                     draw_phase, log_orientation_marginal)
from pycbc.filter.subsample_arrival_time import (correlation_at,
                                                 fractional_correlations)
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
        ``(n_cos, n_psi)`` of the orientation quadrature.
    margin : float
        Terms of the marginal below ``exp(-margin)`` of its total are
        skipped, see
        :func:`pycbc.filter.bstatistic.log_orientation_marginal`.
    """
    name = 'bstatistic_surrogate'

    def __init__(self, variable_params, surrogate, daily_gram,
                 data_spectrum, noise_variance, reference, scales,
                 distance_prior, time_window, oversampling=10,
                 orientation_nodes=(32, 16), margin=40.0, **kwargs):
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
        fine = delays[None, :] + step * numpy.arange(
            self.oversampling)[:, None]
        lo, hi = time_window
        self._window = (fine >= lo) & (fine < hi)
        if not self._window.any():
            raise ValueError("the time window holds no arrival times")
        self.delays = fine[self._window]
        xp = self.xp = surrogate.xp
        self._window_xp = xp.asarray(self._window)
        self._delays_xp = xp.asarray(self.delays)
        self.time_step = step
        self.window_width = float(hi - lo)

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
        fine = fractional_correlations(spectrum, self.oversampling, xp=xp)
        z = xp.stack([fine[:, a, :][self._window_xp] for a in range(2)])
        z = z / self.noise_variance
        gram = self.daily_gram(b, t, rho)
        g = tuple(x / self.noise_variance for x in
                  self.daily_gram.at_delays(gram, self._delays_xp))
        lnb = log_orientation_marginal(z, *g, self.distance_prior,
                                       self.nodes, margin=self.margin,
                                       xp=xp)
        return z, g, lnb, (spectrum, gram)

    def _loglikelihood(self):
        _, _, lnb, _ = self._time_samples(self.current_params)
        xp = self.xp
        top = float(xp.max(lnb))
        return top + float(numpy.log(float(xp.sum(xp.exp(lnb - top)))
                                     * self.time_step / self.window_width))

    def reconstruct(self, rec=None, seed=None):
        """Draw the marginalised parameters at the current point.

        Returns ``rec`` (a new dict by default) with ``delta_tc``,
        ``inclination``, ``polarization`` in ``[0, 2 pi)``, ``distance``,
        ``signal_phase`` (the overall phase of the signal) and
        ``loglikelihood``.
        """
        rng = numpy.random.default_rng(seed)
        rec = {} if rec is None else rec
        params = self.current_params
        _, _, lnb, (spectrum, gram) = self._time_samples(params)
        xp = self.xp
        lnb = _to_numpy(lnb)
        weight = numpy.exp(lnb - lnb.max())
        k = rng.choice(len(weight), p=weight / weight.sum())
        delay = self.delays[k] + (rng.random() - 0.5) * self.time_step
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
        # psi + k pi / 2 with phase + k pi is the same signal
        quarter = rng.integers(4)
        rec.update(delta_tc=float(delay),
                   inclination=float(numpy.arccos(cos_iota)),
                   polarization=float(psi + quarter * numpy.pi / 2),
                   distance=float(self.distance_prior.scale / h),
                   signal_phase=float(numpy.mod(phase + quarter * numpy.pi,
                                                2 * numpy.pi)),
                   loglikelihood=self.loglikelihood)
        return rec
