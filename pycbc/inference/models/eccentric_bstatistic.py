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
