"""Integration boundaries of the per-harmonic relative likelihood."""

import numpy as np
import pytest

from pycbc.inference.models.tdi_harmonic_relbin import HarmonicRelative
from pycbc.types import Array, FrequencySeries
from pycbc.waveform.plugin import add_custom_waveform


def test_harmonic_model_is_available_to_pycbc_inference():
    """A class usable only by direct import is not an inference model."""
    from pycbc.inference.models import get_model

    assert get_model('tdi_harmonic_relative') is HarmonicRelative


def test_shared_query_preserves_each_channels_distinct_grid():
    """The union optimisation must be exact even if channel grids diverge."""
    model = object.__new__(HarmonicRelative)
    model._data = {'A': None, 'E': None}
    model.harmonics = ((2, 2), (3, 3))
    model.f = {
        'A': np.array([1.0, 2.0, 3.0, 4.0]),
        'E': np.array([1.0, 2.5, 3.0, 4.5]),
    }
    model.hquery = {
        'A': {(2, 2): np.array([0, 2]), (3, 3): np.array([1, 3])},
        'E': {(2, 2): np.array([0, 1, 2]),
              (3, 3): np.array([1, 3])},
    }

    model._build_shared_query()

    for harmonic, (channels, union) in model.uquery.items():
        assert channels == ('A', 'E')
        for channel in channels:
            own = model.f[channel][model.hquery[channel][harmonic]]
            assert np.array_equal(union[model.utake[channel][harmonic]], own)


def test_multimode_likelihood_includes_the_cross_terms():
    """Exercise the registered model against a small exact likelihood."""
    name = 'TDIMockOverlappingHarmonics'

    def waveform(**params):
        frequency = np.asarray(params['sample_points'], dtype=float)
        scale = float(params['scale'])
        harmonic = params.get('tdi_harmonic')
        pieces = {
            (2, 2): scale * (1 + 0.04 * frequency)
            * np.exp(0.3j * frequency),
            (3, 3): (1 + 0.6 * (scale - 1))
            * (0.35 + 0.02 * frequency) * np.exp(-0.2j * frequency),
        }
        value = sum(pieces.values()) if harmonic is None else pieces[harmonic]
        return {
            channel: Array(value * factor)
            for channel, factor in params['ifos_factors'].items()
        }

    waveform.required = ('approximant', 'scale', 'ifos_factors')
    add_custom_waveform(name, waveform, 'frequency', sequence=True,
                        has_det_response=True, force=True)

    delta_f, length = 1 / 128, 257
    frequency = np.arange(length) * delta_f
    factors = {'A': 1.0, 'E': 0.7j}
    truth = waveform(
        sample_points=frequency, scale=1.0, ifos_factors=factors)
    data = {
        channel: FrequencySeries(np.asarray(values), delta_f=delta_f)
        for channel, values in truth.items()
    }
    psds = {
        channel: FrequencySeries(np.ones(length), delta_f=delta_f)
        for channel in data
    }
    low = {channel: 0.1 for channel in data}
    high = {channel: 1.8 for channel in data}
    model = HarmonicRelative(
        variable_params=['scale'], data=data, psds=psds,
        low_frequency_cutoff=low, high_frequency_cutoff=high,
        static_params={'approximant': name, 'ifos_factors': factors,
                       'tc': 0.0},
        fiducial_params={'scale': 1.0},
        harmonics=((2, 2), (3, 3)), harmonic_epsilon=1e-4,
        marginalize_phase=False)

    candidate_scale = 1.03
    candidate = waveform(
        sample_points=frequency, scale=candidate_scale,
        ifos_factors=factors)
    kmin, kmax = int(0.1 / delta_f), int(1.8 / delta_f)
    exact = 0.0
    for channel in data:
        observed = np.asarray(data[channel])[kmin:kmax]
        proposed = np.asarray(candidate[channel])[kmin:kmax]
        exact += 4 * delta_f * (
            np.real(np.vdot(proposed, observed))
            - 0.5 * np.vdot(proposed, proposed).real)

    model.update(scale=candidate_scale)
    assert model.loglr == pytest.approx(exact, rel=2e-5)

    without_cross = HarmonicRelative(
        variable_params=['scale'], data=data, psds=psds,
        low_frequency_cutoff=low, high_frequency_cutoff=high,
        static_params={'approximant': name, 'ifos_factors': factors,
                       'tc': 0.0},
        fiducial_params={'scale': 1.0},
        harmonics=((2, 2), (3, 3)), harmonic_epsilon=1e-4,
        cross_terms=False, marginalize_phase=False)
    without_cross.update(scale=candidate_scale)
    assert abs(without_cross.loglr - exact) > 1e-2
