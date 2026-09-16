"""The union of a prior's harmonics, and what it takes for one to count.

Listing a harmonic is not evaluating it. `tdi_harmonic_combine='union'`
bins every harmonic any corner of the prior carries, and the fiducial by
construction does not carry all of them -- relative binning divides by its
reference, so a harmonic whose reference is the fiducial's zero gets no
bins, no summary, and never reaches `_loglr`. A set-membership check passes
throughout. These tests check the binned set and the likelihood instead.
"""

import types

import numpy as np
import pytest

from pycbc.inference.models.tdi_harmonic_relbin import HarmonicRelative
from pycbc.tdi.inference import clear_cache, harmonic_references
from pycbc.types import Array, FrequencySeries
from pycbc.waveform.plugin import add_custom_waveform


#: Below this the mock source drops its highest harmonic, so a prior whose
#: upper edge sits above it carries a harmonic the fiducial does not.
SPLIT = 1.5


def _harmonics(scale):
    # (5, 5) exists above the split but never has support in the analysis
    # band, which is a different thing from being absent and has to stay
    # distinguishable from it.
    base = ((2, 2), (3, 3))
    return base if float(scale) < SPLIT else base + ((4, 4), (5, 5))


def _piece(harmonic, scale, frequency):
    if harmonic == (2, 2):
        return scale * (1 + 0.04 * frequency) * np.exp(0.3j * frequency)
    if harmonic == (3, 3):
        return ((1 + 0.6 * (scale - 1)) * (0.35 + 0.02 * frequency)
                * np.exp(-0.2j * frequency))
    if harmonic == (5, 5):
        return np.zeros(len(frequency), dtype=complex)
    return (scale - 1.4) * (0.2 + 0.01 * frequency) * np.exp(0.5j * frequency)


def _register(monkeypatch):
    """A det-response waveform that answers like the sparse TDI generator.

    The point of the mock is the one behaviour that produced the bug: a
    harmonic the source does not carry comes back as zeros rather than as an
    error, exactly as `sparse_tdi_fd_det_sequence` returns it.
    """
    name = 'TDIMockUnionHarmonics'

    def waveform(**params):
        frequency = np.asarray(params['sample_points'], dtype=float)
        scale = float(params['scale'])
        live = _harmonics(scale)
        harmonic = params.get('tdi_harmonic')
        if harmonic is None:
            value = sum(_piece(h, scale, frequency) for h in live)
        elif tuple(harmonic) not in live:
            value = np.zeros(len(frequency), dtype=complex)
        else:
            value = _piece(tuple(harmonic), scale, frequency)
        return {channel: Array(value * factor)
                for channel, factor in params['ifos_factors'].items()}

    waveform.required = ('approximant', 'scale', 'ifos_factors')
    add_custom_waveform(name, waveform, 'frequency', sequence=True,
                        has_det_response=True, force=True)

    from pycbc.tdi import inference
    clear_cache()
    monkeypatch.setitem(
        inference._SOURCES, 'mock-union',
        lambda **params: types.SimpleNamespace(
            harmonics=_harmonics(params['scale'])))
    return name, waveform


def _static(name):
    return {'approximant': name, 'tc': 0.0,
            'ifos_factors': {'A': 1.0, 'E': 0.7j},
            'tdi_source': 'mock-union',
            'tdi_harmonic_combine': 'union',
            'tdi_coverage_bounds': {'scale': (1.0, 2.0)},
            # A second corner that also carries (4, 4), so 'the first corner
            # that carries it' is a claim the reference test can fail.
            'tdi_coverage_corners': [{'scale': 1.8}]}


def test_harmonic_references_name_the_first_corner_that_carries_it(
        monkeypatch):
    """A harmonic outside the fiducial is binned against a corner, not zero."""
    _register(monkeypatch)
    params = dict(_static('unused'), scale=1.0)

    references = harmonic_references(params)

    assert set(references) == {(2, 2), (3, 3), (4, 4), (5, 5)}
    # An empty mapping means the fiducial itself.
    assert references[(2, 2)] == {}
    assert references[(3, 3)] == {}
    # `tdi_coverage_bounds` declares (low, high) and is walked in that order,
    # so the reference is reproducible from the configuration file rather
    # than from whichever corner the search happened to reach first. The low
    # edge sits at the fiducial and does not carry (4, 4); the high one does,
    # and so does the explicit corner declared after it.
    assert references[(4, 4)] == {'scale': 2.0}
    assert references[(5, 5)] == {'scale': 2.0}


def test_intersection_leaves_every_harmonic_on_the_fiducial(monkeypatch):
    """Only the union mode needs a stand-in, and it should not spread."""
    _register(monkeypatch)
    params = dict(_static('unused'), scale=1.0,
                  tdi_harmonic_combine='intersection')

    references = harmonic_references(params)

    assert references == {(2, 2): {}, (3, 3): {}}


def _model(name, waveform, harmonics=None, **extra):
    delta_f, length = 1 / 128, 257
    frequency = np.arange(length) * delta_f
    factors = {'A': 1.0, 'E': 0.7j}
    truth = waveform(sample_points=frequency, scale=1.7, ifos_factors=factors)
    data = {channel: FrequencySeries(np.asarray(values), delta_f=delta_f)
            for channel, values in truth.items()}
    psds = {channel: FrequencySeries(np.ones(length), delta_f=delta_f)
            for channel in data}
    model = HarmonicRelative(
        variable_params=['scale'], data=data, psds=psds,
        low_frequency_cutoff={channel: 0.1 for channel in data},
        high_frequency_cutoff={channel: 1.8 for channel in data},
        static_params=_static(name), fiducial_params={'scale': 1.0},
        harmonics=harmonics, harmonic_epsilon=1e-4,
        marginalize_phase=False, **extra)
    return model, data, frequency, delta_f, factors


def _exact(data, candidate, delta_f):
    kmin, kmax = int(0.1 / delta_f), int(1.8 / delta_f)
    total = 0.0
    for channel in data:
        observed = np.asarray(data[channel])[kmin:kmax]
        proposed = np.asarray(candidate[channel])[kmin:kmax]
        total += 4 * delta_f * (np.real(np.vdot(proposed, observed))
                                - 0.5 * np.vdot(proposed, proposed).real)
    return total


def test_a_union_harmonic_is_binned_and_reaches_the_likelihood(monkeypatch):
    """The regression: (4, 4) is outside the fiducial and must still count."""
    name, waveform = _register(monkeypatch)
    model, data, frequency, delta_f, factors = _model(name, waveform)

    # Listed -- which is all the previous check established.
    assert (4, 4) in model.harmonics
    # (5, 5) has a reference and still never bins, because its support is
    # outside the band. The model must drop it rather than advertise a mode
    # it does not evaluate -- that confusion is the whole bug.
    assert (5, 5) not in model.harmonics
    # And binned, in every channel. Without a non-zero reference the fiducial
    # for (4, 4) is identically zero, `_prepare_channel` skips it, and this
    # is the assertion that fails.
    for ifo in data:
        assert (4, 4) in model.hedges[ifo]
        assert len(model.hedges[ifo][(4, 4)]) >= 2
        assert np.any(model.h00_h[ifo][(4, 4)] != 0)

    candidate_scale = 1.6
    candidate = waveform(sample_points=frequency, scale=candidate_scale,
                         ifos_factors=factors)
    exact = _exact(data, candidate, delta_f)

    model.update(scale=candidate_scale)
    assert model.loglr == pytest.approx(exact, rel=2e-5)

    # And the harmonic carries real weight, so a model that leaves it out is
    # wrong by much more than that tolerance -- the union is not cosmetic.
    without = _model(name, waveform, harmonics=((2, 2), (3, 3)))[0]
    without.update(scale=candidate_scale)
    assert abs(without.loglr - exact) > 1e-2


def test_a_candidate_below_the_split_contributes_zero_for_the_union_harmonic(
        monkeypatch):
    """A missing mode is worth zero, and saying so must stay exact."""
    name, waveform = _register(monkeypatch)
    model, data, frequency, delta_f, factors = _model(name, waveform)

    candidate_scale = 1.2          # carries (2, 2) and (3, 3) only
    assert (4, 4) not in _harmonics(candidate_scale)
    candidate = waveform(sample_points=frequency, scale=candidate_scale,
                         ifos_factors=factors)
    exact = _exact(data, candidate, delta_f)

    model.update(scale=candidate_scale)
    assert model.loglr == pytest.approx(exact, rel=2e-5)
