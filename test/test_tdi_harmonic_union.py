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
from pycbc.tdi.inference import (clear_cache, harmonic_references,
                                 _coverage_overrides, _coverage_sources,
                                 _harmonic_overrides, _reference_source)
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
        waveform.calls.append(params.get('tdi_harmonic'))
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
    waveform.calls = []
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
            # These points bound response support and train its grid, but do
            # not carry the union-only harmonic.
            'tdi_coverage_bounds': {'scale': (1.0, 1.2)},
            # A second corner that also carries (4, 4), so 'the first corner
            # that carries it' is a claim the reference test can fail. These
            # discover harmonics without making every point train every band.
            'tdi_harmonic_corners': [{'scale': 2.0}, {'scale': 1.8}]}


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
    # The coverage points do not carry (4, 4). The harmonic-only corners are
    # walked in declaration order, so the reference is reproducible from the
    # configuration rather than from whichever corner a search visits first.
    assert references[(4, 4)] == {'scale': 2.0}
    assert references[(5, 5)] == {'scale': 2.0}


def test_harmonic_corners_do_not_train_the_response_grid(monkeypatch):
    """Discovery points and expensive grid-training points are distinct."""
    _register(monkeypatch)
    params = dict(_static('unused'), scale=1.0)

    assert _coverage_overrides(params) == [
        {'scale': 1.0}, {'scale': 1.2}]
    assert _harmonic_overrides(params) == [
        {'scale': 1.0}, {'scale': 1.2},
        {'scale': 2.0}, {'scale': 1.8}]


def test_harmonic_discovery_does_not_retain_corner_sources(monkeypatch):
    """A metadata scan must not fill the large general source cache."""
    _register(monkeypatch)
    from pycbc.tdi import inference
    params = dict(_static('unused'), scale=1.0)

    harmonic_references(params)

    # The reusable fiducial is cached; the four discovery probes are not.
    assert len(inference._SOURCE_CACHE) == 1


def test_source_cache_limit_is_epoch_configurable(monkeypatch):
    """Several reference groups must not force one global memory policy."""
    _register(monkeypatch)
    from pycbc.tdi import inference
    params = dict(_static('unused'), tdi_source_cache_limit=2)

    for scale in (1.0, 1.1, 1.2):
        inference._build_source(dict(params, scale=scale))

    assert len(inference._SOURCE_CACHE) == 2


def test_reference_is_pinned_while_coverage_replaces_whole_groups(monkeypatch):
    """Coverage churn must neither copy nor evict a prepared reference."""
    _register(monkeypatch)
    from pycbc.tdi import inference
    params = dict(_static('unused'), scale=1.0, phase=0.0)

    reference = _reference_source(params)
    first = _coverage_sources(params)
    second = _coverage_sources(dict(params, phase=1.0))

    assert _reference_source(params) is reference
    assert len(first) == len(second) == 2
    assert len(inference._REFERENCE_SOURCES) == 1
    assert len(inference._COVERAGE_SOURCES) == 2
    assert all(left is not right for left, right in zip(first, second))


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
    # The call above constructs synthetic data and is intentionally summed;
    # calls recorded from here on belong to model construction and evaluation.
    waveform.calls.clear()
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


def test_union_parent_bootstraps_with_one_real_harmonic(monkeypatch):
    """The parent must not ask for the impossible mode-summed union."""
    name, waveform = _register(monkeypatch)

    model, _, _, _, _ = _model(name, waveform)

    assert model.harmonics
    assert waveform.calls
    assert None not in waveform.calls
    assert 'tdi_harmonic' not in model.static_params
    assert 'tdi_harmonic' not in model.fid_params


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
