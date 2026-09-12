"""Tests for the sparse on-the-fly TDI evaluator."""

import numpy as np
import pytest

from pycbc.coordinates.space_orbit import LisaEqualArmOrbit
from pycbc.tdi.backends.pytdi_backend import (PyTDICombinationAdapter,
                                              combine_links,
                                              get_pytdi_combination)
from pycbc.tdi.onthefly import (adaptive_time_grid, chain_delay,
                                reconstruct, reconstruct_complex,
                                sparse_channel)
from pycbc.tdi.response import (link_geometry, link_response,
                                sample_constellation)
from pycbc.tdi.sources import (LALFDSource, LALIMRPhenomDSource,
                               LALModesSource, LALTDSource,
                               NewtonianChirp,
                               TimeShiftedHarmonicSource,
                               _positive_frequency_envelope)


def _lal_waveforms_available():
    """Is ``pycbc.waveform`` importable at all in this environment?

    The LAL-backed source needs it, and the plugin scan at import time raises
    if an installed package registers a pycbc.waveform entry point whose
    module is absent from the checked-out PyCBC, which is how a sibling
    feature branch's plugin looks from here. That is an environment defect,
    not a missing optional dependency, so the reason is reported.
    """
    try:
        import pycbc.waveform                                  # noqa: F401
    except Exception as error:                                 # pragma: no cover
        return f"pycbc.waveform is not importable here: {error!r}"
    return None


_NO_LAL = _lal_waveforms_available()


def test_negative_frequency_mode_is_factored_into_a_slow_envelope():
    phase = np.linspace(1.0e6, 1.0e6 + 200.0, 1000)
    slow = (2.0 + 0.25j) * (1.0 + np.linspace(0.0, 1.0e-3, len(phase)))
    native = slow * np.exp(-1j * phase)
    envelope = _positive_frequency_envelope(native, phase)

    # The chosen positive-frequency representation reconstructs the same
    # real strain while removing, rather than doubling, the carrier phase.
    assert np.allclose(
        np.real(envelope * np.exp(1j * phase)), np.real(native),
        rtol=2e-10, atol=2e-14)
    assert np.allclose(envelope, np.conj(slow), rtol=2e-10, atol=2e-14)


class _ClockProbe:
    """Small harmonic source whose values reveal the queried source time."""

    harmonics = (2,)
    t_start = 100.0
    t_end = 200.0

    def amplitude(self, harmonic, time):
        time = np.asarray(time)
        return time + harmonic, 1j * time

    def carrier_phase(self, harmonic, time):
        return 2 * np.asarray(time) + harmonic

    def angular_frequency(self, harmonic, time):
        return np.zeros_like(np.asarray(time)) + harmonic

    def polarizations(self, time):
        time = np.asarray(time)
        return time, -time

    def support_blocks(self, harmonic):
        return ((105.0, 120.0), (150.0, 195.0))


class _LinearCarrier:
    """Constant envelope with an exactly linear carrier phase."""

    harmonics = (2,)

    def __init__(self, frequency=0.0123):
        self.frequency = float(frequency)
        self.plus = 2.0e-21 + 0.5e-21j
        self.cross = -0.3e-21 + 1.1e-21j

    def amplitude(self, harmonic, time):
        shape = np.shape(time)
        return (np.full(shape, self.plus, dtype=complex),
                np.full(shape, self.cross, dtype=complex))

    def carrier_phase(self, harmonic, time):
        return 2 * np.pi * self.frequency * np.asarray(time)

    def angular_frequency(self, harmonic, time):
        return np.full(np.shape(time), 2 * np.pi * self.frequency)


class _LinearFDSource:
    """Minimal native-FD source for response and FrequencySeries tests."""

    harmonics = (2,)

    def frequency_harmonics(self, frequencies):
        frequencies = np.asarray(frequencies, dtype=float)
        return {2: {
            'indices': np.arange(len(frequencies)),
            'frequency': frequencies,
            'time': np.linspace(1e5, 3.0e7, len(frequencies)),
            'plus': np.full(len(frequencies), 2.0e-21 + 0.5e-21j),
            'cross': np.full(len(frequencies), -0.3e-21 + 1.1e-21j),
        }}

    def frequency_polarizations(self, frequencies):
        item = self.frequency_harmonics(frequencies)[2]
        return item['plus'], item['cross']


def test_time_shifted_harmonic_source_translates_every_query():
    native = _ClockProbe()
    shifted = TimeShiftedHarmonicSource(native, offset=120.0,
                                        t_start=0.0, t_end=70.0)
    query = np.array([0.0, 5.0, 70.0])
    source_time = query + 120.0

    amp_plus, amp_cross = shifted.amplitude(2, query)
    assert np.array_equal(amp_plus, source_time + 2)
    assert np.array_equal(amp_cross, 1j * source_time)
    assert np.array_equal(shifted.carrier_phase(2, query),
                          2 * source_time + 2)
    assert np.array_equal(shifted.angular_frequency(2, query),
                          np.full(3, 2.0))
    plus, cross = shifted.polarizations(query)
    assert np.array_equal(plus, source_time)
    assert np.array_equal(cross, -source_time)
    assert native.t_start == 100.0
    assert native.t_end == 200.0


def test_time_shifted_harmonic_source_translates_and_clips_support():
    shifted = TimeShiftedHarmonicSource(
        _ClockProbe(), offset=110.0, t_start=0.0, t_end=70.0)
    assert shifted.support_blocks(2) == ((0.0, 10.0), (40.0, 70.0))
    assert shifted.support(2) == (0.0, 70.0)


def test_time_shifted_harmonic_source_defaults_to_translated_bounds():
    shifted = TimeShiftedHarmonicSource(_ClockProbe(), offset=125.0)
    assert shifted.t_start == -25.0
    assert shifted.t_end == 75.0


def test_time_shifted_frequency_harmonics_shift_phase_and_clip_time():
    native = _LinearFDSource()
    offset = 1e5
    shifted = TimeShiftedHarmonicSource(
        native, offset=offset, t_start=0.0, t_end=1e7)
    frequencies = np.linspace(0.01, 0.02, 11)
    original = native.frequency_harmonics(frequencies)[2]
    result = shifted.frequency_harmonics(frequencies)[2]
    keep = (original['time'] - offset >= 0.0) & \
        (original['time'] - offset <= 1e7)
    phase = np.exp(1j * np.remainder(
        2 * np.pi * original['frequency'][keep] * offset, 2 * np.pi))

    assert np.array_equal(result['indices'], original['indices'][keep])
    assert np.array_equal(result['time'], original['time'][keep] - offset)
    assert np.allclose(result['plus'], original['plus'][keep] * phase)
    assert np.allclose(result['cross'], original['cross'][keep] * phase)


def _terms(name="X2", delta_t=5.0):
    return PyTDICombinationAdapter(
        name, get_pytdi_combination(name), delta_t=delta_t).terms()


def test_chain_delay_reproduces_wang_table_1():
    """Delays summed from the orbit, without build_shifts or dsp.timeshift."""
    orbit = LisaEqualArmOrbit()
    times = np.linspace(1e5, 2e7, 9)
    arm = float(np.mean(sample_constellation(times, orbit).ltt))
    chains = sorted({term.operators for term in _terms("X2")}, key=len)

    for chain in chains:
        delay = chain_delay(orbit, times, chain)
        # every operator in a Michelson chain is one arm
        assert np.allclose(delay / arm, len(chain), rtol=2e-4)

    extent = max(chain_delay(orbit, times, c).mean() for c in chains) \
        - min(chain_delay(orbit, times, c).mean() for c in chains)
    assert np.isclose(extent / arm, 7.0, rtol=1e-3)   # fact 3 / Wang Table 1


def test_chain_delay_prefix_cache_changes_nothing():
    """The memoisation is an optimisation, not a different calculation."""
    orbit = LisaEqualArmOrbit()
    times = np.linspace(1e5, 5e6, 7)
    chains = sorted({term.operators for term in _terms("X2")}, key=len)
    shared = {}
    for chain in chains:
        cached = chain_delay(orbit, times, chain, cache=shared)
        fresh = chain_delay(orbit, times, chain, cache={})
        assert np.array_equal(cached, fresh)


def test_light_cone_solve_survives_a_two_year_mission_time():
    """A fixed 1e-12 s tolerance is below the float64 spacing of t at 6e7 s.

    Before the tolerance was floored at the representable resolution this
    raised 'light-cone solve did not converge' for any late-mission time.
    """
    orbit = LisaEqualArmOrbit()
    for epoch in (1e5, 6.3e7, 3.2e8):
        sample = sample_constellation(
            np.linspace(epoch, epoch + 1e4, 32), orbit)
        assert np.all(np.isfinite(sample.ltt))
        assert np.allclose(sample.ltt, sample.ltt[0, 0], rtol=1e-3)


def test_sparse_matches_dense():
    """The whole point: same channel, two evaluators, one answer."""
    orbit = LisaEqualArmOrbit()
    delta_t, n = 5.0, 120000
    times = np.arange(n) * delta_t + 1e5
    source = NewtonianChirp(3.0e4, times[-1] + 4e5)

    sample = sample_constellation(times, orbit)
    geometry = link_geometry(sample, 0.9, -0.25, velocity_order=1)
    dense = np.asarray(combine_links(
        link_response(source, sample, geometry), sample, channels="XYZ",
        generation=2, interpolation_order=31, delay_order=5)["X"])

    grid = adaptive_time_grid(source, 2, times[0], times[-1],
                              delta_phi=0.5, dt_max=1e9, growth=1.15,
                              max_step_scale=64)
    sparse = reconstruct(source, 2, grid,
                         sparse_channel(source, 2, grid, _terms(), orbit,
                                        0.9, -0.25), times)

    interior = slice(300, -300)
    scale = np.max(np.abs(dense[interior]))
    assert np.max(np.abs(sparse[interior] - dense[interior])) / scale < 1e-7
    assert len(grid) < len(times) / 100          # it is actually sparse


def test_sparse_grid_shrinks_and_error_grows_monotonically():
    """Guards the accuracy/speed knob: a looser grid must cost accuracy.

    delta_phi is the only knob. The geometric step growth it replaced put the
    dense region at the segment start instead of at the merger.
    """
    orbit = LisaEqualArmOrbit()
    delta_t, n = 5.0, 120000
    times = np.arange(n) * delta_t + 1e5
    source = NewtonianChirp(3.0e4, times[-1] + 4e5)
    sample = sample_constellation(times, orbit)
    dense = np.asarray(combine_links(
        link_response(source, sample,
                      link_geometry(sample, 0.9, -0.25, velocity_order=1)),
        sample, channels="XYZ", generation=2,
        interpolation_order=31, delay_order=5)["X"])
    interior = slice(300, -300)
    scale = np.max(np.abs(dense[interior]))

    sizes, errors = [], []
    for cap in (64, 512, 4096):
        grid = adaptive_time_grid(source, 2, times[0], times[-1],
                                  delta_phi=0.5, dt_max=1e9, growth=1.15,
                                  max_step_scale=cap)
        sparse = reconstruct(source, 2, grid,
                             sparse_channel(source, 2, grid, _terms(), orbit,
                                            0.9, -0.25), times)
        sizes.append(len(grid))
        errors.append(np.max(np.abs(sparse[interior] - dense[interior])) / scale)
    assert sizes[0] > sizes[1] > sizes[2]
    assert errors[0] < errors[1] < errors[2]


def test_cached_geometries_agree_with_the_direct_path():
    """The three cached forms are optimisations, not different calculations."""
    from pycbc.tdi.onthefly import (SparseGeometry, StackedGeometry,
                                    TermGeometry, sparse_channel_cached,
                                    sparse_channel_stacked,
                                    sparse_channel_terms)
    orbit = LisaEqualArmOrbit()
    times = np.arange(120000) * 5.0 + 1e5
    source = NewtonianChirp(3.0e4, times[-1] + 4e5)
    terms = _terms()
    grid = adaptive_time_grid(source, 2, times[0], times[-1],
                              delta_phi=0.5, dt_max=1e9, growth=1.15,
                              max_step_scale=512)
    direct = sparse_channel(source, 2, grid, terms, orbit, 0.9, -0.25)
    scale = np.max(np.abs(direct))
    for builder, evaluate, tolerance in (
            (SparseGeometry, sparse_channel_cached, 0.0),
            (StackedGeometry, sparse_channel_stacked, 1e-12),
            (TermGeometry, sparse_channel_terms, 1e-12)):
        got = evaluate(source, 2, builder(orbit, grid, terms), 0.9, -0.25)
        assert np.max(np.abs(got - direct)) / scale <= tolerance


def test_term_geometry_keeps_only_the_pairs_the_channel_uses():
    """The optimisation that mattered: gather (chain, link), do not index."""
    from pycbc.tdi.onthefly import StackedGeometry, TermGeometry
    orbit = LisaEqualArmOrbit()
    grid = np.linspace(1e5, 3e6, 64)
    terms = _terms()
    stacked = StackedGeometry(orbit, grid, terms)
    gathered = TermGeometry(orbit, grid, terms)
    assert gathered.shape == (len(terms), len(grid))
    # the stacked form carries every link of every chain
    assert stacked.n_chain * stacked.n_link > 3 * len(terms)


def test_multichannel_geometry_matches_independent_channel_evaluations():
    """Sharing orbit work must not change any channel's bracket."""
    from pycbc.tdi.onthefly import (MultiChannelTermGeometry, TermGeometry,
                                    sparse_channel_terms,
                                    sparse_channels_terms)
    orbit = LisaEqualArmOrbit()
    grid = np.linspace(1e5, 4e5, 80)
    source = NewtonianChirp(3.0e4, 1e6)
    channel_terms = {name: _terms(f"{name}2") for name in "XYZ"}

    shared = sparse_channels_terms(
        source, 2,
        MultiChannelTermGeometry(orbit, grid, channel_terms),
        0.9, -0.25)
    for name, terms in channel_terms.items():
        separate = sparse_channel_terms(
            source, 2, TermGeometry(orbit, grid, terms), 0.9, -0.25)
        scale = np.max(np.abs(separate))
        assert np.max(np.abs(shared[name] - separate)) / scale < 1e-12


def test_frequency_factors_equal_exact_linear_carrier_brackets():
    """FD delay phases and the time-domain evaluator use one convention."""
    from pycbc.tdi.onthefly import (MultiChannelTermGeometry,
                                    frequency_response_factors,
                                    sparse_channels_terms)
    orbit = LisaEqualArmOrbit()
    grid = np.linspace(1e5, 4e5, 80)
    source = _LinearCarrier()
    channel_terms = {name: _terms(f"{name}2") for name in "XYZ"}
    geometry = MultiChannelTermGeometry(orbit, grid, channel_terms)
    time_domain = sparse_channels_terms(
        source, 2, geometry, 0.9, -0.25)
    factors = frequency_response_factors(
        geometry, np.full(len(grid), source.frequency), 0.9, -0.25)
    for name, (plus, cross) in factors.items():
        frequency_domain = plus * source.plus + cross * source.cross
        scale = np.max(np.abs(time_domain[name]))
        assert np.max(np.abs(frequency_domain - time_domain[name])) / scale \
            < 2e-9


def test_interpolated_frequency_response_controls_error_and_standard_sigma():
    """The accelerated response remains an ordinary PyCBC FD waveform."""
    from pycbc.filter import sigma
    from pycbc.psd import flat_unity
    from pycbc.tdi.onthefly import project_frequency_tdi_series

    orbit = LisaEqualArmOrbit()
    channel_terms = {name: _terms(f"{name}2") for name in "XYZ"}
    arguments = dict(
        source=_LinearFDSource(), orbit=orbit, channel_terms=channel_terms,
        delta_f=2e-4, f_lower=0.021, f_final=0.1,
        lamb=0.9, beta=-0.25)
    exact = project_frequency_tdi_series(**arguments)
    accelerated, diagnostics = project_frequency_tdi_series(
        **arguments, response_tolerance=1e-3,
        response_initial_points=65, return_diagnostics=True)

    for name in "XYZ":
        reference = np.asarray(exact[name])
        difference = np.asarray(accelerated[name]) - reference
        scale = np.sqrt(np.mean(np.abs(reference) ** 2))
        assert np.sqrt(np.mean(np.abs(difference) ** 2)) / scale < 3e-4

        psd = flat_unity(len(reference), arguments['delta_f'],
                         arguments['f_lower'])
        exact_snr = sigma(
            exact[name], psd=psd,
            low_frequency_cutoff=arguments['f_lower'])
        accelerated_snr = sigma(
            accelerated[name], psd=psd,
            low_frequency_cutoff=arguments['f_lower'])
        assert abs(accelerated_snr / exact_snr - 1) < 1e-4

    assert diagnostics[2]['control_points'] < \
        diagnostics[2]['frequency_points']


def test_sparse_response_samples_only_the_requested_arbitrary_times():
    """The representation has no fixed duration or uniform-cadence policy."""
    from pycbc.tdi.onthefly import (TermGeometry,
                                    build_sparse_tdi_response,
                                    sparse_channel_terms)
    orbit = LisaEqualArmOrbit()
    source = NewtonianChirp(3.0e4, 1e6)
    grid = np.linspace(12345.0, 234567.0, 100)
    response = build_sparse_tdi_response(
        source, orbit, {"X": _terms("X2"), "Y": _terms("Y2")},
        0.9, -0.25, {2: grid})
    query = np.array([13000.0, 17777.5, 81000.0, 190123.25, 230000.0])
    got = response.sample(query, channels="X")

    bracket = sparse_channel_terms(
        source, 2, TermGeometry(orbit, grid, _terms("X2")), 0.9, -0.25)
    expected = reconstruct(source, 2, grid, bracket, query,
                           support=(grid[0], grid[-1]))
    assert tuple(got) == ("X",)
    assert np.allclose(got["X"], expected, rtol=1e-12, atol=0.0)
    assert len(got["X"]) == len(query)


def test_sparse_response_reuses_one_carrier_across_channels():
    """Sampling A/E/T must not repeat the source phase solve per channel."""
    from pycbc.tdi.onthefly import build_sparse_tdi_response

    orbit = LisaEqualArmOrbit()
    source = NewtonianChirp(3.0e4, 1e6)
    grid = np.linspace(12345.0, 234567.0, 100)
    response = build_sparse_tdi_response(
        source, orbit, {"X": _terms("X2"), "Y": _terms("Y2")},
        0.9, -0.25, {2: grid})
    original = source.carrier_phase
    calls = []

    def counted(harmonic, times):
        calls.append(np.shape(times))
        return original(harmonic, times)

    source.carrier_phase = counted
    query = np.linspace(13000.0, 230000.0, 1000)
    response.sample(query, channels=("X", "Y"))
    assert calls == [query.shape]


def test_sparse_response_reconstructs_uniform_timeseries_in_chunks():
    """Chunked reconstruction fills caller storage and preserves metadata."""
    from pycbc.tdi.onthefly import build_sparse_tdi_response

    orbit = LisaEqualArmOrbit()
    source = NewtonianChirp(3.0e4, 1e6)
    grid = np.linspace(12345.0, 234567.0, 100)
    response = build_sparse_tdi_response(
        source, orbit, {"X": _terms("X2"), "Y": _terms("Y2")},
        0.9, -0.25, {2: grid})
    start, end, delta_t = 13000.0, 14003.0, 3.25
    size = int(np.ceil(np.nextafter(end - start, 0.0) / delta_t))
    storage = {}

    def allocate(name, count, dtype):
        storage[name] = np.empty(count, dtype=dtype)
        return storage[name]

    series = response.to_timeseries(
        delta_t, t_start=start, t_end=end, channels=("X", "Y"),
        chunk_size=37, out=allocate)
    times = start + np.arange(size) * delta_t
    expected = response.sample(times, channels=("X", "Y"))

    assert tuple(series) == ("X", "Y")
    for name in series:
        assert np.shares_memory(np.asarray(series[name]), storage[name])
        assert np.array_equal(np.asarray(series[name]), expected[name])
        assert float(series[name].delta_t) == delta_t
        assert float(series[name].start_time) == start


def test_adaptive_response_refines_the_envelope_not_the_carrier():
    """A chirp needs a handful of bracket nodes, yet retains dense accuracy."""
    from pycbc.tdi.onthefly import adaptive_sparse_tdi_response
    orbit = LisaEqualArmOrbit()
    delta_t, count = 5.0, 40000
    times = np.arange(count) * delta_t + 1e5
    source = NewtonianChirp(3.0e4, times[-1] + 4e5)
    channel_terms = {name: _terms(f"{name}2") for name in "XYZ"}
    response = adaptive_sparse_tdi_response(
        source, orbit, channel_terms, 0.9, -0.25,
        t_start=times[0], t_end=times[-1], initial_step=86400.0,
        relative_tolerance=1e-5)
    sparse = response.sample(times, channels="X")["X"]

    sample = sample_constellation(times, orbit)
    dense = np.asarray(combine_links(
        link_response(
            source, sample,
            link_geometry(sample, 0.9, -0.25, velocity_order=1)),
        sample, channels="XYZ", generation=2,
        interpolation_order=31, delay_order=5)["X"])
    interior = slice(300, -300)
    scale = np.max(np.abs(dense[interior]))
    assert np.max(np.abs(sparse[interior] - dense[interior])) / scale < 1e-5
    assert len(response.responses[0]['grid']) < len(times) / 1000


def test_real_reconstruction_is_the_real_part_of_analytic_signal():
    source = NewtonianChirp(3.0e4, 1e6)
    times = np.linspace(1e5, 2e5, 1000)
    grid = np.linspace(times[0], times[-1], 40)
    bracket = np.exp(1j * np.linspace(0.0, 0.2, len(grid)))
    analytic = reconstruct_complex(source, 2, grid, bracket, times)
    real = reconstruct(source, 2, grid, bracket, times)
    assert np.array_equal(real, np.real(analytic))


def test_pyefpehm_source_reconstructs_its_native_time_domain():
    """The positive-carrier factorisation preserves pyEFPEHM polarizations."""
    pytest.importorskip("pyEFPEHM")
    from pycbc.tdi.sources import PyEFPEHMSource

    source = PyEFPEHMSource(dict(
        mass1=39.458, mass2=31.719, distance=420.0,
        eccentricity=0.0, spin1z=0.0, spin2z=0.0,
        inclination=1.0, phase=0.0, f22_start=0.02098508,
        f22_ref=0.02098508, f22_end=0.1, Amplitude_tol=1e-4,
    ))
    assert not source.model.params['Compute_hlm_Modes']
    assert source.model.params['Interp_points_per_prec_cycle'] == 0
    times = np.linspace(source.t_start + 1000.0,
                        source.t_end - 1000.0, 1000)
    got_plus, got_cross = source.polarizations(times)
    want_plus, want_cross = source.model.generate_tdomain_waveform(times=times)
    scale = max(np.max(np.abs(want_plus)), np.max(np.abs(want_cross)))
    assert np.max(np.abs(got_plus - want_plus)) / scale < 2e-14
    assert np.max(np.abs(got_cross - want_cross)) / scale < 2e-14

    # The default-angle shortcut uses pyEFPEHM's already projected complex
    # amplitudes. Force the general raw-inertial-mode projection at the same
    # angles and verify that this is an exact call-path optimization.
    harmonic = source.harmonics[0]
    fast = source.harmonic_components(harmonic, times)
    general_source = PyEFPEHMSource(
        source.model.params,
        theta=np.arccos(source.model.cos_theta_JN),
        phi=source.model.phi_JN)
    assert general_source.model.params['Compute_hlm_Modes']
    general = general_source.harmonic_components(harmonic, times)
    for fast_values, general_values in zip(fast, general, strict=True):
        scale = np.max(np.abs(general_values))
        assert np.max(np.abs(fast_values - general_values)) \
            / max(scale, np.finfo(float).tiny) < 2e-14

    frequencies = np.linspace(0.021, 0.099, 1000)
    got_plus, got_cross = source.frequency_polarizations(frequencies)
    want_plus, want_cross = source.model.generate_waveform(frequencies)
    scale = max(np.max(np.abs(want_plus)), np.max(np.abs(want_cross)))
    assert np.max(np.abs(got_plus - want_plus)) / scale < 2e-14
    assert np.max(np.abs(got_cross - want_cross)) / scale < 2e-14


def test_pyefpehm_selected_harmonics_match_multimode_native_oracle():
    """The selected-mode shortcut also handles eccentric negative-m labels."""
    pytest.importorskip("pyEFPEHM")
    from pycbc.tdi.sources import PyEFPEHMSource

    source = PyEFPEHMSource(dict(
        mass1=1.824, mass2=0.739, distance=100.0, eccentricity=0.7,
        spin1x=-0.44, spin1y=-0.26, spin1z=0.48,
        spin2x=-0.31, spin2y=0.01, spin2z=-0.84,
        inclination=1.57, phase=0.4, f22_start=10.0, f22_end=20.0,
        Amplitude_tol=1e-4,
    ))
    times = np.linspace(source.t_start + 1e-6,
                        source.t_end - 1e-6, 128)
    native = source.model.generate_tdomain_modes(
        times=times, return_waveform_pieces=True)
    assert len(native['modes']) > 1
    assert any(harmonic[1] < 0 for harmonic in native['modes'])

    for harmonic, mode in native['modes'].items():
        amp_plus, amp_cross, phase, omega = \
            source.harmonic_components(harmonic, times)
        expected = np.zeros((len(times), 2), dtype=complex)
        contribution = (
            2 * source.model.h0_pref
            * np.asarray(mode['Apc_prec'])
            * np.asarray(mode['Nlm_p'])[:, None])
        indices = np.asarray(mode['time_idxs'])
        expected[indices] = np.conj(contribution)
        scale = max(np.max(np.abs(expected)), np.finfo(float).tiny)
        assert np.max(np.abs(amp_plus - expected[:, 0])) / scale < 2e-14
        assert np.max(np.abs(amp_cross - expected[:, 1])) / scale < 2e-14
        np.testing.assert_allclose(
            phase[indices], mode['phase'], rtol=5e-16, atol=1e-12)
        np.testing.assert_allclose(
            omega[indices], mode['omega'], rtol=5e-16, atol=1e-12)


@pytest.mark.skipif(_NO_LAL is not None, reason=str(_NO_LAL))
def test_lal_imrphenomd_inverse_spa_reproduces_native_fd_waveform():
    """The harmonic adapter is LAL IMRPhenomD, not a surrogate waveform."""
    from pycbc.waveform import get_fd_waveform_sequence

    parameters = dict(
        mass1=35.0, mass2=32.0, spin1z=0.6, spin2z=0.2,
        distance=160.0, inclination=2.7, coa_phase=1.2, f_ref=0.0058,
    )
    source = LALIMRPhenomDSource(
        f_lower=0.0058, duration=3 * 86400.0, n_frequency=512,
        **parameters)
    frequency = source.sample_frequencies
    stationary = source.stationary_times
    relative_time = stationary - stationary[0]
    amp_plus, amp_cross = source.amplitude(2, relative_time)
    phase = source.carrier_phase(2, relative_time)

    # The source clock begins at zero, so compare to the correspondingly
    # time-shifted LAL frequency-domain waveform.
    exponent = phase - 2 * np.pi * frequency * relative_time - np.pi / 4
    scale = 0.5 * amp_plus * np.sqrt(source._dt_df)
    reconstructed = scale * np.exp(1j * exponent)
    native_plus, native_cross = get_fd_waveform_sequence(
        approximant="IMRPhenomD", sample_points=frequency, **parameters)
    time_shift = np.exp(1j * np.remainder(
        2 * np.pi * frequency * stationary[0], 2 * np.pi))
    ratio = reconstructed / (np.asarray(native_plus) * time_shift)

    assert np.max(np.abs(np.abs(ratio) - 1)) < 1e-10
    assert np.max(np.abs(np.angle(ratio))) < 1e-6
    cross_ratio = np.asarray(native_cross) / np.asarray(native_plus)
    assert np.max(np.abs(amp_cross / amp_plus - cross_ratio)) < 1e-12
    # The endpoint is noise-limited, not implementation-limited: the local
    # stationary-time measurement carries an rms of 38 s here (max 127 s),
    # unchanged whether it is smoothed with a degree 6, 12 or 24 fit, so a
    # difference of two of them scatters by tens of seconds on a three-day
    # duration. Anything much tighter would be fitting one draw.
    assert abs(source.t_end - 3 * 86400.0) < 200.0


@pytest.mark.skipif(_NO_LAL is not None, reason=str(_NO_LAL))
def test_lal_imrphenomd_sparse_response_matches_dense_native_response():
    """LAL waveform and native TDI agree before and after sparse factoring."""
    orbit = LisaEqualArmOrbit()
    delta_t, n = 5.0, 20000
    times = np.arange(n) * delta_t + 1e5
    source = LALIMRPhenomDSource(
        mass1=35.0, mass2=32.0, spin1z=0.6, spin2z=0.2,
        distance=160.0, inclination=2.7, coa_phase=1.2, f_ref=0.0058,
        f_lower=0.0058, duration=150000.0, t_start=90000.0,
        n_frequency=256)

    sample = sample_constellation(times, orbit)
    geometry = link_geometry(sample, 0.9, -0.25, velocity_order=1)
    dense = np.asarray(combine_links(
        link_response(source, sample, geometry), sample, channels="XYZ",
        generation=2, interpolation_order=31, delay_order=5)["X"])
    grid = adaptive_time_grid(
        source, 2, times[0], times[-1], delta_phi=0.5, dt_max=1e9,
        growth=1.15, max_step_scale=64)
    sparse = reconstruct(
        source, 2, grid,
        sparse_channel(source, 2, grid, _terms(), orbit, 0.9, -0.25),
        times, support=source.support(2))

    interior = slice(300, -300)
    scale = np.max(np.abs(dense[interior]))
    error = np.max(np.abs(sparse[interior] - dense[interior])) / scale
    assert error < 1e-6
    assert len(grid) < len(times) / 100


@pytest.mark.skipif(_NO_LAL is not None, reason=str(_NO_LAL))
def test_lal_source_generalises_beyond_imrphenomd():
    """The inverse-SPA construction is not tied to one approximant.

    Every dominant-mode frequency-domain model must round-trip back to its own
    native LAL waveform; the adapter must not quietly become a surrogate.
    ROM models are skipped when their data files are absent, an environment
    property and not a code one.
    """
    from pycbc.waveform import get_fd_waveform_sequence

    parameters = dict(mass1=35.0, mass2=32.0, spin1z=0.6, spin2z=0.2,
                      distance=160.0, inclination=2.7, coa_phase=1.2,
                      f_ref=0.0058)
    checked = 0
    for approximant in ("IMRPhenomD", "IMRPhenomXAS", "TaylorF2"):
        try:
            source = LALFDSource(f_lower=0.0058, duration=3 * 86400.0,
                                 n_frequency=512, approximant=approximant,
                                 **parameters)
        except RuntimeError:                      # missing ROM data
            continue
        frequency = source.sample_frequencies
        relative_time = source.stationary_times - source.stationary_times[0]
        amp_plus, _ = source.amplitude(2, relative_time)
        exponent = (source.carrier_phase(2, relative_time)
                    - 2 * np.pi * frequency * relative_time - np.pi / 4)
        reconstructed = (0.5 * amp_plus * np.sqrt(source._dt_df)
                         * np.exp(1j * exponent))
        native, _ = get_fd_waveform_sequence(
            approximant=approximant, sample_points=frequency, **parameters)
        shift = np.exp(1j * np.remainder(
            2 * np.pi * frequency * source.stationary_times[0], 2 * np.pi))
        ratio = reconstructed / (np.asarray(native) * shift)
        assert np.max(np.abs(np.abs(ratio) - 1)) < 1e-9, approximant
        assert np.max(np.abs(np.angle(ratio))) < 1e-5, approximant
        checked += 1
    assert checked >= 2


@pytest.mark.skipif(_NO_LAL is not None, reason=str(_NO_LAL))
def test_lal_source_rejects_what_it_cannot_represent():
    """A model with no frequency sequence, and a fixed-approximant subclass."""
    with pytest.raises(ValueError, match="frequency-sequence"):
        LALFDSource(mass1=35.0, mass2=32.0, f_lower=0.0058,
                    duration=3 * 86400.0, approximant="NotAnApproximant")
    with pytest.raises(ValueError, match="sample_points"):
        LALFDSource(mass1=35.0, mass2=32.0, f_lower=0.0058,
                    duration=3 * 86400.0, sample_points=np.zeros(4))
    with pytest.raises(ValueError, match="fixes approximant"):
        LALIMRPhenomDSource(mass1=35.0, mass2=32.0, f_lower=0.0058,
                            duration=3 * 86400.0, approximant="TaylorF2")
    source = LALIMRPhenomDSource(mass1=35.0, mass2=32.0, f_lower=0.0058,
                                 duration=3 * 86400.0, n_frequency=256)
    assert source.approximant == "IMRPhenomD"
    with pytest.raises(ValueError, match="carrier"):
        source.amplitude(3, np.zeros(2))


@pytest.mark.skipif(_NO_LAL is not None, reason=str(_NO_LAL))
def test_lal_td_source_reaches_merger_and_reproduces_lal():
    """The point of the time-domain route is the part inverse SPA cannot see.

    LALFDSource stops where the stationary-time map does; for a LISA-mass
    binary that is well before the ringdown. This one must run past it and
    must still be the LAL waveform, not an approximation of it.
    """
    from pycbc.waveform import get_td_waveform

    parameters = dict(mass1=60.0, mass2=25.0, spin1z=0.4, spin2z=0.1,
                      distance=500.0, coa_phase=0.0, f_lower=20.0)
    delta_t = 1.0 / 4096
    source = LALTDSource(delta_t=delta_t, inclination=1.0, approximant="IMRPhenomD",
                         **parameters)
    total_mass = (parameters["mass1"] + parameters["mass2"]) * 4.925490947e-6
    probe = np.linspace(source.t_start, source.t_end, 20000)
    frequency = source.angular_frequency(2, probe) / (2 * np.pi)
    # the phase is monotone by construction, but the frequency is not: it
    # falls away again in the decaying ringdown tail
    assert np.all(frequency > 0)
    assert frequency.max() * total_mass > 0.08     # past the ringdown
    assert frequency[0] * total_mass < 0.01        # and starting in inspiral

    plus, cross = get_td_waveform(approximant="IMRPhenomD", delta_t=delta_t,
                                  inclination=1.0, **parameters)
    times = np.asarray(plus.sample_times)
    inside = (times >= source.t_start) & (times <= source.t_end)
    got_plus, got_cross = source.polarizations(times[inside])
    want_plus = np.asarray(plus)[inside]
    want_cross = np.asarray(cross)[inside]
    scale = max(np.max(np.abs(want_plus)), np.max(np.abs(want_cross)))
    assert np.max(np.abs(got_plus - want_plus)) / scale < 1e-7
    assert np.max(np.abs(got_cross - want_cross)) / scale < 1e-7


@pytest.mark.skipif(_NO_LAL is not None, reason=str(_NO_LAL))
def test_lal_td_source_refuses_more_than_one_carrier():
    """Generated on axis, a higher-mode model looks single-carrier.

    The spin-weighted harmonics leave only m = +/-2 at zero inclination, so
    smoothness there proves nothing; what breaks is the angular dependence.
    The check has to go off-axis, and it has to maximise over time and phase,
    because two generations of the same model need not be sample-aligned --
    IMRPhenomXAS puts its two inclinations 0.98 ms apart.
    """
    parameters = dict(mass1=60.0, mass2=25.0, spin1z=0.4, spin2z=0.1,
                      distance=500.0, coa_phase=0.0, f_lower=20.0)
    for approximant in ("IMRPhenomD", "IMRPhenomXAS", "TaylorF2"):
        LALTDSource(delta_t=1.0 / 4096, inclination=1.0,
                    approximant=approximant, **parameters)
    # named by the registry, which knows these families by citation
    for approximant in ("IMRPhenomXHM", "IMRPhenomXPHM", "IMRPhenomHM"):
        with pytest.raises(ValueError, match="besides the .2, 2. carrier"):
            LALTDSource(delta_t=1.0 / 4096, inclination=1.0,
                        approximant=approximant, **parameters)
    # and caught by measurement, which the registry does not cover
    for approximant in ("SEOBNRv4HM", "SEOBNRv4PHM", "IMRPhenomTPHM"):
        with pytest.raises(ValueError, match="single .2, ..-2. carrier"):
            LALTDSource(delta_t=1.0 / 4096, inclination=1.0,
                        approximant=approximant, **parameters)
    # and accepted as the dominant-mode approximation it is
    LALTDSource(delta_t=1.0 / 4096, inclination=1.0,
                approximant="IMRPhenomXHM", check_inclination=None,
                **parameters)


@pytest.mark.skipif(_NO_LAL is not None, reason=str(_NO_LAL))
def test_lal_td_source_is_bandlimited_not_aliased_by_a_coarse_step():
    """A coarse step costs the merger, and it costs it visibly.

    The constructor guards against a carrier advancing more than pi per
    sample, but LAL never lets that happen: it band-limits to the Nyquist of
    the requested delta_t instead of folding. The failure mode is a silently
    shorter waveform, not a corrupt one, which is why the merger test above
    checks how far in Mf the source reaches.
    """
    parameters = dict(mass1=60.0, mass2=25.0, spin1z=0.4, spin2z=0.1,
                      distance=500.0, coa_phase=0.0, f_lower=20.0)
    total_mass = (parameters["mass1"] + parameters["mass2"]) * 4.925490947e-6
    reach = {}
    for delta_t in (1.0 / 4096, 1.0 / 256):
        source = LALTDSource(delta_t=delta_t, inclination=1.0,
                             approximant="IMRPhenomD", **parameters)
        probe = np.linspace(source.t_start, source.t_end, 4000)
        frequency = source.angular_frequency(2, probe) / (2 * np.pi)
        assert np.all(frequency > 0)
        # filled right up to Nyquist: 130.0 Hz against 128.0 at
        # delta_t = 1/256, the excess coming from the edge of the spline
        assert frequency.max() < 1.05 * 0.5 / delta_t
        reach[delta_t] = frequency.max() * total_mass
    assert reach[1.0 / 4096] > 0.08                     # keeps the ringdown
    assert reach[1.0 / 256] < reach[1.0 / 4096] / 2     # loses it


@pytest.mark.skipif(_NO_LAL is not None, reason=str(_NO_LAL))
def test_lal_modes_source_is_lals_own_mode_sum():
    """One harmonic per (l, m), reconstructing LAL's own decomposition.

    The comparison is against pycbc's `sum_modes` over the same modes, not
    against get_td_waveform, so what is tested is the representation and not
    the mode content. The azimuth is not coa_phase: `sum_modes` takes an
    azimuth, and pi/2 - coa_phase is what reproduces get_td_waveform (0.0e+00
    against 6e-02 to 2.0 for coa_phase itself, depending on inclination).
    """
    from pycbc.waveform import get_td_waveform_modes
    from pycbc.waveform.waveform_modes import sum_modes

    parameters = dict(mass1=60.0, mass2=25.0, spin1z=0.4, spin2z=0.1,
                      distance=500.0, f_lower=20.0)
    delta_t, inclination, coa_phase = 1.0 / 4096, 1.0, 0.7
    source = LALModesSource(delta_t=delta_t, inclination=inclination,
                            coa_phase=coa_phase, approximant="SEOBNRv4PHM",
                            **parameters)
    assert source.harmonics
    assert all(mode[1] != 0 for mode in source.harmonics)
    assert any("m = 0" in reason for reason in source.skipped.values())

    modes = get_td_waveform_modes(
        approximant="SEOBNRv4PHM", delta_t=delta_t, inclination=inclination,
        coa_phase=coa_phase, **parameters)
    kept = {mode: np.asarray(modes[mode][0]) + 1j * np.asarray(modes[mode][1])
            for mode in source.harmonics}
    summed = sum_modes(kept, inclination, 0.5 * np.pi - coa_phase)
    times = np.asarray(modes[source.harmonics[0]][0].sample_times)
    inside = ((times >= source.t_start + 0.02)
              & (times <= source.t_end - 0.02))
    got_plus, got_cross = source.polarizations(times[inside])
    want_plus = np.real(summed)[inside]
    want_cross = -np.imag(summed)[inside]
    overlap = (np.sum(got_plus * want_plus) + np.sum(got_cross * want_cross)) \
        / np.sqrt((np.sum(got_plus ** 2) + np.sum(got_cross ** 2))
                  * (np.sum(want_plus ** 2) + np.sum(want_cross ** 2)))
    assert abs(1 - overlap) < 1e-10

    for harmonic in source.harmonics:
        low, high = source.support(harmonic)
        assert high > low
        probe = np.linspace(low, high, 500)
        assert np.all(source.angular_frequency(harmonic, probe) > 0)
        plus, cross = source.amplitude(harmonic, np.array([low - 1.0]))
        assert plus[0] == 0 and cross[0] == 0


@pytest.mark.skipif(_NO_LAL is not None, reason=str(_NO_LAL))
def test_lal_modes_source_carries_a_precessing_waveform():
    """The point of the class: a model LALTDSource has to refuse.

    Against the full get_td_waveform the residual is the m = 0 content it
    cannot carry: non-oscillatory, no carrier to factor out, which is also why
    GW memory goes down the dense path. That is 7e-03 for a precessing 60+25,
    and the assertion pins it to the dropped modes, not to the
    representation.
    """
    from pycbc.waveform import get_td_waveform, get_td_waveform_modes
    from pycbc.waveform.waveform_modes import sum_modes

    parameters = dict(mass1=60.0, mass2=25.0, spin1x=0.6, spin1y=0.2,
                      spin1z=0.3, spin2x=-0.3, spin2y=0.4, spin2z=0.1,
                      distance=500.0, f_lower=20.0)
    delta_t, inclination, coa_phase = 1.0 / 4096, 1.0, 0.7
    with pytest.raises(ValueError):
        LALTDSource(delta_t=delta_t, inclination=inclination,
                    coa_phase=coa_phase, approximant="SEOBNRv4PHM",
                    **parameters)
    source = LALModesSource(delta_t=delta_t, inclination=inclination,
                            coa_phase=coa_phase, approximant="SEOBNRv4PHM",
                            **parameters)
    assert len(source.harmonics) > 20

    modes = get_td_waveform_modes(
        approximant="SEOBNRv4PHM", delta_t=delta_t, inclination=inclination,
        coa_phase=coa_phase, **parameters)
    every = {mode: np.asarray(value[0]) + 1j * np.asarray(value[1])
             for mode, value in modes.items()}
    kept = {mode: every[mode] for mode in source.harmonics}
    azimuth = 0.5 * np.pi - coa_phase
    times = np.asarray(modes[source.harmonics[0]][0].sample_times)
    inside = ((times >= source.t_start + 0.02)
              & (times <= source.t_end - 0.02))
    got_plus, got_cross = source.polarizations(times[inside])

    def mismatch(a, b, c, d):
        return 1 - (np.sum(a * c) + np.sum(b * d)) / np.sqrt(
            (np.sum(a * a) + np.sum(b * b)) * (np.sum(c * c) + np.sum(d * d)))

    partial = sum_modes(kept, inclination, azimuth)
    whole = sum_modes(every, inclination, azimuth)
    plus, cross = get_td_waveform(
        approximant="SEOBNRv4PHM", delta_t=delta_t, inclination=inclination,
        coa_phase=coa_phase, **parameters)
    against_kept = mismatch(got_plus, got_cross, np.real(partial)[inside],
                            -np.imag(partial)[inside])
    against_all = mismatch(got_plus, got_cross, np.asarray(plus)[inside],
                           np.asarray(cross)[inside])
    dropped = mismatch(np.real(partial)[inside], -np.imag(partial)[inside],
                       np.real(whole)[inside], -np.imag(whole)[inside])
    assert abs(against_kept) < 1e-9                  # the representation
    assert abs(against_all - dropped) < 0.2 * abs(dropped)   # all of the rest
    assert 1e-4 < abs(dropped) < 5e-2                # and it is the m = 0 part
