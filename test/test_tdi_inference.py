"""The sparse TDI response as a pycbc_inference detector-response waveform."""
import numpy as np
import pytest

from pycbc.tdi.inference import (channel_terms, clear_cache, register,
                                 sparse_tdi_fd_det_sequence)


def test_orthogonal_channels_are_term_lists_not_a_later_combination():
    """A response asked for "A" has to be able to answer.

    A = (Z - X)/sqrt2 is linear in the channels and hence in the terms, so
    scaling coefficients gives it natively. Building X, Y and Z and combining
    afterwards leaves a multiband response unable to produce A at all, which
    is what `Relative` asks it for.
    """
    michelson = channel_terms(('X', 'Y', 'Z'))
    orthogonal = channel_terms(('A', 'E', 'T'))
    assert len(orthogonal['A']) == len(michelson['X']) + len(michelson['Z'])
    assert len(orthogonal['E']) == sum(
        len(michelson[label]) for label in 'XYZ')

    # Every term keeps its link and operator chain; only the weight moves.
    scale = 2.0 ** -0.5
    by_chain = {}
    for term in orthogonal['A']:
        by_chain.setdefault((term.link, term.operators), 0.0)
        by_chain[(term.link, term.operators)] += term.coefficient
    for label, weight in (('Z', scale), ('X', -scale)):
        for term in michelson[label]:
            assert (term.link, term.operators) in by_chain
    total = sum(abs(term.coefficient) for term in orthogonal['A'])
    expected = scale * sum(
        abs(term.coefficient)
        for label in 'XZ' for term in michelson[label])
    assert total == pytest.approx(expected, rel=1e-12)


def test_an_unknown_channel_is_refused():
    with pytest.raises(ValueError, match="unknown TDI channel"):
        channel_terms(('Q',))


def test_registration_lands_in_the_detector_response_registry():
    """`Relative` switches on membership of `fd_det_sequence`, nothing else."""
    from pycbc.waveform.waveform import fd_det_sequence
    name = register(approximant='TDISparseTest', force=True)
    assert fd_det_sequence[name] is sparse_tdi_fd_det_sequence
    assert 'tdi_source' in sparse_tdi_fd_det_sequence.required


def test_analysis_taper_is_explicit_and_validated():
    from pycbc.tdi.inference import _analysis_time_window

    common = {'t_obs_start': 100.0, 't_obs_end': 200.0}
    assert _analysis_time_window(common) is None
    window = _analysis_time_window(dict(common, tdi_taper_duration=10.0))
    assert np.allclose(window(np.array([100.0, 105.0, 110.0, 150.0,
                                        190.0, 195.0, 200.0])),
                       [0.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.0],
                       rtol=0, atol=1e-15)
    with pytest.raises(ValueError, match='cannot exceed the span'):
        _analysis_time_window(dict(common, tdi_taper_duration=51.0))

    # One edge at a time, for a source that is quiet at one end and not the
    # other. `tdi_taper_duration` is the shorthand for both.
    trailing = _analysis_time_window(dict(common, tdi_taper_end=10.0))
    assert np.allclose(trailing(np.array([100.0, 105.0, 150.0, 195.0])),
                       [1.0, 1.0, 1.0, 0.5], rtol=0, atol=1e-15)
    mixed = _analysis_time_window(dict(common, tdi_taper_duration=10.0,
                                       tdi_taper_end=0.0))
    assert np.allclose(mixed(np.array([105.0, 150.0, 195.0, 200.0])),
                       [0.5, 1.0, 1.0, 1.0], rtol=0, atol=1e-15)
    assert _analysis_time_window(dict(common, tdi_taper_start=0.0,
                                      tdi_taper_end=0.0)) is None
    with pytest.raises(ValueError, match='tdi_taper_end'):
        _analysis_time_window(dict(common, tdi_taper_end=-1.0))


def test_pyefpehm_builder_applies_detector_frame_extrinsics(monkeypatch):
    """A detector-response waveform must consume ``tc`` and polarization.

    ``Relative`` deliberately skips its ground-detector time and antenna
    factors for an approximant in ``fd_det_sequence``.  Leaving either value
    unused in the registered TDI source would therefore make it an inert PE
    parameter while every self-consistent injection test still passed.
    """
    from pycbc.tdi import inference, sources

    class Native:
        t_start = -100.0
        t_end = 900.0
        harmonics = ((2, 2, 2),)

    clear_cache()
    captured = []

    def fake_source(parameters):
        captured.append(dict(parameters))
        return Native()

    monkeypatch.setattr(sources, 'PyEFPEHMSource', fake_source)
    source = inference._pyefpehm(
        mass1=40.0, mass2=30.0, distance=3.0,
        f_lower=0.01, tc=25.0, polarization=0.3)

    assert isinstance(source, sources.PolarizationRotatedHarmonicSource)
    assert source.polarization == pytest.approx(0.3)
    assert isinstance(source.source, sources.TimeShiftedHarmonicSource)
    assert source.source.offset == pytest.approx(-125.0)
    assert source.t_start == pytest.approx(25.0)
    assert source.t_end == pytest.approx(1025.0)
    assert captured[0]['distance'] == pytest.approx(1.0)
    assert captured[0]['f22_start'] == pytest.approx(0.01)
    assert captured[0]['f22_ref'] == pytest.approx(0.01)

    # These are detector-frame wrappers, not inputs to the expensive native
    # evolution.  Varying either must change the exposed source but must not
    # construct pyEFPEHM again.
    moved = inference._pyefpehm(
        mass1=40.0, mass2=30.0, distance=3.0,
        f_lower=0.01, tc=30.0, polarization=0.7)
    assert moved.source.source.source is source.source.source.source
    assert moved.source.offset == pytest.approx(-130.0)
    assert moved.polarization == pytest.approx(0.7)
    assert len(captured) == 1

    # Distance is another cheap wrapper: the canonical native object stays.
    farther = inference._pyefpehm(
        mass1=40.0, mass2=30.0, distance=6.0,
        f_lower=0.01, tc=30.0, polarization=0.7)
    assert farther.source.source.source is source.source.source.source
    assert farther.source.source.scale == pytest.approx(1 / 6.0)
    assert len(captured) == 1

    # An intrinsic change replaces the epoch's native state exactly once.
    inference._pyefpehm(
        mass1=40.1, mass2=30.0, distance=3.0,
        f_lower=0.01, tc=30.0, polarization=0.7)
    assert len(captured) == 2


def test_prepared_geometry_is_model_scoped_and_keyed_on_numerical_knobs(
        monkeypatch):
    """The cache key decides whether a sampler can afford this.

    Preparing costs about half a second and projecting about seven
    milliseconds, so a key that included the sky would make every likelihood
    call pay the preparation. `PreparedMultibandTDI.project` takes the sky, so
    it does not have to.
    """
    from pycbc.tdi import inference

    class Narrow:
        harmonics = (2,)

        def angular_frequency(self, harmonic, times):
            return np.full_like(
                np.asarray(times, dtype=float), 2 * np.pi * 1.5e-3)

    clear_cache()
    assert not inference._PREPARED
    params = dict(tdi_source='lal', tdi_band_edges=[1e-3, 2e-3],
                  t_obs_start=0.0, t_obs_end=1.0e7,
                  eclipticlongitude=0.9, eclipticlatitude=-0.25,
                  tdi_preparation_id='first', tdi_samples_per_cycle=4.0)
    built = []
    monkeypatch.setattr(inference, '_build_source', lambda _: Narrow())
    monkeypatch.setattr(
        inference, 'prepare_sparse_tdi',
        lambda *args, **kwargs: built.append(kwargs) or object())
    terms, orbit = {'A': ()}, object()

    first = inference._preparation(params, terms, orbit)
    # Sky and sampled source parameters are projected after preparation.
    same_epoch = dict(params, eclipticlongitude=2.5, mass1=99.0)
    assert inference._preparation(same_epoch, terms, orbit) is first

    # A different model epoch or a changed preparation knob must not reuse it.
    new_epoch = dict(params, tdi_preparation_id='second')
    assert inference._preparation(new_epoch, terms, orbit) is not first
    finer = dict(params, tdi_samples_per_cycle=16.0)
    assert inference._preparation(finer, terms, orbit) is not first
    denser_geometry = dict(params, tdi_geometry_step=21600.0)
    assert inference._preparation(denser_geometry, terms, orbit) is not first
    assert len(built) == 4
    assert built[-1]['geometry_step'] == 21600.0


def test_two_harmonics_of_one_candidate_get_separate_geometries(monkeypatch):
    """`_signature` is harmonic-blind, so the preparation key must not be.

    A projection is cached under `(epoch, id(prepared), _signature(params))`
    and the signature deliberately drops `tdi_harmonic`, so that the ten
    harmonic requests of one candidate share a source instead of rebuilding
    it ten times. What keeps them apart is the preparation: it keys on
    `tdi_harmonic`, so each harmonic holds a different object and a different
    `id`. That coupling is load-bearing and invisible at both ends -- drop
    the harmonic from the preparation key and every harmonic would silently
    be served the first one's response.
    """
    from pycbc.tdi import inference

    class Narrow:
        harmonics = ((2, 2, 1), (2, 2, 2))

        def angular_frequency(self, harmonic, times):
            return np.full_like(
                np.asarray(times, dtype=float), 2 * np.pi * 1.5e-3)

    clear_cache()
    params = dict(tdi_source='lal', tdi_band_edges=[1e-3, 2e-3],
                  t_obs_start=0.0, t_obs_end=1.0e7,
                  eclipticlongitude=0.9, eclipticlatitude=-0.25,
                  tdi_preparation_id='epoch', tdi_samples_per_cycle=4.0)
    monkeypatch.setattr(inference, '_build_source', lambda _: Narrow())
    monkeypatch.setattr(inference, 'prepare_sparse_tdi',
                        lambda *args, **kwargs: object())
    terms, orbit = {'A': ()}, object()

    first = inference._preparation(dict(params, tdi_harmonic=(2, 2, 1)),
                                   terms, orbit)
    second = inference._preparation(dict(params, tdi_harmonic=(2, 2, 2)),
                                    terms, orbit)
    assert first is not second

    # And the signature really is blind to it, which is what makes the
    # assertion above the only thing separating them.
    assert (inference._signature(dict(params, tdi_harmonic='a'))
            == inference._signature(dict(params, tdi_harmonic='b')))


def test_runtime_source_cache_has_a_small_candidate_bound(monkeypatch):
    """Unique sampler proposals must not retain dozens of large sources."""
    from pycbc.tdi import inference

    clear_cache()
    monkeypatch.setitem(inference._SOURCES, 'cache-test',
                        lambda **params: object())
    for value in range(inference._SOURCE_CACHE_LIMIT + 3):
        inference._build_source(dict(tdi_source='cache-test', value=value))
    assert len(inference._SOURCE_CACHE) == inference._SOURCE_CACHE_LIMIT


def test_new_candidate_releases_previous_runtime_objects(monkeypatch):
    """A PE walk retains one proposal per model, not its recent history."""
    from pycbc.tdi import inference

    clear_cache()
    monkeypatch.setitem(inference._SOURCES, 'cache-test',
                        lambda **params: object())
    first = dict(tdi_source='cache-test', tdi_preparation_id='model-1',
                 value=1.0)
    inference._begin_candidate(first)
    inference._build_source(first)
    signature = inference._signature(first)
    inference._PROJECTED[('model-1', 1, signature)] = object()
    inference._SAMPLED[('model-1', 1, signature)] = object()

    second = dict(first, value=2.0)
    inference._begin_candidate(second)

    assert not inference._SOURCE_CACHE
    assert not inference._PROJECTED
    assert not inference._SAMPLED
    assert inference._RUNTIME_SIGNATURES['model-1'] \
        == inference._signature(second)


def test_band_edges_are_clipped_to_a_single_harmonic():
    """A band's sampling rate comes from its upper edge, so it must fit.

    `samples_per_cycle` over `f_upper` sets the block length, so a harmonic
    living at 5 to 10 mHz inside a band that reaches 90 mHz materialises an
    order of magnitude more samples than its own carrier needs -- and
    materialising the block is half the cost of a frequency evaluation.
    """
    from pycbc.tdi.inference import _harmonic_band_edges

    class Narrow:
        harmonics = (2,)

        def angular_frequency(self, harmonic, times):
            times = np.asarray(times, dtype=float)
            return 2 * np.pi * (5e-3 + 5e-3 * (times - times[0])
                                / max(times[-1] - times[0], 1.0))

    edges = [4e-3, 2.55e-2, 4.7e-2, 6.85e-2, 9e-2]
    clipped = _harmonic_band_edges(Narrow(), 2, edges, 0.0, 1.0e6)
    assert clipped[0] < 5e-3 and clipped[-1] > 1e-2
    assert clipped[-1] < 1.1e-2, clipped
    # None of the global interior edges lie inside 5-10 mHz, so the clipped
    # band is a single interval and its upper edge is the harmonic's own.
    assert len(clipped) == 2, clipped


def test_the_chirp_z_plan_is_reused():
    """`zoom_fft` rebuilds its plan every call; a likelihood asks repeatedly."""
    from pycbc.tdi.multiband import (_ZOOM_PLANS, _zoom_plan,
                                     clear_zoom_plans)
    clear_zoom_plans()
    first = _zoom_plan(1024, 1e-3, 2e-3, 64, 0.2)
    again = _zoom_plan(1024, 1e-3, 2e-3, 64, 0.2)
    assert again is first
    other = _zoom_plan(2048, 1e-3, 2e-3, 64, 0.2)
    assert other is not first
    assert len(_ZOOM_PLANS) == 2
    values = np.random.default_rng(0).normal(size=1024)
    from scipy.signal import zoom_fft
    assert np.allclose(first(values),
                       zoom_fft(values, [1e-3, 2e-3], m=64, fs=0.2,
                                endpoint=True))
    clear_zoom_plans()
    assert not _ZOOM_PLANS
