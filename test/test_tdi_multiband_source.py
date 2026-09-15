"""Tests for pre-response multiband harmonic windows."""

import numpy as np
import pytest

from pycbc.coordinates.space_orbit import LisaEqualArmOrbit
from pycbc.tdi.backends.pytdi_backend import (
    PyTDICombinationAdapter,
    combine_links,
    get_pytdi_combination,
)
from pycbc.tdi.multiband import (
    PreparedMultibandTDI,
    _zoom_frequency_samples,
    multiband_sparse_tdi_response,
    sparse_tdi_response,
    prepare_sparse_tdi,
    raised_cosine_time_window,
)
from pycbc.tdi.onthefly import adaptive_sparse_tdi_response
from pycbc.tdi.response import (
    link_geometry,
    link_response,
    sample_constellation,
)
from pycbc.tdi.sources import frequency_partition_sources
from pycbc.types import TimeSeries


def test_raised_cosine_time_window_has_shared_endpoint_convention():
    times = np.array([-1.0, 0.0, 1.0, 2.0, 5.0, 8.0, 9.0, 10.0, 11.0])
    actual = raised_cosine_time_window(times, 0.0, 10.0, 2.0)
    expected = np.array([0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0])
    assert np.allclose(actual, expected, rtol=0, atol=1e-15)

    rectangular = raised_cosine_time_window(times, 0.0, 10.0, 0.0)
    assert np.array_equal(rectangular,
                          [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="cannot exceed the span"):
        raised_cosine_time_window(times, 0.0, 10.0, 5.1)


def test_the_two_observation_edges_taper_independently():
    """A taper is free only where the signal is already quiet.

    A stellar-origin binary is still chirping at the end of the segment and
    needs the trailing fade. A massive binary that merges inside the segment
    must not have one: a sine-squared fade would drive the merger itself to
    zero, which is where nearly all of its signal-to-noise lives. One
    duration for both edges cannot express that.
    """
    times = np.array([-1.0, 0.0, 1.0, 2.0, 5.0, 8.0, 9.0, 10.0, 11.0])
    symmetric = raised_cosine_time_window(times, 0.0, 10.0, 2.0)
    assert np.array_equal(raised_cosine_time_window(times, 0.0, 10.0,
                                                    (2.0, 2.0)), symmetric)

    leading = raised_cosine_time_window(times, 0.0, 10.0, (2.0, 0.0))
    assert np.allclose(leading,
                       [0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                       rtol=0, atol=1e-15)
    # The point of it: full weight at the last sample, where the merger is.
    assert leading[-2] == 1.0

    trailing = raised_cosine_time_window(times, 0.0, 10.0, (0.0, 2.0))
    assert np.allclose(trailing,
                       [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0],
                       rtol=0, atol=1e-15)

    # Uneven edges are bounded by the whole span, not by half of it.
    raised_cosine_time_window(times, 0.0, 10.0, (8.0, 2.0))
    with pytest.raises(ValueError, match="cannot exceed the span"):
        raised_cosine_time_window(times, 0.0, 10.0, (8.0, 2.1))
    with pytest.raises(ValueError, match="non-negative"):
        raised_cosine_time_window(times, 0.0, 10.0, (-1.0, 2.0))
    with pytest.raises(ValueError, match="pair"):
        raised_cosine_time_window(times, 0.0, 10.0, (1.0, 2.0, 3.0))


class _LinearFrequencySource:
    harmonics = (2,)
    t_start = 0.0
    t_end = 10.0

    def amplitude(self, harmonic, time):
        time = np.asarray(time)
        return 2 + 0.1j * time, -0.3 + 0.2j * time

    def carrier_phase(self, harmonic, time):
        time = np.asarray(time)
        return 2 * np.pi * (time + 0.5 * time ** 2)

    def angular_frequency(self, harmonic, time):
        return 2 * np.pi * (1 + np.asarray(time))

    def polarizations(self, time):
        plus, cross = self.amplitude(2, time)
        carrier = np.exp(1j * self.carrier_phase(2, time))
        return np.real(plus * carrier), np.real(cross * carrier)

    def support_blocks(self, harmonic):
        return ((self.t_start, self.t_end),)


class _CompactChirpSource(_LinearFrequencySource):
    t_end = 4000.0

    def amplitude(self, harmonic, time):
        time = np.asarray(time)
        live = (time >= self.t_start) & (time <= self.t_end)
        return live * (2e-21 + 0.1e-21j), live * (-0.3e-21 + 0.2e-21j)

    def carrier_phase(self, harmonic, time):
        time = np.asarray(time)
        f0 = 1e-3
        slope = 9e-3 / self.t_end
        return 2 * np.pi * (f0 * time + 0.5 * slope * time ** 2)

    def angular_frequency(self, harmonic, time):
        return 2 * np.pi * (
            1e-3 + 9e-3 * np.asarray(time) / self.t_end)


def test_frequency_windows_form_partition_before_delayed_response():
    source = _LinearFrequencySource()
    bands = frequency_partition_sources(
        source, [1.0, 5.0, 8.0, 11.0], overlap=[2.0, 1.0])[2]
    # These stand in for distinct emitter and receiver retardations. Since the
    # partition is applied at each queried source time, any linear link/TDI
    # combination preserves the equality.
    times = np.linspace(0.0, 10.0, 1001)
    for query in (times, np.clip(times - 0.37, 0.0, 10.0)):
        native_plus, native_cross = source.amplitude(2, query)
        pieces = [band.amplitude(2, query) for band in bands]
        assert np.allclose(sum(item[0] for item in pieces), native_plus,
                           rtol=2e-15, atol=2e-15)
        assert np.allclose(sum(item[1] for item in pieces), native_cross,
                           rtol=2e-15, atol=2e-15)


def test_frequency_window_support_is_refined_to_band_edges():
    source = _LinearFrequencySource()
    middle = frequency_partition_sources(
        source, [1.0, 5.0, 8.0, 11.0], overlap=[2.0, 1.0])[2][1]
    # The middle band is live from f=4 to f=8.5. Since f(t)=1+t, its
    # corresponding source-time support is t=3 to t=7.5.
    blocks = middle.support_blocks(2, n_probe=64)
    assert len(blocks) == 1
    assert np.allclose(blocks[0], (3.0, 7.5), rtol=0, atol=2e-13)


def test_frequency_partition_rejects_overlapping_taper_centres():
    source = _LinearFrequencySource()
    try:
        frequency_partition_sources(
            source, [1.0, 2.0, 3.0], overlap=1.1)
    except ValueError as error:
        assert "overlap" in str(error)
    else:  # pragma: no cover
        raise AssertionError("invalid overlap was accepted")


def test_hard_frequency_boundary_has_one_owner():
    bands = frequency_partition_sources(
        _LinearFrequencySource(), [1.0, 5.0, 11.0], overlap=0.0)[2]
    weights = [band.frequency_weight(np.array([5.0]))[0] for band in bands]
    assert weights == [0.0, 1.0]


def test_multiband_tdi_sum_reconstructs_full_time_domain_response():
    source = _CompactChirpSource()
    orbit = LisaEqualArmOrbit(t0=0.0)
    terms = PyTDICombinationAdapter(
        "X2", get_pytdi_combination("X2"), delta_t=25.0).terms()
    channel_terms = {"X": terms}
    common = dict(
        t_start=800.0,
        t_end=3200.0,
        initial_step=200.0,
        relative_tolerance=2e-5,
        velocity_order=1,
    )
    full = adaptive_sparse_tdi_response(
        source, orbit, channel_terms, 1.1, -0.4, **common)
    multiband = multiband_sparse_tdi_response(
        source, orbit, channel_terms, 1.1, -0.4,
        band_edges=[1e-3, 5e-3, 1e-2],
        overlap=1e-3,
        samples_per_cycle=4,
        **common,
    )

    times = np.linspace(1000.0, 3000.0, 301)
    expected = full.sample(times)["X"]
    actual = multiband.sample(times)["X"]
    scale = np.max(np.abs(expected))
    assert np.max(np.abs(actual - expected)) / scale < 2e-4
    assert len(multiband.bands) == 2
    assert multiband.bands[0].delta_t > multiband.bands[1].delta_t

    blocks = list(multiband.iter_timeseries(channels="X", chunk_size=128))
    assert blocks
    assert all(tuple(block.series) == ("X",) for block in blocks)
    assert all(block.series["X"].delta_t == block.band.delta_t
               for block in blocks)

    transformed = multiband.linear_transform([[2.0]], ("twice_X",))
    assert transformed.channels == ("twice_X",)
    assert np.allclose(
        transformed.sample(times)["twice_X"], 2 * actual,
        rtol=2e-15, atol=2e-15 * scale)


def test_prepared_multiband_unions_training_source_coverage():
    """A prior-envelope source extends fixed geometry beyond the fiducial."""
    source = _CompactChirpSource()
    from pycbc.tdi.sources import TimeShiftedHarmonicSource
    candidate = TimeShiftedHarmonicSource(
        source, offset=-400.0, t_start=0.0, t_end=4000.0)
    orbit = LisaEqualArmOrbit(t0=0.0)
    terms = {"X": PyTDICombinationAdapter(
        "X2", get_pytdi_combination("X2"), delta_t=25.0).terms()}
    options = dict(
        source=source, orbit=orbit, channel_terms=terms,
        lamb=1.1, beta=-0.4, band_edges=[1e-3, 5e-3, 1e-2],
        overlap=1e-3, t_start=800.0, t_end=3200.0,
        geometry_step=5.0, minimum_grid_points=16,
    )

    fiducial_only = prepare_sparse_tdi(**options)
    with pytest.raises(ValueError, match="leaves prepared coverage"):
        fiducial_only.project(candidate, 1.1, -0.4)

    prepared = prepare_sparse_tdi(
        **options, coverage_sources=[candidate])
    projected = prepared.project(candidate, 1.1, -0.4)
    times = np.linspace(900.0, 3100.0, 401)
    sample = sample_constellation(times, orbit)
    expected = combine_links(
        link_response(
            candidate, sample,
            link_geometry(sample, 1.1, -0.4, velocity_order=1)),
        sample, channels="XYZ", generation=2,
        interpolation_order=31, delay_order=5)["X"]
    actual = projected.sample(times)["X"]
    interior = slice(40, -40)
    scale = np.max(np.abs(np.asarray(expected)[interior]))
    assert np.max(np.abs(
        actual[interior] - np.asarray(expected)[interior])) / scale < 2e-4


def test_prepared_multiband_rejects_unprepared_harmonic():
    source = _CompactChirpSource()
    prepared = prepare_sparse_tdi(
        source, LisaEqualArmOrbit(t0=0.0),
        {"X": PyTDICombinationAdapter(
            "X2", get_pytdi_combination("X2"), delta_t=25.0).terms()},
        1.1, -0.4, [1e-3, 5e-3, 1e-2], t_start=800.0, t_end=3200.0,
        geometry_step=25.0, harmonics=(2,))

    class ExtraHarmonicSource(_CompactChirpSource):
        harmonics = (2, 3)

    with pytest.raises(ValueError, match="unprepared harmonics"):
        prepared.project(ExtraHarmonicSource(), 1.1, -0.4)


def test_prepared_multiband_rejects_band_absent_from_preparation():
    source = _CompactChirpSource()
    orbit = LisaEqualArmOrbit(t0=0.0)
    terms = {"X": PyTDICombinationAdapter(
        "X2", get_pytdi_combination("X2"), delta_t=25.0).terms()}
    prepared = prepare_sparse_tdi(
        source, orbit, terms, 1.1, -0.4,
        [1e-3, 3e-3, 5e-3, 1e-2], t_start=800.0, t_end=3200.0,
        geometry_step=25.0)
    # Remove one prepared band to exercise the fail-closed invariant directly.
    missing = prepared.prepared_bands[0]
    reduced = PreparedMultibandTDI(
        prepared.response,
        tuple(item for item in prepared.prepared_bands if item is not missing),
        prepared.band_edges, prepared.overlaps, prepared.harmonics)
    with pytest.raises(ValueError, match="unprepared live band"):
        reduced.project(source, 1.1, -0.4)


def test_prepared_multiband_candidate_may_have_an_empty_covered_band():
    class NarrowBand(_CompactChirpSource):
        def carrier_phase(self, harmonic, time):
            time = np.asarray(time)
            return 2 * np.pi * (1e-3 * time + 0.5e-3 / self.t_end
                                * time ** 2)

        def angular_frequency(self, harmonic, time):
            return 2 * np.pi * (
                1e-3 + 1e-3 * np.asarray(time) / self.t_end)

    class HighBand(NarrowBand):
        def carrier_phase(self, harmonic, time):
            return super().carrier_phase(harmonic, time) \
                + 7e-3 * 2 * np.pi * np.asarray(time)

        def angular_frequency(self, harmonic, time):
            return super().angular_frequency(harmonic, time) + 7e-3 * 2*np.pi

    source = NarrowBand()
    distant = HighBand()
    orbit = LisaEqualArmOrbit(t0=0.0)
    terms = {"X": PyTDICombinationAdapter(
        "X2", get_pytdi_combination("X2"), delta_t=25.0).terms()}
    prepared = prepare_sparse_tdi(
        source, orbit, terms, 1.1, -0.4,
        [1e-3, 3e-3, 5e-3, 1e-2], t_start=0.0, t_end=5000.0,
        geometry_step=25.0, coverage_sources=[distant])
    response = prepared.project(source, 1.1, -0.4)
    empty = [band for band in response.bands if not band.support_blocks]
    assert empty
    values = response.sample(np.linspace(100.0, 1000.0, 20))["X"]
    assert np.all(np.isfinite(values))


def test_multiband_frequency_samples_match_dense_time_domain_transform():
    source = _CompactChirpSource()
    orbit = LisaEqualArmOrbit(t0=0.0)
    terms = PyTDICombinationAdapter(
        "X2", get_pytdi_combination("X2"), delta_t=25.0).terms()
    channel_terms = {"X": terms}
    common = dict(
        t_start=800.0,
        t_end=3200.0,
        initial_step=200.0,
        relative_tolerance=1e-5,
        velocity_order=1,
    )
    multiband = multiband_sparse_tdi_response(
        source, orbit, channel_terms, 1.1, -0.4,
        band_edges=[1e-3, 5e-3, 1e-2],
        overlap=1e-3,
        samples_per_cycle=4,
        **common,
    )

    # A fine-grid transform is the continuous-time oracle. This deliberately
    # short chirp has appreciable endpoint quadrature error at four samples per
    # cycle; that error scales away for mission-length observations.
    delta_t = 1.0
    dense = multiband.sample(
        np.arange(common["t_start"], common["t_end"], delta_t))["X"]
    expected = np.fft.rfft(dense) * delta_t
    delta_f = 1.0 / (len(dense) * delta_t)
    indices = np.arange(3, min(24, len(expected)))
    frequencies = indices * delta_f
    actual, diagnostics = multiband.frequency_samples(
        {"X": frequencies}, delta_f={"X": delta_f},
        epoch={"X": common["t_start"]},
        # Include both short-band spectra throughout this deliberately short
        # transform. Production overlaps make the required padding smaller.
        spectral_padding=1e-2,
        heterodyne=False,
        return_diagnostics=True,
    )

    scale = np.max(np.abs(expected[indices]))
    assert np.max(np.abs(actual["X"] - expected[indices])) / scale < 2e-2
    assert diagnostics
    assert sum(item["time_samples"] for item in diagnostics) < len(dense)

    # A shared smooth observation edge makes the dense sampled DFT and the
    # continuous pruned transform converge to the same spectrum.  Without
    # it, a rectangular boundary has broadband aliases which no in-band
    # multirate quadrature can reconstruct from its sparse samples.
    window = raised_cosine_time_window(
        np.arange(common["t_start"], common["t_end"], delta_t),
        common["t_start"], common["t_end"], 400.0)
    tapered_expected = np.fft.rfft(dense * window) * delta_t
    tapered = multiband.frequency_samples(
        {"X": frequencies}, delta_f={"X": delta_f},
        epoch={"X": common["t_start"]}, spectral_padding=1e-2,
        time_window=lambda time: raised_cosine_time_window(
            time, common["t_start"], common["t_end"], 400.0))
    reference = tapered_expected[indices]
    inner = np.vdot(reference, tapered["X"])
    norm = np.linalg.norm(reference) * np.linalg.norm(tapered["X"])
    assert 1 - abs(inner) / norm < 5e-5

    zeroed = multiband.frequency_samples(
        {"X": frequencies}, delta_f={"X": delta_f},
        epoch={"X": common["t_start"]}, spectral_padding=1e-2,
        time_window=lambda time: np.zeros_like(time),
    )
    assert np.all(zeroed["X"] == 0)

    sampler = multiband.prepare_frequency_sampler(
        {"X": frequencies}, delta_f={"X": delta_f},
        epoch={"X": common["t_start"]}, spectral_padding=1e-2,
        max_matrix_bytes=16 * 2 ** 20)
    prepared_values = sampler.evaluate(multiband)
    heterodyned_values = multiband.frequency_samples(
        {"X": frequencies}, delta_f={"X": delta_f},
        epoch={"X": common["t_start"]}, spectral_padding=1e-2)
    assert np.allclose(
        prepared_values["X"], heterodyned_values["X"],
        rtol=2e-12, atol=1e-30)
    assert sampler.diagnostics["matrix_bytes"] > 0

    duplicated = multiband.linear_transform(
        [[1.0], [2.0]], ("X1", "X2"))
    duplicated_sampler = duplicated.prepare_frequency_sampler(
        {"X1": frequencies, "X2": frequencies},
        delta_f={"X1": delta_f, "X2": delta_f},
        epoch={"X1": common["t_start"], "X2": common["t_start"]},
        spectral_padding=1e-2, max_matrix_bytes=16 * 2 ** 20)
    duplicated_values = duplicated_sampler.evaluate(duplicated)
    threaded_values = duplicated_sampler.evaluate(duplicated, workers=2)
    assert duplicated_sampler.diagnostics["kernel_count"] == len(
        duplicated_sampler.tasks)
    assert np.allclose(duplicated_values["X1"], prepared_values["X"])
    assert np.allclose(duplicated_values["X2"], 2 * prepared_values["X"])
    assert np.array_equal(threaded_values["X1"], duplicated_values["X1"])
    assert np.array_equal(threaded_values["X2"], duplicated_values["X2"])


def test_prepared_multiband_reuses_geometry_for_projection():
    """One prepared geometry serves repeated projections of a candidate.

    That the prepared path agrees with the one-shot path is
    `test_fixed_multiband_preparation_reproduces_adaptive_response`'s job;
    this one covers what only projection does -- the channel transform and
    the threaded evaluation must not change the answer.
    """
    source = _CompactChirpSource()
    orbit = LisaEqualArmOrbit(t0=0.0)
    terms = PyTDICombinationAdapter(
        "X2", get_pytdi_combination("X2"), delta_t=25.0).terms()
    prepared = prepare_sparse_tdi(
        source, orbit, {"X": terms}, 1.1, -0.4,
        band_edges=[1e-3, 5e-3, 1e-2], overlap=1e-3, samples_per_cycle=4,
        t_start=800.0, t_end=3200.0, geometry_step=10.0,
        minimum_grid_points=64, velocity_order=1)
    projected = prepared.project(source, 1.1, -0.4)
    assert projected.channels == ("X",)

    times = np.linspace(1000.0, 3000.0, 301)
    values = projected.sample(times)["X"]
    assert np.max(np.abs(values)) > 0

    transformed = prepared.project(
        source, 1.1, -0.4, matrix=[[2.0]], channels=("twice_X",))
    assert transformed.channels == ("twice_X",)
    assert np.allclose(transformed.sample(times)["twice_X"], 2 * values,
                       rtol=2e-12, atol=1e-30)

    threaded = prepared.project(source, 1.1, -0.4, source_workers=2)
    assert np.allclose(threaded.sample(times)["X"], values,
                       rtol=2e-12, atol=1e-30)

def test_fixed_multiband_preparation_reproduces_adaptive_response():
    source = _CompactChirpSource()
    orbit = LisaEqualArmOrbit(t0=0.0)
    terms = PyTDICombinationAdapter(
        "X2", get_pytdi_combination("X2"), delta_t=25.0).terms()
    channel_terms = {"X": terms}
    common = dict(
        band_edges=[1e-3, 5e-3, 1e-2], overlap=1e-3,
        samples_per_cycle=4, t_start=800.0, t_end=3200.0,
        velocity_order=1)
    # The same tolerance on both sides: the point is that one grid policy
    # serves both, so giving the prepared path a different tolerance and a
    # much denser uniform floor would let it pass on the floor alone.
    tolerance = 2e-5
    adaptive = sparse_tdi_response(
        source, orbit, channel_terms, 1.1, -0.4,
        initial_step=200.0, relative_tolerance=tolerance, **common)
    prepared = prepare_sparse_tdi(
        source, orbit, channel_terms, 1.1, -0.4,
        geometry_step=200.0, minimum_grid_points=16,
        relative_tolerance=tolerance, **common)
    fixed = prepared.response

    times = np.linspace(1000.0, 3000.0, 301)
    expected = adaptive.sample(times)["X"]
    actual = fixed.sample(times)["X"]
    scale = np.max(np.abs(expected))
    # 1.46e-04 on the uniform grid this preparation used to build, and
    # 3.22e-05 once the grid also follows the fiducial's carrier. The bound
    # sits where only the phase-following grid reaches, so the two paths
    # cannot drift back into being different response models.
    assert np.max(np.abs(actual - expected)) / scale < 5e-5


def test_zoom_frequency_samples_has_correct_scale_and_epoch():
    delta_t = 0.125
    epoch = 731.25
    times = epoch + np.arange(257) * delta_t
    values = (np.cos(2 * np.pi * 0.37 * times)
              + 0.2 * np.sin(2 * np.pi * 0.81 * times))
    series = TimeSeries(values, delta_t=delta_t, epoch=epoch, copy=False)
    delta_f = 1 / 128.0
    indices = np.arange(11, 71, 6)
    frequencies = indices * delta_f

    actual, diagnostics = _zoom_frequency_samples(
        series, frequencies, indices, delta_f, epoch - 19.0, 64)
    weights = np.ones(len(values))
    weights[[0, -1]] = 0.5
    relative_times = times - (epoch - 19.0)
    expected = delta_t * (
        np.exp(-2j * np.pi * frequencies[:, None] * relative_times)
        @ (weights * values))

    assert np.allclose(actual, expected, rtol=2e-12, atol=2e-12)
    assert diagnostics["zoom_points"] > 0

    direct_indices = indices[[1, -2]]
    direct_frequencies = direct_indices * delta_f
    direct, diagnostics = _zoom_frequency_samples(
        series, direct_frequencies, direct_indices, delta_f,
        epoch - 19.0, 64)
    expected = delta_t * (
        np.exp(-2j * np.pi * direct_frequencies[:, None] * relative_times)
        @ (weights * values))
    assert np.allclose(direct, expected, rtol=2e-12, atol=2e-12)
    assert diagnostics["direct_points"] == 2


def _prepared_grids(prepared):
    """Every band's time grid, as the preparation actually stored it."""
    return [band.prepared.geometries[band.template.harmonic].grid
            for band in prepared.prepared_bands]


def test_a_prior_corner_places_knots_and_not_only_widens_the_interval():
    """Covering a corner in time is not the same as resolving it.

    `coverage_sources` extended the time coverage while only the fiducial
    refined. A corner that leaves the interval is already refused, so the
    gap was the other case: a corner well inside the interval whose brackets
    vary faster than the fiducial's. It was accepted on the interval check
    alone, against a grid that had never been asked to resolve it.

    The corner here keeps the fiducial's support and band occupancy exactly,
    and only wobbles its carrier, so nothing but the knots can differ.
    """
    orbit = LisaEqualArmOrbit(t0=0.0)
    terms = {"X": PyTDICombinationAdapter(
        "X2", get_pytdi_combination("X2"), delta_t=25.0).terms()}
    common = dict(band_edges=[1e-3, 5e-3, 1e-2], overlap=1e-3,
                  samples_per_cycle=4, t_start=800.0, t_end=3200.0,
                  geometry_step=400.0, minimum_grid_points=8,
                  relative_tolerance=2e-5, velocity_order=1)
    fiducial = _CompactChirpSource()

    class _Corner(_CompactChirpSource):
        """Identical carrier, an envelope that varies far faster.

        The band partition is driven by frequency, so leaving the carrier
        alone keeps every band's time support identical and the corner
        inside the coverage. What differs is the bracket the geometry
        interpolates, which is what the knots are there for.
        """

        def amplitude(self, harmonic, time):
            time = np.asarray(time)
            plus, cross = super().amplitude(harmonic, time)
            wobble = 1 + 0.5 * np.sin(2 * np.pi * time / 250.0)
            return plus * wobble, cross * wobble

    corner = _Corner()

    alone = prepare_sparse_tdi(fiducial, orbit, terms, 1.1, -0.4, **common)
    with_corner = prepare_sparse_tdi(
        fiducial, orbit, terms, 1.1, -0.4,
        coverage_sources=[corner], **common)

    knots_alone = sum(len(g) for g in _prepared_grids(alone))
    knots_corner = sum(len(g) for g in _prepared_grids(with_corner))
    assert knots_corner > knots_alone

    # The knots are for the corner: on the fiducial-only geometry it is
    # served worse than on the one that was asked to resolve it.
    times = np.linspace(1000.0, 3000.0, 301)
    truth = sparse_tdi_response(
        corner, orbit, terms, 1.1, -0.4, band_edges=common["band_edges"],
        overlap=common["overlap"], samples_per_cycle=4,
        t_start=common["t_start"], t_end=common["t_end"],
        initial_step=400.0, relative_tolerance=2e-5,
        velocity_order=1).sample(times)["X"]
    scale = max(np.max(np.abs(truth)), np.finfo(float).tiny)
    without = np.max(np.abs(
        alone.project(corner, 1.1, -0.4).sample(times)["X"] - truth)) / scale
    with_it = np.max(np.abs(
        with_corner.project(corner, 1.1, -0.4).sample(times)["X"]
        - truth)) / scale
    assert with_it < without


def test_the_refinement_resolves_the_delay_tail_past_the_support():
    """Without `support_padding` the tail falls to the uniform floor.

    TDI reaches several arms back, so the response continues past the
    source's own support. The refinement only sees that tail if it is told
    the padding; otherwise the padded margin gets nothing but the uniform
    grid, which at the floor's spacing cannot resolve it.

    The assertion is therefore about the margin specifically. An earlier
    version asserted only that the grid was fine *somewhere*, which the
    in-support refinement satisfies on its own -- it survived the mutation
    that removes `support_padding`, and so proved nothing.
    """
    from pycbc.tdi.onthefly import harmonic_windows

    orbit = LisaEqualArmOrbit(t0=0.0)
    terms = {"X": PyTDICombinationAdapter(
        "X2", get_pytdi_combination("X2"), delta_t=25.0).terms()}
    source = _CompactChirpSource()
    floor, padding = 400.0, 150.0
    prepared = prepare_sparse_tdi(
        source, orbit, terms, 1.1, -0.4,
        band_edges=[1e-3, 5e-3, 1e-2], overlap=1e-3, samples_per_cycle=4,
        t_start=800.0, t_end=3200.0, geometry_step=floor,
        minimum_grid_points=8, relative_tolerance=2e-5, padding=padding,
        velocity_order=1)

    checked = 0
    for band in prepared.prepared_bands:
        harmonic = band.template.harmonic
        grid = band.prepared.geometries[harmonic].grid
        # Where the band's own response lives, before padding.
        bare = harmonic_windows(
            band.template.response.source, harmonic, 800.0, 3200.0,
            padding=0.0)
        if not bare:
            continue
        for low, high in bare:
            for margin in ((low - padding, low), (high, high + padding)):
                inside = grid[(grid >= margin[0]) & (grid <= margin[1])]
                if len(inside) < 3:
                    continue
                assert np.min(np.diff(inside)) < 0.5 * floor, (
                    f"margin {margin} spaced no finer than the floor")
                checked += 1
    assert checked, "no padded margin was populated at all"
