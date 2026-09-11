"""Tests for pre-response multiband harmonic windows."""

import numpy as np

from pycbc.coordinates.space_orbit import LisaEqualArmOrbit
from pycbc.tdi.backends.pytdi_backend import (
    PyTDICombinationAdapter,
    get_pytdi_combination,
)
from pycbc.tdi.multiband import (
    _zoom_frequency_samples,
    multiband_sparse_tdi_response,
)
from pycbc.tdi.onthefly import adaptive_sparse_tdi_response
from pycbc.tdi.sources import frequency_partition_sources
from pycbc.types import TimeSeries


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

    zeroed = multiband.frequency_samples(
        {"X": frequencies}, delta_f={"X": delta_f},
        epoch={"X": common["t_start"]}, spectral_padding=1e-2,
        time_window=lambda time: np.zeros_like(time),
    )
    assert np.all(zeroed["X"] == 0)


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
