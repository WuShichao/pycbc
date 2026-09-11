"""Tests for pre-response multiband harmonic windows."""

import numpy as np

from pycbc.coordinates.space_orbit import LisaEqualArmOrbit
from pycbc.tdi.backends.pytdi_backend import (
    PyTDICombinationAdapter,
    get_pytdi_combination,
)
from pycbc.tdi.multiband import multiband_sparse_tdi_response
from pycbc.tdi.onthefly import adaptive_sparse_tdi_response
from pycbc.tdi.sources import frequency_partition_sources


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
