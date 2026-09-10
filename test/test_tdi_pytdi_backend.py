"""Integration tests for the optional PyTDI in-memory adapter."""

import numpy as np
import pytest

pytdi = pytest.importorskip("pytdi")

from pycbc.coordinates.space_orbit import (  # noqa: E402
    LisaEqualArmOrbit,
    NumericOrbits,
)
from pycbc.tdi.backends.pytdi_backend import (  # noqa: E402
    PyTDICombinationAdapter,
    combine_link_combination,
    combine_links,
    get_pytdi_combination,
)
from pycbc.tdi.response import (  # noqa: E402
    link_geometry,
    link_response,
    sample_constellation,
)
from pycbc.tdi.sources import ArrayWaveformSource  # noqa: E402
from pycbc.types import TimeSeries  # noqa: E402


def test_eta_shortcut_matches_pytdi_and_returns_timeseries():
    times = np.arange(1024, dtype=float) * 0.5
    sample = sample_constellation(times, LisaEqualArmOrbit())
    rng = np.random.default_rng(20260903)
    links = rng.normal(size=(len(times), 6))

    actual = combine_links(
        links, sample, channels="XYZ", interpolation_order=15
    )
    etas = {
        f"eta_{i}{j}": links[:, index]
        for index, (i, j) in enumerate(sample.links)
    }
    expected = pytdi.michelson.X2_ETA.build(
        sample.delays, 2.0, order=5
    )(etas, order=15, unit="frequency")
    assert isinstance(actual["X"], TimeSeries)
    assert actual["X"].delta_t == 0.5
    assert float(actual["X"].start_time) == times[0]
    assert np.array_equal(np.asarray(actual["X"]), expected)


def test_end_to_end_velocity_links_to_aet():
    source_times = np.arange(-700.0, 1400.0, 0.5)
    phase = 2 * np.pi * 0.01 * source_times
    source = ArrayWaveformSource(
        source_times, np.cos(phase), 0.2 * np.sin(phase)
    )
    analytic_orbit = LisaEqualArmOrbit()
    orbit_times = np.arange(-100.0, 700.0, 10.0)
    numeric_orbit = NumericOrbits(
        orbit_times,
        analytic_orbit.compute_position(orbit_times),
        analytic_orbit.compute_velocity(orbit_times),
    )
    sample = sample_constellation(
        np.arange(1024, dtype=float) * 0.5, numeric_orbit
    )
    geometry = link_geometry(sample, 0.8, -0.3, velocity_order=1)
    links = link_response(source, sample, geometry)
    channels = combine_links(
        links, sample, channels="AET", interpolation_order=15
    )
    assert set(channels) == {"A", "E", "T"}
    assert all(isinstance(series, TimeSeries) for series in channels.values())
    arrays = [series.numpy() for series in channels.values()]
    assert all(np.all(np.isfinite(array)) for array in arrays)
    assert max(np.max(np.abs(array)) for array in arrays) > 0


def test_combination_adapter_exposes_positive_delay_convention():
    times = np.arange(256, dtype=float) * 2.0
    sample = sample_constellation(times, LisaEqualArmOrbit())
    adapter = PyTDICombinationAdapter(
        "X2", pytdi.michelson.X2_ETA, delta_t=2.0
    )
    assert len(adapter.terms()) == 16
    shifts = adapter.net_shifts(sample)
    assert np.min(shifts[()]) == 0
    delayed = [shift for operators, shift in shifts.items() if operators]
    assert min(np.median(shift) for shift in delayed) > 0
    assert adapter.waveform_support(sample) > adapter.data_delay_extent(sample)


def test_verified_combination_registry_and_extents():
    assert get_pytdi_combination("X2").components \
        == pytdi.michelson.X2_ETA.components
    assert get_pytdi_combination("PD4L-1").rotated().components \
        == get_pytdi_combination("PD4L-2").components
    assert get_pytdi_combination("PD4L-2").rotated().components \
        == get_pytdi_combination("PD4L-3").components
    with pytest.raises(ValueError, match="unknown or unverified"):
        get_pytdi_combination("C12_3")

    times = np.arange(512, dtype=float) * 2.0
    sample = sample_constellation(times, LisaEqualArmOrbit())
    arm = np.median(sample.ltt)
    for name, expected_arms in (("X2", 7), ("UU", 7), ("PD4L-1", 3),
                                ("PD4L-2", 3), ("PD4L-3", 3)):
        adapter = PyTDICombinationAdapter(
            name, get_pytdi_combination(name), delta_t=2.0
        )
        extent = adapter.data_delay_extent(sample) / arm
        support = adapter.waveform_support(sample) / arm
        assert np.isclose(extent, expected_arms, rtol=0.01)
        # exactly one arm longer, not merely longer: the earliest
        # inter-spacecraft link is emitted an arm before it is received, and
        # that arm is what an on-the-fly evaluator has to generate strain
        # over even though no recorded sample reaches back to it
        assert np.isclose(support - extent, 1.0, rtol=0.01)


def test_registered_combination_returns_timeseries():
    times = np.arange(512, dtype=float) * 2.0
    sample = sample_constellation(times, LisaEqualArmOrbit())
    rng = np.random.default_rng(9)
    links = rng.normal(size=(len(times), 6))
    result = combine_link_combination(
        links, sample, "PD4L-1", interpolation_order=15
    )
    assert isinstance(result, TimeSeries)
    assert len(result) == len(times)


def test_wangs_arrow_notation_transcribes_to_pytdi():
    """Wang's X1 path, written out, IS pytdi's second-generation Michelson.

    This is what makes his PD4L strings usable: his ``<-beam`` is an
    unprefixed beam and his ``->beam`` is a '-'-prefixed one, and the only
    way to be sure of that mapping is to transcribe a combination he shares
    with pytdi and compare the monomials. Getting it backwards would leave
    PD4L quietly wrong with no other symptom, since D3 symmetry holds either
    way.
    """
    from pytdi import michelson
    from pytdi.core import LISATDICombination

    transcribed = LISATDICombination.from_string('131212131 -121313121')
    assert transcribed.components == michelson.X2_ETA.components
    # four eta variables, four monomials each: the sixteen the plan counts
    assert set(transcribed.components) == {'eta_12', 'eta_13',
                                           'eta_21', 'eta_31'}
    assert sum(len(v) for v in transcribed.components.values()) == 16

    # and the mapping is not symmetric: swapping which arrow takes the minus
    # sign gives a different combination, so the test above is a real check
    swapped = LISATDICombination.from_string('-131212131 121313121')
    assert swapped.components != michelson.X2_ETA.components
