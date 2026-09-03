"""Tests for the native PyCBC space-detector backend."""

import numpy as np
import pytest


def _relative_difference(a, b):
    """Fractional difference, since np.allclose's atol=1e-8 default swallows
    everything at strain amplitudes of 1e-22."""
    a, b = np.asarray(a), np.asarray(b)
    return np.max(np.abs(a - b)) / np.max(np.abs(a))

from pycbc.detector.space import SpaceDetector
from pycbc.types import TimeSeries


def _strain(n=8192, delta_t=5.0, frequency=0.005):
    t = np.arange(n) * delta_t
    hp = TimeSeries(1e-21 * np.cos(2 * np.pi * frequency * t),
                    delta_t=delta_t, epoch=0.0)
    hc = TimeSeries(0.3e-21 * np.sin(2 * np.pi * frequency * t),
                    delta_t=delta_t, epoch=0.0)
    return hp, hc


@pytest.mark.parametrize("detector", ["LISA", "Taiji", "TianQin"])
def test_native_backend_serves_all_three_constellations(detector):
    """The reason this backend exists: LDC and FLR are LISA-only."""
    hp, hc = _strain()
    channels = SpaceDetector(detector, backend='pycbc').project_wave(
        hp, hc, 0.9, -0.25, polarization=0.3, tdi=2, tdi_chan='AET')
    assert set(channels) == set("AET")
    for series in channels.values():
        assert isinstance(series, TimeSeries)
        assert len(series) == len(hp)
        assert np.all(np.isfinite(np.asarray(series)))
        assert np.max(np.abs(np.asarray(series))) > 0


def test_channel_and_generation_options():
    hp, hc = _strain()
    detector = SpaceDetector('LISA', backend='pycbc')
    assert set(detector.project_wave(hp, hc, 0.9, -0.25,
                                     tdi_chan='XYZ')) == set("XYZ")
    assert set(detector.project_wave(hp, hc, 0.9, -0.25,
                                     tdi_chan='AE')) == set("AE")
    first = detector.project_wave(hp, hc, 0.9, -0.25, tdi=1.5)['A']
    second = detector.project_wave(hp, hc, 0.9, -0.25, tdi=2)['A']
    # second-generation X is first-generation X minus a delayed copy, so the
    # two must not come back identical
    assert _relative_difference(first, second) > 1e-3
    with pytest.raises(ValueError):
        detector.project_wave(hp, hc, 0.9, -0.25, tdi=3)
    with pytest.raises(ValueError):
        detector.project_wave(hp, hc, 0.9, -0.25, tdi_chan='Q')


def test_polarization_is_applied_not_ignored():
    hp, hc = _strain()
    detector = SpaceDetector('LISA', backend='pycbc')
    zero = detector.project_wave(hp, hc, 0.9, -0.25, polarization=0.0)['A']
    turned = detector.project_wave(hp, hc, 0.9, -0.25,
                                   polarization=np.pi / 4)['A']
    assert _relative_difference(zero, turned) > 1e-2


def test_velocity_order_zero_changes_the_answer():
    """velocity_order=0 drops the Doppler weights but keeps the geometry."""
    hp, hc = _strain()
    detector = SpaceDetector('LISA', backend='pycbc')
    full = np.asarray(detector.project_wave(hp, hc, 0.9, -0.25)['A'])
    no_eps = np.asarray(detector.project_wave(hp, hc, 0.9, -0.25,
                                              velocity_order=0)['A'])
    relative = np.max(np.abs(full - no_eps)) / np.max(np.abs(full))
    assert 1e-6 < relative < 1e-1


def test_remove_garbage_trims_both_edges():
    hp, hc = _strain()
    detector = SpaceDetector('LISA', backend='pycbc')
    full = detector.project_wave(hp, hc, 0.9, -0.25)['A']
    cut = detector.project_wave(hp, hc, 0.9, -0.25,
                                remove_garbage=True, t0=1e4)['A']
    assert len(cut) == len(full) - 2 * int(1e4 / full.delta_t)
