"""The single-link response against an independent implementation.

`lisagwresponse` implements the same velocity-order-0, retarded-emitter
equation this project's `velocity_order=0, retard_emitter=True` limit does,
driven by the same orbit positions and light travel times. It is therefore a
strict external oracle rather than a re-derivation, which is what makes the
agreement below worth asserting.

This was an example script that nothing ran automatically. Its Triangle
comparison is not reproduced here: Triangle carries its own value of the
astronomical unit, so a faithful comparison needs an orbit rebuilt on that
constant, and the result is a recorded measurement rather than a pass/fail
gate. The two assertions the script actually made were both about
`lisagwresponse`, and they are the two here.
"""

import os

import numpy as np
import pytest

from pycbc.coordinates.space_orbit import NumericOrbits
from pycbc.tdi.response import (C_SI, LINK_ORDER, link_geometry,
                                link_response,
                                sample_constellation)


DAY = 86400.0
LINK_CODES = np.array([12, 23, 31, 13, 32, 21])
ORBIT_DIR = os.environ.get(
    "PYCBC_TRIANGLE_ORBITS", "/mnt/d/Triangle-Simulator/OrbitData/LDC2BOrbit")
GW_RESPONSE_DIR = os.environ.get("PYCBC_GW_RESPONSE", "/mnt/d/gw-response-v3.0.2")


def _oracle():
    """`lisagwresponse`, or a skip naming what is missing."""
    import sys
    if GW_RESPONSE_DIR and GW_RESPONSE_DIR not in sys.path:
        sys.path.insert(0, GW_RESPONSE_DIR)
    module = pytest.importorskip(
        "lisagwresponse", reason="lisagwresponse is not importable")
    # The unpacked source tree carries no distribution metadata.
    from lisagwresponse import response as response_module
    response_module.importlib_metadata.version = lambda name: "3.0.2"
    return module.ResponseFromStrain


def _orbit_functions(orbit):
    """Positions and light travel times in the shape the oracle wants."""
    def component(sc, xyz):
        return lambda t: orbit.compute_position(
            np.asarray(t), (sc,))[:, 0, xyz]

    positions = {name: {sc: component(sc, xyz) for sc in (1, 2, 3)}
                 for xyz, name in enumerate(("x", "y", "z"))}

    def delay_function(receiver, emitter):
        def delay(t):
            t = np.asarray(t, dtype=float)
            recv = orbit.compute_position(t, (receiver,))[:, 0]
            value = np.linalg.norm(
                recv - orbit.compute_position(t, (emitter,))[:, 0],
                axis=1) / C_SI
            for _ in range(20):
                updated = np.linalg.norm(
                    recv - orbit.compute_position(t - value, (emitter,))[:, 0],
                    axis=1) / C_SI
                if np.max(np.abs(updated - value)) < 1e-12:
                    return updated
                value = updated
            raise RuntimeError("light-cone iteration did not converge")
        return delay

    travel = {int(f"{r}{e}"): delay_function(r, e) for r, e in LINK_ORDER}
    return positions, travel


def _relative_errors(ours, other):
    scale = max(np.max(np.abs(ours)), np.finfo(float).tiny)
    return np.max(np.abs(ours - other), axis=0) / scale


@pytest.fixture(scope="module")
def scan():
    """Both responses on a small grid that still spans a year."""
    if not os.path.isdir(ORBIT_DIR):
        pytest.skip(f"numeric orbit files not found at {ORBIT_DIR}")
    from_strain = _oracle()
    orbit = NumericOrbits.from_triangle_dat_files(ORBIT_DIR)

    class _Monochromatic:
        def __init__(self, frequency):
            self.frequency = float(frequency)

        def polarizations(self, times):
            phase = 2 * np.pi * self.frequency * np.asarray(times)
            return np.cos(phase), np.sin(phase)

    class _OracleSource(from_strain):
        def __init__(self, frequency, **kwargs):
            self.frequency = float(frequency)
            super().__init__(**kwargs)

        def compute_hplus(self, times):
            return np.cos(2 * np.pi * self.frequency * np.asarray(times))

        def compute_hcross(self, times):
            return np.sin(2 * np.pi * self.frequency * np.asarray(times))

    positions, travel = _orbit_functions(orbit)
    # Few points, but still a year apart: the float64 phase-reduction floor
    # the bound is set by comes from phases of order 1e7 radians, and a short
    # span would agree far more tightly and prove less.
    times = np.linspace(2 * DAY, 363 * DAY, 5)
    sample = sample_constellation(times, orbit)
    skies = ((0.7, -0.35), (2.9, 0.15), (4.8, 0.62))

    worst = {"v0": np.zeros(6), "full": np.zeros(6)}
    for frequency in (1e-4, 5e-2):
        source = _Monochromatic(frequency)
        for lamb, beta in skies:
            ours_v0 = link_response(
                source, sample,
                link_geometry(sample, lamb, beta, velocity_order=0))
            ours_full = link_response(
                source, sample,
                link_geometry(sample, lamb, beta, velocity_order=1))
            oracle = _OracleSource(
                frequency, ra=lamb, dec=beta, x=positions["x"],
                y=positions["y"], z=positions["z"], ltt=travel,
                orbits="unused", shift_sun="never",
            ).compute_gw_response(times, LINK_CODES)
            worst["v0"] = np.maximum(
                worst["v0"], _relative_errors(ours_v0, oracle))
            worst["full"] = np.maximum(
                worst["full"], _relative_errors(ours_full, oracle))
    return worst


def test_velocity_order_zero_matches_the_external_oracle(scan):
    """Same equation, two implementations, at the float64 phase floor."""
    assert np.max(scan["v0"]) < 2e-9


def test_the_first_order_response_does_not_collapse_to_the_oracle(scan):
    """Otherwise the agreement above would be measuring nothing.

    The Doppler factors and the emitter retardation are real physics that
    `lisagwresponse` does not carry, so the full response must differ from it
    by far more than the numerical floor.
    """
    assert np.max(scan["full"]) > 1e-6
