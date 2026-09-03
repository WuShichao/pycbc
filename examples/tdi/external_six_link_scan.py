"""Scan the probe single-link response against Triangle and lisagwresponse."""
import os
import sys
import types

os.environ.setdefault("NO_PKGCONFIG", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

PROBE = "/mnt/d/pycbc-tdi-probe"
TRIANGLE = "/mnt/d/Triangle-Simulator"
GWRESPONSE = "/mnt/d/gw-response-v3.0.2"
for path in (PROBE, TRIANGLE, GWRESPONSE):
    if path not in sys.path:
        sys.path.insert(0, path)

# The tutorial environment has an installed PyCBC entry point whose target is
# only present on another development branch.  Triangle imports pycbc.waveform
# even though the GW class used below does not use it.  Supply the inert stub so
# the unrelated entry-point defect does not prevent importing Triangle.GW.
lgwa = types.ModuleType("pycbc.waveform.lgwa")
lgwa.lgwa_fd_response = lambda **kwargs: None
sys.modules[lgwa.__name__] = lgwa

import numpy as np  # noqa: E402
import lisagwresponse.response as lgw_response_module  # noqa: E402
from lisagwresponse.response import ResponseFromStrain  # noqa: E402
from pycbc.coordinates.space_orbit import NumericOrbits  # noqa: E402
from pycbc.tdi.response import (  # noqa: E402
    C_SI,
    LINK_ORDER,
    link_geometry,
    link_response,
    sample_constellation,
)
from Triangle.GW import GW  # noqa: E402
from Triangle.Orbit import Orbit  # noqa: E402

# We are deliberately validating the unpacked v3.0.2 source tree, not an
# installed wheel, so distribution metadata is absent.
lgw_response_module.importlib_metadata.version = lambda name: "3.0.2"

DAY = 86400.0
# Triangle/Constants.py:8; deliberately not the IAU value.
TRIANGLE_AU = 1.496e11
LINK_CODES = np.array([12, 23, 31, 13, 32, 21])
TRIANGLE_LABELS = ["12", "23", "31", "13", "32", "21"]


class Monochromatic:
    def __init__(self, frequency):
        self.frequency = float(frequency)

    def polarizations(self, times):
        phase = 2 * np.pi * self.frequency * np.asarray(times)
        return np.cos(phase), np.sin(phase)

    def hpfunc(self, times):
        return self.polarizations(times)[0]

    def hcfunc(self, times):
        return self.polarizations(times)[1]


class LgwMonochromatic(ResponseFromStrain):
    def __init__(self, frequency, **kwargs):
        self.frequency = float(frequency)
        super().__init__(**kwargs)

    def compute_hplus(self, times):
        return np.cos(2 * np.pi * self.frequency * np.asarray(times))

    def compute_hcross(self, times):
        return np.sin(2 * np.pi * self.frequency * np.asarray(times))


def orbit_functions(orbit):
    def component(sc, xyz):
        return lambda t: orbit.compute_position(np.asarray(t), (sc,))[:, 0, xyz]

    positions = {
        xyz_name: {sc: component(sc, xyz) for sc in (1, 2, 3)}
        for xyz, xyz_name in enumerate(("x", "y", "z"))
    }

    def delay_function(receiver, emitter):
        def delay(t):
            t = np.asarray(t, dtype=float)
            recv = orbit.compute_position(t, (receiver,))[:, 0]
            d = np.linalg.norm(
                recv - orbit.compute_position(t, (emitter,))[:, 0], axis=1
            ) / C_SI
            for _ in range(20):
                updated = np.linalg.norm(
                    recv - orbit.compute_position(t - d, (emitter,))[:, 0], axis=1
                ) / C_SI
                if np.max(np.abs(updated - d)) < 1e-12:
                    return updated
                d = updated
            raise RuntimeError("light-cone iteration did not converge")

        return delay

    ltt = {
        int(f"{r}{e}"): delay_function(r, e)
        for r, e in LINK_ORDER
    }
    return positions, ltt


def relative_errors(a, b):
    scale = max(np.max(np.abs(a)), np.finfo(float).tiny)
    return np.max(np.abs(a - b), axis=0) / scale


def main():
    orbit_dir = f"{TRIANGLE}/OrbitData/LDC2BOrbit"
    native_orbit = NumericOrbits.from_triangle_dat_files(orbit_dir)
    # pn_order=2 keeps LTT2 (v^2/c^2 + solar Shapiro).  Measured to change the
    # residual by <1e-7, so it is not what sets the floor, but there is no
    # reason to validate against a knowingly truncated baseline.
    triangle_orbit = Orbit(orbit_dir, dt=DAY, pn_order=2)
    pos, ltt = orbit_functions(native_orbit)

    # Triangle carries AU to four significant figures (Constants.py:8,
    # 1.496e11) while pycbc uses the IAU value, a relative 1.4233e-5.  The
    # difference is common-mode, so it leaves the arm directions and the
    # light-cone geometry alone and enters only through k.r -- which is ~500 s
    # against an 8.3 s arm, so the 1.4e-5 constant mismatch is amplified into a
    # 2.1e-3 response residual.  Comparing against Triangle on Triangle's AU
    # drops that to 6e-8, the genuine O(v^2/c^2) floor.  lisagwresponse needs no
    # such treatment: it is driven by this orbit's own positions and delays.
    triangle_au_orbit = NumericOrbits(
        np.arange(len(np.loadtxt(f"{orbit_dir}/SCP1.dat"))) * DAY,
        np.stack([np.loadtxt(f"{orbit_dir}/SCP{label}.dat")
                  for label in ("1", "2", "3")], axis=1) * TRIANGLE_AU,
        interp_order=5,
    )

    # Avoid the orbit interpolation margins while spanning essentially a year.
    times = np.linspace(2 * DAY, 363 * DAY, 37)
    frequencies = (1e-4, 1e-3, 1e-2, 5e-2)
    # Deterministic approximately isotropic sky: a longitude lattice and equal
    # spacing in sin(latitude), deliberately including neither poles nor links.
    nsky = 24
    golden = np.pi * (3 - np.sqrt(5))
    skies = [
        ((i * golden) % (2 * np.pi), np.arcsin(-1 + (2 * i + 1) / nsky))
        for i in range(nsky)
    ]

    maxima = {
        "lisagw_v0": np.zeros(6),
        "lisagw_full": np.zeros(6),
        "triangle_equiv": np.zeros(6),
        "triangle_full": np.zeros(6),
    }
    rms_num = {key: np.zeros(6) for key in maxima}
    rms_den = {key: np.zeros(6) for key in maxima}
    count = 0

    sample = sample_constellation(times, native_orbit)
    sample_tri_au = sample_constellation(times, triangle_au_orbit)
    for frequency in frequencies:
        source = Monochromatic(frequency)
        for lamb, beta in skies:
            v0 = link_response(
                source, sample,
                link_geometry(sample, lamb, beta, velocity_order=0),
            )
            v0_unretarded = link_response(
                source,
                sample,
                link_geometry(
                    sample, lamb, beta, velocity_order=0, retard_emitter=False
                ),
            )
            full = link_response(
                source, sample,
                link_geometry(sample, lamb, beta, velocity_order=1),
            )

            lgw = LgwMonochromatic(
                frequency,
                ra=lamb,
                dec=beta,
                x=pos["x"],
                y=pos["y"],
                z=pos["z"],
                ltt=ltt,
                orbits="unused",
                shift_sun="never",
            ).compute_gw_response(times, LINK_CODES)

            tri = GW(
                triangle_orbit,
                [lamb, beta, 0.0],
                GWwaveform=source,
            ).CalculateResponse({str(sc): times for sc in (1, 2, 3)})
            tri = np.column_stack([tri[label] for label in TRIANGLE_LABELS])

            tri_equiv = link_response(
                source, sample_tri_au,
                link_geometry(sample_tri_au, lamb, beta,
                              velocity_order=0, retard_emitter=False),
            )
            tri_full = link_response(
                source, sample_tri_au,
                link_geometry(sample_tri_au, lamb, beta, velocity_order=1),
            )
            comparisons = {
                "lisagw_v0": (v0, lgw),
                "lisagw_full": (full, lgw),
                "triangle_equiv": (tri_equiv, tri),
                "triangle_full": (tri_full, tri),
            }
            for key, (ours, other) in comparisons.items():
                err = relative_errors(ours, other)
                maxima[key] = np.maximum(maxima[key], err)
                rms_num[key] += np.sum(np.abs(ours - other) ** 2, axis=0)
                rms_den[key] += np.sum(np.abs(ours) ** 2, axis=0)
            count += len(times)

    print("External six-link response scan")
    print(f"  Triangle-Simulator: 2c5301d; lisagwresponse: local v3.0.2 source")
    print(f"  grid: {len(times)} times x {len(skies)} skies x {len(frequencies)} frequencies")
    print(f"  frequencies [Hz]: {frequencies}")
    print(f"  directed links: {tuple(LINK_ORDER)}")
    print("  error normalization: per case max_t |ours-ref| / max_all_links,t |ours|")
    for key in maxima:
        rms = np.sqrt(rms_num[key] / rms_den[key])
        print(f"\n  {key}")
        print("    max per link: " + " ".join(f"{v:.6e}" for v in maxima[key]))
        print("    rms per link: " + " ".join(f"{v:.6e}" for v in rms))
        print(f"    global max: {np.max(maxima[key]):.6e}")

    # lisagwresponse implements the same velocity-order-0, retarded-emitter
    # equation.  That is the strict external oracle in this scan.
    # Across a full year, the two codes evaluate trigonometric phases of order
    # 1e7 radians through slightly different arithmetic paths.  The observed
    # 1.8e-9 ceiling is therefore the appropriate float64 phase-reduction
    # floor; short-time cases agree much more tightly.
    assert np.max(maxima["lisagw_v0"]) < 2e-9
    # The full first-order response should *not* accidentally collapse to the
    # velocity-order-0 oracle.
    assert np.max(maxima["lisagw_full"]) > 1e-6


if __name__ == "__main__":
    main()
