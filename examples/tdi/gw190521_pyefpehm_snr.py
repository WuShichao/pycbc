"""End-to-end pyEFPEHM -> first-order links -> PyTDI AET -> PyCBC PSD."""
import argparse
import os
import sys
import time
import types
from pathlib import Path

os.environ.setdefault("NO_PKGCONFIG", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

PROBE = "/mnt/d/pycbc-tdi-probe"
PYTDI = "/mnt/d/pytdi-v2.2.1"
PYEFPE = "/mnt/d/pyEFPEHM_dev/pyEFPEHM"
for path in (PROBE, PYTDI, PYEFPE):
    if path not in sys.path:
        sys.path.insert(0, path)

# Work around an unrelated stale detector-waveform entry point in tutorial.
lgwa = types.ModuleType("pycbc.waveform.lgwa")
lgwa.lgwa_fd_response = lambda **kwargs: None
sys.modules[lgwa.__name__] = lgwa

import numpy as np  # noqa: E402
import pyEFPEHM  # noqa: E402
from pycbc.coordinates.space_orbit import LisaEqualArmOrbit  # noqa: E402
from pycbc.psd.analytical_space import (  # noqa: E402
    analytical_psd_lisa_tdi_AE,
    analytical_psd_lisa_tdi_T,
)
from pycbc.tdi.backends.pytdi_backend import combine_links  # noqa: E402
from pycbc.tdi.response import (  # noqa: E402
    link_geometry,
    link_response,
    sample_constellation,
)

YEAR = 365.25 * 86400.0
C_SI = 299792458.0
ARM_LENGTH = 2.5e9


class OffsetPyEFPE:
    """Evaluate pyEFPE polarizations at arbitrary SSB query times."""

    def __init__(self, waveform, time_offset):
        self.waveform = waveform
        self.time_offset = float(time_offset)

    def polarizations(self, query):
        query = np.asarray(query, dtype=float)
        model_t = query + self.time_offset
        if model_t.ndim == 1:
            hp, hc = self.waveform.generate_tdomain_waveform(model_t)
            return hp, hc
        # Each link column is monotonic.  Evaluating columns independently
        # avoids sorting a large interleaved (time, link) array.
        hp = np.empty_like(model_t)
        hc = np.empty_like(model_t)
        for column in range(model_t.shape[1]):
            hp[:, column], hc[:, column] = (
                self.waveform.generate_tdomain_waveform(model_t[:, column])
            )
        return hp, hc


def process_aet_chunk(source, orbit, start, stop, dt, overlap, generation=1):
    ext_start = start - overlap
    ext_stop = stop + overlap
    times = np.arange(ext_start, ext_stop, dtype=float) * dt
    sample = sample_constellation(times, orbit)
    geometry = link_geometry(sample, 0.9, -0.25, velocity_order=1)
    links = link_response(source, sample, geometry)
    channels = combine_links(
        links,
        sample,
        generation=generation,
        channels="AET",
        interpolation_order=31,
        delay_order=5,
    )
    width = stop - start
    return {
        name: np.asarray(series)[overlap : overlap + width]
        for name, series in channels.items()
    }


def validate_chunking(source, orbit, dt):
    size = 4096
    overlap = 128
    times = np.arange(-overlap, size + overlap, dtype=float) * dt
    sample = sample_constellation(times, orbit)
    links = link_response(
        source, sample, link_geometry(sample, 0.9, -0.25, velocity_order=1)
    )
    full = combine_links(links, sample, generation=1, channels="AET")
    full = {
        name: np.asarray(value)[overlap : overlap + size]
        for name, value in full.items()
    }
    split = {name: np.empty(size) for name in "AET"}
    for start in range(0, size, 777):
        stop = min(size, start + 777)
        piece = process_aet_chunk(source, orbit, start, stop, dt, overlap)
        for name in "AET":
            split[name][start:stop] = piece[name]
    errors = {
        name: np.max(np.abs(split[name] - full[name]))
        / max(np.max(np.abs(full[name])), np.finfo(float).tiny)
        for name in "AET"
    }
    if max(errors.values()) > 2e-10:
        raise AssertionError(f"chunked PyTDI disagrees with full evaluation: {errors}")
    return errors


def tukey_edge_window(size, edge):
    window = np.ones(size)
    edge = min(int(edge), size // 2)
    if edge:
        x = np.arange(edge) / edge
        ramp = 0.5 * (1 - np.cos(np.pi * x))
        window[:edge] = ramp
        window[-edge:] = ramp[::-1]
    return window


def channel_snr(channel, dt, psd_kind, low_frequency=1e-5):
    size = len(channel)
    # One-day cosine edges suppress finite-observation leakage while removing
    # only 0.27% of the two-year duration.
    window = tukey_edge_window(size, round(86400.0 / dt))
    hf = dt * np.fft.rfft(np.asarray(channel) * window)
    delta_f = 1.0 / (size * dt)
    length = len(hf)
    psd_func = (
        analytical_psd_lisa_tdi_AE if psd_kind == "AE"
        else analytical_psd_lisa_tdi_T
    )
    psd = np.asarray(
        psd_func(length, delta_f, delta_f, tdi="1.5")
    )
    frequencies = np.arange(length) * delta_f
    transfer = np.abs(np.sin(2 * np.pi * frequencies * ARM_LENGTH / C_SI))
    valid = (
        (frequencies >= low_frequency)
        & np.isfinite(psd)
        & (psd > 0)
        & (transfer > 1e-6)
    )
    rho2 = 4 * delta_f * np.sum(np.abs(hf[valid]) ** 2 / psd[valid])
    raw_valid = (frequencies >= low_frequency) & np.isfinite(psd) & (psd > 0)
    raw_rho2 = 4 * delta_f * np.sum(
        np.abs(hf[raw_valid]) ** 2 / psd[raw_valid]
    )
    return float(np.sqrt(rho2)), float(np.sqrt(raw_rho2)), int(np.sum(~valid & raw_valid))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=2 * YEAR)
    parser.add_argument("--dt", type=float, default=5.0)
    parser.add_argument("--chunk", type=int, default=131072)
    parser.add_argument("--output", default="/tmp/pyefpehm_gw190521_aet")
    args = parser.parse_args()

    params = {
        # GW190521 discovery medians transformed to detector frame using z=0.82.
        "mass1": 85.0 * 1.82,
        "mass2": 66.0 * 1.82,
        "distance": 5300.0,
        # A common aligned, circular benchmark accepted by both requested models.
        "eccentricity": 0.0,
        "spin1z": 0.0,
        "spin2z": 0.0,
        "inclination": 1.0,
        "phase": 0.0,
        "f22_start": 0.0090784,
        "f22_ref": 0.0090784,
        "f22_end": 0.1,
        "Amplitude_tol": 1e-4,
    }
    begin = time.time()
    waveform = pyEFPEHM.pyEFPE(params)
    # End the exact requested duration at pyEFPE's 0.1-Hz endpoint.  Its
    # internally generated trajectory begins ~1.4 d earlier, providing ample
    # margin for the Sun/arm retardations at the observation start.
    model_offset = waveform.return_end_time() - args.duration
    if model_offset < waveform.return_start_time() + 1000:
        raise ValueError("pyEFPE trajectory lacks the required start margin")
    source = OffsetPyEFPE(waveform, model_offset)
    orbit = LisaEqualArmOrbit(t0=0.0)

    chunk_errors = validate_chunking(source, orbit, args.dt)

    size = int(round(args.duration / args.dt))
    overlap = 128
    paths = {name: Path(f"{args.output}_{name}.dat") for name in "AET"}
    arrays = {
        name: np.memmap(path, dtype="float64", mode="w+", shape=(size,))
        for name, path in paths.items()
    }
    for start in range(0, size, args.chunk):
        stop = min(size, start + args.chunk)
        piece = process_aet_chunk(
            source, orbit, start, stop, args.dt, overlap, generation=1
        )
        for name in "AET":
            arrays[name][start:stop] = piece[name]
        if start == 0 or stop == size or (start // args.chunk) % 10 == 0:
            print(f"processed {stop}/{size} samples", flush=True)
    for array in arrays.values():
        array.flush()

    snrs = {}
    for name in "AET":
        snrs[name] = channel_snr(
            arrays[name], args.dt, "AE" if name in "AE" else "T"
        )
    network = float(np.sqrt(sum(value[0] ** 2 for value in snrs.values())))
    network_raw = float(np.sqrt(sum(value[1] ** 2 for value in snrs.values())))

    print("\npyEFPEHM GW190521 two-year LISA benchmark")
    print(f"  non-base interpreter: {sys.executable}")
    print(f"  params: {params}")
    print(f"  observation: duration={args.duration:.1f} s, dt={args.dt}, N={size}")
    print(f"  model interval: [{model_offset:.6f}, {waveform.return_end_time():.6f}] s")
    print(f"  sky: lambda=0.9, beta=-0.25 rad; TDI-1.5 A/E/T")
    print(f"  chunk/full relative errors: {chunk_errors}")
    for name, (rho, raw, excluded) in snrs.items():
        print(f"  rho_{name}={rho:.9g} (unmasked={raw:.9g}, null bins excluded={excluded})")
    print(f"  rho_network={network:.9g} (unmasked={network_raw:.9g})")
    print(f"  elapsed={time.time() - begin:.3f} s")


if __name__ == "__main__":
    main()
