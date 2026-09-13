"""Compare sparse time-domain TDI with the official Yorsh SOBHB data.

This is deliberately a data-product validation, not a second implementation
of the response.  Waveforms, prepared sparse TDI, A/E/T construction and PSDs
all come from :mod:`pycbc`.  The code here only translates the LDC catalogue,
streams the official HDF5 channels, and forms noise-weighted overlaps.

Two bands are reported because the Yorsh production configuration explicitly
defines 0.05 Hz as the TDI anti-alias passband and 0.1 Hz as its stopband.  A
comparison through 0.1 Hz therefore mixes waveform/response error with the
challenge pipeline's 5 s source sampling, spline upsampling and final Kaiser
filter.  ``passband`` is the physics validation; ``transition`` is retained as
a diagnostic of that data conditioning and is never used as an acceptance
criterion.

The LDC SBBH implementation passes ``Mass1`` and ``Mass2`` directly to
IMRPhenomD.  They are already detector-frame masses.  ``Redshift`` is catalogue
metadata and must not be applied to the masses a second time.

Examples
--------
Run two sources with the independent LAL IMRPhenomD adapter::

    python examples/tdi/validate_yorsh_sobhb.py \
        /path/to/yorsh_1-b-2_training.hdf5 --names sobhb8 sobhb9

Exercise the pyEFPEHM adapter instead::

    python examples/tdi/validate_yorsh_sobhb.py DATA --model pyefpehm
"""

import argparse
import gc
import json
import time

import h5py
import numpy as np
from scipy.fft import rfft

from pycbc.coordinates.space_orbit import (
    LisaEqualArmOrbit,
    LisaKeplerianOrbit,
)
from pycbc.psd.analytical_space import (
    analytical_psd_lisa_tdi_AE,
    analytical_psd_lisa_tdi_T,
)
from pycbc.tdi import (
    PolarizationRotatedHarmonicSource,
    PyEFPEHMSource,
    TimeShiftedHarmonicSource,
    orthogonal_channels,
    prepare_multiband_tdi,
)
from pycbc.tdi.backends.pytdi_backend import (
    PyTDICombinationAdapter,
    get_pytdi_combination,
)
from pycbc.tdi.onthefly import delay_padding

MTSUN_SI = 4.925490947e-6
C_SI = 299_792_458.0
DEFAULT_ARM_LENGTH = 2.5e9
DEFAULT_PASSBAND = 0.05
DEFAULT_STOPBAND = 0.1
SPECTRAL_PADDING = 5e-6
NULL_SINE_MIN = 1e-3


def catalogue_parameters(row):
    """Return scalar catalogue values without applying redshift to masses."""
    result = {}
    for name in row.dtype.names:
        value = row[name].item()
        if isinstance(value, bytes):
            value = value.decode()
        result[name] = value
    return result


def newtonian_end_frequency(mass1, mass2, f_start, duration):
    """Conservative upper-frequency estimate used only to size the source."""
    total = (mass1 + mass2) * MTSUN_SI
    eta = mass1 * mass2 / (mass1 + mass2) ** 2
    chirp = total * eta ** 0.6
    tau = 5 / 256 * chirp ** (-5 / 3) * (np.pi * f_start) ** (-8 / 3)
    remaining = max(tau - duration, 1e-3 * tau)
    return (256 / 5 * chirp ** (5 / 3) * remaining) ** (-3 / 8) / np.pi


def make_source(parameters, duration, model):
    """Build one source on the mission clock using detector-frame masses."""
    mass1 = float(parameters["Mass1"])
    mass2 = float(parameters["Mass2"])
    f_start = float(parameters["InitialFrequency"])
    f_end = min(DEFAULT_STOPBAND * 1.005, max(
        1.01 * f_start,
        1.01 * newtonian_end_frequency(
            mass1, mass2, f_start, duration),
    ))

    common = dict(
        mass1=mass1,
        mass2=mass2,
        distance=float(parameters["Distance"]),
        spin1z=float(parameters["Spin1"]),
        spin2z=float(parameters["Spin2"]),
        inclination=float(parameters["Inclination"]),
    )
    if model == "lal":
        from pycbc.tdi import LALIMRPhenomDSource

        # LDC subtracts the phase at its first sample, so InitialPhase does
        # not enter its carrier.  Its remaining angular convention agrees
        # with coa_phase=0 and the explicit polarization rotation here.
        return LALIMRPhenomDSource(
            **common,
            coa_phase=0.0,
            polarization=float(parameters["Polarization"]),
            f_lower=f_start,
            f_upper=f_end,
            n_frequency=4096,
            t_start=0.0,
        )
    if model != "pyefpehm":
        raise ValueError(f"unknown source model {model!r}")

    native = PyEFPEHMSource(dict(
        **common,
        phase=float(parameters["InitialPhase"]),
        eccentricity=0.0,
        f22_start=f_start,
        f22_ref=f_start,
        f22_end=f_end,
        Amplitude_tol=1e-4,
    ))
    shifted = TimeShiftedHarmonicSource(
        native,
        offset=native.t_start,
        t_start=0.0,
        t_end=duration,
    )
    return PolarizationRotatedHarmonicSource(
        shifted, float(parameters["Polarization"]))


def source_band(source, duration):
    """Return the frequency hull actually live during the observation."""
    lower, upper = np.inf, 0.0
    for harmonic in source.harmonics:
        blocks_method = getattr(source, "support_blocks", None)
        blocks = (blocks_method(harmonic) if blocks_method is not None
                  else (source.support(harmonic),))
        for start, stop in blocks:
            start = max(0.0, float(start))
            stop = min(float(duration), float(stop))
            if stop <= start:
                continue
            values = np.asarray(source.angular_frequency(
                harmonic, np.asarray([start, stop]))) / (2 * np.pi)
            lower = min(lower, float(np.min(values)))
            upper = max(upper, float(np.max(values)))
    if not np.isfinite(lower) or upper <= lower:
        raise ValueError("source has no live frequency support in observation")
    return lower, upper


def _load_field(dataset, name):
    return np.asarray(dataset.fields(name)[:], dtype=np.float64).reshape(-1)


def official_aet_channel(dataset, channel):
    """Read one official channel, keeping peak memory to three real arrays."""
    if channel == "A":
        result = _load_field(dataset, "Z")
        result -= _load_field(dataset, "X")
        result /= np.sqrt(2.0)
    elif channel == "E":
        result = _load_field(dataset, "X")
        result -= 2.0 * _load_field(dataset, "Y")
        result += _load_field(dataset, "Z")
        result /= np.sqrt(6.0)
    elif channel == "T":
        result = _load_field(dataset, "X")
        result += _load_field(dataset, "Y")
        result += _load_field(dataset, "Z")
        result /= np.sqrt(3.0)
    else:
        raise ValueError("channel must be A, E or T")
    return result


def _noise(channel, length, delta_f, low_frequency, acc_noise, oms_noise):
    factory = (analytical_psd_lisa_tdi_T if channel == "T"
               else analytical_psd_lisa_tdi_AE)
    return np.asarray(factory(
        length,
        delta_f,
        low_frequency,
        tdi="2.0",
        acc_noise_level=acc_noise,
        oms_noise_level=oms_noise,
    ))


def _empty_accumulator():
    return {"official_norm": 0.0, "candidate_norm": 0.0, "cross": 0j,
            "bins": 0}


def _add_inner_products(accumulator, official, candidate, psd, mask,
                        delta_f):
    selected_official = official[mask]
    selected_candidate = candidate[mask]
    selected_psd = psd[mask]
    scale = 4 * delta_f
    accumulator["official_norm"] += float(
        scale * np.sum(np.abs(selected_official) ** 2 / selected_psd))
    accumulator["candidate_norm"] += float(
        scale * np.sum(np.abs(selected_candidate) ** 2 / selected_psd))
    accumulator["cross"] += complex(scale * np.sum(
        np.conj(selected_official) * selected_candidate / selected_psd))
    accumulator["bins"] += int(np.count_nonzero(mask))


def finish_metrics(accumulator):
    """Convert accumulated inner products to SNRs and direct mismatch."""
    official_norm = accumulator["official_norm"]
    candidate_norm = accumulator["candidate_norm"]
    if official_norm <= 0 or candidate_norm <= 0:
        return {"official_snr": 0.0, "candidate_snr": 0.0,
                "mismatch": np.nan, "bins": accumulator["bins"]}
    match = abs(accumulator["cross"]) / np.sqrt(
        official_norm * candidate_norm)
    # Roundoff can put an exactly identical pair a few ulps above one.
    match = min(float(match), 1.0)
    return {
        "official_snr": float(np.sqrt(official_norm)),
        "candidate_snr": float(np.sqrt(candidate_norm)),
        "mismatch": 1.0 - match,
        "bins": accumulator["bins"],
    }


def _observation_metadata(handle):
    rows = handle["sky/cat/sobhb"][:].reshape(-1)
    first_name = catalogue_parameters(rows[0])["Name"]
    dataset = handle[f"noisefree/tdi/sobhb/{first_name}"]
    delta_t = float(dataset.attrs["dt"])
    size = int(dataset.shape[0])
    return rows, delta_t, size, delta_t * size


def validate_source(handle, entry, response, band, delta_t, size, duration,
                    passband, acc_noise, oms_noise):
    """Compare one projected source with the official product in two bands."""
    name = entry["parameters"]["Name"]
    delta_f = 1.0 / duration
    lower = max(delta_f, band[0] - SPECTRAL_PADDING)
    upper = min(0.5 / delta_t, band[1] + SPECTRAL_PADDING)
    first = max(1, int(np.floor(lower / delta_f)))
    last = min(size // 2, int(np.ceil(upper / delta_f)))
    frequencies = np.arange(first, last + 1, dtype=float) * delta_f
    phase = 2 * np.pi * frequencies * DEFAULT_ARM_LENGTH / C_SI
    away_from_null = ((np.abs(np.sin(phase)) > NULL_SINE_MIN)
                      & (np.abs(np.sin(2 * phase)) > NULL_SINE_MIN))
    passband_mask = frequencies <= min(passband, upper)
    transition_mask = frequencies <= upper
    accumulators = {
        "passband": _empty_accumulator(),
        "through_nyquist": _empty_accumulator(),
    }
    dataset = handle[f"noisefree/tdi/sobhb/{name}"]

    for channel in "AET":
        candidate = response.frequency_samples(
            frequencies,
            delta_f,
            channels=channel,
            epoch=0.0,
            spectral_padding=SPECTRAL_PADDING,
            chunk_size=131072,
            direct_chunk_size=65536,
        )[channel]
        official_time = official_aet_channel(dataset, channel)
        official = delta_t * rfft(
            official_time, workers=1, overwrite_x=True)[first:last + 1]
        del official_time
        psd = _noise(
            channel, last + 1, delta_f, delta_f, acc_noise, oms_noise,
        )[first:last + 1]
        valid = away_from_null & np.isfinite(psd) & (psd > 0)
        _add_inner_products(
            accumulators["passband"], official, candidate, psd,
            valid & passband_mask, delta_f,
        )
        _add_inner_products(
            accumulators["through_nyquist"], official, candidate, psd,
            valid & transition_mask, delta_f,
        )
        del candidate, official, psd
        gc.collect()

    return {
        "name": name,
        "model": entry["model"],
        "detector_frame_masses": [
            float(entry["parameters"]["Mass1"]),
            float(entry["parameters"]["Mass2"]),
        ],
        "redshift_metadata": float(entry["parameters"]["Redshift"]),
        "source_band_hz": [float(band[0]), float(band[1])],
        "passband": finish_metrics(accumulators["passband"]),
        "through_nyquist": finish_metrics(
            accumulators["through_nyquist"]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", help="Yorsh 1-b-2 training HDF5 file")
    parser.add_argument("--names", nargs="*", help="source names to run")
    parser.add_argument("--model", choices=("lal", "pyefpehm"), default="lal")
    parser.add_argument("--orbit", choices=("equal-arm", "keplerian"),
                        default="equal-arm")
    parser.add_argument("--passband", type=float, default=DEFAULT_PASSBAND)
    parser.add_argument("--acc-noise", type=float, default=2.4e-15)
    parser.add_argument("--oms-noise", type=float, default=7.9e-12)
    parser.add_argument("--accept-mismatch", type=float, default=2e-4,
                        help="maximum passband mismatch; transition is diagnostic")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    with h5py.File(args.data, "r") as handle:
        rows, delta_t, size, duration = _observation_metadata(handle)
        wanted = None if not args.names else set(args.names)
        entries = []
        for row in rows:
            parameters = catalogue_parameters(row)
            if wanted is not None and parameters["Name"] not in wanted:
                continue
            source = make_source(parameters, duration, args.model)
            entries.append({"parameters": parameters, "source": source,
                            "model": args.model})
        if not entries:
            raise ValueError("no selected sources were found in the catalogue")
        if wanted is not None:
            missing = wanted - {item["parameters"]["Name"] for item in entries}
            if missing:
                raise ValueError(
                    f"source names absent from catalogue: {sorted(missing)}")

        orbit_class = (LisaEqualArmOrbit if args.orbit == "equal-arm"
                       else LisaKeplerianOrbit)
        orbit = orbit_class(armlength=DEFAULT_ARM_LENGTH, t0=0.0)
        terms = {
            channel: PyTDICombinationAdapter(
                f"{channel}2", get_pytdi_combination(f"{channel}2"),
                delta_t=delta_t,
            ).terms()
            for channel in "XYZ"
        }
        all_terms = tuple(term for group in terms.values() for term in group)
        padding = delay_padding(
            orbit, np.asarray([0.0, duration]), all_terms,
        )
        bands = [source_band(item["source"], duration) for item in entries]
        global_lower = min(band[0] for band in bands)
        global_upper = max(band[1] for band in bands)
        edges = np.geomspace(0.995 * global_lower, 1.005 * global_upper, 17)
        overlap = np.minimum(
            0.02 * edges[1:-1],
            0.4 * np.minimum(np.diff(edges)[:-1], np.diff(edges)[1:]),
        )
        first = entries[0]
        prepared = prepare_multiband_tdi(
            first["source"],
            orbit,
            terms,
            float(first["parameters"]["EclipticLongitude"]),
            float(first["parameters"]["EclipticLatitude"]),
            edges,
            overlap=overlap,
            t_start=0.0,
            t_end=duration,
            samples_per_cycle=4,
            geometry_step=21600.0,
            minimum_grid_points=16,
            padding=padding,
            coverage_sources=[item["source"] for item in entries[1:]],
        )
        aet_matrix = np.stack(orthogonal_channels(*np.eye(3)))
        print(json.dumps({
            "event": "prepared",
            "sources": len(entries),
            "duration_s": duration,
            "delta_t_s": delta_t,
            "passband_hz": args.passband,
            "nyquist_hz": 0.5 / delta_t,
            "orbit": args.orbit,
            "seconds": time.perf_counter() - started,
        }, sort_keys=True), flush=True)

        results = []
        for entry, band in zip(entries, bands, strict=True):
            source_started = time.perf_counter()
            response = prepared.project(
                entry["source"],
                float(entry["parameters"]["EclipticLongitude"]),
                float(entry["parameters"]["EclipticLatitude"]),
                matrix=aet_matrix,
                channels=tuple("AET"),
            )
            result = validate_source(
                handle, entry, response, band, delta_t, size, duration,
                args.passband, args.acc_noise, args.oms_noise,
            )
            result["seconds"] = time.perf_counter() - source_started
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
            del response
            gc.collect()

    failures = [
        item["name"] for item in results
        if not np.isfinite(item["passband"]["mismatch"])
        or item["passband"]["mismatch"] > args.accept_mismatch
    ]
    summary = {
        "event": "summary",
        "passband_accept_mismatch": args.accept_mismatch,
        "passband_failures": failures,
        "max_passband_mismatch": max(
            item["passband"]["mismatch"] for item in results),
        "max_conditioned_full_band_mismatch": max(
            item["through_nyquist"]["mismatch"] for item in results),
        "seconds": time.perf_counter() - started,
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
