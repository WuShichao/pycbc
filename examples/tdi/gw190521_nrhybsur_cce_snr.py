"""NRHybSur3dq8_CCE compatibility and a two-year-window LISA AET SNR."""
import os
import sys
import time
import types

os.environ.setdefault("NO_PKGCONFIG", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

for path in (
    "/mnt/d/pycbc-tdi-probe",
    "/mnt/d/pytdi-v2.2.1",
):
    if path not in sys.path:
        sys.path.insert(0, path)

lgwa = types.ModuleType("pycbc.waveform.lgwa")
lgwa.lgwa_fd_response = lambda **kwargs: None
sys.modules[lgwa.__name__] = lgwa

import gwsurrogate  # noqa: E402
import gwsurrogate.pycbc as gws_pycbc  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

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
from pycbc.tdi.sources import ArrayWaveformSource  # noqa: E402
from pycbc.waveform import get_td_waveform  # noqa: E402
from pycbc.waveform.plugin import add_custom_waveform  # noqa: E402

YEAR = 365.25 * 86400.0
C_SI = 299792458.0
ARM_LENGTH = 2.5e9
MODEL_PATH = "/tmp/codex-gwsurrogate-models/NRHybSur3dq8_CCE.h5"
M1 = 85.0 * 1.82
M2 = 66.0 * 1.82
MTOTAL = M1 + M2
Q = M1 / M2
DISTANCE = 5300.0
INCLINATION = 1.0


def mode_sum(model, *, dt=None, times=None, mode_list=None):
    return model(
        q=Q,
        chiA0=[0.0, 0.0, 0.0],
        chiB0=[0.0, 0.0, 0.0],
        M=MTOTAL,
        dist_mpc=DISTANCE,
        f_low=0.0,
        f_ref=None,
        dt=dt,
        times=times,
        mode_list=mode_list,
        inclination=INCLINATION,
        phi_ref=0.0,
        units="mks",
    )[:2]


def cosine_fall(times, start, stop):
    out = np.ones(len(times))
    out[times >= stop] = 0.0
    active = (times > start) & (times < stop)
    x = (times[active] - start) / (stop - start)
    out[active] = 0.5 * (1 + np.cos(np.pi * x))
    return out


def resample_to(times, values, up, down, target):
    filtered = resample_poly(values, up, down, padtype="line")
    spacing = (times[1] - times[0]) * down / up
    filtered_times = times[0] + np.arange(len(filtered)) * spacing
    return np.interp(target, filtered_times, filtered.real) + 1j * np.interp(
        target, filtered_times, filtered.imag
    )


def build_lisa_band_source(model, dt=5.0):
    # Use the native grid's (2,2) phase only to identify safe taper times.
    native_t, native_modes = model(
        Q, [0, 0, 0], [0, 0, 0], f_low=0, mode_list=[(2, 2)]
    )[:2]
    m_seconds = MTOTAL * 4.925490947e-6
    native_t_seconds = native_t * m_seconds
    phase = np.unwrap(np.angle(native_modes[(2, 2)]))
    f22 = np.abs(np.gradient(phase, native_t_seconds) / (2 * np.pi))
    taper_start = native_t_seconds[np.flatnonzero(f22 >= 0.12)[0]]
    taper_stop = native_t_seconds[np.flatnonzero(f22 >= 0.18)[0]]
    # Oscillatory modes: sample at 2.5 s (Nyquist 0.2 Hz), taper only after
    # f22=0.12 Hz, low-pass and decimate to the requested 0.1-Hz LISA band.
    osc_modes = [
        (2, 2), (2, 1), (3, 3), (3, 2),
        (4, 4), (4, 3), (5, 5),
    ]
    osc_times, osc = mode_sum(model, dt=2.5, mode_list=osc_modes)
    osc *= cosine_fall(osc_times, taper_start, taper_stop)

    # The m=0 modes carry the memory.  Resolve their merger rise at 0.5 s,
    # anti-alias, and decimate by ten.  Unlike the oscillatory part, preserve
    # the final plateau.
    memory_modes = [(2, 0), (3, 0), (4, 0)]
    # gwsurrogate's coorbital-frame conversion requires (2,2) to be present
    # even when the caller only requests m=0.  Ask for it, then project only
    # the returned memory modes with the package's own exact mode-sum helper.
    mem_times, mem_dict = model(
        q=Q, chiA0=[0, 0, 0], chiB0=[0, 0, 0],
        M=MTOTAL, dist_mpc=DISTANCE, f_low=0.0, dt=0.5,
        mode_list=[(2, 2), *memory_modes], units="mks",
    )[:2]
    selected_memory = {key: mem_dict[key] for key in memory_modes}
    memory = model._mode_sum(
        selected_memory, INCLINATION, np.pi / 2, fake_neg_modes=True
    )

    t_start = max(native_t_seconds[0], osc_times[0], mem_times[0])
    t_end = min(native_t_seconds[-1], osc_times[-1], mem_times[-1])
    target = np.arange(t_start, t_end + 0.5 * dt, dt)
    osc_target = resample_to(osc_times, osc, 1, 2, target)
    osc_target[target > taper_stop] = 0.0
    memory_target = resample_to(mem_times, memory, 1, 10, target)

    return {
        "times": target,
        "total": memory_target + osc_target,
        "memory": memory_target,
        "oscillatory": osc_target,
        "taper_start": float(taper_start),
        "taper_stop": float(taper_stop),
        "f_start": float(f22[0]),
    }


def mission_source(relative_times, strain, mission_duration):
    event_zero = mission_duration / 2
    event_times = relative_times + event_zero
    # Extend constant pre/post-memory levels to the mission boundaries.  A
    # constant strain produces zero one-link Doppler response, while retaining
    # the memory plateau prevents a fictitious second step at array termination.
    times = np.concatenate(([0.0], event_times, [mission_duration]))
    values = np.concatenate(([strain[0]], strain, [strain[-1]]))
    return ArrayWaveformSource(times, values.real, values.imag), event_times


def event_aet(source, event_times, orbit, dt=5.0):
    margin = 512
    start = max(0.0, event_times[0] - margin * dt)
    stop = min(2 * YEAR, event_times[-1] + margin * dt)
    times = np.arange(start, stop + 0.5 * dt, dt)
    sample = sample_constellation(times, orbit)
    geometry = link_geometry(sample, 0.9, -0.25, velocity_order=1)
    links = link_response(source, sample, geometry)
    aet = combine_links(
        links, sample, channels="AET", generation=1,
        interpolation_order=31, delay_order=5,
    )
    arrays = {name: np.asarray(value) for name, value in aet.items()}
    edge = 256
    edge_scale = max(
        np.max(np.abs(value[:edge])) for value in arrays.values()
    ) + max(np.max(np.abs(value[-edge:])) for value in arrays.values())
    peak = max(np.max(np.abs(value)) for value in arrays.values())
    if edge_scale > 1e-8 * peak:
        raise AssertionError("event response window does not have quiet edges")
    return times, arrays


def full_mission_aet(sources, orbit, mission_size, dt=5.0, chunk=131072):
    overlap = 128
    paths = {
        (component, name): f"/tmp/nrhybsur_cce_{component}_{name}.dat"
        for component in sources for name in "AET"
    }
    arrays = {
        key: np.memmap(path, dtype="float64", mode="w+", shape=(mission_size,))
        for key, path in paths.items()
    }
    for start in range(0, mission_size, chunk):
        stop = min(mission_size, start + chunk)
        times = np.arange(start - overlap, stop + overlap, dtype=float) * dt
        sample = sample_constellation(times, orbit)
        geometry = link_geometry(sample, 0.9, -0.25, velocity_order=1)
        for component, source in sources.items():
            links = link_response(source, sample, geometry)
            aet = combine_links(
                links, sample, channels="AET", generation=1,
                interpolation_order=31, delay_order=5,
            )
            width = stop - start
            for name, series in aet.items():
                arrays[(component, name)][start:stop] = np.asarray(series)[
                    overlap : overlap + width
                ]
        if start == 0 or stop == mission_size or (start // chunk) % 10 == 0:
            print(f"processed {stop}/{mission_size} CCE mission samples", flush=True)
    for array in arrays.values():
        array.flush()
    return arrays


def channel_snr_full(values, mission_size, dt, kind):
    if len(values) != mission_size:
        raise ValueError("channel must span the full mission")
    # The one-day edge taper removes transform leakage at the observation
    # boundaries without tapering the mid-mission merger/memory step.
    edge = int(round(86400 / dt))
    window = np.ones(mission_size)
    x = np.arange(edge) / edge
    ramp = 0.5 * (1 - np.cos(np.pi * x))
    window[:edge] = ramp
    window[-edge:] = ramp[::-1]
    hf = dt * np.fft.rfft(np.asarray(values) * window)
    delta_f = 1 / (mission_size * dt)
    length = len(hf)
    psd_func = (
        analytical_psd_lisa_tdi_AE if kind == "AE"
        else analytical_psd_lisa_tdi_T
    )
    psd = np.asarray(psd_func(length, delta_f, delta_f, tdi="1.5"))
    f = np.arange(length) * delta_f
    transfer = np.abs(np.sin(2 * np.pi * f * ARM_LENGTH / C_SI))
    valid = (
        (f >= 1e-5) & np.isfinite(psd) & (psd > 0) & (transfer > 1e-6)
    )
    rho2 = 4 * delta_f * np.sum(np.abs(hf[valid]) ** 2 / psd[valid])
    return float(np.sqrt(rho2))


def pycbc_plugin_smoke(model):
    # gwsurrogate 1.1.9 does not advertise CCE as an entry point even though
    # its generic PyCBC generator supports it. Register the generic generator
    # explicitly and seed its cache with the already loaded official model.
    name = "NRHybSur3dq8_CCE"
    approximant = f"GWS-{name}"
    gws_pycbc._cached_models[name] = model
    add_custom_waveform(
        approximant, gws_pycbc.gws_td_gen, "time", force=True
    )
    hp, hc = get_td_waveform(
        approximant=approximant,
        mass1=M1,
        mass2=M2,
        spin1x=0,
        spin1y=0,
        spin1z=0,
        spin2x=0,
        spin2y=0,
        spin2z=0,
        distance=DISTANCE,
        inclination=INCLINATION,
        coa_phase=0,
        f_lower=0,
        f_ref=0,
        delta_t=5.0,
    )
    if not (len(hp) == len(hc) and len(hp) > 100000):
        raise AssertionError("PyCBC plugin did not return the expected TimeSeries")
    return (
        len(hp), float(hp.delta_t), float(hp.start_time),
        float(np.max(np.abs(np.asarray(hp)))),
    )


def main():
    begin = time.time()
    model = gwsurrogate.LoadSurrogate(MODEL_PATH)
    smoke = pycbc_plugin_smoke(model)
    lowband = build_lisa_band_source(model)
    orbit = LisaEqualArmOrbit(t0=0.0)
    mission_size = int(round(2 * YEAR / 5.0))

    sources = {}
    for component in ("total", "memory"):
        sources[component] = mission_source(
            lowband["times"], lowband[component], 2 * YEAR
        )[0]
    arrays = full_mission_aet(sources, orbit, mission_size)

    all_snrs = {}
    for component in ("total", "memory", "oscillatory"):
        snrs = {}
        for name in "AET":
            values = arrays[("total", name)]
            if component == "memory":
                values = arrays[("memory", name)]
            elif component == "oscillatory":
                values = arrays[("total", name)] - arrays[("memory", name)]
            snrs[name] = channel_snr_full(
                values, mission_size, 5.0, "AE" if name in "AE" else "T"
            )
        snrs["network"] = float(np.sqrt(sum(value * value for value in snrs.values())))
        all_snrs[component] = snrs

    print("NRHybSur3dq8_CCE GW190521 two-year-window LISA benchmark")
    print(f"  non-base interpreter: {sys.executable}")
    print(f"  model: {MODEL_PATH}; md5 catalog=58fa10c2b35d37d0269f9e4b7157c23a")
    print(f"  detector masses: {M1:.6f}, {M2:.6f} Msun; distance={DISTANCE} Mpc")
    print(f"  q={Q:.9f}; aligned zero spins; inclination={INCLINATION}")
    print(f"  model support: {lowband['times'][0]:.3f} to {lowband['times'][-1]:.3f} s")
    print(f"  model duration: {lowband['times'][-1]-lowband['times'][0]:.3f} s")
    print(f"  earliest model f22: {lowband['f_start']:.9g} Hz")
    print(f"  oscillatory taper: {lowband['taper_start']:.3f} to {lowband['taper_stop']:.3f} s")
    print(f"  mission window: {2*YEAR:.1f} s, dt=5 s, event at midpoint")
    print(f"  PyCBC plugin smoke (N, dt, epoch, max|hp|): {smoke}")
    for component, snrs in all_snrs.items():
        print(f"  {component}: " + ", ".join(f"rho_{k}={v:.9g}" for k, v in snrs.items()))
    print(f"  elapsed={time.time()-begin:.3f} s")


if __name__ == "__main__":
    main()
