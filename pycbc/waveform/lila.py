# Copyright (C) 2026  Chayan Chatterjee
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
r"""
Frequency-domain, stationary-phase-approximation (SPA) response for the
LILA Observatory, registered as the `'LILA_response'`
`pycbc.waveform.fd_det`/`fd_det_sequence` plugin. This is what lets
`pycbc.inference.models.relbin.Relative` use LILA data: that model
dispatches purely on `approximant in fd_det_sequence`, so registering
this plugin is sufficient and no LILA-specific code is needed there.

Structurally this follows `pycbc.waveform.lgwa`, and for the same
reasons: relative binning evaluates the waveform at sparse, non-uniform
frequency points, and mid-band sources are in band for hours to years, so
"generate a time-domain waveform, project it, then FFT" is not viable.
Instead the intrinsic waveform is generated directly at the requested
frequencies, mode by mode, and the detector response is applied as a
per-frequency factor at each mode's own stationary-phase arrival time.

Two things differ from the LGWA implementation.

1.  **The propagation delay is evaluated per frequency.** `lgwa.py`
    applies a single constant light-travel offset, justified there on the
    grounds that lunar libration changes only over days. That reasoning
    does not hold: the Moon's barycentric motion is dominated by the
    Earth's ~30 km/s orbit about the SSB, not by libration, so the delay
    drifts by ~44 ms across an hour of signal and ~1 s across a day --
    about ten cycles at 10 Hz. Under the SPA the fix is free, since
    :math:`t_{\ell m}(f)` is computed anyway: the delay is evaluated at
    :math:`t(f)` rather than at a single reference epoch. See
    `_LILA_detector.vertex_delays`.

2.  **Three channels, not two.** LILA's GDC is a triangle of
    interferometers, so `project` returns `LILA_1/2/3`; the LDC baselines
    are available through `channel_set='LDC'`.

The returned waveform is anchored so that `tc` is the arrival time at
`ref_position`, a fixed (SSB-comoving) reference point -- see
`pycbc.coordinates.reference_point` for why a configurable reference
point matters for multiband sampling efficiency.
"""

import numpy as np

from pycbc.coordinates import moon as _moon
from pycbc.coordinates import reference_point as _refpt
from pycbc.detector.space import _LILA_detector
from pycbc.types import Array, FrequencySeries
from pycbc.waveform.waveform import get_fd_waveform_sequence

__all__ = ["lila_fd_response"]

# G * M_sun / c**3 in seconds, used only to pick a safe finite-difference
# step for the local stationary-phase time estimate below.
_SUN_MASS_SECONDS = 4.925490947e-6

# Cache of _LILA_detector instances keyed by the static parameters that
# determine the expensive lunarsky geometry grid. Repeated likelihood
# calls in one process then reuse both the detector and its internal
# coarse-grid cache; rebuilding it per call would dominate the runtime.
_lila_detector_cache = {}


def _get_cached_detector(longitude_site, latitude_site, orientation,
                         arm_length, cadence):
    key = (float(longitude_site), float(latitude_site), float(orientation),
           float(arm_length), float(cadence))
    det = _lila_detector_cache.get(key)
    if det is None:
        det = _LILA_detector(
            "LILA",
            longitude_site=longitude_site,
            latitude_site=latitude_site,
            orientation=orientation,
            arm_length=arm_length,
            cadence=cadence,
        )
        _lila_detector_cache[key] = det
    return det


def _time_of_f_local(fm, base_approximant, mode, cbc_params):
    """
    Per-frequency, unwrap-free stationary-phase time estimate
    :math:`t_{\\ell m}(f)` for one mode, relative to the model's own
    coalescence reference (negative before merger).

    This is the same technique `pycbc.waveform.lgwa` uses, and is
    detector-independent: the time-frequency relation belongs to the
    waveform, not the instrument. It is duplicated here rather than
    imported so the two backends can be reviewed separately; if both are
    accepted it should be lifted into a shared module.

    `numpy.unwrap` is avoided deliberately -- it breaks silently at the
    low frequencies typical of mid-band sources. The step size is scaled
    with the Newtonian chirp-time relation only to keep the phase change
    across it well under 2*pi, not as an estimate of t(f) itself, which
    comes from the waveform's own phase.
    """
    ell, m = mode
    fm = np.asarray(fm, dtype=float)
    mass1 = float(cbc_params["mass1"])
    mass2 = float(cbc_params["mass2"])
    mchirp_sec = ((mass1 * mass2) ** 0.6 / (mass1 + mass2) ** 0.2
                  * _SUN_MASS_SECONDS)

    f_22_equiv = np.maximum(np.abs(fm) * 2.0 / max(abs(m), 1), 1e-8)
    t_est = (5.0 / (256.0 * np.pi ** (8.0 / 3.0))
             / mchirp_sec ** (5.0 / 3.0) / f_22_equiv ** (8.0 / 3.0))
    eps = np.minimum(0.01 / (2 * np.pi * t_est), fm * 1e-4)
    eps = np.maximum(eps, fm * 1e-10)

    dphase = None
    for _ in range(15):
        hp_minus, _ = get_fd_waveform_sequence(
            approximant=base_approximant, mode_array=[[ell, m]],
            sample_points=Array(fm - eps), **cbc_params)
        hp_plus, _ = get_fd_waveform_sequence(
            approximant=base_approximant, mode_array=[[ell, m]],
            sample_points=Array(fm + eps), **cbc_params)
        dphase = np.angle(hp_plus.numpy() * np.conj(hp_minus.numpy()))
        bad = np.abs(dphase) > np.pi / 2
        if not np.any(bad):
            break
        eps = np.where(bad, eps / 2, eps)

    # LALSimulation/pycbc use h(f) = A(f) exp(-i Psi(f)), the opposite
    # phase sign from the t = dPsi/df/(2 pi) convention, so negate to get
    # t_rel negative before merger.
    return -dphase / (2 * eps) / (2 * np.pi)


def _lila_response_core(fm, base_approximant, mode_array, ref_position,
                        longitude_site, latitude_site, orientation,
                        arm_length, cadence, tc, lamb, beta,
                        polarization_ref, channel_set, cbc_params):
    """
    Shared core for the equal-spaced-grid and arbitrary-sample-points
    entry points. Returns a dict of channel name -> complex numpy array
    evaluated at frequencies `fm`.
    """
    ref_position = np.asarray(ref_position, dtype=float)

    # Ref frame -> SSB: sky position and polarization pass through
    # unchanged (the rotation is the identity); only arrival time moves,
    # by a closed-form light-travel correction needing no root-finding.
    t_ssb_merger = _refpt.t_ssb_from_t_ref(tc, lamb, beta, ref_position)

    # SSB ecliptic -> LAL-convention ICRS ra/dec/psi. LAL convention here,
    # unlike the LGWA backend: these response tensors come from PyCBC's
    # own antenna-pattern contraction, which *is* the LAL convention.
    _, ra, dec, psi = _moon.moon_to_geo(
        t_moon=0.0, longitude_moon=lamb, latitude_moon=beta,
        polarization_moon=polarization_ref, lal_convention=True)

    det = _get_cached_detector(longitude_site, latitude_site, orientation,
                               arm_length, cadence)

    requested = channel_set.upper()
    if requested not in ("GDC", "LDC", "ALL"):
        raise ValueError("channel_set must be 'GDC', 'LDC' or 'ALL'")

    # Each entry: (channel names, pattern function, delay-weight map).
    # A GDC vertex sits at one point, so it takes that vertex's delay. An
    # LDC baseline spans two vertices, so it takes their mean: the
    # measured quantity is the differential motion of the two ends, whose
    # natural timing reference is the midpoint. The two ends differ by at
    # most L/c = 133 us, so the choice matters only at that level, but the
    # midpoint is the defensible one.
    sets = []
    if requested in ("GDC", "ALL"):
        sets.append((det.channels, det.antenna_pattern,
                     {c: (c,) for c in det.channels}))
    if requested in ("LDC", "ALL"):
        verts = det.channels
        sets.append((det.ldc_channels, det.ldc_antenna_pattern,
                     {c: (verts[i], verts[(i + 1) % 3])
                      for i, c in enumerate(det.ldc_channels)}))

    modes = mode_array if mode_array else [(2, 2)]
    all_channels = [c for chans, _, _ in sets for c in chans]
    totals = {c: np.zeros(len(fm), dtype=complex) for c in all_channels}

    for ell, m in modes:
        hp_lm, hc_lm = get_fd_waveform_sequence(
            approximant=base_approximant, mode_array=[[ell, m]],
            sample_points=Array(fm), **cbc_params)

        # Each (l, m) harmonic reaches a given frequency at a different
        # time, so the response is evaluated per mode at its own arrival
        # time rather than once on the summed waveform.
        t_rel = _time_of_f_local(fm, base_approximant, (ell, m), cbc_params)
        t_det = t_ssb_merger + t_rel

        delays = det.vertex_delays(lamb, beta, t_det)

        # Anchor the waveform's coalescence reference, which
        # get_fd_waveform_sequence places at t=0, to the SSB merger time.
        shift = np.exp(-2j * np.pi * fm * t_ssb_merger)
        hp_arr = hp_lm.numpy() * shift
        hc_arr = hc_lm.numpy() * shift

        for chans, pattern_fn, delay_map in sets:
            patterns = pattern_fn(ra, dec, psi, t_det)
            for chan in chans:
                fplus, fcross = patterns[chan]
                # A time delay is an exact phase factor in the frequency
                # domain; under the SPA a time-*varying* delay is applied
                # by evaluating it at the epoch each frequency
                # corresponds to.
                refs = delay_map[chan]
                delay = sum(delays[r] for r in refs) / len(refs)
                phase = np.exp(-2j * np.pi * fm * delay)
                totals[chan] += phase * (hp_arr * fplus
                                         + hc_arr * fcross)

    return {"LILA_" + c: totals[c] for c in all_channels}


def lila_fd_response(ifos=None, sample_points=None, delta_f=None,
                     f_final=None, f_lower=None, approximant=None,
                     base_approximant=None, mode_array=None,
                     ref_position=(0.0, 0.0, 0.0),
                     longitude_site=None, latitude_site=None,
                     orientation=0.0, arm_length=4.0e4, cadence=3600.0,
                     channel_set="GDC", tc=0.0, eclipticlongitude=0.0,
                     eclipticlatitude=0.0, polarization=0.0, **params):
    """
    Registered as both `pycbc.waveform.fd_det` (equal-spaced grid, via
    `delta_f`/`f_final`) and `pycbc.waveform.fd_det_sequence` (arbitrary
    `sample_points`), under the approximant name `'LILA_response'`.

    Parameters
    ----------
    base_approximant : str
        Any PyCBC/LAL frequency-domain approximant for the intrinsic
        waveform, e.g. 'IMRPhenomXHM'. Required.
    mode_array : list of (l, m) or None
        Modes to include, each evaluated at its own arrival time.
    longitude_site, latitude_site : float
        Selenodetic longitude/latitude of the triangle *centroid*, in
        radians. Required.
    orientation : float
        Bearing of the first vertex from the centroid, in radians.
    arm_length : float
        Triangle side length in metres; 40 km for LILA Observatory.
    channel_set : {'GDC', 'LDC', 'ALL'}
        Which channels to return: the three GDC vertex
        interferometers, the three LDC baselines, or all six. 'ALL' is
        what a joint two-channel analysis wants, and costs little extra
        since the intrinsic waveform and the stationary-phase times are
        computed once and shared.

    Returns
    -------
    dict of pycbc.types.FrequencySeries
        Keyed 'LILA_1'/'LILA_2'/'LILA_3' for the GDC and
        'LILA_LDC_1'/... for the LDC. Note the two sets have different
        noise spectra, so each must be paired with its own PSD.
    """
    if base_approximant is None:
        raise ValueError("base_approximant is required")
    if longitude_site is None or latitude_site is None:
        raise ValueError(
            "longitude_site and latitude_site (radians, the triangle "
            "centroid) are required: antenna-pattern geometry needs an "
            "actual oriented site on the Moon.")

    cbc_params = {k: v for k, v in params.items()
                  if k not in ("approximant", "base_approximant")}

    # When specific channels are requested, infer which sets are needed
    # from their names rather than relying on channel_set being threaded
    # through. relbin asks for one ifo at a time via
    # get_fd_det_waveform_sequence(ifos=...), with only the model's
    # static_params to go on, so requiring channel_set there would be an
    # easy and silent way to get a KeyError for the LDC channels.
    if ifos is not None:
        wanted = [ifos] if isinstance(ifos, str) else list(ifos)
        has_ldc = any('LDC' in str(name) for name in wanted)
        has_gdc = any('LDC' not in str(name) for name in wanted)
        if has_ldc and has_gdc:
            channel_set = 'ALL'
        elif has_ldc:
            channel_set = 'LDC'
        elif has_gdc:
            channel_set = 'GDC'
    # f_lower/f_ref are named arguments here, so they never reach
    # **params; the intrinsic waveform still needs them.
    cbc_params["f_lower"] = f_lower
    cbc_params.setdefault("f_ref", f_lower)

    if sample_points is not None:
        fm = np.asarray(sample_points, dtype=float)
        equal_spaced = False
    else:
        if delta_f is None or f_final is None:
            raise ValueError(
                "either sample_points, or both delta_f and f_final, "
                "must be given")
        kmin = int(f_lower / delta_f) if f_lower else 0
        kmax = int(f_final / delta_f) + 1
        fm = np.arange(kmin, kmax) * delta_f
        fm[fm == 0.0] = delta_f * 1e-6     # avoid a zero-frequency sample
        equal_spaced = True

    chans = _lila_response_core(
        fm, base_approximant, mode_array, ref_position, longitude_site,
        latitude_site, orientation, arm_length, cadence, tc,
        eclipticlongitude, eclipticlatitude, polarization, channel_set,
        cbc_params)

    if ifos is not None:
        wanted = [ifos] if isinstance(ifos, str) else list(ifos)
        chans = {k: v for k, v in chans.items() if k in wanted}

    out = {}
    for name, arr in chans.items():
        if equal_spaced:
            # Length int(f_final/delta_f)+1, matching the convention of
            # pycbc.waveform.lgwa: the series runs from DC to f_final,
            # not to Nyquist, so a PSD paired with it must be built to
            # the same length.
            full = np.zeros(kmax, dtype=complex)
            full[kmin:kmax] = arr
            out[name] = FrequencySeries(full, delta_f=delta_f, copy=False)
        else:
            # Sequence path returns Array, not FrequencySeries: the
            # sample points are arbitrary, so no single delta_f describes
            # them, and this is what relbin's
            # get_fd_det_waveform_sequence expects.
            out[name] = Array(arr)
    return out


lila_fd_response.required = [
    "f_lower",
    "approximant",
    "base_approximant",
    "mass1",
    "mass2",
    "tc",
    "eclipticlongitude",
    "eclipticlatitude",
    "polarization",
    "longitude_site",
    "latitude_site",
]
