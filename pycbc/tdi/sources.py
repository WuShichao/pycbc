# Copyright (C) 2026  Shichao Wu, Alex Nitz
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Waveform-source interfaces used by the time-domain TDI response."""

from typing import Protocol

import numpy as np


class WaveformSource(Protocol):
    """A source able to evaluate both GW polarizations at arbitrary times."""

    def polarizations(self, t):
        """Return ``(h_plus, h_cross)`` evaluated at ``t``."""
        ...


class ArrayWaveformSource:
    """Linearly interpolate sampled polarizations onto arbitrary times.

    Values requested outside the supplied time range are zero. This makes the
    required waveform padding explicit: callers that need physical data at a
    retarded time must include it in ``times`` rather than relying on spline
    extrapolation.

    Parameters
    ----------
    times : array-like
        Strictly increasing sample times in seconds.
    h_plus, h_cross : array-like
        Polarizations sampled at ``times``.
    """

    def __init__(self, times, h_plus, h_cross):
        self.times = np.asarray(times, dtype=float)
        self.h_plus = np.asarray(h_plus)
        self.h_cross = np.asarray(h_cross)
        if self.times.ndim != 1:
            raise ValueError("times must be one-dimensional")
        if len(self.times) < 2 or np.any(np.diff(self.times) <= 0):
            raise ValueError("times must contain at least two increasing samples")
        if self.h_plus.shape != self.times.shape:
            raise ValueError("h_plus must have the same shape as times")
        if self.h_cross.shape != self.times.shape:
            raise ValueError("h_cross must have the same shape as times")

    @classmethod
    def from_timeseries(cls, h_plus, h_cross):
        """Construct from two uniformly sampled PyCBC ``TimeSeries`` objects."""
        if len(h_plus) != len(h_cross):
            raise ValueError("h_plus and h_cross must have the same length")
        if h_plus.delta_t != h_cross.delta_t:
            raise ValueError("h_plus and h_cross must have the same delta_t")
        if h_plus.start_time != h_cross.start_time:
            raise ValueError("h_plus and h_cross must have the same epoch")
        return cls(
            np.asarray(h_plus.sample_times, dtype=float),
            np.asarray(h_plus),
            np.asarray(h_cross),
        )

    @staticmethod
    def _interpolate(times, values, query):
        shape = np.shape(query)
        flat = np.asarray(query, dtype=float).reshape(-1)
        if np.iscomplexobj(values):
            out = np.interp(flat, times, values.real, left=0.0, right=0.0)
            out = out + 1j * np.interp(
                flat, times, values.imag, left=0.0, right=0.0
            )
        else:
            out = np.interp(flat, times, values, left=0.0, right=0.0)
        return out.reshape(shape)

    def polarizations(self, t):
        """Return linearly interpolated polarizations with the shape of ``t``."""
        return (
            self._interpolate(self.times, self.h_plus, t),
            self._interpolate(self.times, self.h_cross, t),
        )


class NewtonianChirp:
    """Reference `HarmonicSource`: a Newtonian point-particle inspiral.

    Exists so the sparse evaluator can be exercised and benchmarked without
    pulling in a full waveform model. The phase is the closed-form

        Phi(t) = Phi_c - 2 [ (t_c - t) / (5 tau_c) ]^(5/8),  tau_c = G Mc / c^3

    so amplitude and phase are available at ARBITRARY times, which is what the
    sparse path needs -- it queries the waveform at retarded times that are not
    on any grid.

    Parameters
    ----------
    chirp_mass : float
        Detector-frame chirp mass in solar masses.
    t_coalescence : float
        Coalescence time in seconds, on the same clock as the orbit.
    amplitude : float, optional
        Overall scale; the harmonic carries the usual omega^(2/3) growth.
    inclination : float, optional
        Sets the cross/plus ratio.
    """

    TAU_SUN = 4.925490947e-6  # G M_sun / c^3 [s]

    def __init__(self, chirp_mass, t_coalescence, amplitude=1e-21,
                 inclination=0.0):
        self.tau_c = float(chirp_mass) * self.TAU_SUN
        self.t_coalescence = float(t_coalescence)
        self.scale = float(amplitude)
        cosi = np.cos(inclination)
        self.plus_factor = 0.5 * (1 + cosi ** 2)
        self.cross_factor = cosi
        self.harmonics = (2,)

    def _tau(self, t):
        # Clip just before merger: the Newtonian phase diverges at t_c.
        return np.maximum(self.t_coalescence - np.asarray(t, dtype=float),
                          10 * self.tau_c)

    def carrier_phase(self, harmonic, t):
        return -2.0 * (self._tau(t) / (5 * self.tau_c)) ** 0.625

    def angular_frequency(self, harmonic, t):
        tau = self._tau(t)
        return (1.0 / (4 * self.tau_c)) * (tau / (5 * self.tau_c)) ** -0.375

    def amplitude(self, harmonic, t):
        omega = self.angular_frequency(harmonic, t)
        common = self.scale * (self.tau_c * omega) ** (2 / 3)
        return common * self.plus_factor, 1j * common * self.cross_factor

    def polarizations(self, t):
        """Dense-path interface: the real strain at ``t``."""
        amp_p, amp_c = self.amplitude(2, t)
        carrier = np.exp(1j * self.carrier_phase(2, t))
        return np.real(amp_p * carrier), np.real(amp_c * carrier)


class PyEFPEHMSource:
    """`HarmonicSource` over pyEFPEHM's co-precessing (l, m, n) harmonics.

    pyEFPEHM fits the sparse evaluator unusually well: ``generate_tdomain_hlm_modes``
    takes an arbitrary time array, which is what the retarded queries need, and
    ``return_waveform_pieces=True`` hands back ``phase``, ``omega`` and
    ``DomegaDt`` directly, so nothing has to be recovered by unwrapping
    ``arg(hlm)``.

    Grouping. Each co-precessing ``(l, m, n)`` harmonic contributes to every
    inertial ``mp = -l..l``, so a precessing, eccentric configuration reaches
    ~100 pieces. But the ``mp`` projector is a constant at fixed viewing
    angles, and all of a harmonic's ``mp`` columns share one carrier phase, so
    summing over ``mp`` first collapses those to ~16 harmonics -- a 6x saving
    on the sparse cost, with no approximation.

    Convention. This uses the time-domain contract only: ``h = Re[hlm . Proj]``
    with the projector NOT conjugated. The frequency-domain contract conjugates
    it, and pyEFPEHM's own docstrings are explicit that the two must not be
    mixed. The slowly varying amplitude handed to the evaluator is therefore

        A = (hlm . Proj) exp(-i phase),      h = Re[A exp(i phase)]

    Validity windows. Harmonics do not all cover the same times -- pyEFPEHM
    returns ``time_idxs`` per mode -- so queries outside a harmonic's window
    return zero amplitude rather than an extrapolation.

    Parameters
    ----------
    parameters : dict
        Passed to ``pyEFPE``; ``Compute_hlm_Modes=True`` is forced.
    theta, phi : float, optional
        Viewing angles for the projector. Default: the model's own.
    """

    def __init__(self, parameters, theta=None, phi=None):
        from pyEFPEHM.waveform.EFPE import pyEFPE
        parameters = dict(parameters)
        parameters['Compute_hlm_Modes'] = True
        self.model = pyEFPE(parameters)
        self.theta, self.phi = theta, phi
        self.t_start = float(self.model.sol.all_ts[0])
        self.t_end = float(self.model.sol.all_ts[-1])
        probe = np.linspace(self.t_start + 1.0, self.t_end - 1.0, 64)
        self.harmonics = tuple(
            self.model.generate_tdomain_hlm_modes(times=probe)['modes'])
        self._cache = {}

    def _projector(self, l, m):
        from pyEFPEHM.waveform.EFPE import compute_m2_Ylm
        key = ('proj', l, abs(m))
        if key not in self._cache:
            theta = (np.arccos(self.model.cos_theta_JN) if self.theta is None
                     else self.theta)
            phi = self.model.phi_JN if self.phi is None else self.phi
            ylm = compute_m2_Ylm(np.cos(theta), phi, l_array=np.array([l]))[0]
            mirrored = np.conj(ylm[::-1])
            if abs(m) % 2 == 0:
                mirrored[1::2] = -mirrored[1::2]
            else:
                mirrored[::2] = -mirrored[::2]
            self._cache[key] = np.transpose(
                [0.5 * (ylm + mirrored), -0.5j * (ylm - mirrored)])
        return self._cache[key]

    def _evaluate(self, harmonic, t):
        """(A_plus, A_cross, phase, omega) at arbitrary, possibly 2-D ``t``."""
        query = np.asarray(t, dtype=float)
        key = ('eval', harmonic, query.shape, query.ctypes.data,
               float(query.flat[0]), float(query.flat[-1]))
        if key in self._cache:
            return self._cache[key]

        flat = query.reshape(-1)
        order = np.argsort(flat)
        inside = ((flat[order] >= self.t_start) & (flat[order] <= self.t_end))
        amp_p = np.zeros(flat.size, dtype=complex)
        amp_c = np.zeros(flat.size, dtype=complex)
        phase = np.zeros(flat.size)
        omega = np.full(flat.size, np.nan)

        if np.any(inside):
            times = flat[order][inside]
            result = self.model.generate_tdomain_hlm_modes(
                times=times, return_waveform_pieces=True)
            mode = result['modes'].get(harmonic)
            if mode is not None:
                l, m, _ = harmonic
                contribution = np.tensordot(
                    np.asarray(mode['hlm']), self._projector(l, m),
                    axes=(1, 0))                        # (N_mode, 2), complex
                mode_phase = np.asarray(mode['phase'])
                carrier = np.exp(-1j * mode_phase)
                # place back into the sorted-inside slots this mode covers
                slot = np.nonzero(inside)[0][np.asarray(mode['time_idxs'])]
                target = order[slot]
                amp_p[target] = contribution[:, 0] * carrier
                amp_c[target] = contribution[:, 1] * carrier
                phase[target] = mode_phase
                omega[target] = np.asarray(mode['omega'])

        # a harmonic's phase must stay continuous where it is not defined,
        # otherwise exp(i * (phase - phase0)) jumps at the window edge
        if np.any(np.isnan(omega)):
            valid = ~np.isnan(omega)
            if np.any(valid):
                omega[~valid] = np.interp(flat[~valid], flat[valid],
                                          omega[valid])
            else:
                omega[:] = 1.0
        out = tuple(a.reshape(query.shape)
                    for a in (amp_p, amp_c, phase, omega))
        self._cache = {k: v for k, v in self._cache.items()
                       if k[0] in ('proj', 'support')}
        self._cache[key] = out
        return out

    def support(self, harmonic):
        """The time window over which this harmonic is actually defined.

        Harmonics do not all cover the mission: pyEFPEHM returns a per-mode
        ``time_idxs``. Outside its window a harmonic's carrier phase is zero,
        so a grid builder that inverts the phase without knowing the window
        puts all of its points in the flat stretch and resolves nothing.
        """
        key = ('support', harmonic)
        if key not in self._cache:
            probe = np.linspace(self.t_start, self.t_end, 2048)
            _, _, phase, _ = self._evaluate(harmonic, probe)
            live = np.nonzero(phase != 0.0)[0]
            self._cache[key] = ((self.t_start, self.t_end) if live.size == 0
                                else (float(probe[live[0]]),
                                      float(probe[live[-1]])))
        return self._cache[key]

    def amplitude(self, harmonic, t):
        amp_p, amp_c, _, _ = self._evaluate(harmonic, t)
        return amp_p, amp_c

    def carrier_phase(self, harmonic, t):
        return self._evaluate(harmonic, t)[2]

    def angular_frequency(self, harmonic, t):
        return self._evaluate(harmonic, t)[3]

    def polarizations(self, t):
        """Dense-path interface: sum every harmonic's real part."""
        total_p = np.zeros(np.shape(t))
        total_c = np.zeros(np.shape(t))
        for harmonic in self.harmonics:
            amp_p, amp_c, phase, _ = self._evaluate(harmonic, t)
            carrier = np.exp(1j * phase)
            total_p += np.real(amp_p * carrier)
            total_c += np.real(amp_c * carrier)
        return total_p, total_c
