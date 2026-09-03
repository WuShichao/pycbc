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
