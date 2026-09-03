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
