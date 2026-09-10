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


def _windowed_derivative(x, y, window=256, degree=6):
    """``(y_smooth, dy/dx)`` from overlapping polynomial windows.

    Differentiating a spline through ``y`` fails here for a reason that has
    nothing to do with the physics: the measured stationary times carry a
    little roundoff, and once the grid is fine enough that the true change
    between knots drops below it, the derivative goes negative.  A window
    wide enough to average the noise away and short enough to follow the
    curve fixes both ends of that, and unlike one global fit it costs the
    same per knot at any bandwidth.
    """
    count = len(x)
    window = int(min(max(window, degree + 2), count))
    step = max(window // 2, 1)
    derivative = np.zeros(count)
    smoothed = np.zeros(count)
    weight = np.zeros(count)
    for start in range(0, count, step):
        stop = min(start + window, count)
        if stop - start < degree + 2:
            start = max(0, stop - window)
        piece = slice(start, stop)
        centre = 0.5 * (x[start] + x[stop - 1])
        scale = 0.5 * (x[stop - 1] - x[start]) or 1.0
        local = (x[piece] - centre) / scale
        fit = np.polynomial.polynomial.Polynomial.fit(
            local, y[piece], min(degree, stop - start - 1), domain=[-1, 1],
            window=[-1, 1])
        # a raised cosine so neighbouring windows blend instead of stepping
        taper = 0.5 * (1 - np.cos(
            2 * np.pi * (np.arange(stop - start) + 0.5) / (stop - start)))
        derivative[piece] += taper * fit.deriv()(local) / scale
        smoothed[piece] += taper * fit(local)
        weight[piece] += taper
        if stop == count:
            break
    return smoothed / weight, derivative / weight


class LALFDSource:
    r"""Any dominant-mode frequency-domain LAL waveform, for sparse LISA.

    The LISA response code needs a slowly varying complex amplitude and a
    carrier phase that can be evaluated at arbitrary retarded times.  LAL's
    IMRPhenomD interface instead returns frequency-domain polarizations.  This
    adapter connects the two with the stationary-phase map

    .. math::

        t(f) = -\frac{1}{2\pi}\frac{d\arg \tilde h}{df},\qquad
        \Phi(f) = \arg \tilde h + 2\pi f t + \frac{\pi}{4}.

    Importantly, the derivative is measured from close pairs of *native LAL*
    frequency-sequence evaluations.  It is not obtained by unwrapping a
    mission-resolution Fourier grid: for a stellar-mass LISA binary tens of
    years from merger, adjacent mission bins can differ by many phase cycles.

    This is an inverse-SPA representation of the LAL waveform, not a new
    waveform approximant.  It is appropriate for the slowly evolving inspiral
    portion of any such model -- the regime a LISA stellar-origin binary
    spends its whole observation in.

    Applicability is narrower than "a LAL waveform".  The model must be
    available through ``get_fd_waveform_sequence``, must carry the (2, 2)
    carrier alone, and must have a monotone stationary-time map over the
    requested band; the constructor raises when the last fails, which is what
    a higher-mode or precessing model does, since ``arg h_plus`` is then a sum
    of carriers rather than one. Verified against IMRPhenomD, IMRPhenomXAS and
    TaylorF2.

    Parameters
    ----------
    mass1, mass2 : float
        Detector-frame component masses in solar masses.
    f_lower : float
        GW frequency at ``t_start`` in Hz.
    duration : float, optional
        Requested time support in seconds.  The upper frequency is solved
        from the LAL phase derivative.  Exactly one of ``duration`` and
        ``f_upper`` must be supplied.
    f_upper : float, optional
        End frequency in Hz.
    n_frequency : int, optional
        Number of frequency knots, spaced geometrically.  It sets accuracy
        only: the phase is unwrapped against the measured stationary time, so
        the knot count no longer has to scale with duration times bandwidth.
        ``unwrap_margin`` reports how much of the half-cycle budget was used.
    polarization : float, optional
        Polarization rotation in radians.  Sky position belongs to the LISA
        response and is deliberately not part of this source.
    approximant : str, optional
        Any dominant-mode frequency-domain LAL approximant registered in
        ``pycbc.waveform.fd_sequence``.  Default ``IMRPhenomD``.
    waveform_options : keyword arguments
        Passed to :func:`pycbc.waveform.get_fd_waveform_sequence`;
        ``sample_points`` is supplied by the adapter and must not appear.
    """

    TAU_SUN = 4.925490947e-6

    def __init__(self, mass1, mass2, f_lower, duration=None, f_upper=None,
                 n_frequency=4096, t_start=0.0, polarization=0.0,
                 approximant="IMRPhenomD", **waveform_options):
        from scipy.interpolate import CubicSpline, PchipInterpolator

        if (duration is None) == (f_upper is None):
            raise ValueError("supply exactly one of duration and f_upper")
        if mass1 <= 0 or mass2 <= 0 or f_lower <= 0:
            raise ValueError("masses and f_lower must be positive")
        if duration is not None and duration <= 0:
            raise ValueError("duration must be positive")
        if n_frequency < 16:
            raise ValueError("n_frequency must be at least 16")

        if "sample_points" in waveform_options:
            raise ValueError("the adapter supplies sample_points")
        from pycbc.waveform import fd_sequence
        if approximant not in fd_sequence:
            raise ValueError(
                f"{approximant} has no frequency-sequence generator; the "
                "adapter needs one to place its own knots")
        self._parameters = dict(waveform_options)
        self._parameters.update(approximant=str(approximant),
                                mass1=float(mass1), mass2=float(mass2))
        self.approximant = str(approximant)
        self.f_lower = float(f_lower)
        self.t_start = float(t_start)
        self.harmonics = (2,)

        if f_upper is None:
            f_upper = self._solve_upper_frequency(float(duration))
        if f_upper <= f_lower:
            raise ValueError("f_upper must exceed f_lower")
        self.f_upper = float(f_upper)

        frequency = np.geomspace(self.f_lower, self.f_upper,
                                 int(n_frequency))
        h_plus, h_cross = self._fd_waveform(frequency)
        stationary = self._stationary_time(frequency)
        # dt/df comes from the MAIN grid, not from the local windows.  Those
        # span about 1e-10 Hz, over which the curvature contributes
        # (1/2) phi'' eps^2 ~ 2e-8 rad -- far below the roundoff of a phase
        # that is itself ~1e8 rad, so a second derivative measured there is
        # pure noise (it scatters over 1.2e12 to 4.2e12 around a true
        # 9.9e11).  Across the main grid it is smooth, but only if the grid
        # is read in windows: a plain spline derivative turns negative once
        # n_frequency is large enough for roundoff to dominate a single knot
        # spacing, which happened at 2048 knots over a three-day band.
        # The same windows also smooth t(f).  The measured times are not
        # strictly increasing once the grid is fine -- roundoff, not physics
        # -- and CubicSpline demands strictly increasing knots, so the
        # smoothed times are what the splines are built on.
        stationary, dt_df = _windowed_derivative(frequency, stationary)
        stationary_start = stationary[0]
        relative_time = stationary - stationary_start

        if np.any(~np.isfinite(dt_df)) or np.any(dt_df <= 0):
            raise ValueError("LAL stationary-time map is not monotone")
        if np.any(np.diff(relative_time) <= 0):
            raise ValueError(
                "the smoothed stationary times are still not increasing; "
                "the window is too narrow for this grid")

        # Unwrap against the measured t(f) rather than against a removed
        # global linear phase.  The old route needed the phase step to stay
        # under pi/2 after removing 2*pi*f*t_start, which costs
        # 4 * duration * bandwidth knots: fine for a narrowband LISA binary
        # (1.7e4 for a two-year 5.9 mHz source) and impossible for a
        # broadband one (1.5e7 for a stellar-mass binary that merges in
        # band, whose degree-32 Chebyshev design matrix alone is 3.8 GiB).
        # Subtracting the PREDICTED increment first leaves only the
        # prediction error to wrap, so the grid is set by the accuracy wanted
        # and not by the bandwidth.
        # The big linear term is still removed first, modulo 2*pi so that a
        # multi-decade t0 costs no precision, which keeps `phase` on the
        # contract the callers already use.  What is new is that the leftover
        # is unwrapped against the MEASURED time rather than assumed small.
        time_shift_phase = np.remainder(
            2 * np.pi * frequency * stationary_start, 2 * np.pi)
        measured = np.angle(h_plus * np.exp(1j * time_shift_phase))
        predicted = -np.pi * (relative_time[:-1] + relative_time[1:]) \
            * np.diff(frequency)
        residual = np.remainder(
            np.diff(measured) - predicted + np.pi, 2 * np.pi) - np.pi
        margin = np.max(np.abs(residual))
        if margin > 0.5 * np.pi:
            raise ValueError(
                f"the phase prediction is off by {margin:.2f} rad between "
                "knots, more than half a cycle; raise n_frequency")
        self.unwrap_margin = float(margin)
        fd_phase = np.concatenate(
            ([measured[0]], measured[0] + np.cumsum(predicted + residual)))
        # H(f) = A(t_f)/2 sqrt(dt/df)
        amplitude_plus = 2 * np.abs(h_plus) / np.sqrt(dt_df)
        ratio = np.divide(h_cross, h_plus, out=np.zeros_like(h_cross),
                          where=np.abs(h_plus) > 0)
        amplitude_cross = amplitude_plus * ratio
        angle = 2 * float(polarization)
        cpsi, spsi = np.cos(angle), np.sin(angle)
        amplitude_plus, amplitude_cross = (
            cpsi * amplitude_plus - spsi * amplitude_cross,
            spsi * amplitude_plus + cpsi * amplitude_cross,
        )

        phase = fd_phase + 2 * np.pi * frequency * relative_time
        phase += np.pi / 4

        time = relative_time + self.t_start
        self.t_end = float(time[-1])
        self.sample_frequencies = frequency
        self.stationary_times = stationary
        self._dt_df = dt_df
        self._amplitude_plus = CubicSpline(time, amplitude_plus,
                                           extrapolate=False)
        self._amplitude_cross_real = CubicSpline(
            time, amplitude_cross.real, extrapolate=False)
        self._amplitude_cross_imag = CubicSpline(
            time, amplitude_cross.imag, extrapolate=False)
        self._phase = CubicSpline(time, phase, extrapolate=True)
        self._omega = PchipInterpolator(
            time, 2 * np.pi * frequency, extrapolate=True)

    def _fd_waveform(self, frequency):
        from pycbc.waveform import get_fd_waveform_sequence
        h_plus, h_cross = get_fd_waveform_sequence(
            sample_points=np.asarray(frequency), **self._parameters)
        return np.asarray(h_plus), np.asarray(h_cross)

    def _newtonian_time(self, frequency):
        m1, m2 = self._parameters["mass1"], self._parameters["mass2"]
        total = m1 + m2
        eta = m1 * m2 / total ** 2
        chirp_mass = total * eta ** 0.6
        scale = 5 / 256 * (chirp_mass * self.TAU_SUN) ** (-5 / 3)
        return scale * (np.pi * np.asarray(frequency)) ** (-8 / 3)

    def _local_phase_derivatives(self, frequency):
        """``(t, dt/df)`` from local fits, with no global phase unwrapping.

        Nine nearby NATIVE phase values are fitted rather than two nearly
        equal complex numbers differenced.  The local span is about 2.4 rad:
        safely unwrap-able, and large enough to suppress roundoff.  Because
        every window uses the same nine offsets, rescaling to
        ``x = (f - f_i)/epsilon_i`` makes the design matrix common to all of
        them, so the whole set is one least-squares solve instead of a Python
        loop over knots.

        The second derivative it also returns is NOT usable as ``dt/df``: the
        window is far too narrow to see curvature above roundoff.  It is
        returned for diagnosis only, and the caller takes ``dt/df`` from the
        main grid instead.
        """
        frequency = np.atleast_1d(np.asarray(frequency, dtype=float))
        epsilon = 1.2 / (2 * np.pi * self._newtonian_time(frequency))
        epsilon = np.minimum(epsilon, frequency * 1e-7)
        offsets = np.linspace(-1.0, 1.0, 9)
        local_frequency = frequency[:, None] + epsilon[:, None] * offsets
        h_plus, _ = self._fd_waveform(local_frequency.reshape(-1))
        local_phase = np.unwrap(
            np.angle(h_plus.reshape(len(frequency), -1)), axis=1)
        design = np.polynomial.polynomial.polyvander(offsets, 4)
        coefficients = np.linalg.lstsq(design, local_phase.T, rcond=None)[0]
        first = coefficients[1] / epsilon
        second = 2 * coefficients[2] / epsilon ** 2
        return -first / (2 * np.pi), -second / (2 * np.pi)

    def _stationary_time(self, frequency):
        """``-d arg(H)/2pi df``; see `_local_phase_derivatives`."""
        return self._local_phase_derivatives(frequency)[0]

    def _local_curvature(self, frequency):
        """Diagnostic only -- see `_local_phase_derivatives`."""
        return self._local_phase_derivatives(frequency)[1]

    def _solve_upper_frequency(self, duration):
        from scipy.optimize import brentq

        start = self._stationary_time([self.f_lower])[0]
        chirp_time = self._newtonian_time(self.f_lower)
        if duration >= chirp_time:
            raise ValueError("requested duration reaches the Newtonian merger")
        upper = self.f_lower * (chirp_time / (chirp_time - duration)) ** 0.375

        def residual(frequency):
            return self._stationary_time([frequency])[0] - start - duration

        value = residual(upper)
        while value < 0:
            upper *= 1.25
            value = residual(upper)
        return brentq(residual, self.f_lower * (1 + 1e-12), upper,
                      xtol=np.finfo(float).eps * upper * 8, rtol=1e-12)

    def _check_harmonic(self, harmonic):
        if harmonic != 2:
            raise ValueError(
                f"{self.approximant} is used here as a single (2, +/-2) "
                "carrier; there is no other harmonic to ask for")

    def support(self, harmonic):
        self._check_harmonic(harmonic)
        return self.t_start, self.t_end

    def amplitude(self, harmonic, t):
        self._check_harmonic(harmonic)
        query = np.asarray(t, dtype=float)
        # Clip for endpoint roundoff, then restore exact zero outside support.
        # Subtracting the multi-decade stationary time to form this source's
        # local clock loses about 1e-7 s, even though the local times are only
        # days long.
        tolerance = max(1e-6, 32 * np.spacing(max(
            1.0, abs(self.t_start), abs(self.t_end))))
        inside = np.logical_and(query >= self.t_start - tolerance,
                                query <= self.t_end + tolerance)
        sample = np.clip(query, self.t_start, self.t_end)
        plus = np.where(inside, self._amplitude_plus(sample), 0.0)
        cross = np.where(inside, self._amplitude_cross_real(sample), 0.0)
        cross = cross + 1j * np.where(
            inside, self._amplitude_cross_imag(sample), 0.0)
        return plus, cross

    def carrier_phase(self, harmonic, t):
        self._check_harmonic(harmonic)
        query = np.clip(np.asarray(t, dtype=float), self.t_start, self.t_end)
        return self._phase(query)

    def angular_frequency(self, harmonic, t):
        self._check_harmonic(harmonic)
        query = np.clip(np.asarray(t, dtype=float), self.t_start, self.t_end)
        return self._omega(query)

    def polarizations(self, t):
        amplitude_plus, amplitude_cross = self.amplitude(2, t)
        carrier = np.exp(1j * self.carrier_phase(2, t))
        return (np.real(amplitude_plus * carrier),
                np.real(amplitude_cross * carrier))


def _longest_increasing_run(values):
    """``(start, length)`` of the longest strictly increasing stretch.

    ``-inf`` marks a sample as unusable: no run can cross one, because
    ``diff`` across it is not positive on at least one side.
    """
    with np.errstate(invalid='ignore'):
        rising = np.diff(values) > 0
    if not rising.any():
        return 0, 0
    edges = np.flatnonzero(np.diff(np.concatenate(
        ([False], rising, [False]))))
    starts, stops = edges[::2], edges[1::2]
    best = int(np.argmax(stops - starts))
    return int(starts[best]), int(stops[best] - starts[best] + 1)


class LALTDSource:
    r"""A dominant-mode LAL waveform through merger, from its analytic signal.

    `LALFDSource` inverts the stationary-phase map, which needs a monotone
    t(f) and therefore stops before the merger: for a 1.8e6 Msun LISA binary
    it reaches Mf = 0.0017, about ten days early, and only 26% of rho^2 lies
    below that -- half the signal-to-noise.  Merger-ringdown has no stationary
    phase to invert, so no amount of care fixes that route.

    This one takes the amplitude and phase from the time domain instead, where
    they exist throughout.  Generated at zero inclination the two polarizations
    are exactly in quadrature,

    .. math:: h_+ - i h_\times = A(t)\,e^{-i\Phi(t)},

    so A and Phi come straight from the modulus and argument, and inclination
    and polarization re-enter afterwards as the constants they are.  Any
    approximant `pycbc.waveform.get_td_waveform` accepts will do, including
    the frequency-domain ones, which it conditions into the time domain
    itself.

    The cost is that the sparse evaluator's premise -- an amplitude varying on
    the ORBITAL timescale -- fails near merger, so the grid has to densify
    there.  That is a real cost paid where the physics demands it, not an
    approximation.

    Parameters
    ----------
    delta_t : float
        Sampling of the generated waveform in seconds.  It must resolve the
        carrier: the constructor rejects a step that leaves the phase
        advancing by more than pi per sample, since the unwrap would then
        alias silently.
    t_coalescence : float, optional
        Where to place the waveform's own t = 0 (its amplitude peak, by LAL's
        convention) on the mission clock.
    polarization : float, optional
        Polarization rotation in radians.  Sky position belongs to the LISA
        response and is deliberately not part of this source.
    amplitude_floor : float, optional
        Fraction of the peak amplitude below which the waveform is treated as
        absent at either end.  ``get_td_waveform`` pads a frequency-domain
        model out to a power of two and tapers into it, so for a LISA-band
        binary most of the returned array can be padding: a 1.8e6 Msun system
        started at 5e-5 Hz has 332 days of real inspiral inside 971 days of
        array, and the padding sits at 1e-6 of the peak where an inspiral
        would be at (f_start/f_peak)^(2/3) ~ 3e-2.  The default is chosen to
        cut the former and keep the latter.
    approximant : str, optional
        Default ``IMRPhenomD``.
    check_inclination : float or None, optional
        Test inclination in radians at which the dominant-mode projection is
        verified against a real generation, or None to skip.  This is not
        optional book-keeping: generated at zero inclination a higher-mode
        model looks perfectly single-carrier, because the spin-weighted
        harmonics leave only m = +/-2 there, so smoothness proves nothing.
        What fails for such a model is the ANGULAR dependence, and the only
        way to catch it is to go off-axis and look.  Measured mismatches at
        1.0 rad: IMRPhenomD 3e-16, IMRPhenomXAS 3e-16, against IMRPhenomXHM
        2e-03 and SEOBNRv4PHM 1e-01.
    waveform_options : keyword arguments
        Passed to :func:`pycbc.waveform.get_td_waveform`; ``inclination`` is
        used for the projection and is not passed through.
    """

    def __init__(self, delta_t, inclination=0.0, t_coalescence=0.0,
                 polarization=0.0, amplitude_floor=1e-3,
                 approximant="IMRPhenomD", check_inclination=1.0,
                 tolerance=1e-6, **waveform_options):
        from scipy.interpolate import CubicSpline, PchipInterpolator
        from pycbc.waveform import get_td_waveform

        for name in ("delta_t", "approximant", "inclination"):
            if name in waveform_options:
                raise ValueError(f"{name} is a named argument here")
        self.approximant = str(approximant)
        self.harmonics = (2,)
        self.delta_t = float(delta_t)

        plus, cross = get_td_waveform(
            approximant=self.approximant, delta_t=self.delta_t,
            inclination=0.0, **waveform_options)
        analytic = np.asarray(plus) - 1j * np.asarray(cross)
        magnitude = np.abs(analytic)
        peak = magnitude.max()
        if peak <= 0:
            raise ValueError("the generated waveform is identically zero")
        phase = -np.unwrap(np.angle(analytic))

        # Keep the longest stretch that is both loud enough to be signal and
        # strictly monotone in phase.  Either test alone is too weak: the
        # padded head is monotone in a slowly drifting numerical phase, and
        # amplitude alone would keep whatever noise sits above the floor.
        loud = magnitude > float(amplitude_floor) * peak
        first, span = _longest_increasing_run(
            np.where(loud, phase, -np.inf))
        if span < 16:
            raise ValueError(
                "no usable stretch of monotone carrier phase; this source is "
                "for a single (2, +/-2) carrier, and a higher-mode or "
                "precessing model has several")
        analytic = analytic[first:first + span]
        magnitude = magnitude[first:first + span]
        phase = phase[first:first + span]
        times = np.asarray(plus.sample_times)[first:first + span]

        step = np.abs(np.diff(np.angle(analytic)))
        step = np.minimum(step, 2 * np.pi - step)
        if step.max() >= np.pi:
            raise ValueError(
                f"delta_t = {self.delta_t} leaves the carrier advancing "
                f"{step.max():.2f} rad per sample; the unwrap would alias")

        cos_inclination = np.cos(float(inclination))
        plus_factor = 0.5 * (1 + cos_inclination ** 2)
        cross_factor = -1j * cos_inclination
        angle = 2 * float(polarization)
        cos_psi, sin_psi = np.cos(angle), np.sin(angle)
        amplitude_plus = plus_factor * cos_psi - cross_factor * sin_psi
        amplitude_cross = plus_factor * sin_psi + cross_factor * cos_psi

        clock = times + float(t_coalescence)
        self.t_start, self.t_end = float(clock[0]), float(clock[-1])
        self._times, self._magnitude_samples = times, magnitude
        self._phase_samples = phase
        self._magnitude = CubicSpline(clock, magnitude, extrapolate=False)
        self._phase = CubicSpline(clock, phase, extrapolate=True)
        # Read the frequency off a monotonicity-preserving interpolant of the
        # measured samples rather than differentiating the cubic spline.  The
        # phase is monotone at the samples, but a cubic through it need not be
        # between them, and its derivative dips below zero in the faint tails
        # where the amplitude sits at the floor -- at delta_t = 1/256 that is
        # enough to hand `adaptive_time_grid` a negative frequency.  The two
        # therefore differ at interpolation order; angular_frequency is used
        # to size grids, never to build the waveform.
        self._omega = PchipInterpolator(clock, np.gradient(phase, times),
                                        extrapolate=True)
        self._amplitude_plus = amplitude_plus
        self._amplitude_cross = amplitude_cross
        if check_inclination is not None:
            self._verify_projection(float(check_inclination), float(tolerance),
                                    waveform_options)

    def _verify_projection(self, inclination, tolerance, waveform_options):
        """Is this model really one (2, +/-2) carrier off-axis as well?

        Ask first, measure second.  ``pycbc.waveform.waveform_modes`` knows
        the mode content of a few model families by citation -- and only
        those: it raises for everything else, and its own source carries a
        FIXME asking for a lalsimulation call that does not exist yet.  So
        the registry settles the cases it covers, and the measurement below
        covers the rest, including sources that are not LAL models at all.
        """
        from pycbc.waveform import get_td_waveform
        from pycbc.waveform.waveform_modes import default_modes

        try:
            modes = default_modes(self.approximant)
        except (ValueError, KeyError):
            modes = None
        if modes is not None:
            extra = sorted({(l, abs(m)) for l, m in modes} - {(2, 2)})
            if extra:
                raise ValueError(
                    f"{self.approximant} carries {extra} besides the (2, 2) "
                    "carrier, per pycbc.waveform.waveform_modes.default_modes."
                    " Higher modes and precession need a source with one "
                    "harmonic each; pass check_inclination=None only to "
                    "accept the dominant-mode approximation deliberately")

        plus, cross = get_td_waveform(
            approximant=self.approximant, delta_t=self.delta_t,
            inclination=inclination, **waveform_options)
        times = np.asarray(plus.sample_times)
        inside = (times >= self._times[0]) & (times <= self._times[-1])
        magnitude = np.interp(times[inside], self._times,
                              self._magnitude_samples)
        phase = np.interp(times[inside], self._times, self._phase_samples)
        cosine = np.cos(inclination)
        model = magnitude * (0.5 * (1 + cosine ** 2) * np.cos(phase)
                             - 1j * cosine * np.sin(phase))
        reference = np.asarray(plus)[inside] - 1j * np.asarray(cross)[inside]
        # Maximise over time and phase.  The two generations need not be
        # aligned to the sample: IMRPhenomXAS puts its two inclinations 0.98 ms
        # apart, which reads as mismatch 6e-02 unaligned and 3e-09 aligned,
        # and would otherwise be rejected as a higher-mode model. Modes shift
        # the SHAPE, so they survive the maximisation -- IMRPhenomXHM stays at
        # 1.4e-02 -- and the two cases separate by seven orders.
        count = len(model)
        norm = np.sqrt(np.sum(np.abs(model) ** 2)
                       * np.sum(np.abs(reference) ** 2))
        correlation = np.fft.ifft(np.fft.fft(model)
                                  * np.conj(np.fft.fft(reference)))
        mismatch = 1 - np.max(np.abs(correlation)) * count / norm / count
        if not mismatch < tolerance:
            raise ValueError(
                f"{self.approximant} is not a single (2, +/-2) carrier: its "
                f"own waveform at inclination {inclination} differs from the "
                f"dominant-mode projection by mismatch {mismatch:.3e}. Higher "
                "modes and precession need a source with one harmonic each; "
                "pass check_inclination=None only to accept the "
                "dominant-mode approximation deliberately")

    def _check_harmonic(self, harmonic):
        if harmonic != 2:
            raise ValueError(
                f"{self.approximant} is used here as a single (2, +/-2) "
                "carrier; there is no other harmonic to ask for")

    def support(self, harmonic):
        self._check_harmonic(harmonic)
        return self.t_start, self.t_end

    def amplitude(self, harmonic, t):
        self._check_harmonic(harmonic)
        query = np.asarray(t, dtype=float)
        inside = (query >= self.t_start) & (query <= self.t_end)
        magnitude = np.where(
            inside, self._magnitude(np.clip(query, self.t_start, self.t_end)),
            0.0)
        return (magnitude * self._amplitude_plus,
                magnitude * self._amplitude_cross)

    def carrier_phase(self, harmonic, t):
        self._check_harmonic(harmonic)
        return self._phase(np.clip(np.asarray(t, dtype=float),
                                   self.t_start, self.t_end))

    def angular_frequency(self, harmonic, t):
        self._check_harmonic(harmonic)
        return self._omega(np.clip(np.asarray(t, dtype=float),
                                   self.t_start, self.t_end))

    def polarizations(self, t):
        amplitude_plus, amplitude_cross = self.amplitude(2, t)
        carrier = np.exp(1j * self.carrier_phase(2, t))
        return (np.real(amplitude_plus * carrier),
                np.real(amplitude_cross * carrier))


class LALModesSource:
    r"""A higher-mode or precessing LAL waveform, one harmonic per ``(l, m)``.

    `LALTDSource` carries a single carrier and refuses anything else, which
    rules out exactly the models LISA analyses are moving to.  The sparse
    evaluator does not need a single carrier, though -- it needs each piece to
    be a slowly varying amplitude times its own carrier, which is what a mode
    is.  So each ``(l, m)`` becomes its own harmonic, and the response is
    applied to each and summed, as it already is for pyEFPEHM's ``(l, m, n)``.

    The decomposition is LAL's own, through
    :func:`pycbc.waveform.get_td_waveform_modes`, and the reconstruction is
    LAL's own contract: ``h = sum_lm Y_lm h_lm`` with the plus polarization
    the real part and the cross the negative imaginary part.  Writing
    ``h_lm = A_lm exp(i arg h_lm)`` and using ``Re[z] = Re[conj(z)]``, each
    mode contributes ``Re[(A_lm conj(Y_lm)) exp(i Phi_lm)]`` to the plus with
    ``Phi_lm = -arg h_lm``, and the same amplitude times ``-i`` to the cross.
    Whichever sign of ``Phi`` rises is the one kept, so ``m < 0`` modes are
    carried as readily as ``m > 0``.

    ``m = 0`` modes are dropped, and reported in `skipped`.  They are not
    oscillatory, so they have no carrier to factor out and no sparse grid can
    represent them; that is the same reason the plan sends GW memory down the
    dense path.

    Which approximants can be decomposed at all is a separate question from
    which have higher modes, and PyCBC answers only the first:
    ``td_waveform_mode_approximants()`` lists SEOBNRv4PHM, IMRPhenomTPHM,
    NRHybSur3dq8, NRSur7dq2/4 and the Taylor families, while IMRPhenomXPHM --
    which LDC is beginning to use -- is in neither mode list even though
    ``default_modes`` knows its content. Wiring that up belongs upstream in
    ``pycbc.waveform.waveform_modes``, not here.

    Parameters
    ----------
    delta_t : float
        Sampling of the generated modes in seconds.
    inclination, coa_phase : float
        Viewing angles for the spin-weighted harmonics.
    t_coalescence : float, optional
        Where the modes' own t = 0 sits on the mission clock.
    polarization : float, optional
        Polarization rotation in radians.
    mode_array : sequence of (l, m), optional
        Restrict to these modes.  Default: everything the model returns.
    amplitude_floor : float, optional
        Fraction of a MODE's own peak below which it is treated as absent.
    approximant : str, optional
        Default ``SEOBNRv4PHM``.
    """

    def __init__(self, delta_t, inclination, coa_phase, t_coalescence=0.0,
                 polarization=0.0, mode_array=None, amplitude_floor=1e-3,
                 approximant="SEOBNRv4PHM", **waveform_options):
        import lal
        from scipy.interpolate import CubicSpline, PchipInterpolator
        from pycbc.waveform import get_td_waveform_modes

        for name in ("delta_t", "approximant", "inclination", "coa_phase"):
            if name in waveform_options:
                raise ValueError(f"{name} is a named argument here")
        self.approximant = str(approximant)
        self.delta_t = float(delta_t)
        # one dict, (l, m) -> (real TimeSeries, imaginary TimeSeries)
        modes = get_td_waveform_modes(
            approximant=self.approximant, delta_t=self.delta_t,
            inclination=float(inclination), coa_phase=float(coa_phase),
            **waveform_options)

        wanted = (None if mode_array is None
                  else {tuple(mode) for mode in mode_array})
        # The spherical harmonic wants an AZIMUTH, and it is not coa_phase.
        # pycbc.waveform.waveform_modes.sum_modes takes the azimuth as its
        # `phi`; measured against get_td_waveform on the same modes,
        # azimuth = pi/2 - coa_phase reproduces it to 0.0e+00 while coa_phase
        # itself gives mismatch 6e-02 to 2.0 depending on the inclination.
        azimuth = 0.5 * np.pi - float(coa_phase)
        angle = 2 * float(polarization)
        cos_psi, sin_psi = np.cos(angle), np.sin(angle)
        self._modes, self.skipped = {}, {}
        for mode in sorted(modes):
            if wanted is not None and tuple(mode) not in wanted:
                continue
            harmonic = (int(mode[0]), int(mode[1]))
            if harmonic[1] == 0:
                self.skipped[harmonic] = "m = 0 has no carrier to factor out"
                continue
            real, imaginary = modes[mode]
            series = np.asarray(real) + 1j * np.asarray(imaginary)
            magnitude = np.abs(series)
            peak = magnitude.max()
            if peak <= 0:
                self.skipped[harmonic] = "identically zero"
                continue
            loud = magnitude > float(amplitude_floor) * peak
            phase = np.unwrap(np.angle(series))
            sign = 1.0
            first, span = _longest_increasing_run(
                np.where(loud, phase, -np.inf))
            back, back_span = _longest_increasing_run(
                np.where(loud, -phase, -np.inf))
            if back_span > span:
                sign, first, span = -1.0, back, back_span
            if span < 16:
                self.skipped[harmonic] = "no usable stretch of monotone phase"
                continue
            piece = slice(first, first + span)
            spherical = lal.SpinWeightedSphericalHarmonic(
                float(inclination), azimuth, -2, *harmonic)
            # h_lm Y_lm contributes Re[.] to plus and -Im[.] = Re[i .] to
            # cross.  With Phi = +arg(h_lm) that is Re[(A Y) exp(i Phi)] and
            # Re[(i A Y) exp(i Phi)]; with Phi = -arg(h_lm), conjugating
            # inside the real part gives A conj(Y) and -i A conj(Y).
            if sign > 0:
                amplitude_plus = magnitude[piece] * spherical
                amplitude_cross = 1j * amplitude_plus
            else:
                amplitude_plus = magnitude[piece] * np.conj(spherical)
                amplitude_cross = -1j * amplitude_plus
            amplitude_plus, amplitude_cross = (
                cos_psi * amplitude_plus - sin_psi * amplitude_cross,
                sin_psi * amplitude_plus + cos_psi * amplitude_cross)
            clock = (np.asarray(real.sample_times)[piece]
                     + float(t_coalescence))
            carrier = sign * phase[piece]
            self._modes[harmonic] = dict(
                start=float(clock[0]), end=float(clock[-1]),
                plus=CubicSpline(clock, amplitude_plus.real,
                                 extrapolate=False),
                plus_imag=CubicSpline(clock, amplitude_plus.imag,
                                      extrapolate=False),
                cross=CubicSpline(clock, amplitude_cross.real,
                                  extrapolate=False),
                cross_imag=CubicSpline(clock, amplitude_cross.imag,
                                       extrapolate=False),
                phase=CubicSpline(clock, carrier, extrapolate=True),
                omega=PchipInterpolator(clock, np.gradient(carrier, clock),
                                        extrapolate=True))
        if not self._modes:
            raise ValueError(
                f"{self.approximant} yielded no usable mode; skipped "
                f"{self.skipped}")
        self.harmonics = tuple(sorted(self._modes))
        self.t_start = min(m["start"] for m in self._modes.values())
        self.t_end = max(m["end"] for m in self._modes.values())

    def _mode(self, harmonic):
        try:
            return self._modes[tuple(harmonic)]
        except KeyError:
            raise ValueError(
                f"{tuple(harmonic)} is not carried; available "
                f"{self.harmonics}, skipped {self.skipped}") from None

    def support(self, harmonic):
        mode = self._mode(harmonic)
        return mode["start"], mode["end"]

    def amplitude(self, harmonic, t):
        mode = self._mode(harmonic)
        query = np.asarray(t, dtype=float)
        inside = (query >= mode["start"]) & (query <= mode["end"])
        clipped = np.clip(query, mode["start"], mode["end"])

        def value(real_key, imaginary_key):
            return np.where(inside, mode[real_key](clipped), 0.0) \
                + 1j * np.where(inside, mode[imaginary_key](clipped), 0.0)
        return value("plus", "plus_imag"), value("cross", "cross_imag")

    def carrier_phase(self, harmonic, t):
        mode = self._mode(harmonic)
        return mode["phase"](np.clip(np.asarray(t, dtype=float),
                                     mode["start"], mode["end"]))

    def angular_frequency(self, harmonic, t):
        mode = self._mode(harmonic)
        return mode["omega"](np.clip(np.asarray(t, dtype=float),
                                     mode["start"], mode["end"]))

    def polarizations(self, t):
        total_plus = np.zeros(np.shape(t))
        total_cross = np.zeros(np.shape(t))
        for harmonic in self.harmonics:
            amplitude_plus, amplitude_cross = self.amplitude(harmonic, t)
            carrier = np.exp(1j * self.carrier_phase(harmonic, t))
            total_plus += np.real(amplitude_plus * carrier)
            total_cross += np.real(amplitude_cross * carrier)
        return total_plus, total_cross


class LALIMRPhenomDSource(LALFDSource):
    """`LALFDSource` with the approximant fixed to IMRPhenomD."""

    def __init__(self, *args, **kwargs):
        if "approximant" in kwargs:
            raise ValueError("this subclass fixes approximant=IMRPhenomD")
        super().__init__(*args, approximant="IMRPhenomD", **kwargs)


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
                       if k[0] in ('proj', 'blocks')}
        self._cache[key] = out
        return out

    def support_blocks(self, harmonic, n_probe=8192):
        """Every contiguous stretch over which this harmonic has amplitude.

        pyEFPEHM keeps a harmonic only while it carries more than
        ``Amplitude_tol`` of the total, so a harmonic is live over part of the
        trajectory -- and NOT necessarily over one interval. Measured on a
        precessing, eccentric configuration, (2, 1, 2) is live over
        [0.000, 0.009] and again over [0.174, 0.619] of the span, with a dead
        gap of 16% in between. Reporting only the outer hull, as an earlier
        version did, spends grid points on the gap.

        Liveness is read off the amplitude, not the phase: `carrier_phase` is
        now defined everywhere, and even before that a mode's phase passes
        through zero at an interior point.
        """
        key = ('blocks', harmonic)
        if key not in self._cache:
            probe = np.linspace(self.t_start, self.t_end, int(n_probe))
            amp_p, amp_c = self.amplitude(harmonic, probe)
            live = (amp_p != 0) | (amp_c != 0)
            edge = np.diff(live.astype(np.int8))
            starts = np.nonzero(edge == 1)[0] + 1
            stops = np.nonzero(edge == -1)[0]
            if live[0]:
                starts = np.concatenate(([0], starts))
            if live[-1]:
                stops = np.concatenate((stops, [len(live) - 1]))
            self._cache[key] = tuple(
                (float(probe[a]), float(probe[b]))
                for a, b in zip(starts, stops))
        return self._cache[key]

    def support(self, harmonic):
        """Outer hull of `support_blocks`; empty harmonics give a null window."""
        blocks = self.support_blocks(harmonic)
        if not blocks:
            return (self.t_start, self.t_start)
        return (blocks[0][0], blocks[-1][1])

    def amplitude(self, harmonic, t):
        amp_p, amp_c, _, _ = self._evaluate(harmonic, t)
        return amp_p, amp_c

    def _orbital_phases(self, t, derivative):
        """``(lambda, delta_lambda)`` from the model's own ODE solution."""
        query = np.clip(np.asarray(t, dtype=float), self.t_start, self.t_end)
        first, second = self.model.sol(query.reshape(-1),
                                       derivative=derivative, idxs=[2, 3])
        return first.reshape(query.shape), second.reshape(query.shape)

    def carrier_phase(self, harmonic, t):
        """Phi(t) = n lambda + (m - n) delta_lambda, defined EVERYWHERE.

        Not ``_evaluate``'s phase array, which is zero wherever the harmonic
        is not in pyEFPEHM's selected set. The factorisation
        h = Re[B exp(i Phi)] holds for any Phi, but only a smooth one leaves B
        slowly varying: setting Phi = 0 outside the window makes B oscillate at
        the carrier rate there, and no sparse grid can spline that. The
        formula is pyEFPEHM's own (``generate_tdomain_hlm_modes`` builds
        ``phi_t`` this way) and reproduces its ``phase`` output exactly on the
        window; the ODE solution is defined over the whole trajectory, so the
        same expression continues it outside.
        """
        _, m, n = (int(v) for v in harmonic)
        lamb, dlamb = self._orbital_phases(t, 0)
        return n * lamb + (m - n) * dlamb

    def angular_frequency(self, harmonic, t):
        _, m, n = (int(v) for v in harmonic)
        rate, drate = self._orbital_phases(t, 1)
        return n * rate + (m - n) * drate

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
