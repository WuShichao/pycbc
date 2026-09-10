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
        Number of frequency knots used for the inverse-SPA splines.
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

        # After the decades-long linear phase is removed, the largest phase
        # step is about 2*pi*duration*df.  Keep it below pi/2 so unwrap cannot
        # alias even for the full two-year Yorsh band.
        if duration is not None:
            requested_duration = float(duration)
        else:
            end_time = self._stationary_time([f_upper])[0]
            start_time = self._stationary_time([f_lower])[0]
            requested_duration = end_time - start_time
        unwrap_knots = int(np.ceil(
            4 * requested_duration * (self.f_upper - self.f_lower))) + 1
        n_frequency = max(int(n_frequency), unwrap_knots)
        frequency = np.linspace(self.f_lower, self.f_upper, n_frequency)
        h_plus, h_cross = self._fd_waveform(frequency)
        stationary_start = self._stationary_time([self.f_lower])[0]

        # Remove the large linear phase before unwrapping.  Differentiating a
        # spline of the resulting native LAL phase is markedly more stable
        # than storing a finite-difference time at every knot.
        time_shift_phase = np.remainder(
            2 * np.pi * frequency * stationary_start, 2 * np.pi)
        shifted_fd_phase = np.unwrap(np.angle(
            h_plus * np.exp(1j * time_shift_phase)))
        # ``exp(i 2*pi*f*t0)`` itself loses a few ulps when t0 is decades.
        # A local Chebyshev representation removes that harmless point noise
        # before differentiation; a cubic interpolant would amplify it into
        # spurious structure in delayed phases.  A LISA stellar-origin band
        # spans about one percent in frequency, where degree 32 is ample; a
        # band wide enough to need more shows up as the monotonicity failure
        # below rather than silently.
        degree = min(32, n_frequency - 1)
        fd_phase = np.polynomial.Chebyshev.fit(
            frequency, shifted_fd_phase, degree)
        smooth_fd_phase = fd_phase(frequency)
        relative_time = -fd_phase.deriv(1)(frequency) / (2 * np.pi)
        relative_time -= relative_time[0]
        dt_df = -fd_phase.deriv(2)(frequency) / (2 * np.pi)
        invalid_time = np.any(np.diff(relative_time) <= 0)
        invalid_slope = np.any(~np.isfinite(dt_df)) or np.any(dt_df <= 0)
        if invalid_time or invalid_slope:
            raise ValueError("LAL stationary-time map is not monotone")
        stationary = stationary_start + relative_time
        time = relative_time + self.t_start

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

        phase = smooth_fd_phase + 2 * np.pi * frequency * relative_time
        phase += np.pi / 4

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

    def _stationary_time(self, frequency):
        """Measure ``-d arg(H)/2pi df`` without global phase unwrapping."""
        frequency = np.atleast_1d(np.asarray(frequency, dtype=float))
        # Fit nine nearby native phase values instead of differencing only two
        # nearly equal complex numbers.  The full local phase span is about
        # 2.4 rad: safely unwrap-able, but large enough to suppress roundoff.
        epsilon = 1.2 / (2 * np.pi * self._newtonian_time(frequency))
        epsilon = np.minimum(epsilon, frequency * 1e-7)
        offsets = np.linspace(-1.0, 1.0, 9)
        local_frequency = frequency[:, None] + epsilon[:, None] * offsets
        h_plus, _ = self._fd_waveform(local_frequency.reshape(-1))
        local_phase = np.unwrap(
            np.angle(h_plus.reshape(len(frequency), -1)), axis=1)
        derivative = np.empty(len(frequency))
        for index in range(len(frequency)):
            fit = np.polynomial.Chebyshev.fit(
                local_frequency[index], local_phase[index], 4)
            derivative[index] = fit.deriv()(frequency[index])
        return -derivative / (2 * np.pi)

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
