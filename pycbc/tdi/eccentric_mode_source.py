# Copyright (C) 2026 Shichao Wu
# This program is free software under the GNU General Public License,
# version 3 or (at your option) any later version.
"""Frequency-domain TDI source adapter for an eccentric reference cell.

Every eccentric harmonic retains its own stationary time. The adapter
implements the native-FD contract consumed by ``project_frequency_tdi``;
it does not approximate the sum of harmonics by a single phase clock.
"""
import numpy


class EccentricModeFamilySource:
    """One physical point of an EccentricModeFamily as a native-FD source.

    ``d``, ``de`` and ``dq`` are the family's offsets in ln Mc, e and ln q.
    The reference's orientation, aligned spins and distance are retained.
    ``mean_anomaly`` shifts mode (l,m,n) by exp(i (n-m) l0).

    By default the dominant (2,2,2) harmonic reaches ``family.f_ref`` at
    time zero. ``clock_frequency`` and ``clock_time`` choose another clock
    anchor; ``time_shift`` then delays the entire source. The same shift
    is applied to the Fourier phase and every mode's clock. An optional
    ``time_window=(start,stop)`` masks stationary times outside that window
    (an SPA observation window, not an exact time-domain truncation).

    ``circular_method='taylorf2'`` evaluates the analytic PN phase ratio
    without generating a new native waveform at each candidate. Mass/ratio
    offsets require a cell constructed with circular_phase_basis='taylorf2'. The
    explicit ``'pyefpe'`` option regenerates a circular auxiliary waveform
    for off-reference mass coordinates, for offline validation only.

    The source is for FD projection only. Clock accuracy and stencil-grid
    convergence must be checked against native pyEFPEHM at the application's
    reference and offsets before using it in a search.
    """
    def __init__(self, family, d=0., de=0., dq=0., mean_anomaly=0.,
                 clock_frequency=None, clock_time=0., time_shift=0.,
                 time_window=None, circular_method='taylorf2'):
        self.family = family
        self.d, self.de, self.dq = map(float, (d, de, dq))
        self.mean_anomaly = float(mean_anomaly)
        self.clock_frequency = float(family.f_ref if clock_frequency is None
                                     else clock_frequency)
        parameters = [self.d, self.de, self.dq, self.mean_anomaly,
                      self.clock_frequency, clock_time, time_shift]
        if not numpy.all(numpy.isfinite(parameters)):
            raise ValueError("source coordinates and clock must be finite")
        if circular_method not in ('taylorf2', 'pyefpe'):
            raise ValueError("circular_method must be taylorf2 or pyefpe")
        if (self.d != 0. or self.dq != 0.) and circular_method != \
                family.circular_phase_basis:
            raise ValueError('online circular method must match the cell phase basis')
        self.circular_method = circular_method
        self.circular = family.circular_phase_change(
            self.d, family.q0 * numpy.exp(self.dq), method=circular_method)
        anchor = [t[0] for k, _, _, _, t in family.iter_mode_pieces(
            self.d, self.de, self.dq, circular=self.circular,
            frequencies=[self.clock_frequency]) if k == (2, 2, 2)]
        if len(anchor) != 1:
            raise ValueError("clock anchor must lie inside the dominant mode")
        self.shift = float(clock_time + time_shift - anchor[0])
        self.time_window = None
        if time_window is not None:
            a, b = map(float, time_window)
            if not numpy.isfinite(a+b) or not a < b:
                raise ValueError("time window must be finite and increasing")
            self.time_window = (a, b)
        self.harmonics = tuple(family.keys)

    def frequency_harmonics(self, frequencies):
        """Sparse polarizations and per-mode clocks for TDI projection."""
        f = numpy.asarray(frequencies, dtype=float)
        records = {}
        for key, indices, amp, phase, time in self.family.iter_mode_pieces(
                self.d, self.de, self.dq, circular=self.circular,
                frequencies=f):
            time = time + self.shift
            keep = numpy.ones(len(indices), bool)
            if self.time_window is not None:
                lo, hi = self.time_window
                keep &= (time >= lo) & (time < hi)
            idx, t = indices[keep], time[keep]
            if not len(idx):
                continue
            if numpy.any(numpy.diff(t) <= 0):
                raise ValueError("mode stationary times must increase")
            angle = -phase[keep] - 2*numpy.pi*f[idx]*self.shift \
                + (key[2]-key[1])*self.mean_anomaly
            wave = amp[:, keep] * numpy.exp(1j*angle)
            records[key] = dict(indices=idx, frequency=f[idx], time=t,
                                plus=wave[0], cross=wave[1])
        return records

    def frequency_polarizations(self, frequencies):
        """Sum the modes coherently on the requested frequency grid."""
        f = numpy.asarray(frequencies, dtype=float)
        h = numpy.zeros((2, len(f)), complex)
        for record in self.frequency_harmonics(f).values():
            h[0, record['indices']] += record['plus']
            h[1, record['indices']] += record['cross']
        return h[0], h[1]
