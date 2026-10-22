# Copyright (C) 2026 Shichao Wu
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
"""Forward projection of an SSB Fourier waveform onto detector-band bins.

This is a waveform operation, not inversion of a stored data spectrum. The
observation window and the selected frequencies are in detector coordinates.
"""
import numpy

from pycbc.tdi.ssb_yearly_harmonics import (detector_times, ssb_delay,
                                          _smooth_length)


class DetectorBandProjector(object):
    """Evaluate a continuous SSB Fourier series in the detector frame.

    The input contains Fourier *density samples* on ``grid.frequencies``,
    in the same units as a source's native frequency-domain waveform. The
    output samples the detector-window Fourier transform on the original
    data frequencies. Any PSD whitening on those frequencies belongs after
    this projection; a previously whitened input is only an approximation.

    Unlike ``SSBDataMap.recover``, this method does not assume that the input
    was already restricted to the observed data window or range of a data
    map. It uses the full supplied waveform. The periodic Fourier model must
    itself have enough frequency support and duration for the physical
    signal; this operator does not supply missing source physics or edges.
    ``source_support``, when given, is the source's finite interval in SSB
    seconds. Its detector-time endpoints are solved before quadrature, so
    a sharp source edge is not rounded to a sampling bin.
    """

    def __init__(self, grid, orbit, longitude, latitude, oversampling=2,
                 epsilon=1e-12, source_support=None):
        import finufft

        if (not isinstance(oversampling, (int, numpy.integer)) or
                oversampling < 2):
            raise ValueError('detector quadrature needs integer oversampling >= 2')
        if not numpy.isfinite(epsilon) or not 1e-15 <= epsilon < 1e-3:
            raise ValueError('invalid NUFFT accuracy')
        if not numpy.isfinite([longitude, latitude]).all():
            raise ValueError('sky coordinates must be finite')
        self.grid = grid
        self._finufft = finufft
        self.epsilon = float(epsilon)
        duration = grid.stop - grid.start
        if duration <= 0 or grid.size < 1 or grid.data_nbins < 1:
            raise ValueError('invalid detector time or frequency domain')
        lo, hi = grid.start, grid.stop
        if source_support is not None:
            support = numpy.asarray(source_support, dtype=float)
            if (support.shape != (2,) or not numpy.isfinite(support).all() or
                    support[0] >= support[1]):
                raise ValueError('source support needs two increasing finite SSB times')
            endpoints = detector_times(orbit, support, longitude, latitude)
            lo, hi = max(lo, endpoints[0]), min(hi, endpoints[1])
        self.active_duration = max(float(hi - lo), 0.)
        effective_samples = grid.size * self.active_duration / grid.duration
        self.time_samples = (_smooth_length(int(numpy.ceil(oversampling * max(
            effective_samples, grid.data_nbins * self.active_duration / duration))))
                             if self.active_duration else 0)
        detector = lo + (numpy.arange(self.time_samples) + .5) * (
            self.active_duration / max(self.time_samples, 1))
        self.detector_nodes = detector
        ssb = (detector + ssb_delay(orbit, detector, longitude, latitude)
               if detector.size else detector)
        self._x = numpy.ascontiguousarray(numpy.mod(
            2 * numpy.pi * (ssb - grid.start) / grid.duration, 2 * numpy.pi))
        centre = grid.size // 2
        self._padded = grid.data_nbins + grid.data_nbins % 2
        detector_centre = grid.data_first_frequency + (self._padded // 2) * grid.data_delta_f
        self._detector_x = numpy.ascontiguousarray(numpy.mod(
            2 * numpy.pi * grid.data_delta_f * (detector - grid.start), 2 * numpy.pi))
        self._carrier = numpy.exp(
            2j * numpy.pi * ((grid.first_frequency + centre * grid.delta_f)
                              * (ssb - grid.start)
                              - detector_centre * (detector - grid.start)))
        self._scale = grid.delta_f * self.active_duration / max(self.time_samples, 1)

    def project(self, spectrum):
        """Return detector-window/band samples; leading axes are independent."""
        spectrum = numpy.asarray(spectrum, dtype=numpy.complex128)
        if (spectrum.ndim < 1 or spectrum.shape[-1] != self.grid.size or
                not spectrum.size or not numpy.isfinite(spectrum).all()):
            raise ValueError('invalid SSB waveform Fourier array')
        leading = spectrum.shape[:-1]
        if not self.time_samples:
            return numpy.zeros(leading + (self.grid.data_nbins,), complex)
        modes = numpy.ascontiguousarray(spectrum.reshape(-1, self.grid.size))
        values = self._finufft.nufft1d2(
            self._x, modes, isign=1, eps=self.epsilon, nthreads=1)
        transformed = self._finufft.nufft1d1(
            self._detector_x, numpy.ascontiguousarray(values * self._carrier),
            n_modes=self._padded, isign=-1, eps=self.epsilon, nthreads=1) * self._scale
        return transformed[:, :self.grid.data_nbins].reshape(
            leading + (self.grid.data_nbins,))
