# Copyright (C) 2026 Shichao Wu
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
"""Original-detector-band correlations of a periodic response family."""
import numpy

from pycbc.tdi.harmonic_waveform import (harmonic_correlation_spectrum,
                                        harmonic_spectrum)


class DetectorHarmonicLikelihood(object):
    """Keep the physical detector inner product while sharing time FFTs.

    The projector maps an unwhitened SSB Fourier density to the observed
    detector Fourier bins. ``data`` is already whitened and ``weight`` is
    that same detector whitening factor sqrt(4 delta_f / PSD). Independent
    real/imaginary whitened noise components have variance one.

    ``source`` inputs must already use the declared source gauge and finite
    source window. A delay translates that source, including its window;
    the detector observation window remains fixed. This class does not
    build a source, fit its parameters or approximate the Gram.
    """

    def __init__(self, projector, data, weight):
        self.projector = projector
        self.grid = projector.grid
        self.data = numpy.array(data, dtype=complex, copy=True)
        self.weight = numpy.array(weight, dtype=float, copy=True)
        if (self.data.ndim != 2 or self.data.shape != self.weight.shape or
                self.data.shape[-1] != self.grid.data_nbins or
                not numpy.isfinite(self.data).all() or
                not numpy.isfinite(self.weight).all() or numpy.any(self.weight <= 0)):
            raise ValueError('invalid original detector data or whitening')
        self.adjoint_data = projector.adjoint(self.weight * self.data)
        self.data.setflags(write=False)
        self.weight.setflags(write=False)
        self.adjoint_data.setflags(write=False)

    def _periods(self, periods):
        expected = int(self.grid.harmonic_shift)
        if periods is None:
            return expected
        if not isinstance(periods, (int, numpy.integer)) or periods != expected:
            raise ValueError('response harmonic shifts differ from the Fourier grid')
        return expected

    def correlation_spectrum(self, source, responses, orders, periods=None):
        """Complex correlation spectrum; no stationary-time phase is used."""
        return harmonic_correlation_spectrum(source, responses, orders,
                                             self.adjoint_data, self._periods(periods))

    def delays(self, oversampling=1, signed=False):
        """The physical delays of the periodic FFT, optionally signed."""
        if not isinstance(oversampling, (int, numpy.integer)) or oversampling < 1:
            raise ValueError('time oversampling must be a positive integer')
        n = self.grid.size * int(oversampling)
        index = numpy.arange(n)
        if signed:
            index = numpy.where(index <= n // 2, index, index - n)
        return index * (self.grid.duration / n)

    def correlations(self, source, responses, orders, periods=None,
                     oversampling=1, signed=False,
                     max_output_bytes=512 * 1024**2):
        """Return correlations on a zero-padded periodic time FFT.

        Signed negative delays use their physical carrier phase, not the
        phase at the equivalent positive index one Fourier period later.
        The caller still owns the registered, non-wrapping search interval.
        """
        if not isinstance(oversampling, (int, numpy.integer)) or oversampling < 1:
            raise ValueError('time oversampling must be a positive integer')
        if not isinstance(max_output_bytes, (int, numpy.integer)) or max_output_bytes < 1:
            raise ValueError('output memory limit must be positive')
        n = self.grid.size * int(oversampling)
        required = responses.shape[1] * n * numpy.dtype(complex).itemsize
        if required > max_output_bytes:
            raise MemoryError('requested complex correlation fields exceed their memory limit')
        spectrum = self.correlation_spectrum(source, responses, orders, periods)
        delays = self.delays(oversampling, signed)
        carrier = numpy.exp(2j * numpy.pi * self.grid.first_frequency * delays)
        return numpy.fft.ifft(spectrum, n=n, axis=-1) * n * carrier

    def template(self, source, responses, orders, delay, periods=None):
        """Whitened physical detector bases at one finite physical delay."""
        if not numpy.isfinite(delay):
            raise ValueError('delay must be finite')
        shifted = numpy.asarray(source) * numpy.exp(
            -2j * numpy.pi * self.grid.frequencies * delay)
        spectrum = harmonic_spectrum(shifted, responses, orders, self._periods(periods))
        return self.projector.project(spectrum) * self.weight

    def statistics(self, source, responses, orders, delays, periods=None):
        """Exact original-band correlations and Gram on requested delays.

        The explicit projections are a correctness/reference calculation,
        not a fast all-lag Gram algorithm or a search-cost claim.
        """
        delays = numpy.asarray(delays, dtype=float)
        if delays.ndim != 1 or not len(delays) or not numpy.isfinite(delays).all():
            raise ValueError('delays need a nonempty finite vector')
        d, g = [], []
        for delay in delays:
            h = self.template(source, responses, orders, delay, periods)
            d.append(numpy.einsum('acf,cf->a', h.conj(), self.data))
            g.append(numpy.einsum('acf,bcf->ab', h.conj(), h))
        return numpy.array(d), numpy.array(g)
