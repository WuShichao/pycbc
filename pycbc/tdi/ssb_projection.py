# Copyright (C) 2026 Shichao Wu
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
"""Detector-band geometry of the linear SSB data transformation.

The retiming of a finite band is not generally unitary. This module supplies
the forward map K, its adjoint, and a matrix-free solve in the original
detector band. It does not replace the inference model's default noise scale.
"""

import numpy
from scipy.sparse.linalg import LinearOperator, cg

from pycbc.tdi.ssb_yearly_harmonics import detector_times


class SSBProjectionError(RuntimeError):
    """A detector-band recovery did not meet its residual requirement."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


class SSBDataMap(object):
    """The finite-band map used by ``ssb_data_spectrum``, on the CPU.

    Leading array axes label independent channels or templates; the last
    axis holds detector bins for :meth:`forward`, or SSB FFT bins for
    :meth:`adjoint` and :meth:`project`. All operations use complex128.

    The original detector bins must already have the desired whitening.
    No noise variance is inferred from the mean power of the SSB bins.
    ``project`` reports distance to the map's range; ``recover`` refuses
    a spectrum outside it, rather than silently changing the template.
    """

    def __init__(self, grid, orbit, longitude, latitude, epsilon=1e-12):
        import finufft

        if not numpy.isfinite(epsilon) or not 1e-15 <= epsilon < 1e-3:
            raise ValueError('NUFFT epsilon must lie in [1e-15, 1e-3)')
        if not numpy.isfinite([longitude, latitude]).all():
            raise ValueError('sky coordinates must be finite')
        self.grid = grid
        self.epsilon = float(epsilon)
        self.nbins = int(grid.data_nbins)
        self.size = int(grid.size)
        inside = numpy.flatnonzero(grid.in_data)
        self.n_live = len(inside)
        if (not self.nbins or self.n_live < self.nbins or
                not numpy.array_equal(inside, numpy.arange(self.n_live))):
            raise ValueError('SSB window needs consecutive samples and at least M nodes')
        self._finufft = finufft
        self._padded = self.nbins + self.nbins % 2
        centre = self._padded // 2
        ssb = grid.times[:self.n_live]
        detector = detector_times(orbit, ssb, longitude, latitude)
        self._x = numpy.ascontiguousarray(numpy.mod(
            2 * numpy.pi * grid.data_delta_f * (detector - grid.start),
            2 * numpy.pi))
        f0 = grid.data_first_frequency
        self._carrier = numpy.exp(
            2j * numpy.pi * (f0 + centre * grid.data_delta_f) * (detector - ssb)
            + 2j * numpy.pi * (centre * grid.data_delta_f + f0
                              - grid.first_frequency) * (ssb - grid.start))
        self.uniform_correlation_scale = self.size / self.n_live

    @staticmethod
    def _array(value, length):
        value = numpy.asarray(value, dtype=numpy.complex128)
        if value.ndim < 1 or value.shape[-1] != length:
            raise ValueError('unexpected last axis for the SSB map')
        if not value.size or not numpy.isfinite(value).all():
            raise ValueError('SSB map input must be nonempty and finite')
        return value

    def _synthesize(self, bins):
        leading = bins.shape[:-1]
        modes = numpy.zeros((int(numpy.prod(leading)), self._padded), complex)
        modes[:, :self.nbins] = bins.reshape(-1, self.nbins)
        value = self._finufft.nufft1d2(
            self._x, modes, isign=1, eps=self.epsilon, nthreads=1)
        return (value * self._carrier).reshape(leading + (self.n_live,))

    def _sum(self, series):
        leading = series.shape[:-1]
        strengths = numpy.ascontiguousarray(
            (series * self._carrier.conj()).reshape(-1, self.n_live))
        value = self._finufft.nufft1d1(
            self._x, strengths, n_modes=self._padded, isign=-1,
            eps=self.epsilon, nthreads=1)
        return value[:, :self.nbins].reshape(leading + (self.nbins,))

    def forward(self, bins):
        """K times detector bins, with the stored FFT/n_live convention."""
        bins = self._array(bins, self.nbins)
        time = numpy.zeros(bins.shape[:-1] + (self.size,), complex)
        time[..., :self.n_live] = self._synthesize(bins)
        return numpy.fft.fft(time, axis=-1) / self.n_live

    def adjoint(self, spectrum):
        """K^H times an SSB spectrum, not an inverse retiming."""
        spectrum = self._array(spectrum, self.size)
        time = numpy.fft.ifft(spectrum, axis=-1)[..., :self.n_live]
        return self._sum(time) * (self.size / self.n_live)

    def normal(self, bins):
        """K^H K without allocating either full SSB FFT array."""
        bins = self._array(bins, self.nbins)
        return self._sum(self._synthesize(bins)) * (self.size / self.n_live**2)

    def project(self, spectrum, rtol=1e-11, maxiter=128):
        """Least-squares detector bins and per-row solve/range residuals.

        A converged normal equation does not mean the input was in range(K).
        This method returns that residual explicitly. Use ``recover`` when
        the spectrum is supposed to be the image of a detector-band signal.
        No global condition-number or rank certificate is asserted.
        """
        spectrum = self._array(spectrum, self.size)
        if (not numpy.isfinite(rtol) or not 0 < rtol < 1 or
                not isinstance(maxiter, (int, numpy.integer)) or maxiter < 1):
            raise ValueError('invalid normal-equation tolerance or iteration cap')
        rhs = self.adjoint(spectrum).reshape(-1, self.nbins)
        operator = LinearOperator((self.nbins, self.nbins), matvec=self.normal,
                                  dtype=numpy.complex128)
        inverse_scale = 1 / self.uniform_correlation_scale
        preconditioner = LinearOperator(operator.shape,
                                        matvec=lambda v: inverse_scale * v,
                                        dtype=numpy.complex128)
        recovered = numpy.empty_like(rhs)
        diagnostics = []
        for i, row in enumerate(rhs):
            iterations = [0]

            def count(unused):
                iterations[0] += 1

            solution, status = cg(operator, row, rtol=rtol, atol=0.,
                                  maxiter=maxiter, M=preconditioner,
                                  callback=count)
            norm = numpy.linalg.norm(row)
            residual = numpy.linalg.norm(self.normal(solution) - row)
            relative = float(residual / norm) if norm else float(residual)
            record = dict(iterations=iterations[0], solver_status=int(status),
                          normal_residual=relative)
            diagnostics.append(record)
            if status != 0 or relative > max(10 * rtol, 100 * self.epsilon):
                raise SSBProjectionError('SSB normal solve unresolved', diagnostics)
            recovered[i] = solution
        recovered = recovered.reshape(spectrum.shape[:-1] + (self.nbins,))
        mapped = self.forward(recovered).reshape(-1, self.size)
        for row, value, record in zip(spectrum.reshape(-1, self.size), mapped, diagnostics):
            norm = numpy.linalg.norm(row)
            distance = numpy.linalg.norm(value - row)
            record['range_residual'] = float(distance / norm) if norm else float(distance)
        return recovered, diagnostics

    def recover(self, spectrum, rtol=1e-11, maxiter=128, range_tolerance=1e-9):
        """Recover detector bins, rejecting unrepresented SSB components."""
        if not numpy.isfinite(range_tolerance) or range_tolerance < 0:
            raise ValueError('range tolerance must be nonnegative and finite')
        recovered, records = self.project(spectrum, rtol=rtol, maxiter=maxiter)
        if any(r['range_residual'] > range_tolerance for r in records):
            raise SSBProjectionError('spectrum is outside the detector map range', records)
        return recovered, records
