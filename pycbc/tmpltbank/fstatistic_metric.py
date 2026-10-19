# Copyright (C) 2026  Shichao Wu
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
"""The parameter-space metric of the F-statistic.

A signal ``s = r_+ y_+ + r_x y_x`` is a complex combination of two basis
waveforms, the plus and cross polarisations through the detector response;
``r`` holds the orientation. The F-statistic maximises over the four real
amplitudes of that combination and over the arrival time. The mismatch of
a template displaced by ``dxi`` is ``1 - match``, where ``match^2`` is the
fraction of the signal's power the maximised statistic captures, and to
second order it is ``dxi^T M dxi`` with

    M_ij = <P d_i s, P d_j s> / (2 <s, s>),

where ``P`` removes the span of ``y_+, i y_+, y_x, i y_x`` and of
``i 2 pi f s``, and ``<u, v> = Re sum conj(u) v`` over whitened samples. With
``project='orientation'`` only ``s``, ``i s`` and ``i 2 pi f s`` are removed,
the metric of a template whose orientation is fixed.

The inputs are whitened spectra, already multiplied by any data window, so
the metric holds for any detector whose response gives the two bases.
"""

import numpy


def orientation_weights(inclination, polarization):
    """The weights ``r`` of the plus and cross bases for a quadrupolar
    signal of the given inclination and polarization angle."""
    c = numpy.cos(inclination)
    plus, cross = 0.5 * (1 + c ** 2), -1j * c
    c2, s2 = numpy.cos(2 * polarization), numpy.sin(2 * polarization)
    return numpy.array([c2 * plus - s2 * cross, s2 * plus + c2 * cross])


def _real(z):
    z = numpy.asarray(z)
    return numpy.concatenate([z.real.ravel(), z.imag.ravel()])


def fstatistic_metric(basis, derivatives, weights, frequencies,
                      project='fstat'):
    """The metric of the F-statistic at one point of parameter space.

    Parameters
    ----------
    basis : array_like, complex, shape ``(2, ...)``
        The whitened plus and cross bases; the last axis is frequency.
    derivatives : array_like, complex, shape ``(nparams, 2, ...)``
        Their derivatives with respect to each parameter.
    weights : array_like, complex, shape ``(2,)``
        The orientation weights, see :func:`orientation_weights`.
    frequencies : array_like
        The frequency of each sample of the last axis, in Hz.
    project : {'fstat', 'orientation'}
        What the statistic maximises over, see the module docstring.

    Returns
    -------
    numpy.ndarray
        The metric, shape ``(nparams, nparams)``.
    """
    basis = numpy.asarray(basis, dtype=complex)
    derivatives = numpy.asarray(derivatives, dtype=complex)
    weights = numpy.asarray(weights, dtype=complex)
    if basis.shape[0] != 2 or derivatives.shape[1:] != basis.shape:
        raise ValueError("basis must be (2, ...) and derivatives "
                         "(nparams, 2, ...) of the same trailing shape")
    signal = numpy.tensordot(weights, basis, axes=1)
    tangents = numpy.tensordot(derivatives, weights, axes=([1], [0]))
    timing = 2j * numpy.pi * numpy.asarray(frequencies) * signal
    if project == 'fstat':
        removed = [basis[0], 1j * basis[0], basis[1], 1j * basis[1], timing]
    elif project == 'orientation':
        removed = [signal, 1j * signal, timing]
    else:
        raise ValueError("project must be 'fstat' or 'orientation'")
    span, _ = numpy.linalg.qr(numpy.array([_real(b) for b in removed]).T)
    t = numpy.array([_real(d) for d in tangents]).T
    t -= span @ (span.T @ t)
    norm = numpy.sum(_real(signal) ** 2)
    return t.T @ t / (2 * norm)


class MetricChart(object):
    """Coordinates in which a constant metric is the identity.

    ``eta = sqrt(lambda) E^T dxi``, with the eigenvalues ``lambda`` of the
    metric in decreasing order and their eigenvectors ``E``, so that the
    mismatch ``dxi^T M dxi`` is ``|eta|^2`` and the first coordinates are the
    stiffest.

    Parameters
    ----------
    metric : array_like
        A symmetric positive definite metric.
    """
    def __init__(self, metric):
        metric = numpy.asarray(metric, dtype=float)
        if not numpy.allclose(metric, metric.T, rtol=1e-10, atol=0):
            raise ValueError("the metric is not symmetric")
        values, vectors = numpy.linalg.eigh(0.5 * (metric + metric.T))
        if values[0] <= 0:
            raise ValueError("the metric is not positive definite")
        self.metric = metric
        self.eigenvalues = values[::-1]
        self.eigenvectors = vectors[:, ::-1]
        self.dim = len(values)

    def to_chart(self, dxi):
        """Chart coordinates of parameter offsets, shape ``(..., dim)``."""
        return (numpy.asarray(dxi) @ self.eigenvectors) \
            * numpy.sqrt(self.eigenvalues)

    def from_chart(self, eta):
        """Parameter offsets of chart coordinates."""
        return (numpy.asarray(eta) / numpy.sqrt(self.eigenvalues)) \
            @ self.eigenvectors.T

    def mismatch(self, dxi):
        """The quadratic mismatch of parameter offsets."""
        dxi = numpy.asarray(dxi)
        return numpy.einsum('...i,ij,...j->...', dxi, self.metric, dxi)
