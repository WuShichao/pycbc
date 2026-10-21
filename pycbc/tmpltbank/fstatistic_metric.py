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

import sys

import numpy


def orientation_weights(inclination, polarization):
    """The weights ``r`` of the plus and cross bases for a quadrupolar
    signal of the given inclination and polarization angle."""
    c = numpy.cos(inclination)
    plus, cross = 0.5 * (1 + c ** 2), -1j * c
    c2, s2 = numpy.cos(2 * polarization), numpy.sin(2 * polarization)
    return numpy.array([c2 * plus - s2 * cross, s2 * plus + c2 * cross])


def _array_module(array):
    cupy = sys.modules.get('cupy')
    if cupy is not None and isinstance(array, cupy.ndarray):
        return cupy
    return numpy


def tangent_gram(basis, derivatives, frequencies, chunk=1 << 18):
    """The Hermitian inner products ``sum conj(u) v`` of the vectors

        y_a,  2 pi i f y_a,  d_i y_a,

    for each basis ``a`` and parameter ``i``, in that order (``a`` fastest):
    everything :func:`metric_from_gram` needs, in one pass over the samples.

    The arrays may live on numpy or cupy; they are read ``chunk`` samples
    of the last axis at a time, so no copy of their size is made.

    Parameters
    ----------
    basis : array, complex, shape ``(2, ..., N)``
        The whitened plus and cross bases; the last axis is frequency.
    derivatives : array, complex, shape ``(nparams, 2, ..., N)``
        Their derivatives with respect to each parameter.
    frequencies : array_like, shape ``(N,)``
        The frequency of each sample of the last axis, Hz.

    Returns
    -------
    numpy.ndarray
        Shape ``(4 + 2 nparams, 4 + 2 nparams)``, complex.
    """
    xp = _array_module(basis)
    if basis.shape[0] != 2 or tuple(derivatives.shape[1:]) != basis.shape:
        raise ValueError("basis must be (2, ...) and derivatives "
                         "(nparams, 2, ...) of the same trailing shape")
    nparams = derivatives.shape[0]
    size = basis.shape[-1]
    freqs = xp.asarray(frequencies, dtype=xp.float64)
    gram = 0
    for start in range(0, size, chunk):
        cut = slice(start, min(start + chunk, size))
        y = basis[..., cut].astype(xp.complex128)
        timing = 2j * numpy.pi * freqs[cut] * y
        d = derivatives[..., cut].astype(xp.complex128)
        rows = xp.concatenate([y, timing, d.reshape((2 * nparams,)
                                                    + y.shape[1:])])
        rows = rows.reshape(len(rows), -1)
        gram = gram + xp.conj(rows) @ rows.T
    return gram.get() if hasattr(gram, 'get') else numpy.asarray(gram)


def metric_from_gram(gram, weights, project='fstat'):
    """:func:`fstatistic_metric` from :func:`tangent_gram`'s inner products,
    for orientation ``weights``: each vector of the metric is a complex
    combination of the Gram's vectors, so its real inner products follow
    exactly."""
    gram = numpy.asarray(gram)
    nparams = (len(gram) - 4) // 2
    w = numpy.asarray(weights, dtype=complex)

    def vector(entries):
        c = numpy.zeros(len(gram), dtype=complex)
        for index, value in entries:
            c[index] = value
        return c
    signal = vector([(0, w[0]), (1, w[1])])
    timing = vector([(2, w[0]), (3, w[1])])
    if project == 'fstat':
        removed = [vector([(0, 1)]), vector([(0, 1j)]), vector([(1, 1)]),
                   vector([(1, 1j)]), timing]
    elif project == 'orientation':
        removed = [signal, 1j * signal, timing]
    else:
        raise ValueError("project must be 'fstat' or 'orientation'")
    tangents = [vector([(4 + 2 * i, w[0]), (5 + 2 * i, w[1])])
                for i in range(nparams)]
    C = numpy.array(removed + tangents).T
    real = numpy.real(C.conj().T @ gram @ C)
    r = len(removed)
    # the tangents' part orthogonal to the removed span: a Schur complement
    # in units where the removed vectors have unit norm
    scale = 1 / numpy.sqrt(numpy.diag(real)[:r])
    rss = real[:r, :r] * numpy.outer(scale, scale)
    rst = real[:r, r:] * scale[:, None]
    projected = real[r:, r:] - rst.T @ numpy.linalg.solve(rss, rst)
    norm = numpy.real(signal.conj() @ gram @ signal)
    return projected / (2 * norm)


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
    xp = _array_module(basis)
    basis, derivatives = xp.asarray(basis), xp.asarray(derivatives)
    if not xp.iscomplexobj(basis):
        basis = basis.astype(complex)
    if not xp.iscomplexobj(derivatives):
        derivatives = derivatives.astype(complex)
    return metric_from_gram(tangent_gram(basis, derivatives, frequencies),
                            weights, project)


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
