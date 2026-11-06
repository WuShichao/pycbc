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
"""The matched-filter statistic of an eccentric template at every initial
mean anomaly from one short transform.

pyEFPEHM's mode ``(l, m, n)`` changes with the initial mean anomaly ``l0``
only by the phase ``exp(i c l0)``, ``c = n - m`` an integer, its modulus
unchanged. A template ``h(l0) = sum_k exp(i c_k l0) h_k`` therefore has

    z(l0) = <h(l0), d> = sum_c exp(-i c l0) Z_c,     Z_c = sum_{k: c_k = c} z_k
    N(l0) = <h(l0), h(l0)> = sum_d exp(i d l0) M_d,  M_d = sum_{c_l - c_k = d} G_kl

with ``z_k = <h_k, d>`` and ``G_kl = <h_k, h_l>`` (complex inner products,
``<a, b> = sum conj(a) b``), both trigonometric polynomials in ``l0``. On
``n_l0`` equally spaced values both come from one discrete Fourier sum over
the integers ``c``; the phase-maximised statistic ``|z(l0)|^2 / N(l0)`` is
exact on that grid (no interpolation in ``l0``).
"""

import numpy


def mode_offsets(keys):
    """``c = n - m`` of pyEFPEHM mode keys ``(l, m, n)``."""
    return numpy.array([int(k[2]) - int(k[1]) for k in keys], dtype=int)


def mean_anomaly_statistic(z, gram, c, n_l0=None):
    """``|z(l0)|^2 / N(l0)`` on ``n_l0`` equally spaced ``l0`` in
    ``[0, 2 pi)``, and those ``l0``.

    Parameters
    ----------
    z : array, complex, shape ``(K,)`` or ``(K, n)``
        The modes' correlations with the data (``n`` samples, e.g. arrival
        times, share the Gram).
    gram : array, complex, shape ``(K, K)``
        The modes' Gram matrix.
    c : array of int, shape ``(K,)``
        The modes' mean-anomaly harmonic ``n - m``.
    n_l0 : int, optional
        The number of ``l0`` values; by default the smallest power of two
        that resolves the polynomial, at least ``2 (max c - min c) + 1``.
    """
    z = numpy.asarray(z, dtype=complex)
    single = z.ndim == 1
    if single:
        z = z[:, None]
    gram = numpy.asarray(gram, dtype=complex)
    c = numpy.asarray(c, dtype=int)
    span = int(c.max() - c.min())
    need = 2 * span + 1
    if n_l0 is None:
        n_l0 = 1 << int(numpy.ceil(numpy.log2(max(need, 2))))
    elif n_l0 < need:
        # N(l0) has harmonics -span..span: fewer samples alias them
        raise ValueError("n_l0 must be at least 2 (max c - min c) + 1")
    l0 = 2 * numpy.pi * numpy.arange(n_l0) / n_l0
    # z(l0) = sum_c exp(-i c l0) Z_c: a DFT over the integers c (mod n_l0)
    Z = numpy.zeros((n_l0, z.shape[1]), dtype=complex)
    numpy.add.at(Z, c % n_l0, z)
    zl = numpy.fft.fft(Z, axis=0)                 # sum_c Z_c exp(-2 pi i c j / n)
    # N(l0) = sum_{k,l} exp(i (c_l - c_k) l0) G_kl, G_kl = <h_k, h_l>
    M = numpy.zeros(n_l0, dtype=complex)
    d = (c[None, :] - c[:, None]) % n_l0
    numpy.add.at(M, d.ravel(), gram.ravel())
    norm = numpy.fft.ifft(M) * n_l0               # sum_d M_d exp(+2 pi i d j / n)
    stat = numpy.abs(zl) ** 2 / norm.real[:, None]
    return (stat[:, 0] if single else stat), l0
