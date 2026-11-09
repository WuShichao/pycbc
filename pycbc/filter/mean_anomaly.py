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


def mean_anomaly_components(z, gram, c, n_l0=None, xp=numpy):
    """Coherently combine modes into two polarisation correlations and Gram.

    ``z`` has shape ``(K, 2, *samples)`` and ``gram`` has shape
    ``(K, 2, K, 2, *gram_samples)``, with ``gram[k,a,l,b] = <h_ka,h_lb>``.
    The trailing Gram dimensions must broadcast to ``samples`` (an empty
    shape means a shared Gram). This permits lag-dependent Grams; cross-mode
    terms are retained. ``c`` contains the K integer mean-anomaly offsets.

    Returns ``(zl, gl, l0)`` of shapes ``(n_l0, 2, *samples)``,
    ``(n_l0, 2, 2, *gram_samples)`` and ``(n_l0,)``. These components can
    feed either the four-real-amplitude F-statistic or orientation/amplitude
    marginalisation. The modes' amplitudes are tied, not fitted separately.

    ``xp`` is NumPy (default) or CuPy. Arrays and transforms stay on that
    backend; ``c`` is small host-side integer metadata. Validation may
    synchronize device scalars. Complex arrays are promoted to complex128.

    The default grid resolves the Fourier polynomials for z and G. It
    does NOT guarantee convergence of a maximum or integral of the nonlinear
    statistic over continuous l0; check that by increasing ``n_l0``.
    """
    z = xp.asarray(z, dtype=complex)
    gram = xp.asarray(gram, dtype=complex)
    raw_c = numpy.asarray(c)
    if z.ndim < 2 or z.shape[0] == 0 or z.shape[1] != 2:
        raise ValueError("z must have shape (K, 2, *samples), K > 0")
    k = z.shape[0]
    if gram.ndim < 4 or gram.shape[:4] != (k, 2, k, 2):
        raise ValueError("gram must have shape (K, 2, K, 2, *samples)")
    if (raw_c.shape != (k,) or not numpy.all(numpy.isfinite(raw_c))
            or not numpy.all(raw_c == numpy.floor(raw_c))):
        raise ValueError("c must contain K finite integer offsets")
    if not xp.all(xp.isfinite(z)) or not xp.all(xp.isfinite(gram)):
        raise ValueError("correlations and Gram must be finite")
    if numpy.broadcast_shapes(z.shape[2:], gram.shape[4:]) != z.shape[2:]:
        raise ValueError("Gram samples must broadcast to correlation samples")
    c = raw_c.astype(numpy.int64)
    # A common shift only multiplies z by an irrelevant phase. Retain it
    # here so the returned complex correlations also agree with direct sums.
    span = int(c.max() - c.min())
    need = 2 * span + 1
    if n_l0 is None:
        n_l0 = 1 << (max(need, 2) - 1).bit_length()
    if (isinstance(n_l0, (bool, numpy.bool_)) or not numpy.isscalar(n_l0)
            or not numpy.isfinite(n_l0) or int(n_l0) != n_l0
            or n_l0 < need):
        raise ValueError("n_l0 must be an integer >= 2 (max c - min c) + 1")
    n_l0 = int(n_l0)
    Z = xp.zeros((n_l0,) + z.shape[1:], dtype=complex)
    for i in range(k):
        Z[int(c[i] % n_l0)] += z[i]
    zl = xp.fft.fft(Z, axis=0)
    M = xp.zeros((n_l0, 2, 2) + gram.shape[4:], dtype=complex)
    for i in range(k):
        for j in range(k):
            M[int((c[j] - c[i]) % n_l0)] += gram[i, :, j]
    gl = xp.fft.ifft(M, axis=0) * n_l0
    return zl, gl, 2 * numpy.pi * xp.arange(n_l0) / n_l0


def mean_anomaly_twof(z, gram, c, n_l0=None, rcond=1e-12,
                      xp=numpy):
    """Fully coherent ``2F = z^H G^+ z`` on the mean-anomaly grid.

    ``xp`` selects the NumPy or CuPy backend, including the batched
    Hermitian eigensolver. Returned statistics and angles use that backend.

    Inputs follow :func:`mean_anomaly_components`. Noise in each whitened
    complex data bin must have unit variance in EACH real component. At a
    fixed template and l0, the noise law is chi-square with twice the complex
    Gram rank (four for full rank). Maximising over l0 changes that law.

    A Hermitian pseudoinverse handles rank deficiency; eigenvalues no larger
    than ``rcond * max(abs(eigenvalues))`` are discarded. Non-Hermitian or
    materially indefinite Grams are refused. Returns ``(twof, l0)`` with
    ``twof.shape == (n_l0, *samples)``. No time, band or mode powers are
    summed incoherently.
    """
    if not numpy.isfinite(rcond) or not 0 < rcond < 1:
        raise ValueError("rcond must lie strictly between 0 and 1")
    zl, gl, l0 = mean_anomaly_components(z, gram, c, n_l0, xp=xp)
    zz = xp.moveaxis(zl, 1, -1)
    gg = xp.moveaxis(gl, (1, 2), (-2, -1))
    # Insert missing leading sample axes after the l0 axis, keeping the
    # latter paired instead of accidentally broadcasting it as a sample.
    gg = gg.reshape((len(l0),) + (1,) * (zz.ndim - gg.ndim + 1)
                    + gg.shape[1:])
    adjoint = gg.swapaxes(-1, -2).conj()
    scale = xp.max(abs(gg), axis=(-2, -1), keepdims=True)
    tolerance = max(rcond, 32 * numpy.finfo(float).eps)
    if xp.any(abs(gg - adjoint) > tolerance * scale):
        raise ValueError("Gram must be Hermitian")
    values, vectors = xp.linalg.eigh(0.5 * (gg + adjoint))
    cutoff = rcond * xp.max(abs(values), axis=-1, keepdims=True)
    if xp.any(values < -cutoff):
        raise ValueError("Gram must be positive semidefinite")
    retained = values > cutoff
    safe_values = xp.where(retained, values, 1.0)
    inverse = xp.where(retained, 1.0 / safe_values, 0.0)
    projection = xp.einsum('...ai,...a->...i', vectors.conj(), zz)
    return xp.sum(abs(projection) ** 2 * inverse, axis=-1), l0


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



def mean_anomaly_log_bayes(z, gram, c, prior, nodes, n_l0=None,
                           noise_variance=1.0, order=64, table=None, xp=numpy):
    """Physical two-polarization ln B, also marginalized over mean anomaly.

    Inputs z and gram follow mean_anomaly_components and retain all mode
    cross terms. The modes are tied by one common inclination, polarization,
    distance and phase; they do not receive independent amplitude fits.
    prior and nodes are the DistancePowerLaw and OrientationNodes used by
    BStatisticSurrogate. Amplitudes use the template's reference-distance
    normalization. Raw z and G are divided by noise_variance, the variance
    of EACH real component of the whitened complex data.

    Returns (log_b, log_b_by_angle, angles) with shapes (*samples),
    (n_l0,*samples) and (n_l0,). Mean anomaly has a uniform prior; the first
    result is a log mean of Bayes factors, never an average of log factors
    or an incoherent sum of mode powers. Grid convergence must be checked.
    This direct reference also validates the Hermitian/PSD Gram with 2F;
    it does not claim an optimized amplitude-marginal search implementation.
    """
    from pycbc.filter.bstatistic import log_orientation_marginal
    if not numpy.isfinite(noise_variance) or noise_variance <= 0:
        raise ValueError('noise_variance must be positive and finite')
    mean_anomaly_twof(z, gram, c, n_l0, xp=xp)
    zl, gl, angles = mean_anomaly_components(z, gram, c, n_l0, xp=xp)
    shape = zl.shape[2:]
    missing = len(shape) - len(gl.shape[3:])
    gl = xp.broadcast_to(gl.reshape(gl.shape[:3] + (1,)*missing + gl.shape[3:]),
                         (len(angles), 2, 2) + shape)
    zz = xp.moveaxis(zl, 1, 0).reshape(2, -1) / noise_variance
    g00 = gl[:, 0, 0].real.ravel() / noise_variance
    g11 = gl[:, 1, 1].real.ravel() / noise_variance
    g01 = gl[:, 0, 1].ravel() / noise_variance
    # With no observed waveform the normalized likelihood integral is one.
    # Avoid 0/0 in the amplitude integrator's peak/width construction.
    active = xp.flatnonzero(xp.max(abs(gl), axis=(1, 2)).ravel() > 0)
    values = xp.zeros(zl.shape[0:1] + shape, dtype=xp.float64)
    flat = values.reshape(-1)
    if active.size:
        flat[active] = log_orientation_marginal(
            zz[:, active], g00[active], g11[active], g01[active], prior,
            nodes, order=order, margin=None, table=table, xp=xp)
    top = xp.max(values, axis=0)
    result = top + xp.log(xp.mean(xp.exp(values - top), axis=0))
    return result, values, angles
