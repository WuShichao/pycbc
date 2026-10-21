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
"""Exact parameter derivatives of the TaylorF2 phase by Taylor-mode
arithmetic.

Along a straight path ``theta(s) = theta0 + s d`` in (chirp mass, mass
ratio, aligned spins), every quantity is carried as its truncated Taylor
series in ``s`` (a jet), so its derivatives come out exact to rounding,
without finite-difference steps. The phasing coefficients are those of
LALSimulation's ``XLALSimInspiralPNPhasing_F2`` for aligned spins with the
default flags (all spin orders, quadrupole parameter one, no tides).

The source's phase and time are pinned at the lower frequency ``f_l``, as
for a template that reaches ``f_l`` at a fixed time with a fixed phase:

    phase(f) = -[Psi(f) - Psi(f_l) - Psi'(f_l) (f - f_l)],
    t(f) = [Psi'(f) - Psi'(f_l)] / (2 pi),

and the amplitude is the Newtonian ``Mc^(5/6)``. Second derivatives in
several coordinates follow from paths along ``e_i`` and ``e_i + e_j``.
"""

import numpy

MTSUN_SI = 4.925490947641267e-06
EULER_GAMMA = 0.5772156649015329


class Jet(object):
    """A truncated Taylor series in one variable: ``c[k]`` is the ``k``-th
    coefficient, ``x^(k)(0) / k!``; trailing axes broadcast like arrays."""
    __array_priority__ = 100

    def __init__(self, c):
        self.c = numpy.asarray(c)

    @property
    def order(self):
        return self.c.shape[0] - 1

    @classmethod
    def path(cls, value, slope, order):
        c = numpy.zeros(order + 1)
        c[0], c[1] = value, slope
        return cls(c)

    def _lift(self, other):
        if isinstance(other, Jet):
            return other
        c = numpy.zeros((self.order + 1,) + numpy.shape(other),
                        dtype=numpy.result_type(other, float))
        c[0] = other
        return Jet(c)

    def _pair(self, other):
        other = self._lift(other)
        a, b = numpy.broadcast_arrays(*_align(self.c, other.c))
        return a, b

    def __add__(self, other):
        a, b = self._pair(other)
        return Jet(a + b)
    __radd__ = __add__

    def __neg__(self):
        return Jet(-self.c)

    def __sub__(self, other):
        return self + (-self._lift(other))

    def __rsub__(self, other):
        return self._lift(other) - self

    def _scaled(self, other, divide=False):
        arr = numpy.asarray(other)
        c = self.c.reshape(self.c.shape + (1,) * max(
            0, arr.ndim - (self.c.ndim - 1)))
        return Jet(c / arr if divide else c * arr)

    def __mul__(self, other):
        if not isinstance(other, Jet):
            return self._scaled(other)
        a, b = self._pair(other)
        out = numpy.zeros(numpy.broadcast(a, b).shape,
                          dtype=numpy.result_type(a, b))
        for k in range(len(a)):
            out[k] = sum(a[i] * b[k - i] for i in range(k + 1))
        return Jet(out)
    __rmul__ = __mul__

    def __truediv__(self, other):
        if not isinstance(other, Jet):
            return self._scaled(other, divide=True)
        a, b = self._pair(other)
        out = numpy.zeros(numpy.broadcast(a, b).shape,
                          dtype=numpy.result_type(a, b))
        for k in range(len(a)):
            out[k] = (a[k] - sum(b[i] * out[k - i]
                                 for i in range(1, k + 1))) / b[0]
        return Jet(out)

    def __rtruediv__(self, other):
        return self._lift(other) / self

    def __pow__(self, p):
        a = self.c
        out = numpy.zeros_like(a, dtype=float)
        out[0] = a[0] ** p
        for k in range(1, len(a)):
            out[k] = sum(((p + 1) * i - k) * a[i] * out[k - i]
                         for i in range(1, k + 1)) / (k * a[0])
        return Jet(out)

    def log(self):
        a = self.c
        out = numpy.zeros_like(a, dtype=float)
        out[0] = numpy.log(a[0])
        for k in range(1, len(a)):
            out[k] = (a[k] - sum(i * out[i] * a[k - i]
                                 for i in range(1, k)) / k) / a[0]
        return Jet(out)


def _align(a, b):
    """Pad the trailing axes of two coefficient arrays to the same rank."""
    rank = max(a.ndim, b.ndim)
    return (a.reshape(a.shape + (1,) * (rank - a.ndim)) if a.ndim < rank
            else a,
            b.reshape(b.shape + (1,) * (rank - b.ndim)) if b.ndim < rank
            else b)


def _so3(m):
    return m * (25 + 38 / 3 * m)


def _so5(m):
    return -m * (1391.5 / 8.4 - m * (1 - m) * 10 / 3
                 + m * (1276 / 8.1 + m * (1 - m) * 170 / 9))


def _so6(m):
    return numpy.pi * m * (1490 / 3 + m * 260)


def _so7(m):
    e = m * (1 - m)
    return m * (-17097.8035 / 4.8384 + e * 28764.25 / 6.72
                + e * e * 47.35 / 1.44
                + m * (-7189.233785 / 1.524096 + e * 458.555 / 3.024
                       - e * e * 534.5 / 7.2))


def _self2s6(m):
    return (-4108.25 / 6.72 - 108.5 / 1.2 * m + 125.5 / 3.6 * m * m) * m * m


def _qm2s6(m):
    return (4703.5 / 8.4 + 2935 / 6 * m - 120 * m * m) * m * m


def taylorf2_phasing(m1, m2, chi1, chi2):
    """Coefficients of ``v^(k-5)`` and ``v^(k-5) log v``, ``k = 0..7``,
    times ``3 / (128 eta)``; arguments are jets (or numbers)."""
    pi = numpy.pi
    M = m1 + m2
    eta = m1 * m2 / (M * M)
    m1M, m2M = m1 / M, m2 / M
    zero = eta * 0
    v = [zero + 1, zero,
         5 * (74.3 / 8.4 + 11 * eta) / 9,
         zero - 16 * pi,
         5 * (3058.673 / 7.056 + 5429 / 7 * eta + 617 * eta * eta) / 72,
         5 / 9 * (772.9 / 8.4 - 13 * eta) * pi,
         (11583.231236531 / 4.694215680 - 640 / 3 * pi * pi
          - 684.8 / 2.1 * EULER_GAMMA
          + eta * (-15737.765635 / 3.048192 + 225.5 / 1.2 * pi * pi)
          + eta * eta * 76.055 / 1.728 - eta * eta * eta * 127.825 / 1.296
          - 684.8 / 2.1 * numpy.log(4)),
         pi * (770.96675 / 2.54016 + 378.515 / 1.512 * eta
               - 740.45 / 7.56 * eta * eta)]
    vlog = [zero] * 8
    vlog[5] = 5 / 3 * (772.9 / 8.4 - 13 * eta) * pi
    vlog[6] = zero - 684.8 / 2.1
    c11, c22, c12 = chi1 * chi1, chi2 * chi2, chi1 * chi2
    v[7] = v[7] + _so7(m1M) * chi1 + _so7(m2M) * chi2
    v[6] = (v[6] + _so6(m1M) * chi1 + _so6(m2M) * chi2
            + (326.75 / 1.12 + 557.5 / 1.8 * eta) * eta * c12
            + (_qm2s6(m1M) + _self2s6(m1M)) * c11
            + (_qm2s6(m2M) + _self2s6(m2M)) * c22)
    so5 = _so5(m1M) * chi1 + _so5(m2M) * chi2
    v[5] = v[5] + so5
    vlog[5] = vlog[5] + 3 * so5
    v[4] = (v[4] + (247 / 4.8 - 721 / 4.8) * eta * c12
            + (-719 / 9.6 + 233 / 9.6) * m1M * m1M * c11
            + (-719 / 9.6 + 233 / 9.6) * m2M * m2M * c22)
    v[3] = v[3] + _so3(m1M) * chi1 + _so3(m2M) * chi2
    norm = 3 / (128 * eta)
    return [x * norm for x in v], [x * norm for x in vlog]


def taylorf2_phase(coefficients, total_mass, frequencies):
    """``Psi(f)`` and ``dPsi/df`` (rad, rad/Hz) for jets of the phasing
    coefficients and of the total mass in solar masses."""
    a, b = coefficients
    f = numpy.asarray(frequencies, dtype=float)
    v = (total_mass * (numpy.pi * MTSUN_SI) * f) ** (1 / 3)
    logv = v.log()
    psi = dpsi = 0
    for k in range(len(a)):
        n = k - 5
        term = a[k] + b[k] * logv
        psi = psi + term * v ** n
        dpsi = dpsi + (n * term + b[k]) * v ** (n - 1)
    return psi, dpsi * v / (3 * f)


def source_jets(mchirp, q, spin1z, spin2z, direction, frequencies,
                f_lower, order=2):
    """Jets in ``s`` of the source's log amplitude, phase and time along
    ``(mchirp, q, spin1z, spin2z) + s direction``, ``q = m1 / m2``, pinned
    at ``f_lower`` as in the module docstring."""
    x = [Jet.path(v, d, order) for v, d in
         zip((mchirp, q, spin1z, spin2z), direction)]
    mc, q, chi1, chi2 = x
    m1 = mc * (1 + q) ** 0.2 * q ** 0.4
    m2 = m1 / q
    coef = taylorf2_phasing(m1, m2, chi1, chi2)
    M = m1 + m2
    psi, dpsi = taylorf2_phase(coef, M, frequencies)
    psi_l, dpsi_l = taylorf2_phase(coef, M, [f_lower])
    df = numpy.asarray(frequencies, dtype=float) - f_lower
    phase = -(psi - Jet(psi_l.c[:, 0]) - Jet(dpsi_l.c[:, 0]) * df)
    time = (dpsi - Jet(dpsi_l.c[:, 0])) / (2 * numpy.pi)
    logamp = 5 / 6 * mc.log()
    return logamp, phase, time


def taylorf2_derivatives(point, scales, frequencies, f_lower, order=2,
                         chunk=1 << 17):
    """:func:`_taylorf2_derivatives` a chunk of frequencies at a time."""
    f = numpy.asarray(frequencies, dtype=float)
    parts = [_taylorf2_derivatives(point, scales, f[k:k + chunk], f_lower,
                                   order) for k in range(0, len(f), chunk)]
    return tuple(numpy.concatenate(p, axis=-1) for p in zip(*parts))


def _taylorf2_derivatives(point, scales, frequencies, f_lower, order=2):
    """First and second derivatives of ``ln b`` and of ``t`` in shape units
    ``xi_i = (theta_i - theta0_i) / scale_i`` of (mchirp, q, spin1z,
    spin2z): ``(D, T)`` of shape ``(4, N)`` and, for ``order`` 2, ``(D2,
    T2)`` of shape ``(10, N)``, pairs ``i <= j`` in row-major order."""
    names = ('mchirp', 'q', 'spin1z', 'spin2z')
    start = [point[n] for n in names]
    scales = [scales[n] for n in names]

    def along(direction):
        logamp, phase, time = source_jets(*start, direction, frequencies,
                                          f_lower, order=max(order, 1))
        log_b = logamp.c[:, None] + 1j * phase.c if logamp.c.ndim == 1 \
            else logamp.c + 1j * phase.c
        return log_b, time.c

    axes = []
    for i in range(4):
        d = numpy.zeros(4)
        d[i] = scales[i]
        axes.append(along(d))
    D = numpy.array([a[0][1] for a in axes])
    T = numpy.array([a[1][1] for a in axes])
    if order < 2:
        return D, T
    D2, T2 = [], []
    for i in range(4):
        for j in range(i, 4):
            if i == j:
                D2.append(2 * axes[i][0][2])
                T2.append(2 * axes[i][1][2])
                continue
            d = numpy.zeros(4)
            d[i], d[j] = scales[i], scales[j]
            log_b, time = along(d)
            # c2(e_i + e_j) = c2(e_i) + c2(e_j) + H_ij
            D2.append(log_b[2] - axes[i][0][2] - axes[j][0][2])
            T2.append(time[2] - axes[i][1][2] - axes[j][1][2])
    return D, T, numpy.array(D2), numpy.array(T2)


def taylorf2_basis_coefficients(mchirp, q, spin1z, spin2z):
    """The TaylorF2 phase as a sum over fixed frequency functions,

        Psi(f) = sum_n A_n f^(n/3) + B_n f^(n/3) ln f,   n = -5 .. 2,

    with the total mass absorbed into the coefficients (``v = (pi M
    f)^(1/3)``), so that all the parameter dependence is in ``A`` and
    ``B``; and the log amplitude ``5/6 ln mchirp``. Returns ``(A, B,
    logamp)`` with ``A``, ``B`` of length 8, ``q = m1 / m2``."""
    m1 = mchirp * (1 + q) ** 0.2 * q ** 0.4
    m2 = m1 / q
    a, b = taylorf2_phasing(m1, m2, spin1z, spin2z)
    scale = numpy.pi * (m1 + m2) * MTSUN_SI
    log_scale = numpy.log(scale)
    A, B = numpy.zeros(8), numpy.zeros(8)
    for k in range(8):
        n = k - 5
        power = scale ** (n / 3)
        A[k] = power * (float(a[k]) + float(b[k]) * log_scale / 3)
        B[k] = power * float(b[k]) / 3
    return A, B, 5 / 6 * numpy.log(mchirp)


def basis_phase_and_time(A, B, frequencies, f_lower):
    """``phase(f)`` and ``t(f)`` of the module docstring from basis
    coefficients (or their differences, which enter linearly)."""
    f = numpy.asarray(frequencies, dtype=float)

    def psi(x):
        x = numpy.asarray(x, dtype=float)
        lx = numpy.log(x)
        out = dout = 0
        for k in range(8):
            n = (k - 5) / 3
            p = x ** n
            out = out + (A[k] + B[k] * lx) * p
            dout = dout + (A[k] * n + B[k] * (n * lx + 1)) * p / x
        return out, dout
    p, dp = psi(f)
    pl, dpl = psi(f_lower)
    return -(p - pl - dpl * (f - f_lower)), (dp - dpl) / (2 * numpy.pi)


def phase_coordinates(mchirp, q, spin1z, spin2z):
    """Coordinates in which the TaylorF2 phase of a long inspiral is nearly
    linear: the coefficients of ``f^(-5/3)``, ``f^(-1)`` and ``f^(-2/3)``
    of :func:`taylorf2_basis_coefficients` (chirp mass, symmetric mass
    ratio and the 1.5PN spin-orbit combination) and the spin difference
    ``(spin1z - spin2z) / 2``, which the phase barely fixes near equal
    masses. Vectorised over arrays; ``q = m1 / m2``."""
    mchirp, q, spin1z, spin2z = (numpy.asarray(v, dtype=float)
                                 for v in (mchirp, q, spin1z, spin2z))
    m1 = mchirp * (1 + q) ** 0.2 * q ** 0.4
    m2 = m1 / q
    a, b = taylorf2_phasing(m1, m2, spin1z, spin2z)
    scale = numpy.pi * (m1 + m2) * MTSUN_SI
    log_scale = numpy.log(scale)
    out = []
    for k in (0, 2, 3):
        n = k - 5
        out.append(scale ** (n / 3) * (a[k] + b[k] * log_scale / 3))
    out.append(0.5 * (spin1z - spin2z))
    return numpy.array(out)


def from_phase_coordinates(y, guess, iterations=30, tolerance=1e-12,
                           forward=phase_coordinates, scale=None,
                           max_step=None):
    """Invert ``forward`` (default :func:`phase_coordinates`) by Newton
    steps from ``guess`` (``(4, n)`` arrays of mchirp, q, spin1z, spin2z);
    columns that do not converge, or leave ``q >= 1`` or spins within one,
    come back as NaN. The residuals are relative to ``|y|``, or to
    ``scale`` when given; ``max_step`` bounds each parameter's step, which
    keeps the iteration on the root nearest the guess."""
    y = numpy.asarray(y, dtype=float)
    theta = numpy.array(guess, dtype=float, copy=True)
    theta = numpy.broadcast_to(theta, y.shape).copy()
    steps = numpy.array([1e-7, 1e-6, 1e-6, 1e-6])[:, None] * numpy.maximum(
        numpy.abs(theta), 1.0)
    scale = numpy.abs(y) + 1e-30 if scale is None else \
        numpy.broadcast_to(numpy.asarray(scale, dtype=float).reshape(
            (-1,) + (1,) * (y.ndim - 1)), y.shape)
    for _ in range(iterations):
        f0 = forward(*theta)
        res = (f0 - y) / scale
        if numpy.all(numpy.abs(res) < tolerance):
            break
        jac = numpy.empty((4, 4) + y.shape[1:])
        for j in range(4):
            moved = theta.copy()
            moved[j] = moved[j] + steps[j]
            jac[:, j] = (forward(*moved) - f0) / steps[j]
        delta = numpy.linalg.solve(numpy.moveaxis(jac, (0, 1), (-2, -1)),
                                   numpy.moveaxis(f0 - y, 0, -1)[..., None])
        delta = numpy.moveaxis(delta[..., 0], -1, 0)
        if max_step is not None:
            bound = numpy.asarray(max_step, dtype=float).reshape(
                (4,) + (1,) * (y.ndim - 1))
            with numpy.errstate(invalid='ignore', divide='ignore'):
                shrink = numpy.nanmin(numpy.where(
                    numpy.abs(delta) > bound, bound / numpy.abs(delta), 1.0),
                    axis=0)
            delta = delta * shrink
        theta = theta - delta
    with numpy.errstate(invalid='ignore'):
        bad = ~numpy.all(numpy.abs((forward(*theta) - y) / scale) < 1e-9,
                         axis=0) | ~(theta[1] >= 1) | ~(
            numpy.abs(theta[2]) <= 1) | ~(numpy.abs(theta[3]) <= 1)
    theta[:, bad] = numpy.nan
    return theta


def log_jacobian_from_phase_coordinates(theta, forward=phase_coordinates):
    """``ln |det d theta / d y|`` at ``theta`` ``(4, n)`` for the
    coordinates ``forward`` (default :func:`phase_coordinates`)."""
    theta = numpy.asarray(theta, dtype=float)
    steps = numpy.array([1e-7, 1e-6, 1e-6, 1e-6])[:, None] * numpy.maximum(
        numpy.abs(theta), 1.0)
    f0 = forward(*theta)
    jac = numpy.empty((4, 4) + theta.shape[1:])
    for j in range(4):
        moved = theta.copy()
        moved[j] = moved[j] + steps[j]
        jac[:, j] = (forward(*moved) - f0) / steps[j]
    sign, logdet = numpy.linalg.slogdet(numpy.moveaxis(jac, (0, 1),
                                                       (-2, -1)))
    return -logdet


class PhaseCoordinates(object):
    """:func:`phase_coordinates` as the curvilinear coordinates of a
    sampler (see :class:`pycbc.inference.sampler.amr.AMRSampler`): rows of
    (mchirp, q, spin1z, spin2z) to rows of ``y`` and back, the inverse
    started from ``guess``."""
    names = ('mchirp', 'q', 'spin1z', 'spin2z')

    def __init__(self, guess):
        self.guess = numpy.asarray(guess, dtype=float)

    def forward(self, theta):
        return phase_coordinates(*numpy.asarray(theta, dtype=float).T).T

    def inverse(self, y):
        y = numpy.asarray(y, dtype=float)
        return from_phase_coordinates(y.T, self.guess[:, None]).T

    def log_jacobian(self, theta):
        return log_jacobian_from_phase_coordinates(
            numpy.asarray(theta, dtype=float).T)


def basis_coefficients(mchirp, q, spin1z, spin2z):
    """:func:`taylorf2_basis_coefficients`' ``A`` and ``B`` stacked, shape
    ``(16,) + shape``, vectorised over arrays."""
    mchirp, q, spin1z, spin2z = (numpy.asarray(v, dtype=float)
                                 for v in (mchirp, q, spin1z, spin2z))
    m1 = mchirp * (1 + q) ** 0.2 * q ** 0.4
    m2 = m1 / q
    a, b = taylorf2_phasing(m1, m2, spin1z, spin2z)
    scale = numpy.pi * (m1 + m2) * MTSUN_SI
    log_scale = numpy.log(scale)
    A = [scale ** ((k - 5) / 3) * (a[k] + b[k] * log_scale / 3)
         for k in range(8)]
    B = [scale ** ((k - 5) / 3) * b[k] / 3 + 0 * mchirp for k in range(8)]
    return numpy.array([numpy.broadcast_to(v, mchirp.shape)
                        for v in A + B], dtype=float)


class MeasuredPhaseCoordinates(PhaseCoordinates):
    """Curvilinear coordinates ``(s_1, s_2, q, chi_a)`` in which the
    likelihood's degenerate valley is straight: ``s_1`` and ``s_2`` are the
    projections of the TaylorF2 basis coefficients on the two directions
    along which the phase changes most over a sample of points (:meth:`fit`),
    in the rms over frequency weighted by the template power, less the
    constant and linear parts phase and time absorb; the mass ratio and the
    spin difference ``chi_a = (spin1z - spin2z) / 2``, which the phase
    barely fixes, complete them. Points of equal phase then differ in ``q``
    and ``chi_a`` at nearly fixed ``s``, where the fixed coefficients of
    :class:`PhaseCoordinates` change along them.

    Parameters
    ----------
    guess : array_like
        The (mchirp, q, spin1z, spin2z) the inverse starts from, at each
        point's own ``q`` and ``chi_a``.
    frequencies, weights : array_like
        Frequencies, Hz, and the template's power at each; the power of the
        frequencies the data do not cover should be zero.
    f_lower : float
        The frequency at which phase and time are pinned, Hz.
    """
    def __init__(self, guess, frequencies, weights, f_lower, ndir=2):
        super().__init__(guess)
        f = numpy.asarray(frequencies, dtype=float)
        w = numpy.asarray(weights, dtype=float)
        w = w / w.sum()
        lf = numpy.log(f)
        p = [f ** ((k - 5) / 3) for k in range(8)]
        psi = numpy.array(p + [v * lf for v in p])          # (16, nf)
        sw = numpy.sqrt(w)
        X = numpy.vstack([numpy.ones_like(f), f - float(f_lower)]).T
        coef = numpy.linalg.lstsq(X * sw[:, None], (psi * sw).T,
                                  rcond=None)[0]
        weighted = (psi - (X @ coef).T) * sw
        # the metric of coefficient changes: weighted.T = Q R
        self._root = numpy.linalg.qr(weighted.T)[1].T       # (16, 16)
        self.ndir = int(ndir)
        self._vs = None

    def fit(self, theta):
        """Set the directions from rows of (mchirp, q, spin1z, spin2z)."""
        theta = numpy.atleast_2d(numpy.asarray(theta, dtype=float))
        C = basis_coefficients(*theta.T).T                   # (M, 16)
        self._mean = C.mean(axis=0)
        _, _, vs = numpy.linalg.svd((C - self._mean) @ self._root,
                                    full_matrices=False)
        self._vs = vs[:self.ndir]
        return self

    def _forward(self, mchirp, q, spin1z, spin2z):
        if self._vs is None:
            raise RuntimeError("fit the coordinates to a sample first")
        C = basis_coefficients(mchirp, q, spin1z, spin2z)
        shape = C.shape[1:]
        C = C.reshape(16, -1) - self._mean[:, None]
        s = self._vs @ (self._root.T @ C)
        rest = numpy.array([numpy.asarray(q, dtype=float).ravel(),
                            0.5 * (numpy.asarray(spin1z, dtype=float)
                                   - numpy.asarray(spin2z,
                                                   dtype=float)).ravel()])
        return numpy.concatenate([s, rest]).reshape((4,) + shape)

    def forward(self, theta):
        return self._forward(*numpy.asarray(theta, dtype=float).T).T

    def inverse(self, y):
        y = numpy.atleast_2d(numpy.asarray(y, dtype=float))
        # each point starts at its own q and spin difference, equal spins
        # otherwise zero
        guess = numpy.repeat(self.guess[:, None], len(y), axis=1)
        guess[1] = numpy.maximum(y[:, 2], 1.0)
        guess[2], guess[3] = y[:, 3], -y[:, 3]
        return from_phase_coordinates(y.T, guess, iterations=60,
                                      forward=self._forward,
                                      scale=numpy.ones(4),
                                      max_step=[numpy.inf, numpy.inf, 0.2,
                                                0.2]).T

    def log_jacobian(self, theta):
        return log_jacobian_from_phase_coordinates(
            numpy.asarray(theta, dtype=float).T, forward=self._forward)
