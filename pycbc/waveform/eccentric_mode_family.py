# Copyright (C) 2026 Shichao Wu
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
"""A family of eccentric waveforms about one reference, mode by mode.

The pyEFPEHM modes ``(l, m, n)`` of an eccentric binary are each a complex
amplitude ``A_k(f)`` per polarisation and a stationary-phase phase
``psi_k(f)``. About a reference of eccentricity ``e0`` the family takes, for
each mode, the circular phase change in chirp mass, mass ratio and spins
exactly (the circular dominant mode's, times ``m / 2``) and interpolates
only the rest, the amplitudes and the eccentric correction ``psi_k - (m /
2) psi_22,circ``, by the tensor-product quadratic through a 3 x 3 x 3
stencil in ``(ln Mc, e, ln q)``: the correction varies with chirp mass and
mass ratio (``e0`` is defined at a fixed frequency), the circular phase far
faster.
"""

import numpy


def efpe_modes(params, frequencies, f_ref, amplitude_tol=1e-12,
               f22_window=None):
    """pyEFPEHM's modes at ``params`` (masses, spins, ``eccentricity``,
    optionally ``inclination``) on ``frequencies``: ``{key: (amplitude,
    psi, present)}``, ``amplitude`` the complex amplitudes of the plus and
    cross polarisations ``(2, N)``, zero where a mode is absent.

    ``f22_window = (start, end)`` bounds the dominant harmonic's sweep (the
    source's observed time); by default it is the frequencies' own range.
    A window narrower than the frequencies holds the other harmonics of
    the same stretch of the orbit (an eccentric source's band)."""
    from pyEFPEHM.waveform.EFPE import pyEFPE
    f = numpy.asarray(frequencies, dtype=float)
    lo, hi = (f[0], f[-1]) if f22_window is None else f22_window
    p = dict(mass1=params['mass1'], mass2=params['mass2'],
             spin1z=params.get('spin1z', 0.0),
             spin2z=params.get('spin2z', 0.0),
             eccentricity=params.get('eccentricity', 0.0),
             f22_start=lo * 0.999, f22_ref=f_ref, f22_end=hi * 1.001,
             distance=100.0, inclination=params.get('inclination', 0.0),
             Amplitude_tol=amplitude_tol)
    out = pyEFPE(p).generate_modes(f, return_waveform_pieces=True)['modes']
    modes = {}
    for key, m in out.items():
        idx = m['freq_idxs']
        amp = numpy.zeros((2, len(f)), complex)
        psi = numpy.zeros(len(f))
        have = numpy.zeros(len(f), bool)
        amp[:, idx] = numpy.asarray(m['A_SUA']).T[:2]
        psi[idx] = numpy.asarray(m['psi_SPA'])
        have[idx] = True
        modes[tuple(int(v) for v in key)] = (amp, psi, have)
    return modes


class EccentricModeFamily(object):
    """Eccentric waveforms about ``reference`` (``mass1``, ``mass2``,
    ``spin1z``, ``spin2z``, ``eccentricity``) on ``frequencies``, as
    functions of ``d = ln(Mc / Mc0)`` (both masses scaled, ``q`` fixed) and
    ``de = e - e0``.

    Parameters
    ----------
    reference : dict
        The reference binary.
    frequencies : array
        The frequencies, Hz.
    f_ref : float
        The frequency at which the eccentricity is defined, Hz.
    step_mass, step_e, step_q : float
        The stencil's steps in ``ln Mc``, ``e`` and ``ln q``: about a third
        of a cell, so that the quadratic interpolates over the cell rather
        than differentiating at a point (which single-precision amplitudes
        could not bear).
    generator : callable, optional
        ``generator(params)`` giving the modes as :func:`efpe_modes` does;
        by default pyEFPEHM on ``frequencies``.
    f22_window : tuple, optional
        The dominant harmonic's sweep, passed to :func:`efpe_modes`; set it
        to the observed sweep when ``frequencies`` reach further, to hold
        the other harmonics.
    order : {3, 5}
        Stencil points per direction: 3 (the tensor-product quadratic, the
        default) or 5 (quartic), which reaches further from the reference
        where the harmonics' amplitudes and phases curve (bands holding
        n != m).
    support : {'all', 'center'}
        A mode's frequency support: where every stencil point has it
        (default) or where the reference has it.
    phase_scaling : {'m', 'n'}
        How a mode's circular phase change scales with the dominant one's:
        ``m / 2`` (default, right for the n = m modes a single harmonic's
        band holds) or ``n / 2`` (an eccentric harmonic's frequency ratio,
        for bands holding n != m).
    """
    def __init__(self, reference, frequencies, f_ref, step_mass=0.01,
                 step_e=0.002, step_q=0.1, generator=None, f22_window=None,
                 phase_scaling='m', order=3, support='all'):
        self.reference = dict(reference)
        self.f = numpy.asarray(frequencies, dtype=float)
        self.f_ref = float(f_ref)
        self.steps = (float(step_mass), float(step_e), float(step_q))
        self.f22_window = f22_window
        self.generator = generator or (
            lambda p: efpe_modes(p, self.f, self.f_ref,
                                 f22_window=f22_window))
        half = (int(order) - 1) // 2
        if int(order) not in (3, 5):
            raise ValueError("order: 3 (quadratic) or 5 (quartic) points")
        self.order = int(order)
        levels = [tuple(range(-half, half + 1)) if h > 0 else (0,)
                  for h in self.steps]
        # the stencil's amplitudes in single precision (27 points of every
        # mode, both polarisations); the phases, of thousands of radians,
        # in double
        grid = {(i, j, l): {k: (a.astype(numpy.complex64), p, h) for k, (
            a, p, h) in self.generator(self._point(i, j, l)).items()}
                for i in levels[0] for j in levels[1] for l in levels[2]}
        self._circ = {(i, l): self._circular(i, l)
                      for i in levels[0] for l in levels[2]}
        self._circ_amp = _tensor_quadratic(
            {(i, 0, l): c[0] for (i, l), c in self._circ.items()},
            (self.steps[0], 0.0, self.steps[2]))
        self.keys = [k for k in grid[0, 0, 0]
                     if all(k in g for g in grid.values())]
        self._expansions = {}
        for k in self.keys:
            m = k[1]
            if phase_scaling == 'n' and len(k) > 2 and k[2]:
                # an eccentric harmonic (l, m, n) runs at about n / 2 of
                # the dominant harmonic's frequency, its phase likewise
                # (with the sign of m, as the m / 2 default has)
                r = (numpy.sign(m) if m else 1.0) * k[2] / 2.0
            else:
                r = m / 2.0 if m else 1.0
            amp = {ijl: g[k][0] for ijl, g in grid.items()}
            corr = {ijl: g[k][1] - r * self._circ[ijl[0], ijl[2]][1]
                    for ijl, g in grid.items()}
            if support == 'center':
                # a stencil point without the bin has no phase there: take
                # the reference's correction (amplitude zero, as given)
                c0 = corr[(0,) * 3]
                corr = {ijl: numpy.where(grid[ijl][k][2], c, c0)
                        for ijl, c in corr.items()}
            if support == 'center':
                # the reference's own support; a stencil point missing a bin
                # there contributes zero amplitude (pyEFPEHM keeps a
                # harmonic only where it passes its tolerance, so the edges
                # move across the stencil)
                have = grid[(0,) * 3][k][2]
            else:
                have = numpy.logical_and.reduce([g[k][2]
                                                 for g in grid.values()])
            self._expansions[k] = (r, _tensor_quadratic(amp, self.steps),
                                   _tensor_quadratic(corr, self.steps),
                                   grid[0, 0, 0][k][1], have)

    @property
    def q0(self):
        return self.reference['mass1'] / self.reference['mass2']

    def _point(self, i, j, l, eccentricity=None):
        """The reference moved ``i`` steps in ln Mc, ``j`` in e and ``l`` in
        ln q (chirp mass held when q moves)."""
        from pycbc.conversions import mass1_from_mchirp_q, mass2_from_mchirp_q
        r = self.reference
        mc = (r['mass1'] * r['mass2']) ** 0.6 / (r['mass1']
                                                 + r['mass2']) ** 0.2
        mc *= numpy.exp(i * self.steps[0])
        q = self.q0 * numpy.exp(l * self.steps[2])
        e = r.get('eccentricity', 0.0) + j * self.steps[1] \
            if eccentricity is None else eccentricity
        return dict(r, mass1=mass1_from_mchirp_q(mc, q),
                    mass2=mass2_from_mchirp_q(mc, q), eccentricity=e)

    def _circular(self, i, l):
        """The circular binary's dominant mode at stencil point ``(i, l)``:
        pyEFPEHM's key ``(2, 2, 2)``, or the loudest mode."""
        modes = self.generator(self._point(i, 0, l, eccentricity=0.0))
        if (2, 2, 2) in modes:
            return modes[2, 2, 2]
        return max(modes.values(), key=lambda v: numpy.sum(abs(v[0]) ** 2))

    def _circ22(self, i):
        return self._circ[i, 0][1]

    def circular_phase_change(self, d, q=None, spin1z=None, spin2z=None,
                              method='pyefpe'):
        """The circular dominant mode's phase change from the reference to
        ``ln(Mc / Mc0) = d`` and the mass ratio ``q = m1 / m2`` and spins
        given (the reference's by default): pyEFPEHM at e = 0
        (``'pyefpe'``), or the TaylorF2 phase difference pinned in value and
        slope at ``f_ref`` (``'taylorf2'``), analytic and cheap. The
        eccentric correction is taken to depend on chirp mass and e alone."""
        r = self.reference
        q0 = r['mass1'] / r['mass2']
        q = q0 if q is None else float(q)
        s1 = r.get('spin1z', 0.0) if spin1z is None else float(spin1z)
        s2 = r.get('spin2z', 0.0) if spin2z is None else float(spin2z)
        mc0 = (r['mass1'] * r['mass2']) ** 0.6 / (r['mass1']
                                                   + r['mass2']) ** 0.2
        mc = mc0 * numpy.exp(d)
        if method == 'taylorf2':
            from pycbc.waveform.taylorf2_jets import (
                basis_phase_and_time, taylorf2_basis_coefficients)
            A0, B0, _ = taylorf2_basis_coefficients(
                mc0, q0, r.get('spin1z', 0.0), r.get('spin2z', 0.0))
            A, B, _ = taylorf2_basis_coefficients(mc, q, s1, s2)
            phase, _ = basis_phase_and_time(numpy.asarray(A) - A0,
                                            numpy.asarray(B) - B0, self.f,
                                            self.f_ref)
            return -phase
        if d == 0 and q == q0 and s1 == r.get('spin1z', 0.0) and \
                s2 == r.get('spin2z', 0.0):
            return numpy.zeros(len(self.f))
        from pycbc.conversions import mass1_from_mchirp_q, mass2_from_mchirp_q
        p = dict(r, mass1=mass1_from_mchirp_q(mc, q),
                 mass2=mass2_from_mchirp_q(mc, q), spin1z=s1, spin2z=s2,
                 eccentricity=0.0)
        modes = self.generator(p)
        c = modes[2, 2, 2] if (2, 2, 2) in modes else max(
            modes.values(), key=lambda v: numpy.sum(abs(v[0]) ** 2))
        return c[1] - self._circ22(0)

    def modes(self, d, de, dq=0.0, circular=None):
        """The modes at ``(d, de, dq)``, ``dq = ln(q / q0)``: ``{key:
        (amplitude, psi, present)}``. ``circular`` is the circular dominant
        mode's phase change there (spins included, see
        :meth:`circular_phase_change`); pyEFPEHM's when not given."""
        dc = self.circular_phase_change(d, self.q0 * numpy.exp(dq)) \
            if circular is None else circular
        out = {}
        for k, (r, amp, corr, psi0, have) in self._expansions.items():
            a = amp(d, de, dq)
            c = corr(d, de, dq) - corr(0.0, 0.0, 0.0)
            out[k] = (numpy.where(have, a, 0), psi0 + r * dc + c, have)
        return out

    def dominant_correction(self, d, de, dq=0.0, frequencies=None):
        """The dominant mode's eccentric factor at ``(d, de, dq)``: its
        complex amplitude over the circular binary's and ``exp(-i (psi -
        psi_circ))``, on ``frequencies`` (by default the family's),
        interpolated linearly from the family's grid (the factor is smooth
        in frequency) and 1 outside it. Times a circular template it is
        the eccentric one; the circular part is the search's own family."""
        key = (2, 2, 2) if (2, 2, 2) in self._expansions else max(
            self._expansions, key=lambda k: numpy.sum(
                abs(self._expansions[k][1](0.0, 0.0, 0.0)) ** 2))
        r, amp, corr, psi0, have = self._expansions[key]
        # constant and linear parts of the correction (pyEFPEHM's pinning)
        # are a phase and an arrival time, which the search maximises over
        a = amp(d, de, dq)[0]
        a0 = self._circ_amp(d, 0.0, dq)[0]
        ratio = numpy.where(have & (abs(a0) > 0), a / numpy.where(
            abs(a0) > 0, a0, 1), 1)
        factor = ratio * numpy.exp(-1j * numpy.where(have, corr(d, de, dq),
                                                     0))
        if frequencies is None:
            return factor
        f = numpy.asarray(frequencies, dtype=float)
        out = numpy.ones(len(f), dtype=complex)
        inside = (f >= self.f[0]) & (f <= self.f[-1])
        out[inside] = numpy.interp(f[inside], self.f, factor.real) + 1j * \
            numpy.interp(f[inside], self.f, factor.imag)
        return out

    def polarizations(self, d, de, dq=0.0, circular=None):
        """The plus and cross polarisations ``(2, N)``, the modes summed."""
        h = numpy.zeros((2, len(self.f)), complex)
        for amp, psi, have in self.modes(d, de, dq, circular).values():
            h[:, have] += amp[:, have] * numpy.exp(-1j * psi[have])
        return h

    def hplus(self, d, de, dq=0.0, circular=None):
        """The plus polarisation, the modes summed."""
        return self.polarizations(d, de, dq, circular)[0]


def _tensor_quadratic(v, steps):
    """The tensor-product Lagrange polynomial through the stencil
    ``v[i, j, ...]`` at ``(i h0, j h1, ...)``, the indices the stencil's
    keys hold per direction (``-1, 0, 1`` for the quadratic, ``-2 .. 2`` for
    the quartic; only 0 along a direction of step 0): every term up to that
    degree in each direction. A correction ``C(x) e^2`` expanded about
    ``e0`` has a term ``x y^2`` that does not shrink with ``e0``, and a
    total-degree polynomial would drop it."""
    nodes = [sorted({key[d] for key in v}) for d in range(len(steps))]

    def lagrange(t, h, ks):
        if h <= 0:
            return {0: 1.0}
        out = {}
        for k in ks:
            w = 1.0
            for j in ks:
                if j != k:
                    w = w * (t - j * h) / ((k - j) * h)
            out[k] = w
        return out

    def poly(*x):
        ls = [lagrange(t, h, ks) for t, h, ks in zip(x, steps, nodes)]
        out = 0
        for key, value in v.items():
            w = 1.0
            for li, k in zip(ls, key):
                w = w * li[k]
            out = out + value * w
        return out
    return poly
