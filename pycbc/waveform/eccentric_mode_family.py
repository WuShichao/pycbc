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
amplitude ``A_k(f)`` and a stationary-phase phase ``psi_k(f)``. About a
reference of eccentricity ``e0`` the family takes, for each mode, the
circular phase change in chirp mass exactly (the circular 22 mode's, times
``m / 2``) and expands only the rest, the amplitude and the eccentric
correction ``psi_k - (m / 2) psi_22,circ``, on a 3 x 3 stencil in ``(ln Mc, e)``
as the tensor-product quadratic through it: the eccentric correction varies with
chirp mass (``e0`` is defined at a fixed frequency), the circular phase far
faster.
"""

import numpy


def _scaled(reference, scale):
    p = dict(reference)
    p['mass1'] = reference['mass1'] * scale
    p['mass2'] = reference['mass2'] * scale
    return p


def efpe_modes(params, frequencies, f_ref, amplitude_tol=1e-12):
    """pyEFPEHM's modes at ``params`` (masses, spins, ``eccentricity``) on
    ``frequencies``: ``{key: (amplitude, psi, present)}``, the plus
    polarisation's complex amplitude face on, zero where a mode is absent."""
    from pyEFPEHM.waveform.EFPE import pyEFPE
    f = numpy.asarray(frequencies, dtype=float)
    p = dict(mass1=params['mass1'], mass2=params['mass2'],
             spin1z=params.get('spin1z', 0.0),
             spin2z=params.get('spin2z', 0.0),
             eccentricity=params.get('eccentricity', 0.0),
             f22_start=f[0] * 0.999, f22_ref=f_ref, f22_end=f[-1] * 1.001,
             distance=100.0, inclination=0.0, Amplitude_tol=amplitude_tol)
    out = pyEFPE(p).generate_modes(f, return_waveform_pieces=True)['modes']
    modes = {}
    for key, m in out.items():
        idx = m['freq_idxs']
        amp = numpy.zeros(len(f), complex)
        psi = numpy.zeros(len(f))
        have = numpy.zeros(len(f), bool)
        amp[idx] = numpy.asarray(m['A_SUA'])[:, 0]
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
    step_mass, step_e : float
        The central-difference steps in ``ln Mc`` and ``e``.
    generator : callable, optional
        ``generator(params)`` giving the modes as :func:`efpe_modes` does;
        by default pyEFPEHM on ``frequencies``.
    """
    def __init__(self, reference, frequencies, f_ref, step_mass=1e-3,
                 step_e=1e-3, generator=None):
        self.reference = dict(reference)
        self.f = numpy.asarray(frequencies, dtype=float)
        self.f_ref = float(f_ref)
        self.hd, self.he = float(step_mass), float(step_e)
        self.generator = generator or (
            lambda p: efpe_modes(p, self.f, self.f_ref))
        e0 = self.reference.get('eccentricity', 0.0)
        grid = {}
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                p = _scaled(self.reference, numpy.exp(i * self.hd))
                p['eccentricity'] = e0 + j * self.he
                grid[i, j] = self.generator(p)
        self._circ = {i: self._circular(i * self.hd) for i in (-1, 0, 1)}
        self.keys = [k for k in grid[0, 0]
                     if all(k in g for g in grid.values())]
        self._expansions = {}
        for k in self.keys:
            m = k[1]
            r = m / 2.0 if m else 1.0
            amp = {ij: g[k][0] for ij, g in grid.items()}
            corr = {ij: g[k][1] - r * self._circ22(ij[0]) for ij, g in
                    grid.items()}
            have = numpy.logical_and.reduce([g[k][2] for g in grid.values()])
            self._expansions[k] = (r, _biquadratic(amp, self.hd, self.he),
                                   _biquadratic(corr, self.hd, self.he),
                                   grid[0, 0][k][1], have)

    def _circular(self, d):
        """The circular binary's dominant mode at ``ln(Mc / Mc0) = d``:
        pyEFPEHM's key ``(2, 2, 2)``, or the loudest mode."""
        p = _scaled(self.reference, numpy.exp(d))
        p['eccentricity'] = 0.0
        modes = self.generator(p)
        if (2, 2, 2) in modes:
            return modes[2, 2, 2]
        return max(modes.values(), key=lambda v: numpy.sum(abs(v[0]) ** 2))

    def _circ22(self, i):
        return self._circ[i][1]

    def circular_phase_change(self, d):
        """The circular dominant mode's phase change at ``ln(Mc / Mc0) =
        d``, exact (pyEFPEHM at e = 0); the expansion's dominant term."""
        return self._circular(d)[1] - self._circ22(0)

    def modes(self, d, de, circular=None):
        """The modes at ``(d, de)``: ``{key: (amplitude, psi, present)}``.
        ``circular`` is the circular 22 phase change at ``d``; computed
        exactly when not given."""
        dc = self.circular_phase_change(d) if circular is None else circular
        out = {}
        for k, (r, amp, corr, psi0, have) in self._expansions.items():
            a = amp(d, de)
            c = corr(d, de) - corr(0.0, 0.0)
            out[k] = (numpy.where(have, a, 0), psi0 + r * dc + c, have)
        return out

    def hplus(self, d, de, circular=None):
        """The plus polarisation face on, the modes summed."""
        h = numpy.zeros(len(self.f), complex)
        for amp, psi, have in self.modes(d, de, circular).values():
            h[have] += amp[have] * numpy.exp(-1j * psi[have])
        return h


def _biquadratic(v, hx, hy):
    """The tensor-product quadratic through the 3 x 3 stencil ``v[i, j]`` at
    ``(i hx, j hy)``: every term up to ``x^2 y^2``. A correction ``C(x)
    e^2`` expanded about ``e0`` has a term ``x y^2`` that does not shrink
    with ``e0``, and a total-degree quadratic would drop it."""
    def lagrange(t, h):
        return {-1: t * (t - h) / (2 * h * h), 0: 1 - t * t / (h * h),
                1: t * (t + h) / (2 * h * h)}

    def poly(x, y):
        lx, ly = lagrange(x, hx), lagrange(y, hy)
        return sum(v[i, j] * lx[i] * ly[j] for i in (-1, 0, 1)
                   for j in (-1, 0, 1))
    return poly
