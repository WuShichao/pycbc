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
               f22_window=None, taper=None, hlm=False, mode_array=None):
    """pyEFPEHM's modes at ``params`` (masses, spins, ``eccentricity``,
    optionally ``inclination``) on ``frequencies``: ``{key: (amplitude,
    psi, present)}``, ``amplitude`` the complex amplitudes of the plus and
    cross polarisations ``(2, N)``, zero where a mode is absent. At the
    fixed reference distance of 100 Mpc, ``amplitude * exp(-1j * psi)``
    reconstructs pyEFPEHM's native frequency-domain polarizations (before
    the optional taper). The amplitude includes the SPA factor and the
    Fourier-convention conjugation; raw ``A_SUA`` alone is not this amplitude.

    ``f22_window = (start, end)`` bounds the dominant harmonic's sweep (the
    source's observed time); by default it is the frequencies' own range.
    A window narrower than the frequencies holds the other harmonics of
    the same stretch of the orbit (an eccentric source's band).

    ``taper`` (seconds) multiplies each mode by ``sin^2`` ramps over that
    much of its own time ``t(f) = psi'(f) / 2 pi`` from either end of its
    support, as a time-domain taper of the observation would. A mode's
    support edges move with the binary's parameters (pyEFPEHM keeps the
    stretch of the orbit the window sets, at harmonic frequencies that
    depend on e); tapered, its amplitude goes to zero smoothly there.

    ``hlm=True``: each aligned mode ``(l, m, n)``'s multipole moment (its
    inertial ``m' = m`` column of pyEFPEHM's ``generate_hlm_modes``, phase 0,
    mean anomaly 0, no angular factor) in the first row of ``amplitude``,
    zero in the second: the templates of :mod:`pycbc.filter.hm_marginal`,
    which applies the angular factors of every pattern. The modes are
    selected at inclination pi / 3 (where every pattern is non-zero; the
    moments themselves do not depend on it). ``mode_array``: pyEFPEHM's
    ``(l, |m|)`` list (default its own, l <= 4)."""
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
    # in-plane spins (at f_ref), when given: a precessing source. With hlm=True each co-precessing mode (l, m, n)'s
    # m' = m column is then its main row, the co-precessing mode turned by D^l_{mm}(alpha, beta, gamma)(t)
    for k in ('spin1x', 'spin1y', 'spin2x', 'spin2y'):
        if params.get(k, 0.0):
            p[k] = float(params[k])
    if mode_array is not None:
        p['mode_array'] = [list(v) for v in mode_array]
    try:
        return _efpe_modes(p, f, hlm, taper)
    finally:
        # pyEFPEHM's model holds reference cycles, which the collector, counting objects and not bytes, frees
        # late: a cell's 625 calls with hlm grew the memory by 14 GB; collected after each call, by 0.7 GB at 81
        import gc
        gc.collect()


def _efpe_modes(p, f, hlm, taper):
    from pyEFPEHM.waveform.EFPE import pyEFPE
    if hlm:
        p.update(inclination=numpy.pi / 3, Compute_hlm_Modes=True)
        out = pyEFPE(p).generate_hlm_modes(
            f, return_waveform_pieces=True)['modes']
        modes = {}
        for key, m in out.items():
            l, mm = int(key[0]), int(key[1])
            idx = m['freq_idxs']
            amp = numpy.zeros((2, len(f)), complex)
            psi = numpy.zeros(len(f))
            have = numpy.zeros(len(f), bool)
            psi[idx] = numpy.asarray(m['psi_SPA'])
            amp[0, idx] = numpy.asarray(m['hlm'])[:, mm + l] * numpy.exp(
                1j * psi[idx])
            if taper and len(idx) > 1:
                fi = f[idx]
                t = numpy.gradient(psi[idx], fi) / (2 * numpy.pi)
                amp[:, idx] *= _ramp(abs(t - t[0]) / taper) * \
                    _ramp(abs(t[-1] - t) / taper)
            have[idx] = True
            modes[tuple(int(v) for v in key)] = (amp, psi, have)
        return modes
    out = pyEFPE(p).generate_modes(f, return_waveform_pieces=True)['modes']
    modes = {}
    for key, m in out.items():
        idx = m['freq_idxs']
        amp = numpy.zeros((2, len(f)), complex)
        psi = numpy.zeros(len(f))
        have = numpy.zeros(len(f), bool)
        psi[idx] = numpy.asarray(m['psi_SPA'])
        # Native polarizations include sqrt(2 pi) * h0_pref * T_SPA and
        # conjugate A_SUA to the standard Fourier convention. Recover the
        # slowly varying FD amplitude by removing only exp(-i psi).
        amp[:, idx] = numpy.asarray(m['polarizations']) * numpy.exp(
            1j * psi[idx])
        if taper and len(idx) > 1:
            fi = f[idx]
            t = numpy.gradient(psi[idx], fi) / (2 * numpy.pi)
            amp[:, idx] *= _ramp(abs(t - t[0]) / taper) * \
                _ramp(abs(t[-1] - t) / taper)
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
    support : {'all', 'center', 'edges'}
        A mode's frequency support: where every stencil point has it
        (default), where the reference has it, or ``'edges'``: amplitudes
        and phases from a window ``edge_margin`` wider at both ends
        (``wide_generator``), so that every stencil point has them past
        the edges, and the edges themselves, the first and last bin of
        each mode at each stencil point on the nominal window, interpolated
        like the rest. pyEFPEHM's modes are a function of the window only
        through a time and phase shift common to all modes, and where they
        end; the edges move with e and the masses (a harmonic's frequency
        range over the observation depends on them).
    edge_margin : float
        The wider window's relative margin, for ``support='edges'``.
    wide_generator : callable, optional
        ``wide_generator(params)``, the modes on the wider window, for
        ``support='edges'``; by default :func:`efpe_modes` with
        ``amplitude_tol`` on ``f22_window`` (or the frequencies' range)
        widened by ``edge_margin``.
    amplitude_tol : float
        pyEFPEHM's amplitude tolerance for the default generators.
    phase_scaling : {'m', 'n'}
        How a mode's circular phase change scales with the dominant one's:
        ``m / 2`` (default, right for the n = m modes a single harmonic's
        band holds) or ``n / 2`` (an eccentric harmonic's frequency ratio,
        for bands holding n != m).
    taper : float, optional
        Seconds of ``sin^2`` taper at each end of every mode's time, passed
        to :func:`efpe_modes` (with the default generator).
    circular_phase_basis : {'pyefpe', 'taylorf2'}
        Circular phase subtracted from every stencil waveform. Select
        'taylorf2' for analytic online phase ratios without native candidate
        generation. Construction and evaluation must use the same basis.
    """
    def __init__(self, reference, frequencies, f_ref, step_mass=0.01,
                 step_e=0.002, step_q=0.1, generator=None, f22_window=None,
                 phase_scaling='m', order=3, support='all', taper=None,
                 edge_margin=0.01, wide_generator=None,
                 amplitude_tol=1e-12, circular_phase_basis='pyefpe',
                 step_chi=0.0):
        if circular_phase_basis not in ('pyefpe', 'taylorf2'):
            raise ValueError('circular_phase_basis must be pyefpe or taylorf2')
        self.circular_phase_basis = circular_phase_basis
        self.reference = dict(reference)
        self.f = numpy.asarray(frequencies, dtype=float)
        self.f_ref = float(f_ref)
        # the fourth direction: both aligned spins moved together by
        # ``k * step_chi`` (the eccentric correction depends on spin through
        # the periastron advance; the circular phase change stays exact)
        self.steps = (float(step_mass), float(step_e), float(step_q),
                      float(step_chi))
        self.f22_window = f22_window
        self.generator = generator or (
            lambda p: efpe_modes(p, self.f, self.f_ref,
                                 amplitude_tol=amplitude_tol,
                                 f22_window=f22_window, taper=taper))
        if support not in ('all', 'center', 'edges'):
            raise ValueError("support: 'all', 'center' or 'edges'")
        self.support = support
        self._circ_generator = self.generator
        if support == 'edges':
            lo, hi = (self.f[0], self.f[-1]) if f22_window is None \
                else f22_window
            wide = (lo / (1 + edge_margin), hi * (1 + edge_margin))
            self._circ_generator = wide_generator or (
                lambda p: efpe_modes(p, self.f, self.f_ref,
                                     amplitude_tol=amplitude_tol,
                                     f22_window=wide, taper=taper))
        half = (int(order) - 1) // 2
        if int(order) not in (3, 5):
            raise ValueError("order: 3 (quadratic) or 5 (quartic) points")
        self.order = int(order)
        levels = [tuple(range(-half, half + 1)) if h > 0 else (0,)
                  for h in self.steps]
        # the stencil's amplitudes in single precision (27 points of every
        # mode, both polarisations); the phases, of thousands of radians,
        # in double; each mode only on its own window of bins (the union of
        # its supports over the stencil), a fraction of a wide band
        points = [(i, j, l, k) for i in levels[0] for j in levels[1]
                  for l in levels[2] for k in levels[3]]
        edges = {}
        if support == 'edges':
            # each mode's first and last bin on the nominal window
            for ijl in points:
                edges[ijl] = {}
                for k, (_, _, h) in self.generator(
                        self._point(*ijl)).items():
                    idx = numpy.flatnonzero(h)
                    edges[ijl][k] = numpy.array([idx[0], idx[-1]], float)
        gen = self._circ_generator
        grid = {}
        for ijl in points:
            grid[ijl] = {}
            for k, (a, p, h) in gen(self._point(*ijl)).items():
                idx = numpy.flatnonzero(h)
                if len(idx):
                    w = slice(idx[0], idx[-1] + 1)
                    grid[ijl][k] = (int(idx[0]), a[:, w].astype(
                        numpy.complex64), p[w].copy(), h[w].copy())
        self._circ = {(i, l, k): self._circular(i, l, k)
                      for i in levels[0] for l in levels[2]
                      for k in levels[3]}
        self._circ_amp = _tensor_quadratic(
            {(i, 0, l, k): c[0] for (i, l, k), c in self._circ.items()},
            (self.steps[0], 0.0, self.steps[2], self.steps[3]))
        self.keys = [k for k in grid[0, 0, 0, 0]
                     if all(k in g for g in grid.values())
                     and all(k in e for e in edges.values())]
        self._edges = {k: _tensor_quadratic(
            {ijl: e[k] for ijl, e in edges.items()}, self.steps)
            for k in self.keys} if edges else {}
        self._bins = numpy.arange(len(self.f))
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
            lo = min(g[k][0] for g in grid.values())
            hi = max(g[k][0] + len(g[k][3]) for g in grid.values())
            sl = slice(lo, hi)
            amp, psi, hv = {}, {}, {}
            for ijl, g in grid.items():
                a0, a, ph, h = g.pop(k)
                w = slice(a0 - lo, a0 - lo + len(h))
                amp[ijl] = numpy.zeros((2, hi - lo), numpy.complex64)
                amp[ijl][:, w] = a
                psi[ijl] = numpy.zeros(hi - lo)
                psi[ijl][w] = ph
                hv[ijl] = numpy.zeros(hi - lo, bool)
                hv[ijl][w] = h
            corr = {ijl: ph - r * self._circ[ijl[0], ijl[2], ijl[3]][1][sl]
                    for ijl, ph in psi.items()}
            psi0 = psi[(0,) * 4]
            del psi
            if support == 'center':
                # a stencil point without the bin has no phase there: take
                # the reference's correction (amplitude zero, as given)
                c0 = corr[(0,) * 4]
                corr = {ijl: numpy.where(hv[ijl], c, c0)
                        for ijl, c in corr.items()}
            if support == 'center':
                # the reference's own support; a stencil point missing a bin
                # there contributes zero amplitude (pyEFPEHM keeps a
                # harmonic only where it passes its tolerance, so the edges
                # move across the stencil)
                have = hv[(0,) * 4]
            else:
                have = numpy.logical_and.reduce(list(hv.values()))
            self._expansions[k] = (r, _tensor_quadratic(amp, self.steps),
                                   _tensor_quadratic(corr, self.steps),
                                   psi0, have, sl)
        del grid

    def __getstate__(self):
        """The family without its closures, for pickling (a cell built once, cached on disk): every stencil
        polynomial (:func:`_tensor_quadratic`) as its stencil values, the generators dropped (needed only to build
        the stencil, or for ``circular_phase_change(method='pyefpe')``)."""
        def encode(v):
            if callable(v) and hasattr(v, 'stacked') and hasattr(v, 'index'):
                return ('__tensor_quadratic__', [tuple(int(i) for i in row) for row in v.index], v.stacked,
                        tuple(v.steps))
            if isinstance(v, dict):
                return {k: encode(x) for k, x in v.items()}
            if isinstance(v, tuple):
                return tuple(encode(x) for x in v)
            if isinstance(v, list):
                return [encode(x) for x in v]
            if callable(v):
                return None
            return v
        state = {k: encode(v) for k, v in self.__dict__.items() if k != '_windows_last'}
        return state

    def __setstate__(self, state):
        def decode(v):
            if isinstance(v, tuple) and len(v) == 4 and v[0] == '__tensor_quadratic__':
                _, keys, stacked, steps = v
                return _tensor_quadratic(dict(zip(keys, stacked)), steps)
            if isinstance(v, dict):
                return {k: decode(x) for k, x in v.items()}
            if isinstance(v, tuple):
                return tuple(decode(x) for x in v)
            if isinstance(v, list):
                return [decode(x) for x in v]
            return v
        self.__dict__.update({k: decode(v) for k, v in state.items()})

    @property
    def q0(self):
        return self.reference['mass1'] / self.reference['mass2']

    def _point(self, i, j, l, k=0, eccentricity=None):
        """The reference moved ``i`` steps in ln Mc, ``j`` in e, ``l`` in
        ln q (chirp mass held when q moves) and ``k`` in both aligned spins."""
        from pycbc.conversions import mass1_from_mchirp_q, mass2_from_mchirp_q
        r = self.reference
        mc = (r['mass1'] * r['mass2']) ** 0.6 / (r['mass1']
                                                 + r['mass2']) ** 0.2
        mc *= numpy.exp(i * self.steps[0])
        q = self.q0 * numpy.exp(l * self.steps[2])
        e = r.get('eccentricity', 0.0) + j * self.steps[1] \
            if eccentricity is None else eccentricity
        chi = k * self.steps[3]
        return dict(r, mass1=mass1_from_mchirp_q(mc, q),
                    mass2=mass2_from_mchirp_q(mc, q), eccentricity=e,
                    spin1z=r.get('spin1z', 0.0) + chi,
                    spin2z=r.get('spin2z', 0.0) + chi)

    def _circular(self, i, l, k=0):
        """The circular binary's dominant mode at stencil point ``(i, l)``:
        pyEFPEHM's key ``(2, 2, 2)``, or the loudest mode."""
        modes = self._circ_generator(self._point(i, 0, l, k,
                                                 eccentricity=0.0))
        result = modes[2, 2, 2] if (2, 2, 2) in modes else max(
            modes.values(), key=lambda v: numpy.sum(abs(v[0]) ** 2))
        if self.circular_phase_basis == 'taylorf2':
            point = self._point(i, 0, l, k, eccentricity=0.0)
            return result[0], self._pn_phase(point), result[2]
        return result

    def _pn_phase(self, point):
        # Define the analytic basis on the entire eccentric-mode frequency
        # grid. Native circular phases are zero outside their own support;
        # mixing that residual with a global PN change creates phase/clock
        # discontinuities. Construction and evaluation must use one basis.
        from pycbc.waveform.taylorf2_jets import (
            basis_phase_and_time, taylorf2_basis_coefficients)
        mc = (point['mass1'] * point['mass2']) ** 0.6 / (
            point['mass1'] + point['mass2']) ** 0.2
        A, B, _ = taylorf2_basis_coefficients(
            mc, point['mass1'] / point['mass2'],
            point.get('spin1z', 0.0), point.get('spin2z', 0.0))
        phase, _ = basis_phase_and_time(A, B, self.f, self.f_ref)
        return -phase

    def _circ22(self, i):
        return self._circ[i, 0, 0][1]

    def circular_phase_change(self, d, q=None, spin1z=None, spin2z=None,
                              method=None):
        """The circular dominant mode's phase change from the reference to
        ``ln(Mc / Mc0) = d`` and the mass ratio ``q = m1 / m2`` and spins
        given (the reference's by default): pyEFPEHM at e = 0
        (``'pyefpe'``), or the TaylorF2 phase difference pinned in value and
        slope at ``f_ref`` (``'taylorf2'``), analytic and cheap. The
        eccentric correction is taken to depend on chirp mass and e alone.
        The default method is the basis selected when constructing the cell."""
        method = self.circular_phase_basis if method is None else method
        if method not in ('pyefpe', 'taylorf2'):
            raise ValueError('method must be pyefpe or taylorf2')
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
        modes = self._circ_generator(p)
        c = modes[2, 2, 2] if (2, 2, 2) in modes else max(
            modes.values(), key=lambda v: numpy.sum(abs(v[0]) ** 2))
        return c[1] - self._circ22(0)

    def modes(self, d, de, dq=0.0, circular=None, dchi=0.0):
        """The modes at ``(d, de, dq)``, ``dq = ln(q / q0)``: ``{key:
        (amplitude, psi, present)}``. ``circular`` is the circular dominant
        mode's phase change there (spins included, see
        :meth:`circular_phase_change`); pyEFPEHM's when not given."""
        n = len(self.f)
        out = {}
        for k, sl, a, psi, have in self._windows(d, de, dq, circular,
                                                 dchi=dchi):
            amp = numpy.zeros((2, n), complex)
            amp[:, sl] = a
            ph = numpy.zeros(n)
            ph[sl] = psi
            h = numpy.zeros(n, bool)
            h[sl] = have
            out[k] = (amp, ph, h)
        return out

    def _windows(self, d, de, dq=0.0, circular=None, spin_phase=None,
                 dchi=0.0):
        """The modes on their windows (see :meth:`_windows_uncached`); the
        last result is kept, since a source and its device evaluation ask
        for the same point twice."""
        def digest(a):
            return None if a is None else hash(numpy.ascontiguousarray(a).tobytes())
        key = (float(d), float(de), float(dq), float(dchi), digest(circular), digest(spin_phase))
        cached = getattr(self, '_windows_last', None)
        if cached is None or cached[0] != key:
            cached = (key, list(self._windows_uncached(d, de, dq, circular, spin_phase, dchi)))
            self._windows_last = cached
        return iter(cached[1])

    def _windows_uncached(self, d, de, dq=0.0, circular=None, spin_phase=None,
                          dchi=0.0):
        """Each mode at ``(d, de, dq)`` on its own window of bins: ``(key,
        slice, amplitude, psi, present)``, the amplitude zero where the mode
        is absent.

        ``spin_phase`` (on the family's grid, as a function of the dominant
        harmonic's frequency) is a circular phase change the stencil does
        not interpolate, such as an aligned-spin change: a harmonic of phase
        scale ``r`` at frequency ``f`` lies at the orbital frequency of the
        dominant harmonic's ``f / |r|``, so it gets ``r * spin_phase(f /
        |r|)``. (Mass and ratio changes are applied at ``f`` itself; the
        difference is part of the interpolated eccentric correction.)"""
        if circular is None:
            ref = self.reference
            dc = self.circular_phase_change(
                d, self.q0 * numpy.exp(dq),
                spin1z=ref.get('spin1z', 0.0) + dchi,
                spin2z=ref.get('spin2z', 0.0) + dchi)
        else:
            dc = circular
        for k, (r, amp, corr, psi0, have, sl) in self._expansions.items():
            a = amp(d, de, dq, dchi)
            c = corr(d, de, dq, dchi) - corr(0.0, 0.0, 0.0, 0.0)
            if spin_phase is not None and r != 0:
                c = c + r * numpy.interp(self.f[sl] / abs(r), self.f,
                                         spin_phase)
            if self._edges:
                lo, hi = numpy.round(self._edges[k](d, de, dq, dchi))
                b = self._bins[sl]
                have = have & (b >= lo) & (b <= hi)
            yield k, sl, numpy.where(have, a, 0), psi0 + r * dc[sl] + c, have

    def dominant_correction(self, d, de, dq=0.0, frequencies=None,
                            dchi=0.0):
        """The dominant mode's eccentric factor at ``(d, de, dq)``: its
        complex amplitude over the circular binary's and ``exp(-i (psi -
        psi_circ))``, on ``frequencies`` (by default the family's),
        interpolated linearly from the family's grid (the factor is smooth
        in frequency) and 1 outside it. Times a circular template it is
        the eccentric one; the circular part is the search's own family."""
        key = (2, 2, 2) if (2, 2, 2) in self._expansions else max(
            self._expansions, key=lambda k: numpy.sum(
                abs(self._expansions[k][1](0.0, 0.0, 0.0)) ** 2))
        r, amp, corr, psi0, have, sl = self._expansions[key]
        # constant and linear parts of the correction (pyEFPEHM's pinning)
        # are a phase and an arrival time, which the search maximises over
        a = amp(d, de, dq, dchi)[0]
        a0 = self._circ_amp(d, 0.0, dq, dchi)[0][sl]
        ratio = numpy.where(have & (abs(a0) > 0), a / numpy.where(
            abs(a0) > 0, a0, 1), 1)
        factor = numpy.ones(len(self.f), dtype=complex)
        factor[sl] = ratio * numpy.exp(-1j * numpy.where(
            have, corr(d, de, dq, dchi), 0))
        if frequencies is None:
            return factor
        f = numpy.asarray(frequencies, dtype=float)
        out = numpy.ones(len(f), dtype=complex)
        inside = (f >= self.f[0]) & (f <= self.f[-1])
        out[inside] = numpy.interp(f[inside], self.f, factor.real) + 1j * \
            numpy.interp(f[inside], self.f, factor.imag)
        return out

    def iter_mode_pieces(self, d, de, dq=0.0, circular=None,
                         frequencies=None, spin_phase=None, dchi=0.0):
        """Sparse FD modes and their individual stationary-time clocks.

        Yield ``(key, indices, amplitude, phase, time)`` on ``frequencies``
        (the family's grid by default), only where each mode is supported.
        ``amplitude`` has shape ``(2, len(indices))``; the waveform is
        ``amplitude * exp(-1j * phase)`` and ``time = phase' / (2 pi)``.
        The clock retains the generator's time origin. A caller applying
        an epoch shift must translate the phase and time consistently.

        Cubic splines interpolate the unwrapped SPA phase and complex FD
        amplitudes within each contiguous support interval. They never
        interpolate a wrapped waveform or bridge gaps. This permits a
        reference stencil on a smaller grid than the coherent data grid,
        but its resolution and the clock derivative must be validated for
        the application. No extrapolation beyond the sampled support is
        performed. Singleton support bins cannot supply a clock and raise.
        """
        from scipy.interpolate import CubicSpline
        f = self.f if frequencies is None else numpy.asarray(
            frequencies, dtype=float)
        if (f.ndim != 1 or not len(f) or not numpy.all(numpy.isfinite(f))
                or numpy.any(f <= 0) or numpy.any(numpy.diff(f) <= 0)):
            raise ValueError("frequencies must be positive, finite and increasing")
        for key, sl, amp, phase, have in self._windows(d, de, dq, circular,
                                                       spin_phase, dchi):
            active = numpy.flatnonzero(have)
            if not len(active):
                continue
            intervals = numpy.split(active, numpy.flatnonzero(
                numpy.diff(active) != 1) + 1)
            chunks = []
            for bins in intervals:
                x = self.f[sl][bins]
                lo, hi = numpy.searchsorted(f, [x[0], x[-1]])
                hi = numpy.searchsorted(f, x[-1], side='right')
                if hi <= lo:
                    continue
                if len(bins) < 2:
                    raise ValueError("a mode needs at least two supported bins for its clock")
                idx = numpy.arange(lo, hi)
                phase_spline = CubicSpline(x, phase[bins], extrapolate=False)
                a = CubicSpline(x, amp[:, bins], axis=-1,
                                extrapolate=False)(f[idx])
                chunks.append((idx, a, phase_spline(f[idx]),
                               phase_spline(f[idx], 1) / (2 * numpy.pi)))
            if chunks:
                yield (key, numpy.concatenate([p[0] for p in chunks]),
                       numpy.concatenate([p[1] for p in chunks], axis=-1),
                       numpy.concatenate([p[2] for p in chunks]),
                       numpy.concatenate([p[3] for p in chunks]))

    def polarizations(self, d, de, dq=0.0, circular=None, dchi=0.0):
        """The plus and cross polarisations ``(2, N)``, the modes summed."""
        h = numpy.zeros((2, len(self.f)), complex)
        for _, sl, amp, psi, have in self._windows(d, de, dq, circular,
                                                   dchi=dchi):
            hs = h[:, sl]
            hs[:, have] += amp[:, have] * numpy.exp(-1j * psi[have])
        return h

    def hplus(self, d, de, dq=0.0, circular=None, dchi=0.0):
        """The plus polarisation, the modes summed."""
        return self.polarizations(d, de, dq, circular, dchi)[0]


def _ramp(x):
    """``sin^2(pi x / 2)`` on ``[0, 1]``, 1 beyond."""
    return numpy.sin(0.5 * numpy.pi * numpy.clip(x, 0.0, 1.0)) ** 2


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

    # the stencil values stacked once, so an evaluation is one weighted sum
    # (a matrix-vector product) instead of a Python loop over the points
    keys = list(v)
    stacked = numpy.stack([numpy.asarray(v[k]) for k in keys])
    real = numpy.zeros((), stacked.dtype).real.dtype
    index = numpy.array(keys, dtype=int).reshape(len(keys), -1)

    def weights(*x):
        w = numpy.ones(len(keys))
        for d, (t, h, ks) in enumerate(zip(x, steps, nodes)):
            ld = lagrange(t, h, ks)
            w = w * numpy.array([ld[k] for k in index[:, d]])
        return w

    def weights_many(X):
        """``weights`` at each row of ``X`` (M, ndim): (M, S)."""
        X = numpy.atleast_2d(numpy.asarray(X, dtype=float))
        w = numpy.ones((len(X), len(keys)))
        for d, (h, ks) in enumerate(zip(steps, nodes)):
            if h <= 0:
                continue
            t = X[:, d]
            L = numpy.ones((len(X), len(ks)))
            for a, k in enumerate(ks):
                for j in ks:
                    if j != k:
                        L[:, a] *= (t - j * h) / ((k - j) * h)
            pos = numpy.searchsorted(numpy.asarray(ks), index[:, d])
            w *= L[:, pos]
        return w

    def poly(*x):
        return numpy.tensordot(weights(*x).astype(real), stacked, axes=(0, 0))
    # the polynomial is linear in the stencil values: poly(x) = weights(x) . stacked (for callers that project the
    # stacked values once and then need only the weights)
    poly.weights, poly.weights_many, poly.stacked = weights, weights_many, stacked
    # the stencil's layout: each stacked row's node indices per direction, the nodes and the steps (for callers that
    # contract a projection of the stacked values as a tensor, one direction at a time)
    poly.index, poly.nodes, poly.steps = index, nodes, tuple(steps)
    return poly


class EfpeMainRowGenerator(object):
    """pyEFPEHM's modes for :func:`pycbc.waveform.precession_main_row.
    build_precession_main_row`: ``(lam, inplane)`` with ``lam = (Mc, e, chi,
    q, chi1_perp, chi2_perp, dphi)`` (both aligned spins at ``chi``; in-plane
    spins at azimuths 0 and ``-dphi`` at ``f_ref``, a common rotation being the
    observer's azimuth) to ``efpe_modes(..., hlm=True)``, the precessing source
    (``inplane``) or the aligned one."""

    def __init__(self, frequencies, f_ref, f22_window):
        self.f = numpy.asarray(frequencies, dtype=float)
        self.f_ref = float(f_ref)
        self.f22_window = tuple(float(v) for v in f22_window)

    def __call__(self, lam, inplane):
        from pycbc.conversions import mass1_from_mchirp_q, mass2_from_mchirp_q
        mc, e, chi, q, s1, s2, dphi = (float(v) for v in lam)
        p = dict(mass1=float(mass1_from_mchirp_q(mc, q)),
                 mass2=float(mass2_from_mchirp_q(mc, q)), spin1z=chi,
                 spin2z=chi, eccentricity=e)
        if inplane:
            p.update(spin1x=s1, spin1y=0.0, spin2x=s2 * numpy.cos(dphi),
                     spin2y=-s2 * numpy.sin(dphi))
        return efpe_modes(p, self.f, self.f_ref, f22_window=self.f22_window,
                          hlm=True)
