# Copyright (C) 2026  Shichao Wu, Alex Nitz
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Sparse "TDI on the fly" evaluation of TDI channels.

The dense path costs ~25-38 us per output sample and is dominated (85-90%) by
PyTDI's Lagrange interpolation, not by the response: a two-year LISA channel at
dt = 5 s takes about eight minutes, which is five to six orders too slow for
parameter estimation.

Cornish & Littenberg (PRD 112, 102007) remove that cost rather than optimise
it. For a quasi-monochromatic harmonic h(t) = Re[A(t) exp(i Phi(t))], every
delayed copy the combination needs can be written as

    h(t - D) = Re[ A(t - D) exp(i Phi(t - D)) ]
             = Re[ exp(i Phi(t)) * A(t - D) exp(i (Phi(t - D) - Phi(t))) ]

so pulling exp(i Phi(t)) out of the whole sum leaves a bracket that varies on
the orbital timescale rather than the GW period. That bracket is sampled on a
grid of a few hundred points per year and splined; the carrier is reattached at
full cadence at the end. No intermediate eta series is ever built and
`dsp.timeshift` is never called -- which is why Layer 2 has to expose
(coefficient, net shift) pairs rather than only "give me eta, take channels".

The delays go on the WAVEFORM's time argument, and the orbit quantities are
evaluated at the shifted times too, so this is not a frozen-constellation
approximation: it is exact up to the interpolation of a slowly varying complex
amplitude.
"""

import numpy as np

from pycbc.tdi.response import (C_SI, LINK_ORDER, antenna_pattern,
                                doppler_factors, polarization_basis,
                                sample_constellation)

YEAR = 3.15581498e7


class HarmonicSource:
    """Protocol: a source whose harmonics are amplitude times a carrier.

    Implementations must provide, for a harmonic label ``n``:

    ``amplitude(n, t)``  complex A(t), the two polarisations as a pair
    ``carrier_phase(n, t)``  real Phi(t)
    ``angular_frequency(n, t)``  real dPhi/dt, used to size the grid
    """


def delay_padding(orbit, times, terms, links=LINK_ORDER):
    """How far outside its own window a harmonic still reaches in a channel.

    The channel at output time ``t`` queries the waveform at
    ``t - net_shift - tau`` with ``tau = L + k.r_emit/c`` for the emission term
    and ``k.r_recv/c`` for the reception term. So a harmonic that pyEFPEHM
    switches off at ``t_off`` still contributes to the channel for another
    ``max|tau|`` afterwards, and starts contributing that much before it
    switches on. Bounding ``|k.r| <= |r|`` makes the result sky-independent,
    which is what a grid held fixed across a likelihood run needs.

    For LISA this is about 570 s: 500 s of light time across the orbit, one
    arm, and the combination's own chain delay (58 s for X2).

    Widening the grid by this much is correct but measured NOT to help: with
    pyEFPEHM at ``Amplitude_tol = 1e-4`` it moves the 16-harmonic mismatch from
    4.83e-04 to 5.34e-04, and at 1e-5 it changes nothing. The reach is real;
    what limits the reconstruction there is the amplitude step itself, and
    extra nodes just past the step buy ringing rather than accuracy. Hence
    ``padding`` defaults to zero.
    """
    probe = np.linspace(times[0], times[-1], 32)
    cache = {}
    worst = max(np.max(np.abs(chain_delay(orbit, probe, chain, links,
                                          cache=cache)))
                for chain in {term.operators for term in terms})
    sample = sample_constellation(probe, orbit, links=links)
    radius = np.max(np.linalg.norm(sample.position, axis=-1))
    return float(worst + np.max(sample.ltt) + radius / C_SI)


def adaptive_time_grid(source, harmonic, t_start, t_end, delta_phi=0.5,
                       dt_max=None, n_probe=4096, growth=1.0,
                       max_step_scale=np.inf, padding=0.0):
    """Grid on which the carrier advances by ``delta_phi`` per step.

    The definition is exactly that -- points where Phi increases by delta_phi --
    so instead of stepping through time and asking for omega at each point, the
    carrier phase is evaluated ONCE on a probe grid and inverted. Because omega
    rises toward merger, the result densifies there automatically.

    Two bugs this replaces, both found on a real pyEFPEHM harmonic:

    * the first version grew the step geometrically from ``t_start``, which put
      the dense region at the start of the segment and the coarse region at the
      merger -- backwards for a chirp. It survived testing only because the
      test source's merger lay beyond the segment, where omega barely moves.
    * the second called ``source.angular_frequency`` once per grid point inside
      a Python loop. For an analytic source that is free; for pyEFPEHM every
      call re-evaluates the inspiral solution, and building one grid took
      longer than the dense path it was meant to replace.

    ``dt_max`` still caps the step, for the stretches where the carrier is so
    slow that the constellation's own motion becomes the limit.

    The grid covers `harmonic_windows`, one sub-grid per window, so a harmonic
    with a dead gap in the middle of its span gets no points there. Pass the
    SAME ``padding`` and windows to `reconstruct`: the spline is only valid
    inside them, and across a gap it interpolates between two blocks with a
    single cubic.
    """
    if dt_max is None:
        dt_max = 86400.0
    pieces = []
    for low, high in harmonic_windows(source, harmonic, t_start, t_end,
                                      padding):
        pieces.append(_grid_over_window(
            source, harmonic, low, high, delta_phi, dt_max, n_probe, growth,
            max_step_scale))
    if not pieces:
        return np.array([])
    return np.unique(np.concatenate(pieces))


def harmonic_windows(source, harmonic, t_start, t_end, padding=0.0):
    """The stretches of [t_start, t_end] a harmonic can contribute to.

    Prefers ``support_blocks`` over ``support``: a harmonic's live set is not
    always one interval, and the outer hull then spans a dead gap. ``padding``
    widens each block by the channel's delay spread -- see `delay_padding`.
    """
    blocks = getattr(source, 'support_blocks', None)
    if blocks is not None:
        windows = blocks(harmonic)
    else:
        support = getattr(source, 'support', None)
        windows = [support(harmonic)] if support is not None else [(t_start,
                                                                    t_end)]
    out = []
    for low, high in windows:
        low = max(t_start, low - padding)
        high = min(t_end, high + padding)
        if high > low:
            if out and low <= out[-1][1]:
                out[-1] = (out[-1][0], max(out[-1][1], high))
            else:
                out.append((low, high))
    return out


def _grid_over_window(source, harmonic, t_start, t_end, delta_phi, dt_max,
                      n_probe, growth, max_step_scale):
    probe = np.linspace(t_start, t_end, int(n_probe))
    phase = np.asarray(source.carrier_phase(harmonic, probe), dtype=float)
    if not np.all(np.diff(phase) > 0):          # phase must be monotone to invert
        phase = np.maximum.accumulate(phase)
    total = phase[-1] - phase[0]
    if not np.isfinite(total) or total <= 0:
        return np.linspace(t_start, t_end, 2)

    # Walk the carrier phase BACKWARDS from the merger, letting the step grow
    # away from it. Cornish & Littenberg's own optimisation, and the direction
    # matters: growing forwards from t_start puts the dense region where the
    # signal is slowest and the coarse region at the merger.
    if growth <= 1.0:
        wanted = np.arange(phase[0], phase[-1], float(delta_phi))
    else:
        wanted, value, step = [], phase[-1], float(delta_phi)
        limit = float(max_step_scale) * float(delta_phi)
        while value > phase[0]:
            wanted.append(value)
            value -= step
            step = min(step * growth, limit)
        wanted = np.asarray(wanted[::-1])
    grid = np.interp(wanted, phase, probe)
    grid = np.unique(np.concatenate(([t_start], grid, [t_end])))

    gaps = np.diff(grid)
    if np.any(gaps > dt_max):                   # refill where the carrier is slow
        extra = [np.arange(a, b, dt_max)
                 for a, b in zip(grid[:-1], grid[1:]) if b - a > dt_max]
        grid = np.unique(np.concatenate([grid] + extra))
    return grid[(grid >= t_start) & (grid <= t_end)]


def _link_factors(sample, lamb, beta, velocity_order=1):
    """Prefactor, Doppler weights and the two sampling delays, per link."""
    u_hat, v_hat, k_hat = polarization_basis(lamb, beta)
    xi_plus, xi_cross = antenna_pattern(sample.n_hat, u_hat, v_hat)
    denominator = 1 - np.einsum('nla,a->nl', sample.n_hat, k_hat)
    prefactor = np.stack((xi_plus, xi_cross), axis=-1) / (
        2 * denominator[..., None])
    receivers = np.array([link[0] - 1 for link in sample.links])
    emitters = np.array([link[1] - 1 for link in sample.links])
    tau_emit = sample.ltt + np.einsum(
        'nla,a->nl', sample.r_emit, k_hat) / C_SI
    tau_recv = np.einsum(
        'nla,a->nl', sample.position[:, receivers], k_hat) / C_SI
    if velocity_order:
        eps1, eps2 = doppler_factors(
            k_hat, sample.n_hat, sample.velocity[:, emitters],
            sample.velocity[:, receivers])
    else:
        eps1 = eps2 = np.zeros_like(sample.ltt)
    return prefactor, 1 + eps1, 1 + eps2, tau_emit, tau_recv


def chain_delay(orbit, times, chain, links=LINK_ORDER, iterations=3,
                cache=None):
    """Net delay of one operator chain, summed from retarded light times.

    ``D_ij`` delays by the light travel time of the link received at i from j,
    evaluated at the time the chain has already reached:

        (D_ab D_cd) x (t) = x(t - L_ab(t) - L_cd(t - L_ab(t)))

    This deliberately does not call ``build_shifts``: it needs the delay at a
    few hundred sparse times, not a full-cadence array, and PyTDI's composition
    goes through ``dsp.timeshift``, which requires one delay sample per output
    sample. Summing retarded light times directly is the alternative the plan
    calls for, and Speri does the same thing for response purposes.

    Two things make this cheap enough to sit inside a likelihood:

    * only the single link being traversed is solved, not all six -- a naive
      version asked ``sample_constellation`` for the whole constellation and
      threw five sixths of it away;
    * ``cache`` memoises on the chain PREFIX. A Michelson combination's chains
      are nested (``D_12``, ``D_12 D_21``, ``D_12 D_21 D_13``, ...), so
      recomputing each from scratch repeats almost all of the work. With the
      cache the cost is the number of distinct prefixes, not the sum of the
      chain lengths.
    """
    if cache is None:
        cache = {}
    chain = tuple(chain)
    if chain in cache:
        return cache[chain]
    if not chain:
        cache[chain] = np.zeros(len(times))
        return cache[chain]

    total = chain_delay(orbit, times, chain[:-1], links, iterations, cache)
    kind, indices = chain[-1].split('_')
    link = (int(indices[0]), int(indices[1]))
    step = np.zeros(len(times))
    for _ in range(iterations):
        probe = sample_constellation(times - total - step, orbit,
                                     links=(link,))
        step = probe.ltt[:, 0]
    cache[chain] = total + (step if kind == 'D' else -step)
    return cache[chain]


def sparse_channel(source, harmonic, grid, terms, orbit, lamb, beta,
                   velocity_order=1, links=LINK_ORDER):
    """Evaluate one harmonic's contribution to one channel on ``grid``.

    Returns the complex bracket B(t) with the carrier factored out, so the
    channel is ``Re[B(t) exp(i Phi(t))]``. B varies on the orbital timescale,
    which is what makes the sparse grid legitimate.
    """
    link_index = {tuple(link): i for i, link in enumerate(links)}
    chains = sorted({term.operators for term in terms}, key=len)
    bracket = np.zeros(len(grid), dtype=complex)
    phi0 = source.carrier_phase(harmonic, grid)
    cache = {}

    for chain in chains:
        shifted = grid - chain_delay(orbit, grid, chain, links, cache=cache)
        order = np.argsort(shifted)
        inverse = np.argsort(order)
        sample = sample_constellation(shifted[order], orbit, links=links)
        pref, w1, w2, tau_e, tau_r = _link_factors(
            sample, lamb, beta, velocity_order)
        pref, w1, w2 = pref[inverse], w1[inverse], w2[inverse]
        tau_e, tau_r = tau_e[inverse], tau_r[inverse]

        for term in terms:
            if term.operators != chain:
                continue
            j = link_index[tuple(term.link)]
            for tau, weight, sign in ((tau_e[:, j], w1[:, j], +1.0),
                                      (tau_r[:, j], w2[:, j], -1.0)):
                t_query = shifted - tau
                amp_p, amp_c = source.amplitude(harmonic, t_query)
                dphi = source.carrier_phase(harmonic, t_query) - phi0
                bracket += (term.coefficient * sign * weight
                            * (pref[:, j, 0] * amp_p + pref[:, j, 1] * amp_c)
                            * np.exp(1j * dphi))
    return bracket


def reconstruct_complex(source, harmonic, grid, bracket, times,
                        support=None):
    """Spline the slow bracket onto ``times`` and reattach the carrier.

    Returns the complex analytic signal; `reconstruct` is its real part. The
    analytic form is what a narrow-band search wants: it can be heterodyned
    and sampled at the envelope's bandwidth rather than at the cadence needed
    to resolve the GW carrier.

    The real and imaginary parts of the bracket are splined SEPARATELY rather
    than its amplitude and phase. Cornish & Littenberg spline amplitude and
    phase and then need explicit zero-crossing handling, because
    ``unwrap(angle(B))`` jumps wherever ``|B|`` passes through zero --
    measured here at over a radian between adjacent grid points for a real
    waveform's strongest harmonic, which a cubic spline then turns into
    garbage. The bracket is slowly varying by construction, so its real and
    imaginary parts are too, and splining them needs no unwrapping and no sign
    bookkeeping.

    ``support`` restricts the output to where the harmonic can contribute at
    all, so that the spline is never read outside the grid it was built on. It
    takes one ``(low, high)`` pair or a sequence of them, since a harmonic's
    live set need not be one interval. Pass the windows `harmonic_windows`
    returns for the same ``padding`` the grid was built with.
    """
    from scipy.interpolate import CubicSpline
    times = np.asarray(times, dtype=float)
    carrier = np.exp(1j * source.carrier_phase(harmonic, times))
    if support is None:
        value = (CubicSpline(grid, np.real(bracket))(times)
                 + 1j * CubicSpline(grid, np.imag(bracket))(times))
        return value * carrier

    # One spline per window, not one across all of them: a harmonic with a
    # dead gap has no grid points in it, and a single spline would then join
    # the two blocks with one cubic reaching across the gap. Its end
    # conditions leak back into the block edges, which is exactly where the
    # error lives. Measured on the 16-harmonic case, splining per window
    # instead of once takes the mismatch from 5.69e-04 to 4.83e-04.
    out = np.zeros(times.shape, dtype=complex)
    for low, high in ([support] if np.ndim(support) == 1 else support):
        nodes = (grid >= low) & (grid <= high)
        want = (times >= low) & (times <= high)
        if np.count_nonzero(nodes) < 4 or not np.any(want):
            continue
        piece = grid[nodes]
        value = (CubicSpline(piece, np.real(bracket)[nodes])(times[want])
                 + 1j * CubicSpline(piece, np.imag(bracket)[nodes])(times[want]))
        out[want] = value * carrier[want]
    return out


def reconstruct(source, harmonic, grid, bracket, times, support=None):
    """The real TDI channel: the real part of `reconstruct_complex`.

    Defined this way rather than duplicating the spline, so the two cannot
    drift apart.
    """
    return np.real(reconstruct_complex(source, harmonic, grid, bracket, times,
                                       support))


class SparseGeometry:
    """Everything on the sparse grid that does not depend on sky or waveform.

    Profiling the first working version showed ~90% of its time inside the
    orbit: light-cone solves, position and velocity splines. None of that
    depends on the source. In parameter estimation the orbit is fixed and the
    grid can be held fixed too, so all of it belongs outside the likelihood.

    What is left inside the likelihood is one einsum per sky direction and the
    waveform evaluations themselves.

    Build once per (orbit, grid, combination); reuse across every likelihood
    call.
    """

    def __init__(self, orbit, grid, terms, links=LINK_ORDER,
                 velocity_order=1):
        self.grid = np.asarray(grid, dtype=float)
        self.links = tuple(tuple(link) for link in links)
        self.velocity_order = velocity_order
        link_index = {link: i for i, link in enumerate(self.links)}
        cache = {}
        self.chains = {}

        for chain in sorted({term.operators for term in terms}, key=len):
            shifted = self.grid - chain_delay(orbit, self.grid, chain,
                                              self.links, cache=cache)
            order = np.argsort(shifted)
            inverse = np.argsort(order)
            sample = sample_constellation(shifted[order], orbit,
                                          links=self.links)
            receivers = [link[0] - 1 for link in self.links]
            emitters = [link[1] - 1 for link in self.links]
            entry = {
                'shifted': shifted,
                'n_hat': sample.n_hat[inverse],
                'ltt': sample.ltt[inverse],
                'r_emit': sample.r_emit[inverse],
                'r_recv': sample.position[:, receivers][inverse],
                'terms': [(link_index[tuple(t.link)], t.coefficient)
                          for t in terms if t.operators == chain],
            }
            if velocity_order:
                entry['v_emit'] = sample.velocity[:, emitters][inverse]
                entry['v_recv'] = sample.velocity[:, receivers][inverse]
            self.chains[chain] = entry

    @property
    def size(self):
        return len(self.grid)


def sparse_channel_cached(source, harmonic, geometry, lamb, beta):
    """`sparse_channel` with the orbit work lifted out into `geometry`.

    Identical output; the only difference is that the light-cone solves and
    the orbit splines happened once, when the geometry was built.
    """
    u_hat, v_hat, k_hat = polarization_basis(lamb, beta)
    bracket = np.zeros(geometry.size, dtype=complex)
    phi0 = source.carrier_phase(harmonic, geometry.grid)

    for entry in geometry.chains.values():
        n_hat = entry['n_hat']
        xi_plus, xi_cross = antenna_pattern(n_hat, u_hat, v_hat)
        denominator = 2 * (1 - np.einsum('nla,a->nl', n_hat, k_hat))
        pref_plus, pref_cross = xi_plus / denominator, xi_cross / denominator
        tau_emit = entry['ltt'] + np.einsum(
            'nla,a->nl', entry['r_emit'], k_hat) / C_SI
        tau_recv = np.einsum('nla,a->nl', entry['r_recv'], k_hat) / C_SI
        if geometry.velocity_order:
            eps1, eps2 = doppler_factors(k_hat, n_hat, entry['v_emit'],
                                         entry['v_recv'])
            w1, w2 = 1 + eps1, 1 + eps2
        else:
            w1 = w2 = np.ones_like(tau_emit)

        shifted = entry['shifted']
        for j, coefficient in entry['terms']:
            for tau, weight, sign in ((tau_emit[:, j], w1[:, j], +1.0),
                                      (tau_recv[:, j], w2[:, j], -1.0)):
                query = shifted - tau
                amp_p, amp_c = source.amplitude(harmonic, query)
                phase = source.carrier_phase(harmonic, query) - phi0
                bracket += (coefficient * sign * weight
                            * (pref_plus[:, j] * amp_p
                               + pref_cross[:, j] * amp_c)
                            * np.exp(1j * phase))
    return bracket


class StackedGeometry:
    """`SparseGeometry` with every operator chain stacked into one array.

    Profiling the per-chain version showed half its remaining time inside
    `numpy.einsum` -- 135 calls per channel, each contracting a (n_grid, 6, 3)
    array against a 3-vector. At that size the dispatch costs more than the
    arithmetic. Stacking the chains turns those into a handful of large
    matrix products, and lets the waveform be queried once per channel instead
    of twice per term.

    Layout, with C chains, G grid points and L links:

        n_hat, r_emit, r_recv, v_*   (C, G, L, 3)
        ltt, shifted                 (C, G, L) and (C, G)
        term table                   (chain index, link index, coefficient)
    """

    def __init__(self, orbit, grid, terms, links=LINK_ORDER,
                 velocity_order=1):
        base = SparseGeometry(orbit, grid, terms, links, velocity_order)
        self.grid = base.grid
        self.links = base.links
        self.velocity_order = velocity_order
        chains = list(base.chains)
        self.n_chain, self.n_grid = len(chains), len(base.grid)
        self.n_link = len(self.links)

        def stack(key):
            return np.stack([base.chains[c][key] for c in chains])

        self.n_hat = stack('n_hat')
        self.r_emit = stack('r_emit')
        self.r_recv = stack('r_recv')
        self.ltt = stack('ltt')
        self.shifted = stack('shifted')
        if velocity_order:
            self.v_emit = stack('v_emit')
            self.v_recv = stack('v_recv')

        rows = [(i, j, coefficient)
                for i, chain in enumerate(chains)
                for j, coefficient in base.chains[chain]['terms']]
        self.term_chain = np.array([r[0] for r in rows])
        self.term_link = np.array([r[1] for r in rows])
        self.term_coefficient = np.array([r[2] for r in rows], dtype=float)

    def _project(self, array, vector):
        """Contract a (C, G, L, 3) stack against one 3-vector, in one call."""
        return (array.reshape(-1, 3) @ vector).reshape(
            self.n_chain, self.n_grid, self.n_link)


def sparse_channel_stacked(source, harmonic, geometry, lamb, beta):
    """One channel, one waveform call, a handful of matrix products."""
    u_hat, v_hat, k_hat = polarization_basis(lamb, beta)
    n_dot_u = geometry._project(geometry.n_hat, u_hat)
    n_dot_v = geometry._project(geometry.n_hat, v_hat)
    n_dot_k = geometry._project(geometry.n_hat, k_hat)

    denominator = 2 * (1 - n_dot_k)
    pref_plus = (n_dot_u ** 2 - n_dot_v ** 2) / denominator
    pref_cross = (2 * n_dot_u * n_dot_v) / denominator
    tau_emit = geometry.ltt + geometry._project(geometry.r_emit, k_hat) / C_SI
    tau_recv = geometry._project(geometry.r_recv, k_hat) / C_SI

    if geometry.velocity_order:
        k_emit = geometry._project(geometry.v_emit, k_hat)
        k_recv = geometry._project(geometry.v_recv, k_hat)
        n_v_recv = np.einsum('cgla,cgla->cgl', geometry.n_hat, geometry.v_recv)
        n_v_mix = np.einsum('cgla,cgla->cgl', geometry.n_hat,
                            geometry.v_emit - 2 * geometry.v_recv)
        w1 = 1 + (-k_emit + n_v_recv) / C_SI
        w2 = 1 + (-k_recv + n_v_mix) / C_SI
    else:
        w1 = w2 = np.ones_like(tau_emit)

    chain, link = geometry.term_chain, geometry.term_link
    base = geometry.shifted[chain]                       # (T, G)
    # emission and reception queries for every term, in one array
    query = np.concatenate((base - tau_emit[chain, :, link],
                            base - tau_recv[chain, :, link]))
    amp_p, amp_c = source.amplitude(harmonic, query)
    phase = source.carrier_phase(harmonic, query)
    phase -= source.carrier_phase(harmonic, geometry.grid)[None, :]

    weight = np.concatenate((w1[chain, :, link], -w2[chain, :, link]))
    plus = np.concatenate([pref_plus[chain, :, link]] * 2)
    cross = np.concatenate([pref_cross[chain, :, link]] * 2)
    coefficient = np.concatenate([geometry.term_coefficient] * 2)[:, None]
    contribution = (coefficient * weight * (plus * amp_p + cross * amp_c)
                    * np.exp(1j * phase))
    return contribution.sum(axis=0)


class TermGeometry:
    """Geometry gathered down to the (chain, link) pairs a channel uses.

    `StackedGeometry` keeps every link of every chain and then indexes out the
    ones the terms need. For a Michelson combination that is 15 chains x 6
    links = 90 entries to build the sky projections for, of which 16 are used:
    two thirds of the arithmetic is thrown away. Gathering at construction
    leaves arrays of shape (T, G) with T the term count, and the per-likelihood
    work becomes proportional to the terms rather than to chains x links.

    This is the form to hand a likelihood: build once per (orbit, grid,
    combination), then each call is a few (T, G) matrix products, one waveform
    evaluation and one complex exponential.
    """

    def __init__(self, orbit, grid, terms, links=LINK_ORDER,
                 velocity_order=1):
        stacked = StackedGeometry(orbit, grid, terms, links, velocity_order)
        chain, link = stacked.term_chain, stacked.term_link
        self.grid = stacked.grid
        self.velocity_order = velocity_order
        self.coefficient = stacked.term_coefficient[:, None]
        self.shifted = stacked.shifted[chain]                    # (T, G)
        self.ltt = stacked.ltt[chain, :, link]                   # (T, G)
        self.n_hat = stacked.n_hat[chain, :, link]               # (T, G, 3)
        self.r_emit = stacked.r_emit[chain, :, link]
        self.r_recv = stacked.r_recv[chain, :, link]
        if velocity_order:
            self.v_emit = stacked.v_emit[chain, :, link]
            self.v_recv = stacked.v_recv[chain, :, link]
            self.n_dot_v_recv = np.einsum('tga,tga->tg', self.n_hat,
                                          self.v_recv)
            self.n_dot_v_mix = np.einsum('tga,tga->tg', self.n_hat,
                                         self.v_emit - 2 * self.v_recv)
        self.shape = self.ltt.shape

    def _project(self, array, vector):
        return (array.reshape(-1, 3) @ vector).reshape(self.shape)


def sparse_channel_terms(source, harmonic, geometry, lamb, beta):
    """The narrowest per-likelihood path: work scales with the term count."""
    u_hat, v_hat, k_hat = polarization_basis(lamb, beta)
    n_dot_u = geometry._project(geometry.n_hat, u_hat)
    n_dot_v = geometry._project(geometry.n_hat, v_hat)
    denominator = 2 * (1 - geometry._project(geometry.n_hat, k_hat))
    pref_plus = (n_dot_u ** 2 - n_dot_v ** 2) / denominator
    pref_cross = (2 * n_dot_u * n_dot_v) / denominator

    tau_emit = geometry.ltt + geometry._project(geometry.r_emit, k_hat) / C_SI
    tau_recv = geometry._project(geometry.r_recv, k_hat) / C_SI
    if geometry.velocity_order:
        w1 = 1 + (-geometry._project(geometry.v_emit, k_hat)
                  + geometry.n_dot_v_recv) / C_SI
        w2 = 1 + (-geometry._project(geometry.v_recv, k_hat)
                  + geometry.n_dot_v_mix) / C_SI
    else:
        w1 = w2 = 1.0

    query = np.concatenate((geometry.shifted - tau_emit,
                            geometry.shifted - tau_recv))
    amp_p, amp_c = source.amplitude(harmonic, query)
    phase = source.carrier_phase(harmonic, query)
    phase -= source.carrier_phase(harmonic, geometry.grid)[None, :]

    weight = np.concatenate((np.broadcast_to(w1, geometry.shape),
                             -np.broadcast_to(w2, geometry.shape)))
    plus = np.concatenate((pref_plus, pref_plus))
    cross = np.concatenate((pref_cross, pref_cross))
    coefficient = np.concatenate((geometry.coefficient,
                                  geometry.coefficient))
    return ((coefficient * weight * (plus * amp_p + cross * amp_c)
             * np.exp(1j * phase))).sum(axis=0)
