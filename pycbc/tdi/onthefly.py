# Copyright (C) 2026  Shichao Wu, Alex Nitz
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Sparse "TDI on the fly" evaluation of TDI channels.

Cornish & Littenberg, PRD 112, 102007. For a harmonic
h(t) = Re[A(t) exp(i Phi(t))],

    h(t - D) = Re[ exp(i Phi(t)) * A(t - D) exp(i (Phi(t - D) - Phi(t))) ]

so factoring exp(i Phi(t)) out of the whole combination leaves a bracket
varying on the orbital timescale. The bracket is sampled on a few hundred
points per year and splined; the carrier is reattached at full cadence.

Delays are applied to the waveform's time argument and the orbit quantities
are evaluated at the shifted times, so the constellation is not frozen. No
intermediate eta series is built and `dsp.timeshift` is never called, which
is why the combination layer exposes (coefficient, net shift) pairs.
"""

from dataclasses import dataclass

import numpy as np

from pycbc.tdi.response import (C_SI, LINK_ORDER, antenna_pattern,
                                doppler_factors, polarization_basis,
                                sample_constellation)

YEAR = 3.15581498e7


@dataclass(frozen=True)
class _ChannelTerm:
    """A TDI term tagged with its output channel index."""

    link: tuple
    coefficient: float
    operators: tuple
    channel: int


class HarmonicSource:
    """Protocol: a source whose harmonics are amplitude times a carrier.

    Implementations must provide, for a harmonic label ``n``:

    ``amplitude(n, t)``  complex A(t), the two polarisations as a pair
    ``carrier_phase(n, t)``  real Phi(t)
    ``angular_frequency(n, t)``  real dPhi/dt, used to size the grid
    """


def delay_padding(orbit, times, terms, links=LINK_ORDER):
    """How far outside its own window a harmonic still reaches in a channel.

    The channel at ``t`` queries the waveform at ``t - net_shift - tau``,
    tau being ``L + k.r_emit/c`` for the emission term and ``k.r_recv/c`` for
    the reception term. Bounding ``|k.r| <= |r|`` drops the sky dependence, so
    one grid serves a whole likelihood run. About 570 s for LISA.

    Padding the grid by this much does not improve the reconstruction, so
    callers default to zero; see test_tdi_onthefly.
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

    The carrier phase is evaluated once on a probe grid and inverted, so the
    grid densifies wherever omega rises. ``dt_max`` caps the step where the
    carrier is slow enough that the constellation's motion sets the scale.

    One sub-grid per window of `harmonic_windows`. Pass the same ``padding``
    to `reconstruct`; the spline is valid only inside the windows.
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

    Uses ``support_blocks`` where the source has it. A live set spanning two
    intervals with a dead gap between them would otherwise be covered by its
    outer hull. ``padding`` widens each block by `delay_padding`.
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
    if not np.all(np.diff(phase) > 0):          # must be monotone to invert
        phase = np.maximum.accumulate(phase)
    total = phase[-1] - phase[0]
    if not np.isfinite(total) or total <= 0:
        return np.linspace(t_start, t_end, 2)

    # Walk the carrier phase backwards from the merger, letting the step grow
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
    if np.any(gaps > dt_max):                   # refill where the carrier lags
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

    ``D_ij`` delays by the travel time of the link received at i from j,
    evaluated at the time the chain has already reached:

        (D_ab D_cd) x (t) = x(t - L_ab(t) - L_cd(t - L_ab(t)))

    ``build_shifts`` is unusable here: PyTDI composes through
    ``dsp.timeshift``, which wants one delay sample per output sample, while
    this needs a few hundred sparse times.

    Only the link being traversed is solved. ``cache`` memoises on the chain
    prefix, so nested Michelson chains cost one solve per distinct prefix.
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

    Returns the bracket B(t) with the carrier factored out, so the channel is
    ``Re[B(t) exp(i Phi(t))]``.
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

    Returns the analytic signal, of which `reconstruct` is the real part. A
    narrow-band search can heterodyne it and sample at the envelope's
    bandwidth instead of the carrier's.

    Real and imaginary parts are splined separately. Cornish & Littenberg
    spline amplitude and phase, which needs explicit handling wherever |B|
    crosses zero and ``unwrap(angle(B))`` jumps; on a real waveform those
    jumps exceed a radian between adjacent grid points.

    ``support`` takes one ``(low, high)`` pair or a sequence of them, since a
    harmonic's live set need not be one interval. Pass what
    `harmonic_windows` returned for the padding the grid was built with.
    """
    from scipy.interpolate import CubicSpline
    times = np.asarray(times, dtype=float)
    carrier = np.exp(1j * source.carrier_phase(harmonic, times))
    if support is None:
        value = (CubicSpline(grid, np.real(bracket))(times)
                 + 1j * CubicSpline(grid, np.imag(bracket))(times))
        return value * carrier

    # A dead gap holds no grid points, so one spline across all the windows
    # joins the blocks with a single cubic whose end conditions leak back into
    # the block edges. One spline per window instead.
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
    """The real TDI channel: the real part of `reconstruct_complex`."""
    return np.real(reconstruct_complex(source, harmonic, grid, bracket, times,
                                       support))


class SparseGeometry:
    """Everything on the sparse grid that does not depend on sky or waveform.

    The light-cone solves and orbit splines are ~90% of a channel evaluation
    and none of it depends on the source, so it is lifted out of the
    likelihood. Build once per (orbit, grid, combination) and reuse.
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

    The per-chain form spends half its time in `numpy.einsum` dispatch: 135
    calls per channel, each contracting a (n_grid, 6, 3) array against a
    3-vector. Stacking turns them into a few large matrix products and lets
    the waveform be queried once per channel.

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

        chain_index = {chain: i for i, chain in enumerate(chains)}
        link_index = {link: i for i, link in enumerate(self.links)}
        rows = [(chain_index[term.operators], link_index[tuple(term.link)],
                 term.coefficient, getattr(term, 'channel', 0))
                for term in terms]
        self.term_chain = np.array([r[0] for r in rows])
        self.term_link = np.array([r[1] for r in rows])
        self.term_coefficient = np.array([r[2] for r in rows], dtype=float)
        self.term_channel = np.array([r[3] for r in rows], dtype=int)

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

    `StackedGeometry` builds sky projections for every link of every chain:
    90 entries for a Michelson combination, of which the terms use 16.
    Gathering at construction leaves (T, G) arrays for T terms, so the
    per-call work scales with the term count.

    The form to hand a likelihood. Each call is then a few (T, G) matrix
    products, one waveform evaluation and one complex exponential.
    """

    def __init__(self, orbit, grid, terms, links=LINK_ORDER,
                 velocity_order=1):
        terms = tuple(terms)
        if not terms:
            raise ValueError("terms must not be empty")
        self.grid = np.asarray(grid, dtype=float)
        self.velocity_order = velocity_order
        self.coefficient = np.asarray(
            [term.coefficient for term in terms], dtype=float)[:, None]
        self.term_channel = np.asarray(
            [getattr(term, 'channel', 0) for term in terms], dtype=int)

        allowed = {tuple(link) for link in links}
        if any(tuple(term.link) not in allowed for term in terms):
            raise ValueError("a TDI term uses a link absent from links")
        cache = {}
        gathered = {key: [None] * len(terms) for key in (
            'shifted', 'ltt', 'n_hat', 'r_emit', 'r_recv')}
        if velocity_order:
            gathered.update(v_emit=[None] * len(terms),
                            v_recv=[None] * len(terms))

        # Evaluate only the links actually referenced by each chain.  The
        # previous implementation constructed all six links for every chain
        # and then gathered these same rows, multiplying the dominant orbit
        # work and peak memory by almost six for Michelson TDI-2.
        chains = sorted({term.operators for term in terms}, key=len)
        for chain in chains:
            selected = [(row, term) for row, term in enumerate(terms)
                        if term.operators == chain]
            used_links = tuple(dict.fromkeys(tuple(term.link)
                                             for _, term in selected))
            shifted = self.grid - chain_delay(
                orbit, self.grid, chain, links, cache=cache)
            order = np.argsort(shifted)
            inverse = np.argsort(order)
            sample = sample_constellation(
                shifted[order], orbit, links=used_links)
            index = {link: i for i, link in enumerate(used_links)}
            for row, term in selected:
                link = tuple(term.link)
                column = index[link]
                receiver, emitter = link[0] - 1, link[1] - 1
                gathered['shifted'][row] = shifted
                gathered['ltt'][row] = sample.ltt[inverse, column]
                gathered['n_hat'][row] = sample.n_hat[inverse, column]
                gathered['r_emit'][row] = sample.r_emit[inverse, column]
                gathered['r_recv'][row] = sample.position[inverse, receiver]
                if velocity_order:
                    gathered['v_emit'][row] = sample.velocity[inverse, emitter]
                    gathered['v_recv'][row] = sample.velocity[inverse, receiver]

        for key, values in gathered.items():
            setattr(self, key, np.stack(values))
        if velocity_order:
            self.n_dot_v_recv = np.einsum('tga,tga->tg', self.n_hat,
                                          self.v_recv)
            self.n_dot_v_mix = np.einsum('tga,tga->tg', self.n_hat,
                                         self.v_emit - 2 * self.v_recv)
        self.shape = self.ltt.shape

    def _project(self, array, vector):
        return (array.reshape(-1, 3) @ vector).reshape(self.shape)


class MultiChannelTermGeometry(TermGeometry):
    """One cached geometry shared by several TDI output channels.

    ``channel_terms`` maps channel labels to their delayed-link terms.  Orbit
    and light-cone work is performed once for the union of their operator
    chains, while each term retains the channel to which it contributes.
    Observation duration and sample cadence are deliberately absent from the
    interface: ``grid`` may be any increasing set of mission times.
    """

    def __init__(self, orbit, grid, channel_terms, links=LINK_ORDER,
                 velocity_order=1):
        self.channel_names = tuple(channel_terms)
        if not self.channel_names:
            raise ValueError("channel_terms must contain at least one channel")
        if len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError("channel names must be unique")
        tagged = []
        for channel, name in enumerate(self.channel_names):
            terms = tuple(channel_terms[name])
            if not terms:
                raise ValueError(f"channel {name!r} has no terms")
            tagged.extend(_ChannelTerm(term.link, term.coefficient,
                                       term.operators, channel)
                          for term in terms)
        super().__init__(orbit, grid, tagged, links=links,
                         velocity_order=velocity_order)


def _sparse_term_projection(geometry, lamb, beta):
    """Return delayed queries and polarization weights for sparse terms."""
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
    weight = np.concatenate((np.broadcast_to(w1, geometry.shape),
                             -np.broadcast_to(w2, geometry.shape)))
    plus = np.concatenate((pref_plus, pref_plus))
    cross = np.concatenate((pref_cross, pref_cross))
    coefficient = np.concatenate((geometry.coefficient,
                                  geometry.coefficient))
    return (query, coefficient * weight * plus,
            coefficient * weight * cross)


def _sparse_term_contributions(source, harmonic, geometry, lamb, beta):
    """Return each gathered term before reducing it into output channels."""
    query, plus, cross = _sparse_term_projection(geometry, lamb, beta)
    amp_p, amp_c = source.amplitude(harmonic, query)
    phase = source.carrier_phase(harmonic, query)
    phase -= source.carrier_phase(harmonic, geometry.grid)[None, :]
    return ((plus * amp_p + cross * amp_c)
            * np.exp(1j * phase))


def _reduce_sparse_contributions(contribution, geometry):
    """Reduce gathered term contributions into their output channels."""
    channels = np.concatenate((geometry.term_channel,
                               geometry.term_channel))
    output = np.zeros((len(geometry.channel_names), geometry.shape[1]),
                      dtype=complex)
    for index in range(len(output)):
        output[index] = contribution[channels == index].sum(axis=0)
    return dict(zip(geometry.channel_names, output, strict=True))


def sparse_channel_terms(source, harmonic, geometry, lamb, beta):
    """The narrowest one-channel path: work scales with the term count."""
    return _sparse_term_contributions(
        source, harmonic, geometry, lamb, beta).sum(axis=0)


def sparse_channels_terms(source, harmonic, geometry, lamb, beta):
    """Evaluate several channels with one source call and shared geometry.

    Parameters are the same as :func:`sparse_channel_terms`, except
    ``geometry`` must be a :class:`MultiChannelTermGeometry`.  The returned
    mapping contains one complex, carrier-factored bracket per channel.
    """
    if not isinstance(geometry, MultiChannelTermGeometry):
        raise TypeError("geometry must be a MultiChannelTermGeometry")
    contribution = _sparse_term_contributions(
        source, harmonic, geometry, lamb, beta)
    return _reduce_sparse_contributions(contribution, geometry)


def sparse_windowed_channels_terms(sources, harmonics, geometries,
                                   lamb, beta, source_batch_size=1):
    """Batch source evaluation across several pre-response frequency bands.

    Every ``source`` must be a frequency-window view exposing ``source`` and
    ``frequency_weight``. The shared underlying harmonic is evaluated once
    on the concatenated delayed queries, after which each window is applied
    at those same retarded source times. Thus batching changes call layout,
    not the order of the physical response operations.
    """
    sources = tuple(sources)
    harmonics = tuple(harmonics)
    geometries = tuple(geometries)
    source_batch_size = int(source_batch_size)
    if source_batch_size < 1:
        raise ValueError("source_batch_size must be positive")
    if not (len(sources) == len(harmonics) == len(geometries)):
        raise ValueError("sources, harmonics and geometries must match")
    outputs = [None] * len(sources)
    groups = {}
    group_counts = {}
    for index, (source, harmonic, geometry) in enumerate(zip(
            sources, harmonics, geometries, strict=True)):
        if not isinstance(geometry, MultiChannelTermGeometry):
            raise TypeError("geometry must be a MultiChannelTermGeometry")
        if not hasattr(source, "source") or not hasattr(
                source, "frequency_weight"):
            raise TypeError("sources must be frequency-window views")
        base_key = (id(source.source), harmonic)
        count = group_counts.get(base_key, 0)
        key = (*base_key, count // source_batch_size)
        groups.setdefault(key, []).append(index)
        group_counts[base_key] = count + 1

    for (_, harmonic, _), positions in groups.items():
        base = sources[positions[0]].source
        if any(sources[index].source is not base for index in positions):
            raise ValueError("frequency windows in one group must share a source")
        projections = [
            _sparse_term_projection(geometries[index], lamb, beta)
            for index in positions
        ]
        query_sizes = [query.size for query, _, _ in projections]
        grid_sizes = [len(geometries[index].grid) for index in positions]
        all_query = np.concatenate([
            query.reshape(-1) for query, _, _ in projections])
        all_grids = np.concatenate([
            geometries[index].grid for index in positions])
        combined = getattr(base, "harmonic_components", None)
        if combined is None:
            amp_p, amp_c = base.amplitude(harmonic, all_query)
            query_phase = base.carrier_phase(harmonic, all_query)
            omega = base.angular_frequency(harmonic, all_query)
        else:
            amp_p, amp_c, query_phase, omega = combined(
                harmonic, all_query)
        frequency = omega / (2 * np.pi)
        # The combined pyEFPEHM phase is defined only where a mode is live.
        # A grid point just outside that window can still factor a nonzero
        # delayed query, so use the source's continuous carrier on grids.
        grid_phase = base.carrier_phase(harmonic, all_grids)

        query_offset = grid_offset = 0
        for position, projection, query_size, grid_size in zip(
                positions, projections, query_sizes, grid_sizes, strict=True):
            query, plus, cross = projection
            query_slice = slice(query_offset, query_offset + query_size)
            grid_slice = slice(grid_offset, grid_offset + grid_size)
            shape = query.shape
            weight = sources[position].frequency_weight(
                frequency[query_slice]).reshape(shape)
            phase = query_phase[query_slice].reshape(shape)
            phase -= grid_phase[grid_slice][None, :]
            contribution = (
                plus * amp_p[query_slice].reshape(shape) * weight
                + cross * amp_c[query_slice].reshape(shape) * weight)
            contribution *= np.exp(1j * phase)
            outputs[position] = _reduce_sparse_contributions(
                contribution, geometries[position])
            query_offset += query_size
            grid_offset += grid_size
    return tuple(outputs)


def frequency_response_factors(geometry, frequencies, lamb, beta):
    """Adiabatic FD response factors on stationary-time geometry.

    The orbit and all TDI operator delays remain the same light-cone values
    used by the time-domain evaluator.  A harmonic at frequency ``f`` turns
    each total delay ``D`` into ``exp(-2 pi i f D)``.  The approximation is
    only that the source amplitude and the slowly moving constellation are
    held at that harmonic's stationary time, matching the SPA/SUA contract
    of frequency-domain pyEFPEHM.

    Returns
    -------
    dict
        ``channel -> (R_plus, R_cross)`` arrays.
    """
    if not isinstance(geometry, MultiChannelTermGeometry):
        raise TypeError("geometry must be a MultiChannelTermGeometry")
    frequencies = np.asarray(frequencies, dtype=float)
    if frequencies.shape != (geometry.shape[1],):
        raise ValueError("frequencies must have one value per geometry time")

    u_hat, v_hat, k_hat = polarization_basis(lamb, beta)
    n_dot_u = geometry._project(geometry.n_hat, u_hat)
    n_dot_v = geometry._project(geometry.n_hat, v_hat)
    denominator = 2 * (1 - geometry._project(geometry.n_hat, k_hat))
    pref_plus = (n_dot_u ** 2 - n_dot_v ** 2) / denominator
    pref_cross = (2 * n_dot_u * n_dot_v) / denominator
    tau_emit = (geometry.ltt
                + geometry._project(geometry.r_emit, k_hat) / C_SI)
    tau_recv = geometry._project(geometry.r_recv, k_hat) / C_SI
    if geometry.velocity_order:
        w1 = 1 + (-geometry._project(geometry.v_emit, k_hat)
                  + geometry.n_dot_v_recv) / C_SI
        w2 = 1 + (-geometry._project(geometry.v_recv, k_hat)
                  + geometry.n_dot_v_mix) / C_SI
    else:
        w1 = w2 = 1.0

    chain_delay_value = geometry.grid[None, :] - geometry.shifted
    omega = 2 * np.pi * frequencies[None, :]
    transfer = geometry.coefficient * (
        w1 * np.exp(-1j * omega * (chain_delay_value + tau_emit))
        - w2 * np.exp(-1j * omega * (chain_delay_value + tau_recv)))
    output = {}
    for index, name in enumerate(geometry.channel_names):
        selected = geometry.term_channel == index
        output[name] = (
            (transfer[selected] * pref_plus[selected]).sum(axis=0),
            (transfer[selected] * pref_cross[selected]).sum(axis=0),
        )
    return output


def interpolated_frequency_response_factors(
        orbit, channel_terms, times, frequencies, lamb, beta,
        relative_tolerance=1e-4, initial_points=513, amplitude_floor=1e-2,
        max_refinements=20, velocity_order=1, links=LINK_ORDER):
    """Evaluate slow FD response factors on an error-controlled subset.

    Source harmonics themselves are never interpolated here.  Only the two
    geometric factors multiplying their native ``h_plus`` and ``h_cross``
    are splined; exact midpoint evaluations recursively bisect intervals that
    miss the requested tolerance.
    """
    from scipy.interpolate import CubicSpline

    times = np.asarray(times, dtype=float)
    frequencies = np.asarray(frequencies, dtype=float)
    if times.shape != frequencies.shape or times.ndim != 1:
        raise ValueError("times and frequencies must be matching 1-D arrays")
    if len(times) < 2 or np.any(np.diff(frequencies) <= 0):
        raise ValueError("at least two increasing frequencies are required")
    if relative_tolerance <= 0:
        raise ValueError("relative_tolerance must be positive")
    if not 0 <= amplitude_floor <= 1:
        raise ValueError("amplitude_floor must lie in [0, 1]")
    if max_refinements < 0:
        raise ValueError("max_refinements must not be negative")

    channel_names = tuple(channel_terms)

    def evaluate(indices):
        geometry = MultiChannelTermGeometry(
            orbit, times[indices], channel_terms, links=links,
            velocity_order=velocity_order)
        factors = frequency_response_factors(
            geometry, frequencies[indices], lamb, beta)
        return np.stack([value for name in channel_names
                         for value in factors[name]])

    count = len(frequencies)
    control = np.unique(np.linspace(
        0, count - 1, min(count, max(2, int(initial_points))), dtype=int))
    values = evaluate(control)
    evaluated = len(control)
    deepest = 0
    active_left, active_right = control[:-1], control[1:]
    for depth in range(int(max_refinements) + 1):
        # A midpoint-only test can alias a smooth oscillation.  Probe the
        # quarter points as well, then split every failed interval at all of
        # its probes.  This remains much cheaper than evaluating every source
        # bin, while making the error estimate substantially harder to fool.
        probes = []
        owners = []
        for owner, (left, right) in enumerate(zip(
                active_left, active_right, strict=True)):
            candidates = np.unique(np.asarray((
                (3 * left + right) // 4,
                (left + right) // 2,
                (left + 3 * right) // 4), dtype=int))
            candidates = candidates[
                (candidates > left) & (candidates < right)]
            probes.extend(candidates)
            owners.extend([owner] * len(candidates))
        probes = np.asarray(probes, dtype=int)
        owners = np.asarray(owners, dtype=int)
        if not len(probes):
            break
        exact = evaluate(probes)
        evaluated += len(probes)
        predicted = np.stack([
            CubicSpline(frequencies[control], row)(frequencies[probes])
            for row in values
        ])
        peak = np.maximum(np.max(np.abs(values), axis=1),
                          np.max(np.abs(exact), axis=1))[:, None]
        local = np.maximum(np.abs(exact), amplitude_floor * peak)
        failed_probe = np.any(
            np.abs(exact - predicted) > relative_tolerance * local, axis=0)
        failed_interval = np.zeros(len(active_left), dtype=bool)
        failed_interval[owners[failed_probe]] = True
        if not np.any(failed_interval):
            break
        if depth == max_refinements:
            raise RuntimeError(
                "frequency-response interpolation did not converge")
        selected = failed_interval[owners]
        insert = probes[selected]
        joined = np.concatenate((control, insert))
        joined_values = np.concatenate((values, exact[:, selected]), axis=1)
        order = np.argsort(joined)
        control, values = joined[order], joined_values[:, order]
        next_left, next_right = [], []
        for owner in np.flatnonzero(failed_interval):
            interior = np.sort(probes[owners == owner])
            edges = np.concatenate((
                [active_left[owner]], interior, [active_right[owner]]))
            usable = np.diff(edges) > 1
            next_left.extend(edges[:-1][usable])
            next_right.extend(edges[1:][usable])
        active_left = np.asarray(next_left, dtype=int)
        active_right = np.asarray(next_right, dtype=int)
        deepest = depth + 1

    complete = np.stack([
        CubicSpline(frequencies[control], row)(frequencies)
        for row in values
    ])
    output = {
        name: (complete[2 * index], complete[2 * index + 1])
        for index, name in enumerate(channel_names)
    }
    diagnostics = {
        'frequency_points': count,
        'control_points': len(control),
        'evaluated_points': evaluated,
        'deepest_refinement': deepest,
        'relative_tolerance': relative_tolerance,
    }
    return output, diagnostics


def project_frequency_tdi(source, orbit, channel_terms, frequencies,
                          lamb, beta, velocity_order=1, links=LINK_ORDER,
                          chunk_size=4096, response_tolerance=None,
                          response_initial_points=513,
                          return_diagnostics=False):
    """Project source FD harmonics directly into TDI frequency samples.

    ``frequencies`` may be any increasing grid. The source supplies its
    native SPA/SUA harmonics and stationary times; the response is evaluated
    only at those times. No inverse FFT or observation-length array is
    created. By default every response sample is evaluated exactly. Setting
    ``response_tolerance`` enables error-controlled interpolation of only the
    geometric plus/cross factors, never of the source harmonics.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    if (frequencies.ndim != 1 or len(frequencies) < 2
            or np.any(np.diff(frequencies) <= 0)):
        raise ValueError("frequencies must be one-dimensional and increasing")
    harmonic_method = getattr(source, 'frequency_harmonics', None)
    if harmonic_method is None:
        raise TypeError("source has no frequency_harmonics method")
    if chunk_size is None:
        chunk_size = len(frequencies)
    chunk_size = int(chunk_size)
    if chunk_size < 2:
        raise ValueError("chunk_size must be at least two")
    output = {name: np.zeros(len(frequencies), dtype=complex)
              for name in channel_terms}
    diagnostics = {}
    if response_tolerance is not None:
        for harmonic, item in harmonic_method(frequencies).items():
            indices = np.asarray(item['indices'], dtype=int)
            if not len(indices):
                continue
            factors, detail = interpolated_frequency_response_factors(
                orbit, channel_terms, item['time'], item['frequency'],
                lamb, beta, relative_tolerance=response_tolerance,
                initial_points=response_initial_points,
                velocity_order=velocity_order, links=links)
            diagnostics[harmonic] = detail
            for name, (plus, cross) in factors.items():
                contribution = plus * item['plus'] + cross * item['cross']
                np.add.at(output[name], indices, contribution)
        return (output, diagnostics) if return_diagnostics else output

    for start in range(0, len(frequencies), chunk_size):
        stop = min(len(frequencies), start + chunk_size)
        # pyEFPEHM's frequency interface requires at least two increasing
        # points. Borrow the preceding point for a one-sample final block and
        # discard it afterwards.
        block_start = start if stop - start > 1 else start - 1
        block = frequencies[block_start:stop]
        discard = start - block_start
        for item in harmonic_method(block).values():
            indices = np.asarray(item['indices'], dtype=int)
            keep = indices >= discard
            indices = indices[keep]
            if not len(indices):
                continue
            geometry = MultiChannelTermGeometry(
                orbit, np.asarray(item['time'])[keep], channel_terms,
                links=links, velocity_order=velocity_order)
            item_frequency = np.asarray(item['frequency'])[keep]
            factors = frequency_response_factors(
                geometry, item_frequency, lamb, beta)
            destination = indices + block_start
            for name, (plus, cross) in factors.items():
                contribution = (
                    plus * np.asarray(item['plus'])[keep]
                    + cross * np.asarray(item['cross'])[keep])
                np.add.at(output[name], destination, contribution)
    return (output, diagnostics) if return_diagnostics else output


def frequency_polarization_series(source, delta_f, f_lower, f_final,
                                  chunk_size=4096):
    """Return source polarizations as ordinary PyCBC ``FrequencySeries``.

    The analytic FD source is sampled directly; ``delta_f`` is therefore an
    integration grid and need only be refined until the requested statistic
    converges.  It is not forced to ``1 / observation_duration`` by an FFT.
    """
    from pycbc.types import FrequencySeries

    delta_f = float(delta_f)
    if not np.isfinite(delta_f) or delta_f <= 0:
        raise ValueError("delta_f must be positive and finite")
    chunk_size = int(chunk_size)
    if chunk_size < 2:
        raise ValueError("chunk_size must be at least two")
    kmin = max(1, int(np.ceil(float(f_lower) / delta_f)))
    kmax = int(np.floor(float(f_final) / delta_f))
    if kmax <= kmin:
        raise ValueError("frequency bounds must contain at least two bins")
    frequencies = np.arange(kmin, kmax + 1, dtype=float) * delta_f
    active_plus = np.zeros(len(frequencies), dtype=complex)
    active_cross = np.zeros(len(frequencies), dtype=complex)
    method = getattr(source, 'frequency_polarizations', None)
    if method is None:
        raise TypeError("source has no frequency_polarizations method")
    for start in range(0, len(frequencies), chunk_size):
        stop = min(len(frequencies), start + chunk_size)
        block_start = start if stop - start > 1 else start - 1
        plus, cross = method(frequencies[block_start:stop])
        discard = start - block_start
        active_plus[start:stop] = np.asarray(plus)[discard:]
        active_cross[start:stop] = np.asarray(cross)[discard:]
    plus = np.zeros(kmax + 1, dtype=complex)
    cross = np.zeros(kmax + 1, dtype=complex)
    plus[kmin:] = active_plus
    cross[kmin:] = active_cross
    return (FrequencySeries(plus, delta_f=delta_f, copy=False),
            FrequencySeries(cross, delta_f=delta_f, copy=False))


def project_frequency_tdi_series(
        source, orbit, channel_terms, delta_f, f_lower, f_final, lamb, beta,
        velocity_order=1, links=LINK_ORDER, chunk_size=4096,
        response_tolerance=None, response_initial_points=513,
        return_diagnostics=False):
    """Return direct FD TDI channels as standard PyCBC ``FrequencySeries``."""
    from pycbc.types import FrequencySeries

    delta_f = float(delta_f)
    if not np.isfinite(delta_f) or delta_f <= 0:
        raise ValueError("delta_f must be positive and finite")
    kmin = max(1, int(np.ceil(float(f_lower) / delta_f)))
    kmax = int(np.floor(float(f_final) / delta_f))
    if kmax <= kmin:
        raise ValueError("frequency bounds must contain at least two bins")
    frequencies = np.arange(kmin, kmax + 1, dtype=float) * delta_f
    projected = project_frequency_tdi(
        source, orbit, channel_terms, frequencies, lamb, beta,
        velocity_order=velocity_order, links=links, chunk_size=chunk_size,
        response_tolerance=response_tolerance,
        response_initial_points=response_initial_points,
        return_diagnostics=return_diagnostics)
    if return_diagnostics:
        active, diagnostics = projected
    else:
        active = projected
    output = {}
    for name, values in active.items():
        series = np.zeros(kmax + 1, dtype=complex)
        series[kmin:] = values
        output[name] = FrequencySeries(
            series, delta_f=delta_f, copy=False)
    return (output, diagnostics) if return_diagnostics else output


class SparseTDIResponse:
    """Carrier-factored TDI harmonics sampled on arbitrary sparse grids.

    This is a representation, not a two-year (or any other fixed-duration)
    array.  :meth:`sample` evaluates only the mission times requested by the
    caller.  Every harmonic may use a different grid and support window.
    """

    def __init__(self, source, responses, diagnostics=None):
        from scipy.interpolate import CubicSpline

        self.source = source
        self.responses = tuple(responses)
        self.diagnostics = {} if diagnostics is None else diagnostics
        names = {tuple(item['brackets']) for item in self.responses}
        if len(names) > 1:
            raise ValueError("every harmonic must contain the same channels")
        self.channels = next(iter(names), ())
        self._splines = []
        for item in self.responses:
            grid = np.asarray(item['grid'], dtype=float)
            support = item.get('support')
            if support is None:
                windows = ((float(grid[0]), float(grid[-1])),)
            elif np.ndim(support) == 1:
                windows = (tuple(support),)
            else:
                windows = tuple(tuple(window) for window in support)
            channel_splines = {}
            for name, bracket in item['brackets'].items():
                bracket = np.asarray(bracket)
                pieces = []
                for low, high in windows:
                    nodes = (grid >= low) & (grid <= high)
                    if np.count_nonzero(nodes) < 4:
                        continue
                    pieces.append((
                        float(low),
                        float(high),
                        CubicSpline(grid[nodes], bracket[nodes]),
                    ))
                channel_splines[name] = tuple(pieces)
            self._splines.append(channel_splines)

    def sample_brackets(self, times, channels=None):
        """Return each response record before reattaching its carrier.

        The tuple order matches :attr:`responses`; every item is a mapping of
        selected channel names to complex carrier-factored brackets. This is
        useful when a consumer can batch carrier evaluation across several
        independently windowed views of the same source harmonic.
        """
        times = np.asarray(times, dtype=float)
        if times.ndim != 1:
            raise ValueError("times must be one-dimensional")
        if channels is None:
            selected = self.channels
        elif isinstance(channels, str):
            selected = (channels,)
        else:
            selected = tuple(channels)
        unknown = set(selected) - set(self.channels)
        if unknown:
            raise ValueError(f"unknown channels: {sorted(unknown)}")
        records = []
        for channel_splines in self._splines:
            output = {}
            for name in selected:
                bracket = np.zeros(len(times), dtype=complex)
                for low, high, spline in \
                        channel_splines[name]:
                    want = (times >= low) & (times <= high)
                    if np.any(want):
                        bracket[want] = spline(times[want])
                output[name] = bracket
            records.append(output)
        return tuple(records)

    def sample(self, times, channels=None, complex_output=False):
        """Evaluate selected channels at arbitrary increasing mission times."""
        times = np.asarray(times, dtype=float)
        brackets = self.sample_brackets(times, channels=channels)
        selected = self.channels if channels is None else (
            (channels,) if isinstance(channels, str) else tuple(channels))
        dtype = complex if complex_output else float
        output = {name: np.zeros(len(times), dtype=dtype) for name in selected}
        for item, record in zip(self.responses, brackets, strict=True):
            carrier = np.exp(1j * self.source.carrier_phase(
                item['harmonic'], times))
            for name in selected:
                value = record[name] * carrier
                output[name] += value if complex_output else np.real(value)
        return output

    def to_timeseries(self, delta_t, t_start=None, t_end=None, channels=None,
                      chunk_size=262144, out=None, complex_output=False):
        """Reconstruct selected channels on a uniform grid in bounded memory.

        The sparse response remains the source of truth.  Dense samples are
        produced only in ``chunk_size`` blocks and may be written directly to
        caller-supplied arrays or memory maps through ``out``.  The returned
        objects are ordinary PyCBC :class:`~pycbc.types.TimeSeries` instances.

        Parameters
        ----------
        delta_t : float
            Output cadence in seconds.
        t_start, t_end : float, optional
            Half-open output interval.  Defaults to the source bounds.
        channels : sequence or str, optional
            Channels to reconstruct.  Defaults to every response channel.
        chunk_size : int, optional
            Maximum number of dense samples evaluated at once.
        out : mapping or callable, optional
            Preallocated one-dimensional arrays keyed by channel.  Each must
            have exactly the required output length.  Alternatively, a
            callable receives ``(channel, size, dtype)`` and allocates each
            array.  Memory maps are allowed in either form.
        complex_output : bool, optional
            Preserve the analytic carrier instead of returning its real part.
        """
        from pycbc.types import TimeSeries

        delta_t = float(delta_t)
        if not np.isfinite(delta_t) or delta_t <= 0:
            raise ValueError("delta_t must be positive and finite")
        if t_start is None:
            t_start = self.source.t_start
        if t_end is None:
            t_end = self.source.t_end
        t_start, t_end = float(t_start), float(t_end)
        if not t_end > t_start:
            raise ValueError("t_end must be greater than t_start")
        chunk_size = int(chunk_size)
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")

        if channels is None:
            selected = self.channels
        elif isinstance(channels, str):
            selected = (channels,)
        else:
            selected = tuple(channels)
        unknown = set(selected) - set(self.channels)
        if unknown:
            raise ValueError(f"unknown channels: {sorted(unknown)}")
        if not selected:
            return {}

        span = np.nextafter(t_end - t_start, 0.0)
        size = int(np.ceil(span / delta_t))
        dtype = complex if complex_output else float
        if out is None:
            arrays = {name: np.empty(size, dtype=dtype) for name in selected}
        elif callable(out):
            arrays = {
                name: np.asarray(out(name, size, np.dtype(dtype)))
                for name in selected
            }
        else:
            if not hasattr(out, 'keys'):
                raise TypeError("out must be a mapping keyed by channel")
            missing = set(selected) - set(out)
            if missing:
                raise ValueError(f"out is missing channels: {sorted(missing)}")
            arrays = {name: np.asarray(out[name]) for name in selected}
            for name, values in arrays.items():
                if values.shape != (size,):
                    raise ValueError(
                        f"out[{name!r}] must have shape ({size},)")
                if complex_output and not np.issubdtype(
                        values.dtype, np.complexfloating):
                    raise ValueError(
                        f"out[{name!r}] needs a complex dtype")

        for first in range(0, size, chunk_size):
            stop = min(size, first + chunk_size)
            times = t_start + np.arange(first, stop, dtype=float) * delta_t
            block = self.sample(
                times, channels=selected, complex_output=complex_output)
            for name in selected:
                arrays[name][first:stop] = block[name]

        return {
            name: TimeSeries(values, delta_t=delta_t, epoch=t_start,
                             copy=False)
            for name, values in arrays.items()
        }

    def linear_transform(self, matrix, channels):
        """Return new channels formed from a constant linear transformation."""
        matrix = np.asarray(matrix, dtype=float)
        channels = tuple(channels)
        if matrix.shape != (len(channels), len(self.channels)):
            raise ValueError(
                "matrix shape must be (new channels, existing channels)")
        records = []
        for item in self.responses:
            old = np.stack([item['brackets'][name]
                            for name in self.channels])
            new = matrix @ old
            records.append({
                'harmonic': item['harmonic'],
                'grid': item['grid'],
                'brackets': dict(zip(channels, new, strict=True)),
                'support': item['support'],
            })
        return SparseTDIResponse(
            self.source, records, diagnostics=self.diagnostics.copy())


class PreparedSparseTDI:
    """Source-independent orbit and delay work cached on harmonic grids.

    A prepared object is reusable for every source exposing the same harmonic
    labels and time coordinate.  Only sky projection and waveform evaluation
    remain in :meth:`project`; no orbit spline or light-cone solve is repeated.
    """

    def __init__(self, orbit, channel_terms, grids, velocity_order=1,
                 links=LINK_ORDER):
        self.channel_terms = channel_terms
        self.velocity_order = velocity_order
        self.links = links
        self.geometries = {}
        for harmonic, grid in grids.items():
            grid = np.asarray(grid, dtype=float)
            if (grid.ndim != 1 or len(grid) < 4
                    or np.any(np.diff(grid) <= 0)):
                raise ValueError(
                    "each harmonic grid needs four increasing times")
            self.geometries[harmonic] = MultiChannelTermGeometry(
                orbit, grid, channel_terms, links=links,
                velocity_order=velocity_order)

    @property
    def channels(self):
        """Output-channel labels in stable caller-supplied order."""
        geometry = next(iter(self.geometries.values()), None)
        return () if geometry is None else geometry.channel_names

    def project(self, source, lamb, beta, support_padding=0.0,
                matrix=None, channels=None):
        """Evaluate one source while reusing all prepared geometric work.

        ``support_padding`` must match the padding used to choose a prepared
        grid for a frequency-windowed source. It keeps delayed link terms live
        just outside the source window without rebuilding any geometry.

        ``matrix`` and ``channels`` optionally apply a constant channel
        transform to the carrier-factored brackets before spline construction.
        This avoids constructing an intermediate set of native-channel
        splines for every likelihood candidate.
        """
        support_padding = float(support_padding)
        if support_padding < 0:
            raise ValueError("support_padding must be non-negative")
        if matrix is None:
            if channels is not None:
                raise ValueError("channels requires a channel-transform matrix")
            output_channels = self.channels
        else:
            matrix = np.asarray(matrix, dtype=float)
            if channels is None:
                raise ValueError("matrix requires output channel names")
            output_channels = tuple(channels)
            if matrix.shape != (len(output_channels), len(self.channels)):
                raise ValueError(
                    "matrix shape must be (output channels, native channels)")
            if len(set(output_channels)) != len(output_channels):
                raise ValueError("output channel names must be distinct")
        missing = set(source.harmonics) - set(self.geometries)
        if missing:
            raise ValueError(f"no prepared grids for harmonics: {missing}")
        records = []
        for harmonic in source.harmonics:
            geometry = self.geometries[harmonic]
            brackets = sparse_channels_terms(
                source, harmonic, geometry, lamb, beta)
            if matrix is not None:
                native = np.stack([
                    brackets[name] for name in self.channels])
                transformed = matrix @ native
                brackets = dict(zip(
                    output_channels, transformed, strict=True))
            records.append({
                'harmonic': harmonic,
                'grid': geometry.grid,
                'brackets': brackets,
                'support': harmonic_windows(
                    source, harmonic, float(geometry.grid[0]),
                    float(geometry.grid[-1]), padding=support_padding),
            })
        return SparseTDIResponse(source, records)


def build_sparse_tdi_response(source, orbit, channel_terms, lamb, beta,
                              grids, velocity_order=1, links=LINK_ORDER):
    """Build a reusable sparse response from caller-selected harmonic grids.

    ``grids`` may be a mapping from harmonic labels to grids or a callable
    ``grids(source, harmonic)``.  Keeping grid policy separate from response
    evaluation prevents an observation-length assumption from entering this
    low-level API.
    """
    selected = {
        harmonic: (grids(source, harmonic) if callable(grids)
                   else grids[harmonic])
        for harmonic in source.harmonics
    }
    prepared = PreparedSparseTDI(
        orbit, channel_terms, selected, velocity_order=velocity_order,
        links=links)
    return prepared.project(source, lamb, beta)


def adaptive_sparse_tdi_response(
        source, orbit, channel_terms, lamb, beta, t_start=None, t_end=None,
        initial_step=86400.0, relative_tolerance=1e-4,
        amplitude_floor=1e-3, max_refinements=24, velocity_order=1,
        links=LINK_ORDER, support_padding=0.0):
    """Build an error-controlled response-envelope representation.

    Refinement tests the *carrier-factored TDI brackets*, not the carrier
    phase.  Each active interval is probed at its quarter, midpoint and
    three-quarter points so a smooth oscillation cannot hide behind one
    symmetric midpoint.  Passing intervals are not revisited; failing
    intervals alone are split.  Thus a chirp may become dense near merger
    without imposing that cadence on an earlier multi-year inspiral.

    Parameters
    ----------
    source, orbit, channel_terms, lamb, beta
        As in :func:`build_sparse_tdi_response`.
    t_start, t_end : float, optional
        Mission-time interval. Defaults to the source bounds; no duration is
        built into the implementation.
    initial_step : float, optional
        Largest interval before error-controlled refinement, in seconds.
    relative_tolerance : float, optional
        Maximum probe interpolation error relative to each channel's peak
        bracket amplitude.
    amplitude_floor : float, optional
        Fraction of the peak used as a local scale near response zeros.
    max_refinements : int, optional
        Maximum number of local bisections along any initial interval.
    support_padding : float, optional
        Widen source support blocks on the mission-time grid. Restricted
        frequency bands need a bound from :func:`delay_padding`, because a
        delayed source sample can contribute just outside its native support.
    """
    from scipy.interpolate import CubicSpline

    if t_start is None:
        t_start = source.t_start
    if t_end is None:
        t_end = source.t_end
    t_start, t_end = float(t_start), float(t_end)
    if not t_end > t_start:
        raise ValueError("t_end must be greater than t_start")
    if initial_step <= 0:
        raise ValueError("initial_step must be positive")
    if relative_tolerance <= 0:
        raise ValueError("relative_tolerance must be positive")
    if not 0 <= amplitude_floor <= 1:
        raise ValueError("amplitude_floor must lie in [0, 1]")
    if support_padding < 0:
        raise ValueError("support_padding must be non-negative")

    channel_names = tuple(channel_terms)
    records, diagnostics = [], {}

    def evaluate(harmonic, times):
        geometry = MultiChannelTermGeometry(
            orbit, times, channel_terms, links=links,
            velocity_order=velocity_order)
        values = sparse_channels_terms(
            source, harmonic, geometry, lamb, beta)
        return np.stack([values[name] for name in channel_names])

    for harmonic in source.harmonics:
        windows = harmonic_windows(
            source, harmonic, t_start, t_end, padding=support_padding)
        harmonic_grids, harmonic_values = [], []
        tested = 0
        deepest = 0
        for low, high in windows:
            count = max(4, int(np.ceil((high - low) / initial_step)) + 1)
            grid = np.linspace(low, high, count)
            values = evaluate(harmonic, grid)
            active_left, active_right = grid[:-1], grid[1:]

            for depth in range(int(max_refinements) + 1):
                if not len(active_left):
                    break
                fractions = np.asarray((0.25, 0.5, 0.75))
                probes = (active_left[:, None]
                          + fractions[None, :]
                          * (active_right - active_left)[:, None])
                if np.any((probes == active_left[:, None])
                          | (probes == active_right[:, None])):
                    raise RuntimeError(
                        "response refinement reached floating-point spacing")
                probes = probes.reshape(-1)
                owners = np.repeat(np.arange(len(active_left)), 3)
                exact = evaluate(harmonic, probes)
                predicted = np.stack([
                    CubicSpline(grid, row)(probes) for row in values
                ])
                peak = np.maximum(np.max(np.abs(values), axis=1),
                                  np.max(np.abs(exact), axis=1))[:, None]
                local = np.maximum(np.abs(exact), amplitude_floor * peak)
                failed_probe = np.any(
                    np.abs(exact - predicted)
                    > relative_tolerance * local, axis=0)
                failed_interval = np.zeros(len(active_left), dtype=bool)
                failed_interval[owners[failed_probe]] = True
                tested += len(probes)
                if not np.any(failed_interval):
                    active_left = active_right = np.array([])
                    break
                if depth == max_refinements:
                    raise RuntimeError(
                        "TDI response grid did not reach the requested "
                        f"tolerance after {max_refinements} refinements")

                selected = failed_interval[owners]
                insert_t = probes[selected]
                insert_v = exact[:, selected]
                joined_t = np.concatenate((grid, insert_t))
                joined_v = np.concatenate((values, insert_v), axis=1)
                order = np.argsort(joined_t)
                grid, values = joined_t[order], joined_v[:, order]
                next_left, next_right = [], []
                for owner in np.flatnonzero(failed_interval):
                    interior = probes[owners == owner]
                    edges = np.concatenate((
                        [active_left[owner]], interior,
                        [active_right[owner]]))
                    next_left.extend(edges[:-1])
                    next_right.extend(edges[1:])
                active_left = np.asarray(next_left)
                active_right = np.asarray(next_right)
                deepest = max(deepest, depth + 1)

            harmonic_grids.append(grid)
            harmonic_values.append(values)

        if not harmonic_grids:
            continue
        grid = np.concatenate(harmonic_grids)
        values = np.concatenate(harmonic_values, axis=1)
        records.append({
            'harmonic': harmonic,
            'grid': grid,
            'brackets': dict(zip(channel_names, values, strict=True)),
            'support': windows,
        })
        diagnostics[harmonic] = {
            'grid_points': len(grid),
            'tested_points': tested,
            'deepest_refinement': deepest,
            'relative_tolerance': relative_tolerance,
        }
    return SparseTDIResponse(source, records, diagnostics=diagnostics)
