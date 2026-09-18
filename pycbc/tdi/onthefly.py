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

from pycbc.tdi.response import (C_SI, LINK_ORDER, antenna_prefactor,
                                doppler_factors, polarization_basis,
                                sample_constellation)

YEAR = 3.15581498e7

try:                                             # built by setup.py
    from pycbc.tdi import sparse_cpu as _sparse_cpu
except ImportError:                              # pragma: no cover
    _sparse_cpu = None


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
                       dt_max=2.0e5, n_probe=4096, growth=1.1,
                       max_step_scale=np.inf, padding=0.0, plateau=0.0,
                       reference=None, refinements=2):
    """Grid on which the carrier advances by ``delta_phi`` per step.

    Cornish & Littenberg's prescription (arXiv:2506.08093 Sec. III.2): anchor
    at the merger, step by ``delta_phi`` of carrier phase, and once the anchor
    is further than ``plateau`` away let ``delta_phi`` grow by ``growth`` per
    step, with the step capped at ``dt_max``.

    **``growth`` is what bounds the largest interval, and ``delta_phi`` is
    not.** The step compounds geometrically with distance from the anchor, so
    refining ``delta_phi`` adds knots where they are already dense and leaves
    the far intervals where they were. Measured on a Sangria MBHB over the
    four days before merger: lowering ``delta_phi`` from 0.125 to 0.000488
    takes the grid from 481 knots to 105,983 and the median gap from 9.4 s to
    0.0 s, while the **largest** gap moves only from 42,602 s to 41,490 s.
    Lowering ``growth`` to 1.002 instead closes that gap with 2,872 knots and
    is what takes all fifteen Sangria MBHB inside
    :math:`\rho^2 R/2 < 0.1`; at the 1.1 default the loudest reads 0.163.

    The default stays at the paper's 1.1 because a source that does not
    concentrate its signal-to-noise near an anchor -- a galactic binary, a
    stellar-origin binary -- would pay five to nine times the knots for
    nothing. Choose it per source class, and see
    `test_growth_not_delta_phi_bounds_the_largest_interval`. Their published values are
    ``delta_phi = 0.5``, ``growth = 1.1``, ``dt_max = 2e5`` (2.3 days) and a
    plateau of 100 M in seconds; those are the defaults here except for the
    plateau, which needs a total mass this layer does not have. Pass it.

    Stepping the phase rather than the time is a deliberate difference. The
    paper writes ``dt = delta_phi / omega(t)`` with omega at the *current*
    time, which is the first-order form of advancing the phase by exactly
    ``delta_phi``; the two part company over a step near merger, where omega
    moves appreciably within one step, and the phase form is the one that
    delivers the constant carrier resolution both are after.

    The inversion from phase back to time is Newton-corrected
    (``t += (wanted - phase(t)) / omega(t)``, ``refinements`` times). Without
    it the map is read off a uniform probe table, which over a year of data
    is spaced in hours -- coarser than the entire merger it is meant to
    resolve.

    ``reference`` is the anchor; the default is the time of largest ``|omega|``
    in each window, which is the merger for a monotone chirp and harmless for
    a monochromatic source.

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
            max_step_scale, plateau, reference, refinements))
    if not pieces:
        return np.array([])
    return np.unique(np.concatenate(pieces))


def _edge_fades(source, harmonic, low, high, tolerance):
    """``(front, back)``: does the harmonic reach zero at each block edge?

    An edge where the amplitude is still a finite fraction of the block's own
    peak is a cut, not a fade.
    """
    span = high - low
    if not span > 0:
        return True, True
    offset = max(1e-9 * span, 4 * np.spacing(max(abs(low), abs(high))))
    probe = np.concatenate((np.linspace(low, high, 33),
                            [low + offset, high - offset]))
    amp_p, amp_c = source.amplitude(harmonic, probe)
    magnitude = np.hypot(np.abs(amp_p), np.abs(amp_c))
    peak = float(np.max(magnitude[:33]))
    if not peak > 0:
        return True, True
    return (float(magnitude[33]) <= tolerance * peak,
            float(magnitude[34]) <= tolerance * peak)


def harmonic_windows(source, harmonic, t_start, t_end, padding=0.0,
                     edge_tolerance=1e-6):
    """The stretches of [t_start, t_end] a harmonic can contribute to.

    Uses ``support_blocks`` where the source has it. A live set spanning two
    intervals with a dead gap between them would otherwise be covered by its
    outer hull.

    ``padding`` widens each block by `delay_padding`. That is what a harmonic
    fading out at its own edge needs, because the channel still carries
    delayed copies of it for one more padding.

    An edge the source cuts while the harmonic is still loud is a different
    object. pyEFPEHM drops a harmonic once it falls under ``Amplitude_tol``,
    at up to 60% of that harmonic's own peak, and every term of the delay
    chain then steps as its own query crosses that cut. The bracket acquires
    as many staggered steps as the channel has terms, which no spline fits and
    no refinement resolves.

    Those steps straddle the edge. A second-generation chain carries
    advancements as well as delays, and ``k.r/c`` changes sign over a year, so
    a term's query runs from ``t - padding`` to ``t + padding`` and the
    stepped interval is ``[edge - padding, edge + padding]``. Such an edge is
    therefore trimmed by a padding on the inside: the window begins one
    padding after a cut-on and ends one before a cut-off. The response over
    those two paddings is not represented.
    """
    blocks = getattr(source, 'support_blocks', None)
    if blocks is not None:
        windows = blocks(harmonic)
    else:
        support = getattr(source, 'support', None)
        windows = [support(harmonic)] if support is not None else [(t_start,
                                                                    t_end)]
    padding = float(padding)
    out = []
    for low, high in windows:
        first, last = low - padding, high + padding
        # A cut edge steps every term somewhere in [edge, edge + padding].
        # Only if that interval reaches into the observation does the trim
        # matter, so a block spanning the whole of it costs no source call.
        front_bites = low + padding > t_start
        back_bites = high - padding < t_end
        if padding > 0 and (front_bites or back_bites):
            fades_front, fades_back = _edge_fades(
                source, harmonic, low, high, edge_tolerance)
            if front_bites and not fades_front:
                first = low + padding
            if back_bites and not fades_back:
                last = high - padding
        low = max(t_start, first)
        high = min(t_end, last)
        if high > low:
            if out and low <= out[-1][1]:
                out[-1] = (out[-1][0], max(out[-1][1], high))
            else:
                out.append((low, high))
    return out


def _grid_over_window(source, harmonic, t_start, t_end, delta_phi, dt_max,
                      n_probe, growth, max_step_scale, plateau=0.0,
                      reference=None, refinements=2):
    probe = np.linspace(t_start, t_end, int(n_probe))
    phase = np.asarray(source.carrier_phase(harmonic, probe), dtype=float)
    if not np.all(np.diff(phase) > 0):          # must be monotone to invert
        phase = np.maximum.accumulate(phase)
    total = phase[-1] - phase[0]
    if not np.isfinite(total) or total <= 0:
        return np.linspace(t_start, t_end, 2)

    # Anchor at the merger and let the step grow away from it, in both
    # directions. The direction matters: growing forwards from t_start puts
    # the dense region where the signal is slowest and the coarse region at
    # the merger.
    if reference is None:
        rate = np.abs(np.asarray(
            source.angular_frequency(harmonic, probe), dtype=float))
        usable = np.isfinite(rate) & (rate > 0)
        anchor_phase = (phase[np.argmax(np.where(usable, rate, -np.inf))]
                        if usable.any() else phase[-1])
    else:
        anchor_phase = float(np.interp(
            min(max(float(reference), t_start), t_end), probe, phase))

    if growth <= 1.0:
        wanted = np.arange(phase[0], phase[-1], float(delta_phi))
    else:
        limit = float(max_step_scale) * float(delta_phi)
        # The plateau is the paper's "hold delta_phi until |t - tc| > 100 M",
        # counted in steps rather than seconds because the times are not known
        # until after the inversion. Near the anchor omega is by construction
        # close to its largest value, so plateau * omega / delta_phi is the
        # step count that covers it.
        anchor_rate = float(np.interp(
            anchor_phase, phase,
            np.abs(np.asarray(source.angular_frequency(harmonic, probe),
                              dtype=float))))
        flat = 0
        if plateau > 0 and np.isfinite(anchor_rate) and anchor_rate > 0:
            flat = int(np.ceil(plateau * anchor_rate / float(delta_phi)))
        wanted = [anchor_phase]
        for sign, bound in ((-1.0, phase[0]), (1.0, phase[-1])):
            value, step, taken = anchor_phase, float(delta_phi), 0
            while (value - bound) * sign < 0:
                value += sign * step
                wanted.append(value)
                taken += 1
                if taken >= flat:
                    step = min(step * growth, limit)
        wanted = np.unique(np.asarray(wanted))
        wanted = wanted[(wanted >= phase[0]) & (wanted <= phase[-1])]
    grid = np.interp(wanted, phase, probe)

    # The probe table is uniform, so over a year it is spaced in hours -- far
    # coarser than the merger the grid exists to resolve. Newton on the phase
    # itself removes that: the correction is (wanted - phase(t)) / omega(t),
    # and the source can evaluate both at arbitrary times.
    for _ in range(int(refinements)):
        current = np.asarray(source.carrier_phase(harmonic, grid), dtype=float)
        rate = np.asarray(
            source.angular_frequency(harmonic, grid), dtype=float)
        ok = np.isfinite(current) & np.isfinite(rate) & (np.abs(rate) > 0)
        if not ok.any():
            break
        step = np.zeros_like(grid)
        step[ok] = (wanted[ok] - current[ok]) / rate[ok]
        grid = np.clip(grid + step, t_start, t_end)
        grid = np.maximum.accumulate(grid)

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
    pref_plus, pref_cross = antenna_prefactor(
        sample.n_hat, u_hat, v_hat, k_hat)
    prefactor = np.stack((pref_plus, pref_cross), axis=-1)
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
        pref_plus, pref_cross = antenna_prefactor(
            n_hat, u_hat, v_hat, k_hat)
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
    pref_plus, pref_cross = antenna_prefactor(
        geometry.n_hat, u_hat, v_hat, k_hat)
    tau_emit = geometry.ltt + geometry._project(geometry.r_emit, k_hat) / C_SI
    tau_recv = geometry._project(geometry.r_recv, k_hat) / C_SI

    if geometry.velocity_order:
        k_emit = geometry._project(geometry.v_emit, k_hat)
        k_recv = geometry._project(geometry.v_recv, k_hat)
        n_v_recv = np.einsum('cgla,cgla->cgl', geometry.n_hat,
                             geometry.v_recv)
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
                 velocity_order=1, delay_expansion=None,
                 reference_delay=False, threads=1):
        self.threads = max(1, int(threads))
        if delay_expansion not in (None, 2, 3):
            raise ValueError("delay_expansion must be None, 2 or 3")
        self.delay_expansion = delay_expansion
        self.reference_delay = bool(reference_delay)
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
            'shifted', 'chain_delay', 'ltt', 'n_hat', 'r_emit', 'r_recv')}
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
            net_shift = chain_delay(
                orbit, self.grid, chain, links, cache=cache)
            shifted = self.grid - net_shift
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
                gathered['chain_delay'][row] = net_shift
                gathered['ltt'][row] = sample.ltt[inverse, column]
                gathered['n_hat'][row] = sample.n_hat[inverse, column]
                gathered['r_emit'][row] = sample.r_emit[inverse, column]
                gathered['r_recv'][row] = sample.position[inverse, receiver]
                if velocity_order:
                    gathered['v_emit'][row] = sample.velocity[inverse, emitter]
                    gathered['v_recv'][row] = sample.velocity[inverse, receiver]

        for key, values in gathered.items():
            setattr(self, key, np.stack(values))
        if self.reference_delay:
            self.barycentre = np.mean(np.asarray(
                orbit.compute_position(self.grid, (1, 2, 3)), dtype=float),
                axis=1)
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
                 velocity_order=1, delay_expansion=None,
                 reference_delay=False, threads=1):
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
                         velocity_order=velocity_order,
                         delay_expansion=delay_expansion,
                         reference_delay=reference_delay, threads=threads)


def _reference_delay(geometry, lamb, beta):
    """k . R / c for the constellation barycentre: the delay every term shares.

    A term samples the source at ``t - d`` with ``d`` up to about 570 s, and
    almost all of that is the light time from the barycentre to the solar
    system barycentre -- 499 s, sweeping through its own range once a year.
    The rest is the TDI chain (67 s at most), the arm (8 s), and how far one
    spacecraft sits from the middle of the three (5 s).

    Factoring the common part out of the carrier leaves the bracket varying at
    the rate of the residual delay rather than of the whole one, an order of
    magnitude slower, and the adaptive grid follows.
    """
    if not getattr(geometry, "reference_delay", False):
        return None
    _, _, k_hat = polarization_basis(lamb, beta)
    return (geometry.barycentre @ k_hat) / C_SI


def barycentre_delay(orbit, times, lamb, beta):
    """`_reference_delay` for callers holding an orbit instead of a geometry."""
    _, _, k_hat = polarization_basis(lamb, beta)
    position = np.asarray(orbit.compute_position(
        np.asarray(times, dtype=float), (1, 2, 3)), dtype=float)
    return (np.mean(position, axis=1) @ k_hat) / C_SI


def _sparse_term_projection(geometry, lamb, beta, offset=None):
    """Return delayed queries, delays and polarization weights.

    The delay is returned rather than left to be recovered downstream as
    ``anchor - query``. Both of those are absolute mission epochs of order
    1e7 s while their difference is at most 570 s, so differencing them
    discards nine significant digits and leaves the delay quantised at
    ULP(1e7 s) = 1.9e-9 s. The carrier turns that into 2 pi f = 6e-12 rad of
    jitter per term, invisible in A/E but 1e-6 of the T channel's peak, which
    is a cancellation residual four orders smaller. It is not an
    interpolation error, so refinement never removes it: measured on a 1 mHz
    galactic binary, the grid error stops falling at 1e-34 and 64x refinement
    changes nothing, and the adaptive grid then exhausts max_grid_points
    rather than converging. Composing the delay from the chain shift, the
    light-cone term and the reference delay keeps every quantity below 600 s
    and puts the floor back on float64 epsilon.
    """
    u_hat, v_hat, k_hat = polarization_basis(lamb, beta)
    n_dot_u = geometry._project(geometry.n_hat, u_hat)
    n_dot_v = geometry._project(geometry.n_hat, v_hat)
    n_dot_k = geometry._project(geometry.n_hat, k_hat)
    gap = 1 - n_dot_k
    transverse = n_dot_u ** 2 + n_dot_v ** 2
    scale = np.divide(0.5 * (1 + n_dot_k), transverse,
                      out=np.zeros_like(transverse), where=transverse > 0)
    direct = gap >= 1e-4
    pref_plus = np.where(
        direct, (n_dot_u ** 2 - n_dot_v ** 2) / (2 * gap),
        scale * (n_dot_u ** 2 - n_dot_v ** 2))
    pref_cross = np.where(
        direct, n_dot_u * n_dot_v / gap,
        scale * (2 * n_dot_u * n_dot_v))

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
    shift = geometry.chain_delay
    if offset is not None:
        shift = shift - offset
    delay = np.concatenate((shift + tau_emit, shift + tau_recv))
    weight = np.concatenate((np.broadcast_to(w1, geometry.shape),
                             -np.broadcast_to(w2, geometry.shape)))
    plus = np.concatenate((pref_plus, pref_plus))
    cross = np.concatenate((pref_cross, pref_cross))
    coefficient = np.concatenate((geometry.coefficient,
                                  geometry.coefficient))
    return (query, delay, coefficient * weight * plus,
            coefficient * weight * cross)


def _source_blocks(source, harmonic):
    """(low, high) arrays for the stretches the source is defined over."""
    blocks = getattr(source, "support_blocks", None)
    if blocks is not None:
        windows = tuple(blocks(harmonic))
        if windows:
            return (np.asarray([item[0] for item in windows], dtype=float),
                    np.asarray([item[1] for item in windows], dtype=float))
    support = getattr(source, "support", None)
    if support is not None:
        low, high = support(harmonic)
        return np.asarray([low], dtype=float), np.asarray([high], dtype=float)
    return None, None


def _delay_stencil(source, harmonic, grid, omega):
    """``(ahead, back, drop, rise)``: where to difference the source.

    The step is a tenth of a carrier radian. Smaller and the difference of two
    nearly equal amplitudes loses its significant digits; larger and the
    second derivative of a chirping frequency stops being resolved.

    A point closer than a step to the end of its own block gets both of its
    stencil points moved to the other side, so the source is never asked for a
    time it answers zero at. The three nodes stay distinct, which is all
    `_three_point_derivatives` needs -- one of ``drop``, ``rise`` is then
    negative and the formula becomes the one-sided one. A source declaring no
    support is defined everywhere and is never moved.
    """
    step = np.clip(0.1 / np.maximum(np.abs(omega), 1e-30), 1e-3, np.inf)
    low, high = _source_blocks(source, harmonic)
    ahead, back = grid + step, grid - step
    if low is not None:
        index = np.clip(np.searchsorted(low, grid, side="right") - 1,
                        0, len(low) - 1)
        first, last = low[index], high[index]
        step = np.minimum(step, 0.01 * np.maximum(last - first, 1e-30))
        ahead, back = grid + step, grid - step
        under = back < first
        over = ahead > last
        ahead = np.where(under, grid + 2 * step,
                         np.where(over, grid - step, ahead))
        back = np.where(under, grid + step,
                        np.where(over, grid - 2 * step, back))
    drop, rise = grid - back, ahead - grid
    if not np.all((drop != 0) & (rise != 0) & (drop + rise != 0)):
        raise ValueError(
            "a source block is too short to difference the delay expansion "
            "over; evaluate this harmonic without delay_expansion")
    return ahead, back, drop, rise


def _three_point_derivatives(back, here, ahead, drop, rise):
    """First and second derivatives from three unevenly spaced samples.

    The stencil is shortened on one side at a block edge, so the even-spacing
    formulas do not apply. These are the derivatives of the Lagrange
    interpolant through ``(t - drop, t, t + rise)``.
    """
    total = drop * rise * (drop + rise)
    first = (-rise ** 2 * back + (rise ** 2 - drop ** 2) * here
             + drop ** 2 * ahead) / total
    second = 2 * (rise * back - (drop + rise) * here + drop * ahead) / total
    return first, second


def expansion_coefficients(source, harmonic, anchor, order=3):
    """Amplitude/frequency delay coefficients on ``anchor``.

    A source may provide the coefficients analytically.  Otherwise one
    stencil supplies the derivatives with a number of source calls that does
    not depend on the channel term count.
    """
    analytic = getattr(source, 'delay_expansion_coefficients', None)
    if analytic is not None:
        coefficients = analytic(harmonic, anchor, order)
        if coefficients is not None:
            return tuple(
                np.ascontiguousarray(value, dtype=(
                    complex if index < 6 else float))
                for index, value in enumerate(coefficients))
    amplitude_frequency = getattr(source, 'amplitude_frequency', None)
    combined = getattr(source, 'harmonic_components', None)
    if amplitude_frequency is not None:
        amp_p, amp_c, omega = amplitude_frequency(harmonic, anchor)
        omega = np.asarray(omega, dtype=float)
        ahead, back, drop, rise = _delay_stencil(
            source, harmonic, anchor, omega)
        pair_p, pair_c, pair_omega = amplitude_frequency(
            harmonic, np.concatenate((ahead, back)))
        count = len(anchor)
        ahead_p, back_p = pair_p[:count], pair_p[count:]
        ahead_c, back_c = pair_c[:count], pair_c[count:]
        omega_ahead = np.asarray(pair_omega[:count], dtype=float)
        omega_back = np.asarray(pair_omega[count:], dtype=float)
    elif combined is None:
        omega = np.asarray(source.angular_frequency(harmonic, anchor),
                           dtype=float)
        ahead, back, drop, rise = _delay_stencil(
            source, harmonic, anchor, omega)
        amp_p, amp_c = source.amplitude(harmonic, anchor)
        ahead_p, ahead_c = source.amplitude(harmonic, ahead)
        back_p, back_c = source.amplitude(harmonic, back)
        omega_ahead = np.asarray(
            source.angular_frequency(harmonic, ahead), dtype=float)
        omega_back = np.asarray(
            source.angular_frequency(harmonic, back), dtype=float)
    else:
        # One pass gives amplitude and frequency together, and the two
        # stencil sides go in one array: two source calls where the separate
        # accessors need six. On pyEFPEHM that is 0.42 us per grid point
        # against 0.72.
        amp_p, amp_c, _, omega = combined(harmonic, anchor)
        omega = np.asarray(omega, dtype=float)
        ahead, back, drop, rise = _delay_stencil(
            source, harmonic, anchor, omega)
        pair_p, pair_c, _, pair_omega = combined(
            harmonic, np.concatenate((ahead, back)))
        count = len(anchor)
        ahead_p, back_p = pair_p[:count], pair_p[count:]
        ahead_c, back_c = pair_c[:count], pair_c[count:]
        omega_ahead = np.asarray(pair_omega[:count], dtype=float)
        omega_back = np.asarray(pair_omega[count:], dtype=float)
    rate, curve = _three_point_derivatives(
        omega_back, omega, omega_ahead, drop, rise)
    slope_p, bend_p = _three_point_derivatives(
        back_p, amp_p, ahead_p, drop, rise)
    slope_c, bend_c = _three_point_derivatives(
        back_c, amp_c, ahead_c, drop, rise)
    return (np.ascontiguousarray(amp_p, dtype=complex),
            np.ascontiguousarray(slope_p, dtype=complex),
            np.ascontiguousarray(bend_p, dtype=complex),
            np.ascontiguousarray(amp_c, dtype=complex),
            np.ascontiguousarray(slope_c, dtype=complex),
            np.ascontiguousarray(bend_c, dtype=complex),
            np.ascontiguousarray(omega), np.ascontiguousarray(rate),
            np.ascontiguousarray(curve))


def _expanded_source(source, harmonic, geometry, delay, order, offset):
    """Delayed amplitudes and relative phase, expanded about the grid.

    Every delayed time the channel asks for lies within one `delay_padding` of
    a grid point, and asking the source for all of them is 64 to 87 per cent
    of a pyEFPEHM likelihood call. Expanding about the grid instead,

        Phi(t - d) - Phi(t) = -omega d + omega' d^2 / 2 - omega'' d^3 / 6,
        A(t - d)            = A - d A' + d^2 A'' / 2,

    leaves the source evaluated on the grid and on one point either side of
    it, whatever the channel term count. Both derivatives come off that same
    stencil, so third order costs no more source calls than second.

    `delay` arrives already composed from the chain shift, the light-cone
    term and the reference delay. It is deliberately not recovered here as
    ``anchor - query``: see `_sparse_term_projection` for what that costs.

    Where the expansion is centred matters more than how far it is carried.
    With `_reference_delay` on, the centre moves to ``t - tau_0`` and the
    parameter becomes ``d - tau_0``, at most 80 s where ``d`` reaches 570, so
    second order is enough. Without it, second order leaves 6.1e-05 of the
    loudest bracket on ten eccentric harmonics -- and third order is no cure,
    because it multiplies the stencil roundoff in the second derivative of
    omega by ``d^3``: on one of those harmonics that noise sends the
    refinement past 30,000 points where second order converges at 3,240.
    """
    anchor = geometry.grid if offset is None else geometry.grid - offset
    (amp_p, slope_p, bend_p, amp_c, slope_c, bend_c,
     omega, rate, curve) = expansion_coefficients(
         source, harmonic, anchor, order=order)

    phase = delay * (0.5 * rate[None, :] * delay - omega[None, :])
    out_p = amp_p[None, :] - delay * slope_p[None, :]
    out_c = amp_c[None, :] - delay * slope_c[None, :]
    if order >= 3:
        squared = delay ** 2
        phase = phase - (curve[None, :] / 6.0) * squared * delay
        out_p = out_p + 0.5 * squared * bend_p[None, :]
        out_c = out_c + 0.5 * squared * bend_c[None, :]
    return out_p, out_c, phase


def _sparse_term_contributions(source, harmonic, geometry, lamb, beta):
    """Return each gathered term before reducing it into output channels."""
    order = getattr(geometry, "delay_expansion", None)
    offset = _reference_delay(geometry, lamb, beta)
    query, delay, plus, cross = _sparse_term_projection(
        geometry, lamb, beta, offset if order is not None else None)
    if order is None:
        amp_p, amp_c = source.amplitude(harmonic, query)
        anchor = (geometry.grid if offset is None
                  else geometry.grid - offset)
        phase = source.carrier_phase(harmonic, query)
        phase -= source.carrier_phase(harmonic, anchor)[None, :]
    else:
        amp_p, amp_c, phase = _expanded_source(
            source, harmonic, geometry, delay, order, offset)
    return ((plus * amp_p + cross * amp_c)
            * np.exp(1j * phase))


def _reduce_sparse_contributions(contribution, geometry):
    """Reduce gathered term contributions into their output channels."""
    term_count = len(geometry.term_channel)
    if contribution.shape[0] == 2 * term_count:
        # Keep the physical one-link subtraction together.  Near n.k = 1
        # the emission and reception pieces are individually O(h), while
        # their difference carries the vanishing (1 - n.k) factor.  Summing
        # every emission side before every reception side needlessly loses
        # that cancellation before the TDI terms are even combined.
        contribution = (contribution[:term_count]
                        + contribution[term_count:])
    elif contribution.shape[0] != term_count:
        raise ValueError("contribution rows do not match the TDI terms")
    channels = geometry.term_channel
    output = np.zeros((len(geometry.channel_names), geometry.shape[1]),
                      dtype=complex)
    for index in range(len(output)):
        output[index] = contribution[channels == index].sum(axis=0)
    return dict(zip(geometry.channel_names, output, strict=True))


def sparse_channel_terms(source, harmonic, geometry, lamb, beta):
    """The narrowest one-channel path: work scales with the term count."""
    contribution = _sparse_term_contributions(
        source, harmonic, geometry, lamb, beta)
    term_count = len(geometry.term_channel)
    return (contribution[:term_count]
            + contribution[term_count:]).sum(axis=0)


_EMPTY = np.zeros((1, 1, 3))


def _fused_channels(source, harmonic, geometry, lamb, beta):
    """One compiled pass, or ``None`` if this configuration cannot take it."""
    order = getattr(geometry, 'delay_expansion', None)
    if order is None or _sparse_cpu is None:
        return None
    offset = _reference_delay(geometry, lamb, beta)
    anchor = np.ascontiguousarray(
        geometry.grid if offset is None else geometry.grid - offset)
    # The kernel composes the delay from small quantities; `anchor` stays an
    # absolute epoch because the source has to be evaluated there.
    reference = np.ascontiguousarray(
        np.zeros(len(geometry.grid)) if offset is None else offset)
    coefficients = expansion_coefficients(
        source, harmonic, anchor, order=order)
    u_hat, v_hat, k_hat = polarization_basis(lamb, beta)
    # The compiled loop accumulates emission and reception sides directly.
    # Around the removable n.k = 1 limit, use the Python path that pairs the
    # two sides of each physical link before reducing the TDI polynomial.
    if np.any(1 - geometry._project(geometry.n_hat, k_hat) < 1e-4):
        return None
    boosted = bool(geometry.velocity_order)
    out = np.empty((len(geometry.channel_names), len(geometry.grid)),
                   dtype=complex)
    _sparse_cpu.sparse_brackets(
        geometry.n_hat, geometry.r_emit, geometry.r_recv,
        geometry.v_emit if boosted else _EMPTY,
        geometry.v_recv if boosted else _EMPTY,
        geometry.n_dot_v_recv if boosted else _EMPTY[0],
        geometry.n_dot_v_mix if boosted else _EMPTY[0],
        geometry.ltt, np.ascontiguousarray(geometry.chain_delay),
        np.ascontiguousarray(geometry.coefficient[:, 0]),
        np.ascontiguousarray(geometry.term_channel, dtype=np.int64),
        reference,
        np.ascontiguousarray(u_hat), np.ascontiguousarray(v_hat),
        np.ascontiguousarray(k_hat),
        *coefficients, int(order), int(boosted), C_SI, out,
        int(getattr(geometry, 'threads', 1)))
    return dict(zip(geometry.channel_names, out, strict=True))


def sparse_channels_terms(source, harmonic, geometry, lamb, beta):
    """Evaluate several channels with one source call and shared geometry.

    Parameters are the same as :func:`sparse_channel_terms`, except
    ``geometry`` must be a :class:`MultiChannelTermGeometry`.  The returned
    mapping contains one complex, carrier-factored bracket per channel.
    """
    if not isinstance(geometry, MultiChannelTermGeometry):
        raise TypeError("geometry must be a MultiChannelTermGeometry")
    fused = _fused_channels(source, harmonic, geometry, lamb, beta)
    if fused is not None:
        return fused
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
        if getattr(geometry, "reference_delay", False):
            raise ValueError(
                "this path reattaches the carrier at t, and a bracket reduced "
                "against the reference delay needs it at t - tau_0; build the "
                "geometry with reference_delay=False")
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
        query_sizes = [query.size for query, _, _, _ in projections]
        grid_sizes = [len(geometries[index].grid) for index in positions]
        all_query = np.concatenate([
            query.reshape(-1) for query, _, _, _ in projections])
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
            query, _, plus, cross = projection
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
    pref_plus, pref_cross = antenna_prefactor(
        geometry.n_hat, u_hat, v_hat, k_hat)
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

    chain_delay_value = geometry.chain_delay
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


def interpolant(x, y, order=3, extrapolate=False):
    """``order``-degree interpolating spline through ``(x, y)``.

    Cubic is the default. A quintic resolves an oscillation with fewer knots,
    but scipy builds one at 0.21 us per point against 0.06, so whether it pays
    depends on how much the grid actually shrinks.
    """
    from scipy.interpolate import CubicSpline, make_interp_spline

    if order == 3:
        return CubicSpline(x, y, extrapolate=extrapolate or None)
    return make_interp_spline(x, y, k=order)


class SparseTDIResponse:
    """Carrier-factored TDI harmonics sampled on arbitrary sparse grids.

    This is a representation, not a two-year (or any other fixed-duration)
    array.  :meth:`sample` evaluates only the mission times requested by the
    caller.  Every harmonic may use a different grid and support window.
    """

    def __init__(self, source, responses, diagnostics=None,
                 interpolation_order=3):
        if interpolation_order not in (3, 5):
            raise ValueError("interpolation_order must be 3 or 5")
        self.interpolation_order = interpolation_order
        self.source = source
        self.responses = tuple(responses)
        self.diagnostics = {} if diagnostics is None else diagnostics
        names = {tuple(item['brackets']) for item in self.responses}
        if len(names) > 1:
            raise ValueError("every harmonic must contain the same channels")
        self.channels = next(iter(names), ())
        # Splines are built on demand and kept. A consumer that reads one
        # channel of a ten-harmonic response pays for one, and one that reads
        # none -- asking only for diagnostics, or for the grids -- pays for
        # nothing. Building all of them eagerly was 31.6 ms of a 204.6 ms
        # candidate.
        self._splines = [{} for _ in self.responses]
        self._windows = []
        self._grids = []
        for item in self.responses:
            grid = np.asarray(item['grid'], dtype=float)
            support = item.get('support')
            if support is None:
                windows = ((float(grid[0]), float(grid[-1])),)
            elif len(support) == 0:
                windows = ()
            elif np.ndim(support) == 1:
                windows = (tuple(support),)
            else:
                windows = tuple(tuple(window) for window in support)
            self._grids.append(grid)
            self._windows.append(windows)
        self._offset = self._shared_offset()

    def _shared_offset(self):
        """k . R / c does not depend on the harmonic, so build it once.

        Ten harmonics carry ten samplings of one function of time. The widest
        grid covers the rest, and the others are checked against it rather
        than trusted: a caller is free to hand in records whose offsets really
        do differ, and then each keeps its own.
        """
        from scipy.interpolate import CubicSpline

        offsets = [item.get('reference_delay') for item in self.responses]
        if all(offset is None for offset in offsets):
            return None
        widest = max(
            (index for index, offset in enumerate(offsets)
             if offset is not None),
            key=lambda index: self._grids[index][-1] - self._grids[index][0])
        shared = CubicSpline(self._grids[widest],
                             np.asarray(offsets[widest], dtype=float),
                             extrapolate=True)
        for index, offset in enumerate(offsets):
            if offset is None:
                return [None if item is None
                        else CubicSpline(self._grids[i],
                                         np.asarray(item, dtype=float),
                                         extrapolate=True)
                        for i, item in enumerate(offsets)]
            grid = self._grids[index]
            probe = grid[::max(1, len(grid) // 8)]
            wanted = np.interp(probe, grid, np.asarray(offset, dtype=float))
            scale = max(np.max(np.abs(wanted)), 1e-30)
            if np.max(np.abs(shared(probe) - wanted)) > 1e-9 * scale:
                return [CubicSpline(self._grids[i],
                                    np.asarray(item, dtype=float),
                                    extrapolate=True)
                        for i, item in enumerate(offsets)]
        return shared

    def _offset_for(self, index):
        """The reference delay of one record, shared or its own."""
        if self._offset is None:
            return None
        if isinstance(self._offset, list):
            return self._offset[index]
        return self._offset

    def _channel_spline(self, index, name):
        """Build, and keep, the pieces of one channel of one harmonic."""
        cached = self._splines[index].get(name)
        if cached is not None:
            return cached
        grid = self._grids[index]
        bracket = np.asarray(self.responses[index]['brackets'][name])
        order = self.interpolation_order
        pieces = []
        for low, high in self._windows[index]:
            nodes = (grid >= low) & (grid <= high)
            if np.count_nonzero(nodes) < order + 1:
                continue
            pieces.append((float(low), float(high),
                           interpolant(grid[nodes], bracket[nodes], order)))
        self._splines[index][name] = tuple(pieces)
        return self._splines[index][name]

    def sample_brackets(self, times, channels=None):
        """Return each response record before reattaching its carrier.

        The tuple order matches :attr:`responses`; every item is a mapping of
        selected channel names to complex carrier-factored brackets. This is
        useful when a consumer can batch carrier evaluation across several
        independently windowed views of the same source harmonic.

        A caller reattaching the carrier itself has to read it at
        ``times - carrier_offset(times, index)``, not at ``times``: a bracket
        reduced against the reference delay carries the difference. That
        method returns zeros when there is nothing to correct, so it is always
        safe to subtract.
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
        for index in range(len(self.responses)):
            output = {}
            for name in selected:
                bracket = np.zeros(len(times), dtype=complex)
                for low, high, spline in self._channel_spline(index, name):
                    want = (times >= low) & (times <= high)
                    if np.any(want):
                        bracket[want] = spline(times[want])
                output[name] = bracket
            records.append(output)
        return tuple(records)

    def carrier_offset(self, times, index):
        """How far back one record's carrier is read, at ``times``.

        Zero unless the bracket was reduced against the reference delay. See
        :meth:`sample_brackets`.
        """
        offset = self._offset_for(index)
        times = np.asarray(times, dtype=float)
        return (np.zeros(len(times)) if offset is None
                else np.asarray(offset(times), dtype=float))

    def sample(self, times, channels=None, complex_output=False):
        """Evaluate selected channels at arbitrary increasing mission times."""
        times = np.asarray(times, dtype=float)
        brackets = self.sample_brackets(times, channels=channels)
        selected = self.channels if channels is None else (
            (channels,) if isinstance(channels, str) else tuple(channels))
        dtype = complex if complex_output else float
        output = {name: np.zeros(len(times), dtype=dtype) for name in selected}
        for index, (item, record) in enumerate(
                zip(self.responses, brackets, strict=True)):
            offset = self._offset_for(index)
            # the bracket was reduced against Phi(t - tau_0), so the carrier
            # has to be read at the same shifted time
            anchor = times if offset is None else times - offset(times)
            carrier = np.exp(1j * self.source.carrier_phase(
                item['harmonic'], anchor))
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
                'reference_delay': item.get('reference_delay'),
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
                 links=LINK_ORDER, delay_expansion=None,
                 reference_delay=False, threads=1, interpolation_order=3):
        self.interpolation_order = interpolation_order
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
                velocity_order=velocity_order,
                delay_expansion=delay_expansion,
                reference_delay=reference_delay, threads=threads)

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
                'reference_delay': _reference_delay(geometry, lamb, beta),
            })
        return SparseTDIResponse(
            source, records, interpolation_order=self.interpolation_order)


def adaptive_sparse_tdi_response(
        source, orbit, channel_terms, lamb, beta, t_start=None, t_end=None,
        initial_step=86400.0, relative_tolerance=1e-4,
        amplitude_floor=1e-3, max_refinements=24, velocity_order=1,
        links=LINK_ORDER, support_padding=0.0, max_grid_points=1000000,
        delay_expansion=None, reference_delay=False, threads=1,
        interpolation_order=3, stall_refusal_factor=100.0,
        stall_patience=3, evaluation_chunk_size=16384):
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
        The source to project, the orbit and per-channel `Term` lists, and
        the ecliptic sky position in radians.
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
        Usually `delay_padding`. Note what the caller must do with it as well:
        the response at ``t`` reads the source over ``[t - padding,
        t + padding]``, so a window reaching closer than one padding to the
        end of support is not represented, and a window ending exactly one
        padding short leaves the spline one-sided there. About one and a half
        paddings is the useful setting -- measured on Sangria's massive black
        hole binaries, anything from 1.25 to 1.75 gives mismatches between
        7.3e-09 and 2.6e-08 with no sharp optimum, while a larger margin loses
        signal monotonically. Four paddings ends the window before the
        amplitude peak and costs three quarters of rho.
        Widen source support blocks on the mission-time grid. Restricted
        frequency bands need a bound from :func:`delay_padding`, because a
        delayed source sample can contribute just outside its native support.
    max_grid_points : int, optional
        Give up once one harmonic's grid passes this size. Active intervals
        quadruple per refinement, so a tolerance asked for by mistake fills
        memory long before ``max_refinements`` stops it.
    delay_expansion : {None, 2, 3}, optional
        Expand the source about each grid point in the link and TDI delays
        instead of evaluating it at every delayed time, to second or third
        order. See `_expanded_source`. Both orders read one stencil, so 3
        costs no extra source calls. ``None``, the default, is exact.
    reference_delay : bool, optional
        Reduce the bracket against the delay every term shares, the light
        time to the constellation barycentre, and carry that delay in the
        carrier instead. See `_reference_delay`. The bracket then varies an
        order of magnitude more slowly and the grid shrinks with it.
    stall_refusal_factor : float, optional
        A refinement can stop improving at a response cancellation because
        float64 phase reduction sets a floor even while the reconstructed
        waveform is accurate.  Such an interval is recorded in diagnostics
        and accepted only if its error relative to the channel peak is below
        ``stall_refusal_factor * relative_tolerance``.  The default 100
        separates the measured arithmetic floor from a discontinuous source
        cutoff.  Set this to 1 for a strictly literal tolerance.
    stall_patience : int, optional
        Consecutive refinements that must fail to reduce an interval's error
        by a factor of two before it is classified as stalled.  The default
        preserves the inexpensive narrow-band guard.  Sharply non-stationary
        merger signals may need a larger value so a day-scale initial
        interval can reach their physical time scale before classification.
    evaluation_chunk_size : int, optional
        Maximum number of trial times held in one temporary delay-geometry
        object.  Adaptive refinement can test several hundred thousand points
        at once near merger; chunking bounds that temporary memory without
        changing the accepted grid or its error test.
    """
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
    stall_refusal_factor = float(stall_refusal_factor)
    if not np.isfinite(stall_refusal_factor) or stall_refusal_factor < 1:
        raise ValueError("stall_refusal_factor must be finite and at least one")
    stall_patience = int(stall_patience)
    if stall_patience < 1:
        raise ValueError("stall_patience must be at least one")
    evaluation_chunk_size = int(evaluation_chunk_size)
    if evaluation_chunk_size < 2:
        raise ValueError("evaluation_chunk_size must be at least two")
    if not 0 <= amplitude_floor <= 1:
        raise ValueError("amplitude_floor must lie in [0, 1]")
    if support_padding < 0:
        raise ValueError("support_padding must be non-negative")
    max_grid_points = int(max_grid_points)
    if max_grid_points < 4:
        raise ValueError("max_grid_points must be at least four")

    channel_names = tuple(channel_terms)
    records, diagnostics = [], {}

    def evaluate(harmonic, times):
        times = np.asarray(times, dtype=float)
        pieces = []
        for first in range(0, len(times), evaluation_chunk_size):
            stop = min(first + evaluation_chunk_size, len(times))
            # Constellation sampling needs two times.  If a batch leaves one
            # point, repeat the preceding point in this last geometry and
            # discard its duplicate output.
            duplicate = first > 0 and stop - first == 1
            selected = times[first - int(duplicate):stop]
            geometry = MultiChannelTermGeometry(
                orbit, selected, channel_terms, links=links,
                velocity_order=velocity_order,
                delay_expansion=delay_expansion,
                reference_delay=reference_delay, threads=threads)
            values = sparse_channels_terms(
                source, harmonic, geometry, lamb, beta)
            piece = np.stack([values[name] for name in channel_names])
            pieces.append(piece[:, 1:] if duplicate else piece)
        if not pieces:
            return np.empty((len(channel_names), 0), dtype=complex)
        return pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=1)

    for harmonic in source.harmonics:
        windows = harmonic_windows(
            source, harmonic, t_start, t_end, padding=support_padding)
        harmonic_grids, harmonic_values = [], []
        tested = 0
        deepest = 0
        stalled_count = 0
        stalled_error = 0.0
        stalled_location = float('nan')
        stalled_width = 0.0
        for low, high in windows:
            count = max(interpolation_order + 1,
                        int(np.ceil((high - low) / initial_step)) + 1)
            grid = np.linspace(low, high, count)
            values = evaluate(harmonic, grid)
            active_left, active_right = grid[:-1], grid[1:]
            active_error = np.full(len(active_left), np.inf)
            active_stall = np.zeros(len(active_left), dtype=int)

            for depth in range(int(max_refinements) + 1):
                if not len(active_left):
                    break
                fractions = np.asarray((0.25, 0.5, 0.75))
                probes = (active_left[:, None]
                          + fractions[None, :]
                          * (active_right - active_left)[:, None])
                # Two ways an interval runs out of floats, and both have
                # to be caught here: a probe landing on an endpoint, and
                # two probes landing on each other. The second leaves the
                # probe array non-monotone, which `sample_constellation`
                # rejects further down with a message about the time grid
                # rather than about the refinement that produced it.
                collapsed = (
                    np.any((probes == active_left[:, None])
                           | (probes == active_right[:, None]), axis=1)
                    | np.any(np.diff(probes, axis=1) <= 0.0, axis=1))
                if np.any(collapsed):
                    # The extreme of the stall the block below handles, and
                    # it arrives the other way round. There the error stops
                    # falling; here it keeps falling while the criterion
                    # falls with it, because `local` shrinks toward the
                    # amplitude floor as the bracket approaches a zero, so
                    # splitting reaches float spacing before it wins.
                    # Measured on the last mission window of an 8.3-year
                    # source: harmonic (2, 2, 6), depth 20, an interval
                    # 6e-08 s wide at 4.3e-07 of the channel peak.
                    #
                    # Account for it as stalling rather than raising. The
                    # refusal threshold below already says how much of this
                    # a grid may carry, and a stated tolerance is a better
                    # answer than a crash at one mission epoch.
                    index = np.flatnonzero(collapsed)
                    worst = index[np.argmax(active_error[index])]
                    if active_error[worst] > stalled_error:
                        stalled_error = float(active_error[worst])
                        stalled_location = float(
                            0.5 * (active_left[worst] + active_right[worst]))
                        stalled_width = float(
                            active_right[worst] - active_left[worst])
                    stalled_count += int(len(index))
                    keep = ~collapsed
                    active_left, active_right = (active_left[keep],
                                                 active_right[keep])
                    active_error, active_stall = (active_error[keep],
                                                  active_stall[keep])
                    if not len(active_left):
                        break
                    probes = (active_left[:, None]
                              + fractions[None, :]
                              * (active_right - active_left)[:, None])
                probes = probes.reshape(-1)
                owners = np.repeat(np.arange(len(active_left)), 3)
                exact = evaluate(harmonic, probes)
                predicted = np.stack([
                    interpolant(grid, row, interpolation_order)(probes)
                    for row in values
                ])
                peak = np.maximum(np.max(np.abs(values), axis=1),
                                  np.max(np.abs(exact), axis=1))[:, None]
                local = np.maximum(np.abs(exact), amplitude_floor * peak)
                error = np.abs(exact - predicted)
                failed_probe = np.any(error > relative_tolerance * local,
                                      axis=0)
                failed_interval = np.zeros(len(active_left), dtype=bool)
                failed_interval[owners[failed_probe]] = True
                tested += len(probes)
                if not np.any(failed_interval):
                    active_left = active_right = np.array([])
                    break

                # An interval whose error stops falling is not under-resolved.
                # Where the terms of a channel cancel -- a band straddling
                # c/2L is the case that brought this up -- the carrier phase,
                # a difference of two values of order 1e7 rad, carries that
                # cancellation's worth of roundoff into the bracket, and no
                # splitting reaches below it. Measured on such a band: seven
                # orders of interval width for no change in an error already
                # at 1.5e-8 of the channel peak.
                #
                # Three consecutive splits have to miss a factor of two before
                # an interval is given up on. A cubic over a resolved stretch
                # gains 4**4 per split, so that margin is 1e7 wide; one split
                # is not, because an early probe can land on a worse point
                # than its parent did and every harmonic of a ten-harmonic
                # source was then abandoned on the first bump.
                scaled = np.max(error / peak, axis=0)
                interval_error = np.zeros(len(active_left))
                np.maximum.at(interval_error, owners, scaled)
                stalling = np.where(interval_error > 0.5 * active_error,
                                    active_stall + 1, 0)
                stuck = failed_interval & (stalling >= stall_patience)
                if np.any(stuck):
                    stuck_indices = np.flatnonzero(stuck)
                    worst = stuck_indices[
                        np.argmax(interval_error[stuck_indices])]
                    if interval_error[worst] > stalled_error:
                        stalled_error = float(interval_error[worst])
                        stalled_location = float(
                            0.5 * (active_left[worst]
                                   + active_right[worst]))
                        stalled_width = float(
                            active_right[worst] - active_left[worst])
                    stalled_count += int(np.count_nonzero(stuck))
                    failed_interval &= ~stuck
                if not np.any(failed_interval):
                    active_left = active_right = np.array([])
                    break
                if depth == max_refinements:
                    raise RuntimeError(
                        "TDI response grid did not reach the requested "
                        f"tolerance after {max_refinements} refinements")

                selected = failed_interval[owners]
                if len(grid) + int(np.count_nonzero(selected)) > max_grid_points:
                    raise RuntimeError(
                        f"harmonic {harmonic}: the grid would pass "
                        f"max_grid_points={max_grid_points:,} at refinement "
                        f"{depth} with {int(np.count_nonzero(failed_interval)):,}"
                        " intervals still failing; loosen relative_tolerance "
                        "or split the window")
                insert_t = probes[selected]
                insert_v = exact[:, selected]
                joined_t = np.concatenate((grid, insert_t))
                joined_v = np.concatenate((values, insert_v), axis=1)
                order = np.argsort(joined_t)
                grid, values = joined_t[order], joined_v[:, order]
                next_left, next_right = [], []
                next_error, next_stall = [], []
                for owner in np.flatnonzero(failed_interval):
                    interior = probes[owners == owner]
                    edges = np.concatenate((
                        [active_left[owner]], interior,
                        [active_right[owner]]))
                    next_left.extend(edges[:-1])
                    next_right.extend(edges[1:])
                    next_error.extend([interval_error[owner]]
                                      * (len(edges) - 1))
                    next_stall.extend([stalling[owner]] * (len(edges) - 1))
                active_left = np.asarray(next_left)
                active_right = np.asarray(next_right)
                active_error = np.asarray(next_error)
                active_stall = np.asarray(next_stall, dtype=int)
                deepest = max(deepest, depth + 1)

            harmonic_grids.append(grid)
            harmonic_values.append(values)

        if not harmonic_grids:
            continue
        # What the refusal is for is a step: a source that drops this
        # harmonic mid-window leaves one at 0.1 to 1 of the channel peak,
        # four orders above a tolerance anyone asks for. Arithmetic stalls sit
        # at 1e-8. Refusing at the tolerance itself put a hard stop in between
        # -- 6.3e-05 against a requested 1e-05 on one Yorsh source, a grid
        # that was perfectly usable -- so the refusal keeps a factor of a
        # hundred of margin and anything short of that is recorded instead.
        if stalled_error > stall_refusal_factor * relative_tolerance:
            raise RuntimeError(
                f"harmonic {harmonic}: {stalled_count} interval(s) stopped "
                f"improving with an interpolation error of "
                f"{stalled_error:.2e} of the channel peak, above "
                f"{stall_refusal_factor:g} times the requested "
                f"{relative_tolerance:.1e}; the worst interval is centred "
                f"at t={stalled_location:.9g} s and is "
                f"{stalled_width:.3g} s wide. The bracket is most "
                "likely stepped rather than under-resolved; check whether "
                "the source cuts this harmonic off inside the window")
        grid = np.concatenate(harmonic_grids)
        values = np.concatenate(harmonic_values, axis=1)
        records.append({
            'harmonic': harmonic,
            'grid': grid,
            'brackets': dict(zip(channel_names, values, strict=True)),
            'support': windows,
            'reference_delay': (barycentre_delay(orbit, grid, lamb, beta)
                                if reference_delay else None),
        })
        diagnostics[harmonic] = {
            'grid_points': len(grid),
            'tested_points': tested,
            'deepest_refinement': deepest,
            'relative_tolerance': relative_tolerance,
            'stalled_intervals': stalled_count,
            'stalled_error': stalled_error,
            'stalled_location': stalled_location,
            'stalled_width': stalled_width,
            'stall_refusal_factor': stall_refusal_factor,
            'stall_patience': stall_patience,
            'evaluation_chunk_size': evaluation_chunk_size,
        }
    return SparseTDIResponse(source, records, diagnostics=diagnostics,
                             interpolation_order=interpolation_order)
