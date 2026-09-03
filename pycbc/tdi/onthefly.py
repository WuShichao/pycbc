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


def adaptive_time_grid(source, harmonic, t_start, t_end, delta_phi=0.5,
                       growth=1.1, dt_max=None, t_ref=None,
                       max_step_scale=4096.0):
    """Cornish & Littenberg Sec. III.B grid: dense at merger, growing outward.

    The step is set by the carrier, ``dt = delta_phi / omega``, then allowed to
    grow geometrically once far from the reference time. ``dt_max`` caps it;
    the cap that matters is the constellation's own timescale, not the
    waveform's, so it defaults to a day.
    """
    if dt_max is None:
        dt_max = 86400.0
    t_ref = t_end if t_ref is None else t_ref
    times, t, step_scale = [t_start], t_start, delta_phi
    while t < t_end:
        omega = float(np.abs(source.angular_frequency(harmonic, np.array([t]))[0]))
        dt = step_scale / omega if omega > 0 else dt_max
        dt = min(max(dt, 1e-3), dt_max)
        t = t + dt
        # The step is allowed to grow far past a GW period: once the carrier
        # is factored out, what is being sampled varies on the ORBITAL
        # timescale. Capping the growth at a few GW periods, as a first version
        # of this did, throws away most of the available speedup.
        step_scale = min(step_scale * growth, max_step_scale * delta_phi)
        times.append(min(t, t_end))
    grid = np.unique(np.asarray(times))
    return grid[grid <= t_end]


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


def reconstruct(source, harmonic, grid, bracket, times):
    """Spline the slow bracket onto ``times`` and reattach the carrier."""
    from scipy.interpolate import CubicSpline
    amplitude = np.abs(bracket)
    phase = np.unwrap(np.angle(bracket))
    amp = CubicSpline(grid, amplitude)(times)
    ang = CubicSpline(grid, phase)(times)
    return np.real(amp * np.exp(1j * (ang + source.carrier_phase(harmonic,
                                                                 times))))
