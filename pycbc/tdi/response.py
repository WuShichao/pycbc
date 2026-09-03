# Copyright (C) 2026  Shichao Wu, Alex Nitz
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""First-order time-domain response of individual laser links."""

from dataclasses import dataclass

import numpy as np
from astropy.constants import c as SPEED_OF_LIGHT

C_SI = SPEED_OF_LIGHT.value
LINK_ORDER = ((1, 2), (2, 3), (3, 1), (1, 3), (3, 2), (2, 1))


@dataclass(frozen=True)
class ConstellationSample:
    """Sky-independent orbit data evaluated on one reception-time grid."""

    t: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    n_hat: np.ndarray
    r_emit: np.ndarray
    ltt: np.ndarray
    links: tuple = LINK_ORDER

    @property
    def delays(self):
        """Return light travel times with PyTDI's ``d_ij`` labels."""
        return {
            f"d_{receiver}{emitter}": self.ltt[:, index]
            for index, (receiver, emitter) in enumerate(self.links)
        }


@dataclass(frozen=True)
class LinkGeometry:
    """Sky-dependent factors for each reception time and directed link."""

    prefactor: np.ndarray
    tau_emit: np.ndarray
    tau_recv: np.ndarray
    weight_emit: np.ndarray
    weight_recv: np.ndarray


def _validate_links(links):
    links = tuple(tuple(link) for link in links)
    if len(set(links)) != len(links):
        raise ValueError("links must be unique")
    for receiver, emitter in links:
        if receiver == emitter or receiver not in (1, 2, 3) \
                or emitter not in (1, 2, 3):
            raise ValueError(f"invalid directed link {(receiver, emitter)}")
    return links


def sample_constellation(t, orbit, *, ltt_order=1, links=LINK_ORDER,
                         tolerance=1e-12, max_iterations=20):
    """Sample an orbit and solve each directed link's light-cone equation.

    ``ltt_order=0`` uses the simultaneous-distance light travel time but still
    constructs the direction from the corresponding emission event.
    ``ltt_order=1`` (the default) iterates

    ``c L_ij = |r_i(t) - r_j(t - L_ij)|``

    to the requested absolute tolerance in seconds. Solar Shapiro and other
    second-order corrections are intentionally not guessed here;
    ``ltt_order=2`` therefore raises ``NotImplementedError``.
    """
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or len(t) < 2:
        raise ValueError("t must be a one-dimensional array with at least two samples")
    if np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing")
    if ltt_order not in (0, 1, 2):
        raise ValueError("ltt_order must be 0, 1, or 2")
    if ltt_order == 2:
        raise NotImplementedError(
            "ltt_order=2 requires a specified solar-Shapiro convention"
        )
    links = _validate_links(links)
    # The light-cone iteration evaluates the emitter at t - delay, so the
    # achievable precision is set by the spacing of t itself, not by the
    # delay. At a two-year mission time t ~ 6.3e7 s the float64 spacing is
    # 7.5e-9 s, which a fixed 1e-12 s tolerance can never reach -- the solve
    # then exhausts max_iterations and raises on a converged answer. Floor the
    # tolerance at the representable resolution.
    tolerance = max(float(tolerance),
                    4 * float(np.spacing(np.max(np.abs(t)))))
    position = np.asarray(orbit.compute_position(t, (1, 2, 3)), dtype=float)
    velocity = np.asarray(orbit.compute_velocity(t, (1, 2, 3)), dtype=float)
    expected = (len(t), 3, 3)
    if position.shape != expected or velocity.shape != expected:
        raise ValueError(
            "orbit compute_position/compute_velocity must return shape "
            f"{expected}; got {position.shape} and {velocity.shape}"
        )

    n_hat = np.empty((len(t), len(links), 3))
    r_emit = np.empty_like(n_hat)
    ltt = np.empty((len(t), len(links)))
    for index, (receiver, emitter) in enumerate(links):
        r_recv = position[:, receiver - 1]
        r_emit_now = position[:, emitter - 1]
        delay = np.linalg.norm(r_recv - r_emit_now, axis=-1) / C_SI
        if ltt_order == 1:
            for _ in range(max_iterations):
                emitted = np.asarray(
                    orbit.compute_position(t - delay, (emitter,)), dtype=float
                )[:, 0]
                updated = np.linalg.norm(r_recv - emitted, axis=-1) / C_SI
                if np.max(np.abs(updated - delay)) <= tolerance:
                    delay = updated
                    break
                delay = updated
            else:
                raise RuntimeError(
                    f"light-cone solve did not converge for link "
                    f"{receiver}{emitter} in {max_iterations} iterations"
                )
        emitted = np.asarray(
            orbit.compute_position(t - delay, (emitter,)), dtype=float
        )[:, 0]
        separation = r_recv - emitted
        distance = np.linalg.norm(separation, axis=-1)
        n_hat[:, index] = separation / distance[:, None]
        r_emit[:, index] = emitted
        ltt[:, index] = delay

    return ConstellationSample(
        t=t,
        position=position,
        velocity=velocity,
        n_hat=n_hat,
        r_emit=r_emit,
        ltt=ltt,
        links=links,
    )


def polarization_basis(lamb, beta):
    """Return ``(u_hat, v_hat, k_hat)`` in the SSB ecliptic frame."""
    sin_lamb, cos_lamb = np.sin(lamb), np.cos(lamb)
    sin_beta, cos_beta = np.sin(beta), np.cos(beta)
    u_hat = np.array((sin_lamb, -cos_lamb, 0.0))
    v_hat = np.array(
        (-sin_beta * cos_lamb, -sin_beta * sin_lamb, cos_beta)
    )
    k_hat = np.array(
        (-cos_beta * cos_lamb, -cos_beta * sin_lamb, -sin_beta)
    )
    return u_hat, v_hat, k_hat


def antenna_pattern(n_hat, u_hat, v_hat):
    """Return the plus and cross link contractions for ``n_hat``."""
    n_dot_u = np.einsum("...a,a->...", n_hat, u_hat)
    n_dot_v = np.einsum("...a,a->...", n_hat, v_hat)
    return n_dot_u ** 2 - n_dot_v ** 2, 2 * n_dot_u * n_dot_v


def doppler_factors(k_hat, n_hat, v_emitter, v_receiver):
    """Return the frequency-independent Speri Eq. (D2)-(D3) factors."""
    k_dot_emit = np.einsum("a,...a->...", k_hat, v_emitter)
    k_dot_recv = np.einsum("a,...a->...", k_hat, v_receiver)
    n_dot_recv = np.einsum("...a,...a->...", n_hat, v_receiver)
    n_dot_combined = np.einsum(
        "...a,...a->...", n_hat, v_emitter - 2 * v_receiver
    )
    eps1 = (-k_dot_emit + n_dot_recv) / C_SI
    eps2 = (-k_dot_recv + n_dot_combined) / C_SI
    return eps1, eps2


def link_geometry(sample, lamb, beta, *, velocity_order=1,
                  retard_emitter=True):
    """Build first-order sky-dependent response factors.

    ``velocity_order=0`` removes only the explicit Doppler weights. It does
    not replace the light-cone arm direction with a simultaneous arm.
    """
    if velocity_order not in (0, 1):
        raise ValueError("velocity_order must be 0 or 1")
    u_hat, v_hat, k_hat = polarization_basis(lamb, beta)
    xi_plus, xi_cross = antenna_pattern(sample.n_hat, u_hat, v_hat)
    denominator = 1 - np.einsum("nla,a->nl", sample.n_hat, k_hat)
    if np.any(np.abs(denominator) < 64 * np.finfo(float).eps):
        raise ValueError(
            "source is numerically collinear with a link; the time-domain "
            "prefactor has a removable 0/0 that requires a joint limit"
        )
    prefactor = np.stack((xi_plus, xi_cross), axis=-1)
    prefactor /= 2 * denominator[..., None]

    receivers = np.array([link[0] - 1 for link in sample.links])
    emitters = np.array([link[1] - 1 for link in sample.links])
    r_recv = sample.position[:, receivers]
    if retard_emitter:
        r_emit = sample.r_emit
    else:
        r_emit = sample.position[:, emitters]
    tau_emit = sample.ltt + np.einsum("nla,a->nl", r_emit, k_hat) / C_SI
    tau_recv = np.einsum("nla,a->nl", r_recv, k_hat) / C_SI

    if velocity_order == 1:
        eps1, eps2 = doppler_factors(
            k_hat,
            sample.n_hat,
            sample.velocity[:, emitters],
            sample.velocity[:, receivers],
        )
    else:
        eps1 = np.zeros_like(sample.ltt)
        eps2 = np.zeros_like(sample.ltt)
    return LinkGeometry(
        prefactor=prefactor,
        tau_emit=tau_emit,
        tau_recv=tau_recv,
        weight_emit=1 + eps1,
        weight_recv=1 + eps2,
    )


def link_response(source, sample, geometry, *, links=LINK_ORDER):
    """Evaluate fractional-frequency responses in directed-link order."""
    links = tuple(tuple(link) for link in links)
    if links != sample.links:
        raise ValueError("links must match the order used to build sample")
    emit_time = sample.t[:, None] - geometry.tau_emit
    recv_time = sample.t[:, None] - geometry.tau_recv
    hp_emit, hc_emit = source.polarizations(emit_time)
    hp_recv, hc_recv = source.polarizations(recv_time)
    delta_plus = geometry.weight_emit * hp_emit \
        - geometry.weight_recv * hp_recv
    delta_cross = geometry.weight_emit * hc_emit \
        - geometry.weight_recv * hc_recv
    return geometry.prefactor[..., 0] * delta_plus \
        + geometry.prefactor[..., 1] * delta_cross
