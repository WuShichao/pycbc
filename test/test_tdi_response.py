"""Tests for the native directed-link response."""

import numpy as np

from pycbc.coordinates.space_orbit import LisaEqualArmOrbit
from pycbc.tdi.response import (
    C_SI,
    antenna_pattern,
    doppler_factors,
    link_geometry,
    link_response,
    polarization_basis,
    sample_constellation,
)


class MonochromaticSource:
    """Analytic source that supports arbitrary multidimensional time arrays."""

    def __init__(self, frequency):
        self.frequency = frequency

    def polarizations(self, t):
        phase = 2 * np.pi * self.frequency * np.asarray(t)
        return np.cos(phase), 0.3 * np.sin(phase)


def test_light_cone_and_velocity_response():
    times = np.arange(512, dtype=float) * 2.0
    orbit = LisaEqualArmOrbit()
    sample = sample_constellation(times, orbit)

    receivers = np.array([link[0] - 1 for link in sample.links])
    separation = sample.position[:, receivers] - sample.r_emit
    residual = np.linalg.norm(separation, axis=-1) - C_SI * sample.ltt
    assert np.max(np.abs(residual)) < 1e-5
    assert np.allclose(np.linalg.norm(sample.n_hat, axis=-1), 1.0)

    geometry = link_geometry(sample, 1.1, -0.4, velocity_order=1)
    no_velocity = link_geometry(sample, 1.1, -0.4, velocity_order=0)
    assert np.all(no_velocity.weight_emit == 1)
    assert np.all(no_velocity.weight_recv == 1)
    assert np.max(np.abs(geometry.weight_emit - 1)) > 1e-6
    assert np.max(np.abs(geometry.weight_emit - 1)) < 1e-3

    source = MonochromaticSource(0.01)
    links = link_response(source, sample, geometry)
    links_without_velocity = link_response(source, sample, no_velocity)
    assert links.shape == (len(times), 6)
    assert np.all(np.isfinite(links))
    assert np.max(np.abs(links - links_without_velocity)) > 0


def test_doppler_formula_and_basis_are_orthonormal():
    u_hat, v_hat, k_hat = polarization_basis(0.7, 0.2)
    basis = np.stack((u_hat, v_hat, k_hat))
    assert np.allclose(basis @ basis.T, np.eye(3), atol=1e-15)

    n_hat = np.array([[[1.0, 0.0, 0.0]]])
    v_emitter = np.array([[[10.0, 20.0, 30.0]]])
    v_receiver = np.array([[[40.0, 50.0, 60.0]]])
    eps1, eps2 = doppler_factors(
        np.array([0.0, 0.0, 1.0]), n_hat, v_emitter, v_receiver
    )
    assert np.allclose(eps1, (-30.0 + 40.0) / C_SI)
    assert np.allclose(eps2, (-60.0 + 10.0 - 80.0) / C_SI)


def test_monochromatic_response_matches_frequency_domain_expression():
    frequency = 0.013
    times = np.arange(128, dtype=float) * 2.0
    sample = sample_constellation(times, LisaEqualArmOrbit())
    geometry = link_geometry(sample, 0.9, -0.25, velocity_order=1)

    class ComplexPlusSource:
        def polarizations(self, query):
            phase = np.exp(2j * np.pi * frequency * np.asarray(query))
            return phase, np.zeros_like(phase)

    actual = link_response(ComplexPlusSource(), sample, geometry)
    carrier = np.exp(
        2j * np.pi * frequency * (sample.t[:, None] - geometry.tau_recv)
    )
    delay_difference = geometry.tau_recv - geometry.tau_emit
    expected = geometry.prefactor[..., 0] * carrier * (
        geometry.weight_emit
        * np.exp(2j * np.pi * frequency * delay_difference)
        - geometry.weight_recv
    )
    assert np.allclose(actual, expected, rtol=2e-14, atol=2e-14)


def test_antenna_pattern_is_the_polarization_tensor_contraction():
    """``xi_p = n^a n^b e^p_ab``, by the tensors rather than by the shortcut.

    `antenna_pattern` returns (n.u)^2 - (n.v)^2 and 2 (n.u)(n.v), which is
    Speri Eq (7)-(8) and Costa Barroso Eq (A5)-(A6) -- cheaper than building
    3x3 polarization tensors, and the same form the coronagraphic-kappa
    coefficients need, so one implementation has to serve both.  Checking it
    against the contraction it stands in for is the only way that claim is
    load-bearing rather than a comment.
    """
    generator = np.random.default_rng(20260910)
    for _ in range(8):
        lamb = generator.uniform(0, 2 * np.pi)
        beta = np.arcsin(generator.uniform(-1, 1))
        u_hat, v_hat, k_hat = polarization_basis(lamb, beta)
        plus_tensor = np.outer(u_hat, u_hat) - np.outer(v_hat, v_hat)
        cross_tensor = np.outer(u_hat, v_hat) + np.outer(v_hat, u_hat)

        direction = generator.normal(size=(32, 3))
        direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
        xi_plus, xi_cross = antenna_pattern(direction, u_hat, v_hat)
        contracted_plus = np.einsum('na,ab,nb->n', direction, plus_tensor,
                                    direction)
        contracted_cross = np.einsum('na,ab,nb->n', direction, cross_tensor,
                                     direction)
        assert np.max(np.abs(xi_plus - contracted_plus)) < 1e-14
        assert np.max(np.abs(xi_cross - contracted_cross)) < 1e-14

        # transverse-traceless: a wave carries no response along its own
        # propagation direction
        along_plus, along_cross = antenna_pattern(k_hat[None, :], u_hat, v_hat)
        assert abs(along_plus[0]) < 1e-14 and abs(along_cross[0]) < 1e-14
