"""Finite-interval correlations, analytic derivatives and physical time mass."""
import numpy as np
import pytest

from pycbc.filter.correlation_polynomial import (
    CorrelationPolynomial, interval_weights)


def direct(coefficients, first, spacing, t, derivative=0, reference=0.):
    f = first - reference + np.arange(coefficients.shape[1]) * spacing
    return (coefficients * (2j * np.pi * f)**derivative) @ np.exp(2j * np.pi * f[:, None] * t)


@pytest.mark.parametrize('bounds', [(-.41, .37), (0., .5), (-.6, -.02)])
def test_signed_endpoints_and_derivatives_equal_independent_direct_sums(bounds):
    rng = np.random.default_rng(20267270)
    values = rng.normal(size=(2, 49)) + 1j * rng.normal(size=(2, 49))
    polynomial = CorrelationPolynomial(values, .173, .75)
    for reference in (0., 15.7):
        t, fields = polynomial.interval(*bounds, 117, derivatives=(0, 1, 2),
                                        reference_frequency=reference)
        assert t[0] == bounds[0] and t[-1] == bounds[1]
        for order, field in fields.items():
            np.testing.assert_allclose(field,
                direct(values, .173, .75, t, order, reference), rtol=1e-10, atol=2e-8)


def test_long_physical_grid_does_not_need_full_period_time_arrays():
    rng = np.random.default_rng(20267271)
    values = np.zeros((2, 131072), complex)
    nodes = rng.choice(values.shape[1], 37, replace=False)
    values[:, nodes] = rng.normal(size=(2, 37)) + 1j * rng.normal(size=(2, 37))
    first, spacing = .0145, 1 / (4 * 31557600.)
    polynomial = CorrelationPolynomial(values, first, spacing)
    reference = first + (values.shape[1] - 1) * spacing / 2
    t, fields = polynomial.interval(-259200., 259200., 65,
        derivatives=(0, 1), reference_frequency=reference)
    f = first - reference + nodes * spacing
    for order, field in fields.items():
        wanted = (values[:, nodes] * (2j * np.pi * f)**order) @ np.exp(2j * np.pi * f[:, None] * t)
        np.testing.assert_allclose(field, wanted, rtol=1e-8, atol=1e-9)


def test_hermite_remainder_controls_real_and_imaginary_components():
    values = np.array([[1. + .3j, -.2j, -.4, .1], [.2, .4j, .3, -.1j]])
    polynomial = CorrelationPolynomial(values, 3., .7)
    errors = []
    for count in (33, 65):
        spline, bound = polynomial.hermite(-.2, .3, count, reference_frequency=4.)
        t = np.linspace(-.2, .3, 4097)
        residual = spline(t) - direct(values, 3., .7, t, reference=4.)
        assert np.all(np.max(np.abs(residual.real), axis=1) <= bound + 1e-12)
        assert np.all(np.max(np.abs(residual.imag), axis=1) <= bound + 1e-12)
        assert np.isnan(spline([-.3, .4])).all()
        errors.append(np.max(np.abs(residual)))
    assert errors[1] < errors[0] / 12


def test_endpoint_inclusive_time_prior_preserves_mass_and_linear_integrals():
    for t in (np.linspace(-259200., 259200., 267),
              np.array([-259200., -1000., 0., 400., 259200.])):
        weights = interval_weights(t)
        assert np.all(weights > 0)
        assert abs(weights.sum() - 1) < 1e-14
        assert abs(weights @ t) < 1e-10


def test_coefficients_are_owned_and_immutable_and_common_phase_is_not_fitted():
    values = np.array([[1. + 2j, .1, -.3j]])
    polynomial = CorrelationPolynomial(values, .15, .01)
    values[:] = 0
    t, ordinary = polynomial.interval(-5., 7., 33)
    _, shifted = polynomial.interval(-5., 7., 33, reference_frequency=.17)
    np.testing.assert_allclose(shifted[0] * np.exp(2j * np.pi * .17 * t), ordinary[0], atol=1e-12)
    with pytest.raises(ValueError):
        polynomial.coefficients[0, 0] = 0


def test_invalid_inputs_and_explicit_output_budget_are_rejected():
    with pytest.raises(ValueError):
        CorrelationPolynomial(np.ones(3), 0., 1.)
    polynomial = CorrelationPolynomial(np.ones((2, 8)), .1, .2)
    for arguments in [(0., 0., 3), (0., np.inf, 3), (0., 1., 1)]:
        with pytest.raises(ValueError):
            polynomial.interval(*arguments)
    for orders in [(), (0, 0), (-1,), (.5,)]:
        with pytest.raises(ValueError):
            polynomial.interval(0., 1., 3, derivatives=orders)
    with pytest.raises(MemoryError):
        polynomial.interval(0., 1., 10, max_output_bytes=1)
    with pytest.raises(ValueError):
        polynomial.derivative_bound(-1)
    with pytest.raises(ValueError):
        interval_weights([0., 1., 1.])
