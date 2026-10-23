"""Endpoint stability on prepared grids without fitting waveform parameters."""
import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from pycbc.tdi.onthefly import (
    _local_endpoint_derivative, interpolant, SparseTDIResponse)


def clustered_knots():
    return np.array([0., 1., 2., 3., 3. + 1e-6, 3. + 2e-6, 4.])


def test_default_preserves_existing_scipy_representation():
    x = clustered_knots()
    y = np.sin(x / 10.) + 1j * np.cos(x / 10.)
    t = np.linspace(0., 4., 513)
    np.testing.assert_array_equal(interpolant(x, y)(t), CubicSpline(x, y)(t))


@pytest.mark.parametrize('reverse', [False, True])
def test_cubic_exactness_on_clustered_endpoints_and_large_epochs(reverse):
    x = clustered_knots()
    if reverse:
        x = 4. - x[::-1]
    y = (1. + 2j) + (3. - 4j) * x + .5 * x**2 - .02j * x**3
    t = np.linspace(0., 4., 513)
    wanted = (1. + 2j) + (3. - 4j) * t + .5 * t**2 - .02j * t**3
    for epoch, scale in [(0., 1.), (1e7, 20000.)]:
        actual = interpolant(epoch + scale * x, y,
            endpoint_condition='local-cubic')(epoch + scale * t)
        np.testing.assert_allclose(actual, wanted, atol=2e-8, rtol=2e-9)


def test_stencil_controls_bounded_sample_perturbations():
    x = clustered_knots()
    eta = 1e-10
    for right in (False, True):
        baseline = _local_endpoint_derivative(x, np.sin(x / 10.), right)
        for i in range(len(x)):
            y = np.sin(x / 10.)
            y[i] += eta
            actual = _local_endpoint_derivative(x, y, right)
            assert abs(actual - baseline) <= 64 * eta


def test_known_smooth_counterexample_is_not_amplified_at_the_endpoint():
    x = clustered_knots()
    y0 = np.sin(x / 10.)
    y = y0.copy()
    eta = 1e-10
    y[-3] += eta
    t = np.linspace(0., 4., 4097)
    stable = interpolant(x, y, endpoint_condition='local-cubic')(t)
    baseline = interpolant(x, y0, endpoint_condition='local-cubic')(t)
    # Retaining every perturbed knot still forces large local slopes. Check
    # their actual linear sensitivity, not an assumed all-spline error bound
    # from the separately bounded endpoint derivative.
    local_weights = interpolant(x, np.eye(len(x)),
        endpoint_condition='local-cubic')(t)
    old_weights = interpolant(x, np.eye(len(x)))(t)
    predicted = eta * np.max(np.abs(local_weights[:, -3]))
    assert np.max(np.abs(stable - baseline)) <= predicted + 1e-10
    assert np.max(np.sum(np.abs(local_weights), axis=1)) < np.max(
        np.sum(np.abs(old_weights), axis=1)) / 1000


def test_complex_rotation_linearity_and_every_node_are_preserved():
    x = clustered_knots()
    y = np.sin(x / 10.) + 1j * np.cos(x / 10.)
    phase = np.exp(.7j)
    t = np.linspace(0., 4., 513)
    first = interpolant(x, y, endpoint_condition='local-cubic')
    rotated = interpolant(x, phase * y, endpoint_condition='local-cubic')
    np.testing.assert_allclose(first(x), y, atol=1e-12)
    np.testing.assert_allclose(rotated(t), phase * first(t), atol=2e-9)


def test_sparse_response_keeps_the_declared_support():
    from types import SimpleNamespace
    x = clustered_knots()
    source = SimpleNamespace(carrier_phase=lambda h, t: np.zeros_like(t))
    record = dict(harmonic=2, grid=x, support=[0., 4.],
                  brackets={'A': np.sin(x / 10.)})
    response = SparseTDIResponse(source, [record], endpoint_condition='local-cubic')
    actual = response.sample(np.array([-.1, 0., 1., 4., 4.1]))['A']
    assert actual[0] == 0. and actual[-1] == 0.
    np.testing.assert_allclose(actual[1:-1], np.sin(np.array([0., 1., 4.]) / 10.))


@pytest.mark.parametrize('x', [[0., 1., 2.], [0., 1., 1., 2.],
                              [0., 1., 2., np.inf]])
def test_invalid_endpoint_data_are_rejected(x):
    with pytest.raises(ValueError):
        _local_endpoint_derivative(x, np.ones(len(x)))


def test_unknown_or_inapplicable_boundary_modes_are_rejected():
    with pytest.raises(ValueError, match='unknown endpoint'):
        interpolant(np.arange(6.), np.ones(6), endpoint_condition='guess')
    with pytest.raises(ValueError, match='require cubic'):
        interpolant(np.arange(6.), np.ones(6), order=5,
                    endpoint_condition='local-cubic')


def test_inference_endpoint_modes_do_not_share_candidate_projections(monkeypatch):
    from types import SimpleNamespace
    from pycbc.tdi import inference
    class Prepared:
        harmonics = (2,)
        def __init__(self):
            self.calls = []
        def project(self, source, lon, lat, **kwargs):
            self.calls.append(kwargs)
            return object()
    prepared = Prepared()
    source = SimpleNamespace(harmonics=(2,))
    params = dict(tdi_preparation_id='endpoint-test',
                  eclipticlongitude=1., eclipticlatitude=.2)
    inference.clear_cache()
    monkeypatch.setattr(inference, '_build_source', lambda p: source)
    monkeypatch.setattr(inference, '_narrow', lambda s, p: s)
    try:
        old = inference._projection(params, prepared)
        stable = inference._projection(dict(params, tdi_endpoint_condition='local-cubic'), prepared)
        assert old is not stable
        assert inference._projection(params, prepared) is old
        assert inference._projection(dict(params, tdi_endpoint_condition='local-cubic'), prepared) is stable
        assert prepared.calls == [{}, dict(endpoint_condition='local-cubic')]
        with pytest.raises(ValueError, match='tdi_endpoint_condition'):
            inference._projection(dict(params, tdi_endpoint_condition='guess'), prepared)
    finally:
        inference.clear_cache()
