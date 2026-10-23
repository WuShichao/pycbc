"""Finite-frequency streaming controls against the existing dense operators."""
import weakref

import numpy as np
import pytest

from pycbc.tdi.harmonic_waveform import harmonic_correlation_spectrum, harmonic_spectrum
from pycbc.tdi.streamed_harmonics import (
    InterpolatedHarmonicResponse, streamed_correlation_spectrum,
    streamed_harmonic_spectrum)


@pytest.mark.parametrize('periods', [1, 2, 4])
def test_interpolated_stream_equals_dense_operators_without_frequency_wrap(periods):
    rng = np.random.default_rng(20267310)
    coarse, fine = np.linspace(.013, .021, 13), np.linspace(.013, .021, 73)
    orders = np.array([-100, -3, 0, 2, 100])
    rho = rng.normal(size=(5, 2, 2, 13)) + 1j*rng.normal(size=(5, 2, 2, 13))
    lazy = InterpolatedHarmonicResponse(orders, coarse, fine, rho)
    dense = np.array([lazy.at_index(i) for i in range(len(orders))])
    source = rng.normal(size=len(fine)) + 1j*rng.normal(size=len(fine))
    data = rng.normal(size=(2, len(fine))) + 1j*rng.normal(size=(2, len(fine)))
    np.testing.assert_allclose(streamed_harmonic_spectrum(source, lazy, orders, periods),
                               harmonic_spectrum(source, dense, orders, periods), atol=1e-12)
    np.testing.assert_allclose(streamed_correlation_spectrum(source, lazy, orders, data, periods),
                               harmonic_correlation_spectrum(source, dense, orders, data, periods), atol=1e-12)


def test_response_arrays_are_owned_and_extrapolation_or_oversized_blocks_fail():
    orders, coarse, fine = np.array([-1, 0, 1]), np.linspace(0, 1, 5), np.linspace(0, 1, 13)
    rho = np.ones((3, 2, 2, 5), complex)
    lazy = InterpolatedHarmonicResponse(orders, coarse, fine, rho)
    rho[:] = 0.; orders[:] = 99; coarse[:] = 10.; fine[:] = 10.
    np.testing.assert_array_equal(lazy.at_index(1), np.ones((2, 2, 13)))
    with pytest.raises(ValueError):
        lazy.coarse_responses[:] = 0.
    with pytest.raises(ValueError):
        InterpolatedHarmonicResponse([0], [0., 1.], [-.1, .5], np.ones((1, 2, 2, 2)))
    with pytest.raises(MemoryError):
        InterpolatedHarmonicResponse([0], [0., 1.], [0., .5], np.ones((1, 2, 2, 2)), max_block_bytes=1)


def test_stream_does_not_materialize_or_keep_previous_harmonics():
    class Response:
        orders = np.arange(-3, 4)
        shape = (7, 2, 2, 31)
        dtype = np.dtype(complex)
        def __init__(self):
            self.references = []
        def __array__(self, *args, **kwargs):
            raise AssertionError('the full response must never be materialized')
        def at_index(self, index):
            assert not any(r() is not None for r in self.references)
            value = np.full(self.shape[1:], index+1j)
            self.references.append(weakref.ref(value))
            return value
    for operation in ('template', 'correlation'):
        lazy = Response(); source = np.ones(31)
        if operation == 'template':
            streamed_harmonic_spectrum(source, lazy, lazy.orders)
        else:
            streamed_correlation_spectrum(source, lazy, lazy.orders, np.ones((2, 31)))
        assert len(lazy.references) == len(lazy.orders)
        assert not any(r() is not None for r in lazy.references)


def test_mismatched_orders_and_bad_generated_harmonics_are_rejected():
    lazy = InterpolatedHarmonicResponse([0], [0., 1.], [0., .5], np.ones((1, 2, 2, 2)))
    with pytest.raises(ValueError):
        streamed_harmonic_spectrum(np.ones(2), lazy, [1])
    with pytest.raises(ValueError):
        streamed_correlation_spectrum(np.ones(2), lazy, [0], np.ones((3, 2)))
    lazy.at_index = lambda index: np.full((2, 2, 2), np.nan)
    with pytest.raises(ValueError):
        streamed_harmonic_spectrum(np.ones(2), lazy, [0])


def test_streamed_original_band_likelihood_matches_dense_time_and_gram():
    pytest.importorskip('finufft')
    from pycbc.filter.detector_harmonic import DetectorHarmonicLikelihood
    from pycbc.filter.streamed_detector_harmonic import StreamedDetectorHarmonicLikelihood
    from pycbc.tdi.detector_projection import DetectorBandProjector
    from pycbc.tdi.ssb_yearly_harmonics import SSBGrid
    from test_ssb_yearly_harmonics import _StillOrbit, SKY, DAY
    rng = np.random.default_rng(20267311)
    grid = SSBGrid(0., DAY, .015, 9, 1/DAY, periods=2, period=DAY, guard=3)
    projection = DetectorBandProjector(grid, _StillOrbit(), *SKY, oversampling=4)
    data = rng.normal(size=(2, 9)) + 1j*rng.normal(size=(2, 9))
    weight = rng.uniform(.3, 2., size=data.shape)
    source = rng.normal(size=grid.size) + 1j*rng.normal(size=grid.size)
    orders = np.array([-2, 0, 1])
    coarse = np.linspace(grid.frequencies[0], grid.frequencies[-1], 7)
    rho = rng.normal(size=(3, 2, 2, 7)) + 1j*rng.normal(size=(3, 2, 2, 7))
    lazy = InterpolatedHarmonicResponse(orders, coarse, grid.frequencies, rho)
    dense = np.array([lazy.at_index(i) for i in range(3)])
    a = DetectorHarmonicLikelihood(projection, data, weight)
    b = StreamedDetectorHarmonicLikelihood(projection, data, weight)
    ta, ga = a.statistics(source, dense, orders, [-.3*DAY, 0., .2*DAY])
    tb, gb = b.statistics(source, lazy, orders, [-.3*DAY, 0., .2*DAY])
    np.testing.assert_allclose(ta, tb, atol=1e-12)
    np.testing.assert_allclose(ga, gb, atol=1e-12)
    pa = a.correlation_polynomial(source, dense, orders)
    pb = b.correlation_polynomial(source, lazy, orders)
    _, xa = pa.interval(-.3*DAY, .2*DAY, 13)
    _, xb = pb.interval(-.3*DAY, .2*DAY, 13)
    np.testing.assert_allclose(xa[0], xb[0], atol=1e-12)
