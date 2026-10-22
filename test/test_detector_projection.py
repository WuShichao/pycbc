"""Independent Fourier quadrature of detector-frame waveform projection."""
import numpy
import pytest

pytest.importorskip('finufft')

from pycbc.tdi.detector_projection import DetectorBandProjector
from pycbc.tdi.ssb_yearly_harmonics import SSBGrid, ssb_delay
from test_ssb_yearly_harmonics import ORBIT, SKY, DAY, _StillOrbit


def test_stationary_frame_recovers_original_tones_with_absolute_scale():
    duration, m = 30 * DAY, 20
    grid = SSBGrid(0., duration, .015, m, 1 / duration, periods=4,
                   period=duration, guard=3)
    rng = numpy.random.default_rng(20267234)
    bins = rng.normal(size=(2, m)) + 1j * rng.normal(size=(2, m))
    spectrum = numpy.zeros((2, grid.size), complex)
    # Native FD density of a discrete tone scales with transform duration.
    where = grid.guard + 4 * numpy.arange(m)
    spectrum[:, where] = 4 * bins
    for oversampling in (2, 4):
        projection = DetectorBandProjector(grid, _StillOrbit(), *SKY,
                                            oversampling=oversampling)
        numpy.testing.assert_allclose(projection.project(spectrum), bins,
                                      rtol=1e-10, atol=1e-10)


def test_retarded_waveform_agrees_with_direct_dense_detector_quadrature():
    duration = 2 * DAY
    grid = SSBGrid(1e6, 1e6 + duration, .015, 9, 1 / duration, periods=2,
                   period=duration, guard=2)
    projector = DetectorBandProjector(grid, ORBIT, *SKY, oversampling=4)
    rng = numpy.random.default_rng(20267235)
    spectrum = rng.normal(size=grid.size) + 1j * rng.normal(size=grid.size)
    detector = projector.detector_nodes
    ssb = detector + ssb_delay(ORBIT, detector, *SKY)
    values = numpy.exp(2j * numpy.pi * (ssb - grid.start)[:, None] * grid.frequencies[None]) @ spectrum
    frequencies = grid.data_first_frequency + numpy.arange(grid.data_nbins) * grid.data_delta_f
    expected = grid.delta_f / grid.data_delta_f / len(detector) * (
        numpy.exp(-2j * numpy.pi * frequencies[:, None] * (detector - grid.start)[None]) @ values)
    numpy.testing.assert_allclose(projector.project(spectrum), expected, rtol=1e-9, atol=1e-9)


def test_phase_linearity_and_leading_network_axes():
    duration = 30 * DAY
    grid = SSBGrid(0., duration, .015, 20, 1 / duration, periods=2, period=duration)
    projector = DetectorBandProjector(grid, _StillOrbit(), *SKY)
    rng = numpy.random.default_rng(20267236)
    x = rng.normal(size=(2, 2, grid.size)) + 1j * rng.normal(size=(2, 2, grid.size))
    result = projector.project(x)
    numpy.testing.assert_allclose(projector.project(2j * x), 2j * result, rtol=1e-10, atol=1e-10)
    for a in range(2):
        for b in range(2):
            numpy.testing.assert_allclose(projector.project(x[a, b]), result[a, b], rtol=1e-10, atol=1e-10)


def test_invalid_precision_sampling_and_spectrum_are_rejected():
    duration = 30 * DAY
    grid = SSBGrid(0., duration, .015, 20, 1 / duration, periods=2, period=duration)
    for options in (dict(oversampling=1), dict(oversampling=2.5), dict(epsilon=0.)):
        with pytest.raises(ValueError):
            DetectorBandProjector(grid, _StillOrbit(), *SKY, **options)
    projector = DetectorBandProjector(grid, _StillOrbit(), *SKY)
    for x in (numpy.zeros(grid.size + 1), numpy.full(grid.size, numpy.nan)):
        with pytest.raises(ValueError):
            projector.project(x)


def test_finite_source_support_is_integrated_on_its_actual_endpoints():
    duration, m = 30 * DAY, 20
    grid = SSBGrid(0., duration, .015, m, 1 / duration, periods=4,
                   period=duration, guard=3)
    spectrum = numpy.zeros(grid.size, complex)
    spectrum[grid.guard] = 4.
    lo, hi = .17 * duration, .74 * duration
    projector = DetectorBandProjector(grid, _StillOrbit(), *SKY,
                                      source_support=(lo, hi), oversampling=8)
    projected = projector.project(spectrum)
    j = numpy.arange(m)
    fraction = (hi - lo) / duration
    expected = fraction * numpy.sinc(j * fraction) * numpy.exp(
        -2j * numpy.pi * j * (lo + hi) / (2 * duration))
    numpy.testing.assert_allclose(projected, expected, rtol=2e-2, atol=2e-4)
    assert abs(projected[0] - fraction) < 1e-10
    finer = DetectorBandProjector(grid, _StillOrbit(), *SKY,
                                  source_support=(lo, hi), oversampling=16)
    fine_error = numpy.max(numpy.abs(finer.project(spectrum) - expected))
    assert fine_error < .3 * numpy.max(numpy.abs(projected - expected))
    empty = DetectorBandProjector(grid, _StillOrbit(), *SKY,
                                   source_support=(2 * duration, 3 * duration))
    assert numpy.count_nonzero(empty.project(spectrum)) == 0
