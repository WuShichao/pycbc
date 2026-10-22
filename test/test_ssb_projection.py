"""Dense and direct-frequency controls of the SSB map and its adjoint."""
import numpy
import pytest

pytest.importorskip('finufft')

from pycbc.tdi.ssb_projection import SSBDataMap, SSBProjectionError
from pycbc.tdi.ssb_yearly_harmonics import (SSBGrid, detector_times,
                                            resample_to_ssb)
from test_ssb_yearly_harmonics import ORBIT, SKY, DAY, _StillOrbit


def setup(guard=3, orbit=ORBIT):
    grid = SSBGrid(2e6, 2e6 + 30 * DAY, .0151, 35, 1 / (30 * DAY),
                   periods=1, guard=guard)
    return SSBDataMap(grid, orbit, *SKY), orbit


def dense(mapper, orbit):
    grid = mapper.grid
    ssb = grid.times[:mapper.n_live]
    times = detector_times(orbit, ssb, *SKY)
    frequencies = grid.data_first_frequency + numpy.arange(mapper.nbins) * grid.data_delta_f
    basis = numpy.exp(2j * numpy.pi * (times - grid.start)[:, None] * frequencies[None])
    basis *= numpy.exp(-2j * numpy.pi * grid.first_frequency * (ssb - grid.start))[:, None]
    padded = numpy.zeros((grid.size, mapper.nbins), complex)
    padded[:mapper.n_live] = basis
    return numpy.fft.fft(padded, axis=0) / mapper.n_live


@pytest.mark.parametrize('guard', [0, 3, 11])
def test_forward_matches_existing_resampler_and_dense_fourier_map(guard):
    mapper, orbit = setup(guard=guard)
    rng = numpy.random.default_rng(20267227)
    bins = rng.normal(size=(2, mapper.nbins)) + 1j * rng.normal(size=(2, mapper.nbins))
    expected = dense(mapper, orbit)
    actual = mapper.forward(bins)
    numpy.testing.assert_allclose(actual, bins @ expected.T, rtol=1e-10, atol=1e-9)
    direct = numpy.fft.fft(resample_to_ssb(mapper.grid, bins, orbit, *SKY), axis=-1) / mapper.n_live
    numpy.testing.assert_allclose(actual, direct, rtol=1e-9, atol=1e-8)


def test_adjoint_and_normal_equal_dense_conjugate_transpose():
    mapper, orbit = setup()
    matrix = dense(mapper, orbit)
    rng = numpy.random.default_rng(20267228)
    bins = rng.normal(size=mapper.nbins) + 1j * rng.normal(size=mapper.nbins)
    spectrum = rng.normal(size=mapper.size) + 1j * rng.normal(size=mapper.size)
    numpy.testing.assert_allclose(mapper.adjoint(spectrum), matrix.conj().T @ spectrum,
                                  rtol=1e-10, atol=1e-9)
    assert numpy.vdot(mapper.forward(bins), spectrum) == pytest.approx(
        numpy.vdot(bins, mapper.adjoint(spectrum)), rel=1e-10, abs=1e-9)
    numpy.testing.assert_allclose(mapper.normal(bins), matrix.conj().T @ matrix @ bins,
                                  rtol=1e-10, atol=1e-9)


def test_recovery_preserves_network_statistics_in_the_original_band():
    mapper, _ = setup()
    rng = numpy.random.default_rng(20267229)
    h = rng.normal(size=(2, 2, mapper.nbins)) + 1j * rng.normal(size=(2, 2, mapper.nbins))
    d = rng.normal(size=(2, mapper.nbins)) + 1j * rng.normal(size=(2, mapper.nbins))
    recovered, info = mapper.recover(mapper.forward(h))
    recovered_data, _ = mapper.recover(mapper.forward(d))
    numpy.testing.assert_allclose(recovered, h, rtol=1e-9, atol=1e-9)
    assert max(row['range_residual'] for row in info) < 1e-9
    z = numpy.einsum('acf,cf->a', h.conj(), d)
    g = numpy.einsum('acf,bcf->ab', h.conj(), h)
    zr = numpy.einsum('acf,cf->a', recovered.conj(), recovered_data)
    gr = numpy.einsum('acf,bcf->ab', recovered.conj(), recovered)
    numpy.testing.assert_allclose(zr, z, rtol=1e-9, atol=1e-9)
    numpy.testing.assert_allclose(gr, g, rtol=1e-9, atol=1e-9)
    assert numpy.vdot(zr, numpy.linalg.solve(gr, zr)).real == pytest.approx(
        numpy.vdot(z, numpy.linalg.solve(g, z)).real, rel=1e-9)


def test_range_failure_is_not_silently_renormalized_or_projected_away():
    mapper, _ = setup()
    rng = numpy.random.default_rng(20267230)
    invalid = rng.normal(size=mapper.size) + 1j * rng.normal(size=mapper.size)
    _, info = mapper.project(invalid)
    assert info[0]['range_residual'] > .1
    with pytest.raises(SSBProjectionError, match='outside') as error:
        mapper.recover(invalid)
    assert error.value.diagnostics[0]['range_residual'] > .1


def test_zero_signal_and_uniform_correlation_scale():
    mapper, _ = setup(orbit=_StillOrbit())
    zero, info = mapper.recover(numpy.zeros((2, mapper.size), complex))
    assert numpy.count_nonzero(zero) == 0
    assert all(row['range_residual'] == row['normal_residual'] == 0 for row in info)
    # The actual sidereal grid need not exactly divide the data span. This
    # check uses the normal map itself, not a claim of exact uniformity.
    matrix = mapper.normal(numpy.eye(mapper.nbins))
    assert numpy.linalg.eigvalsh(matrix).min() > 0


@pytest.mark.parametrize('value', [numpy.nan, 0., -1., 1.])
def test_invalid_nufft_tolerance_rejected(value):
    mapper, orbit = setup()
    with pytest.raises(ValueError):
        SSBDataMap(mapper.grid, orbit, *SKY, epsilon=value)


def test_invalid_input_and_unresolved_solve_are_explicit():
    mapper, _ = setup()
    for value in (numpy.zeros(mapper.nbins + 1), numpy.full(mapper.nbins, numpy.nan), []):
        with pytest.raises(ValueError):
            mapper.forward(value)
    with pytest.raises(ValueError):
        mapper.recover(numpy.zeros(mapper.size), rtol=0.)
    rng = numpy.random.default_rng(20267231)
    value = mapper.forward(rng.normal(size=mapper.nbins))
    with pytest.raises(SSBProjectionError, match='unresolved'):
        mapper.recover(value, rtol=1e-14, maxiter=1)
