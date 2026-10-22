"""Shared sky correlations against full, independently assembled templates."""

import numpy
import pytest

from pycbc.filter.sky_block_fft import SkyBlockFFT, network_twof
from test_linear_exponent_surrogate import build


class ComplexBlock(object):
    def __init__(self, family):
        rng = numpy.random.default_rng(20267221)
        shape = (3, len(family.orders), 2, 2, len(family.coarse_frequencies))
        self._basis = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        self.rank, self.orders = 3, family.orders

    def inside(self, offsets):
        return (numpy.abs(offsets) <= 1).all(axis=1)

    def _unit(self, offsets):
        return offsets

    def _coords(self, offsets):
        x, y = offsets.T
        return numpy.column_stack([1 + .2j * x, x + 1j * y, .4j + x * y])

    def responses(self, offsets, xp=numpy):
        return xp.asarray(numpy.einsum('pr,r...->p...', self._coords(offsets), self._basis))


def setup():
    fam, data, _ = build()
    fam.sky_block = ComplexBlock(fam)
    offsets = numpy.array([[0., 0.], [.2, -.3], [1., 1.], [-1., -1.]])
    return fam, data, offsets


def test_shared_correlations_equal_full_responses_at_all_requested_lags():
    fam, data, offsets = setup()
    corr = SkyBlockFFT(fam, data, offsets)
    xi = numpy.array([.2, -.1, .4, -.3])
    fields, b, t = corr.basis_correlations(xi)
    lags = numpy.array([0, 1, 17, fam.size // 2, fam.size - 3])
    result = numpy.empty((len(offsets), 2, len(lags)), complex)
    for sky, selected, value in corr.chunks(fields, lags, sky_chunk=3, lag_chunk=2):
        result[sky, :, numpy.searchsorted(lags, selected)] = value
    for j, rho in enumerate(fam.responses(offsets)):
        full = fam.correlate(data, b, t, rho)
        numpy.testing.assert_allclose(result[j], full[:, lags], rtol=1e-12, atol=1e-10)
        direct_rows = []
        for k, lag in enumerate(lags):
            signed = lag if lag < fam.size // 2 else lag - fam.size
            template = fam.template(b, t, rho, signed * fam.grid.delta_t)
            direct = numpy.einsum('acf,cf->a', template.conj(), data)
            direct_rows.append(direct)
        # Normalize the whole checked network, including weak correlation lags.
        direct_rows = numpy.stack(direct_rows, axis=-1)
        relative = numpy.linalg.norm(result[j] - direct_rows) / numpy.linalg.norm(direct_rows)
        assert relative <= 1e-10


def test_missing_conjugation_and_fft_normalization_are_detectable():
    fam, data, offsets = setup()
    corr = SkyBlockFFT(fam, data, offsets)
    fields, b, t = corr.basis_correlations(numpy.zeros(4))
    direct = fam.correlate(data, b, t, fam.responses(offsets)[1])
    good = numpy.einsum('r,ran->an', corr.weights[1], fields)
    wrong = numpy.einsum('r,ran->an', corr.weights[1].conj(), fields)
    assert numpy.linalg.norm(wrong - direct) > .1 * numpy.linalg.norm(direct)
    numpy.testing.assert_allclose(good, direct, rtol=1e-12, atol=1e-10)
    numpy.testing.assert_allclose(good / fam.size * fam.size, direct, rtol=1e-12, atol=1e-10)
    assert numpy.linalg.norm(good / fam.size - direct) > .9 * numpy.linalg.norm(direct)


def test_source_pieces_follow_each_template_not_the_reference():
    fam, data, offsets = setup()
    corr = SkyBlockFFT(fam, data, offsets)
    _, b0, t0 = corr.basis_correlations(numpy.zeros(4))
    _, b1, t1 = corr.basis_correlations(numpy.ones(4))
    assert not numpy.array_equal(b0, b1)
    assert not numpy.array_equal(t0, t1)
    expected_b, expected_t = fam._source(numpy.ones(4))
    numpy.testing.assert_array_equal(b1, expected_b)
    numpy.testing.assert_array_equal(t1, expected_t)


@pytest.mark.parametrize('offsets', [[[1.01, 0]], [[numpy.nan, 0]], [[0]], []])
def test_outside_or_invalid_sky_is_rejected(offsets):
    fam, data, _ = setup()
    with pytest.raises(ValueError):
        SkyBlockFFT(fam, data, offsets)


def test_no_block_bad_network_and_memory_limits_are_explicit():
    fam, data, offsets = setup()
    with pytest.raises(MemoryError):
        SkyBlockFFT(fam, data, offsets, field_memory_limit=1)
    with pytest.raises(ValueError):
        SkyBlockFFT(fam, data[:1], offsets)
    fam.sky_block = None
    with pytest.raises(ValueError):
        SkyBlockFFT(fam, data, offsets)


def test_twof_equals_the_independent_complex_solve_and_gaussian_maximum():
    rng = numpy.random.default_rng(20267222)
    h = rng.normal(size=(2, 2, 15)) + 1j * rng.normal(size=(2, 2, 15))
    d = rng.normal(size=(2, 15)) + 1j * rng.normal(size=(2, 15))
    z = numpy.einsum('acf,cf->a', h.conj(), d)
    gram = numpy.einsum('acf,bcf->ab', h.conj(), h)
    fitted = numpy.linalg.solve(gram, z)
    signal = numpy.einsum('a,acf->cf', fitted, h)
    variance = 2.5
    direct_loglr = (numpy.vdot(signal, d).real - numpy.vdot(signal, signal).real / 2) / variance
    value, status = network_twof(z, gram, variance)
    assert status == 0
    assert value == pytest.approx(2 * direct_loglr, abs=1e-10)
    rotation = numpy.array([[1, 1j], [1j, 1]]) / numpy.sqrt(2)
    h2, d2 = numpy.einsum('ij,ajf->aif', rotation, h), rotation @ d
    z2 = numpy.einsum('acf,cf->a', h2.conj(), d2)
    g2 = numpy.einsum('acf,bcf->ab', h2.conj(), h2)
    other, _ = network_twof(z2, g2, variance)
    assert other == pytest.approx(value, abs=1e-10)


def test_rank_labels_preserve_small_positive_directions():
    z = numpy.array([[1, 1e-8], [1, 0], [0, 0], [1, 1e-20], [1, 0]], complex)
    grams = numpy.array([numpy.diag([1, 1e-16]), numpy.diag([1, 0]),
                         numpy.zeros((2, 2)), numpy.diag([1, 0]), numpy.diag([1, -1e-16])])
    value, status = network_twof(z, grams)
    numpy.testing.assert_array_equal(status, [0, 1, 2, 3, 3])
    numpy.testing.assert_allclose(value[:3], [2, 1, 0], atol=1e-10)
    assert numpy.isnan(value[3:]).all()


def test_unresolved_nonfinite_and_nonhermitian_statistics_not_made_negative():
    for gram in (numpy.array([[1, 1j], [1j, 1]]), numpy.full((2, 2), numpy.nan)):
        value, status = network_twof(numpy.ones(2), gram)
        assert status == 3 and numpy.isnan(value)


def test_network_shape_errors_are_rejected():
    with pytest.raises(ValueError):
        network_twof(numpy.ones(2), numpy.eye(3))
    with pytest.raises(ValueError):
        network_twof(numpy.ones(2), numpy.eye(2), noise_variance=0)
