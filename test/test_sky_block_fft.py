"""Shared sky correlations against full, independently assembled templates."""

import numpy
import pytest

from pycbc.filter.sky_block_fft import (SkyBlockFFT, network_twof,
                                       search_sky_block_fft)
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


def test_roundtrip_corner_is_accepted_without_accepting_outside_points():
    fam, data, _ = setup()
    corner = numpy.array([[1 + 2 * numpy.finfo(float).eps, -1.]])
    SkyBlockFFT(fam, data, corner)
    with pytest.raises(ValueError):
        SkyBlockFFT(fam, data, [[1 + 1e-10, 0]])


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


def test_search_global_top_equals_unshared_exhaustive_grid():
    fam, data, offsets = setup()
    engine = SkyBlockFFT(fam, data, offsets)
    points = numpy.array([[0., 0., 0., 0.], [.1, -.2, .3, .4]])
    lags = numpy.array([fam.size - 1, 10, 1, 0, fam.size // 2])
    visited = []

    def matrix(index, selected):
        out = numpy.zeros((len(selected), 2, 2), complex)
        out[:, 0, 0], out[:, 1, 1] = 2 + index, 3 + index
        out[:, 0, 1], out[:, 1, 0] = .2j, -.2j
        return out

    def prepare(index, intrinsic, b, t):
        expected = fam._source(intrinsic)
        numpy.testing.assert_array_equal(b, expected[0])
        numpy.testing.assert_array_equal(t, expected[1])
        visited.append(index)

        def at(sky, selected):
            return numpy.broadcast_to(matrix(index, selected),
                                      (sky.stop - sky.start, len(selected), 2, 2))
        return at

    full = []
    for i, point in enumerate(points):
        b, t = fam._source(point)
        for p, rho in enumerate(fam.responses(offsets)):
            z = fam.correlate(data, b, t, rho)[:, lags].T
            fitted = numpy.linalg.solve(matrix(i, lags), z[..., None])[..., 0]
            values = numpy.einsum('la,la->l', z.conj(), fitted).real / 2.5
            for lag, value in zip(lags, values):
                full.append(dict(template=i, sky=p, lag=int(lag), twof=float(value)))
    full.sort(key=lambda c: (-c['twof'], c['template'], c['sky'], c['lag']))
    for sky_chunk, lag_chunk in ((1, 2), (3, 3), (4, 20)):
        result = search_sky_block_fft(engine, points, prepare, lags,
                                     noise_variance=2.5, top=7,
                                     sky_chunk=sky_chunk, lag_chunk=lag_chunk)
        assert result['complete']
        assert result['evaluated'] == 40
        assert result['rank_counts'] == {'0': 40, '1': 0, '2': 0, '3': 0}
        for actual, expected in zip(result['candidates'], full[:7]):
            assert [actual[k] for k in ('template', 'sky', 'lag')] == [
                expected[k] for k in ('template', 'sky', 'lag')]
            assert actual['twof'] == pytest.approx(expected['twof'], rel=1e-10)
    assert visited == [0, 1] * 3


def test_search_unresolved_entry_cannot_be_a_complete_negative_result():
    fam, data, offsets = setup()
    engine = SkyBlockFFT(fam, data, offsets)

    def prepare(index, intrinsic, b, t):
        def at(sky, selected):
            g = numpy.broadcast_to(numpy.eye(2),
                                   (sky.stop - sky.start, len(selected), 2, 2)).copy()
            if sky.start == 0 and selected[0] == 0:
                g[0, 0] = numpy.nan
            return g
        return at
    result = search_sky_block_fft(engine, [[0.] * 4], prepare, [0, 1, 5],
                                 sky_chunk=2, lag_chunk=2)
    assert not result['complete']
    assert result['rank_counts'] == {'0': 11, '1': 0, '2': 0, '3': 1}
    assert len(result['candidates']) == 11


@pytest.mark.parametrize('lags', [[], [0, 0], [0., 1.], [-1], [10**8]])
def test_search_rejects_invalid_time_domains(lags):
    fam, data, offsets = setup()
    with pytest.raises(ValueError):
        search_sky_block_fft(SkyBlockFFT(fam, data, offsets), [[0.] * 4],
                             None, lags)


def test_search_rejects_invalid_points_and_chunk_sizes():
    fam, data, offsets = setup()
    engine = SkyBlockFFT(fam, data, offsets)
    for points in ([], [0, 1], [[numpy.nan] * 4]):
        with pytest.raises(ValueError):
            search_sky_block_fft(engine, points, None, [0])
    for options in (dict(top=0), dict(sky_chunk=.5), dict(lag_chunk=0)):
        with pytest.raises(ValueError):
            search_sky_block_fft(engine, [[0.] * 4], None, [0], **options)
