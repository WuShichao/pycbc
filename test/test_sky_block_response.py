"""The response harmonics over a sky block, against the exact response."""
import numpy
import pytest

from pycbc.coordinates.space_orbit import LisaEqualArmOrbit
from pycbc.tdi.inference import DEFAULT_ARM, channel_terms
from pycbc.tdi.sky_block_response import SkyBlockResponse
from pycbc.tdi.ssb_yearly_harmonics import yearly_response_samples

ORBIT = LisaEqualArmOrbit(armlength=DEFAULT_ARM, t0=0.0)
TERMS = channel_terms(('A', 'E'), generation=2, delta_t=5.0)
F = numpy.linspace(0.0136, 0.0157, 12)
CENTRE = (4.0, -0.7)
SCALES = (0.02, 0.01)


@pytest.fixture(scope='module')
def block():
    return SkyBlockResponse(ORBIT, TERMS, F, CENTRE, SCALES, radius=0.03,
                            kmax=8, samples=32, points=300, test_points=60,
                            tolerance=1e-3)


def test_the_exact_harmonics_are_the_familys_at_the_centre(block):
    samples = yearly_response_samples(ORBIT, TERMS, F, *CENTRE, *CENTRE,
                                      samples=32)
    rho = numpy.fft.fft(samples, axis=0) / 32
    expected = rho[block.orders % 32]
    numpy.testing.assert_allclose(block.exact([[0.0, 0.0]])[0], expected,
                                  rtol=1e-10, atol=1e-12)


def test_the_block_meets_its_tolerance_at_fresh_points(block):
    assert block.error <= block.tolerance
    rng = numpy.random.default_rng(5)
    unit = rng.uniform(-1, 1, (20, 2))
    off = block._offsets(unit)
    assert block.inside(off).all()
    exact, approx = block.exact(off), block.responses(off)
    err = (numpy.linalg.norm((approx - exact).reshape(20, -1), axis=1)
           / numpy.linalg.norm(exact.reshape(20, -1), axis=1))
    assert err.max() < 2 * block.tolerance
    # the first-order sky model is far worse over the same block
    assert block.rank > 1


def test_a_family_takes_its_harmonics_from_the_block(block):
    from pycbc.tdi.linear_exponent_surrogate import LinearExponentSurrogate
    fam = LinearExponentSurrogate.__new__(LinearExponentSurrogate)
    fam.xp, fam.dtype, fam.orders = numpy, numpy.complex128, block.orders
    fam.sky_block = block
    off = numpy.array([[0.5, -1.0], [-1.2, 0.3]])
    numpy.testing.assert_allclose(fam.responses(off), block.responses(off))
    numpy.testing.assert_allclose(fam._response(off[0]),
                                  block.responses(off[:1])[0])
    fam.orders = numpy.arange(-4, 5)
    with pytest.raises(ValueError):
        fam.responses(off)


def test_the_basis_interpolated_in_frequency_keeps_its_error_honest():
    fine = numpy.linspace(0.0136, 0.0157, 40)
    blk = SkyBlockResponse(ORBIT, TERMS, fine, CENTRE, SCALES, radius=0.02,
                           kmax=8, samples=32, points=200, test_points=30,
                           tolerance=1e-3, exact_frequencies=10)
    assert len(blk.frequencies) == 10 and blk.interpolation_error > 0
    full = SkyBlockResponse(ORBIT, TERMS, fine, CENTRE, SCALES, radius=0.02,
                            kmax=8, samples=32, points=40, test_points=4,
                            tolerance=1.0)
    off = blk._offsets(numpy.array([[0.3, -0.6], [-0.8, 0.9]]))
    exact = full.exact(off)
    err = (numpy.linalg.norm((blk.responses(off) - exact).reshape(2, -1),
                             axis=1)
           / numpy.linalg.norm(exact.reshape(2, -1), axis=1))
    assert err.max() < 3 * max(blk.error, 1e-6)


def test_responses_in_single_precision():
    blk = SkyBlockResponse(ORBIT, TERMS, F, CENTRE, SCALES, radius=0.03,
                           kmax=4, points=200, test_points=40)
    off = numpy.array([[0.3, -0.2], [-1.0, 0.5]]) * 0.03 / numpy.array(SCALES)
    double = blk.responses(off)
    single = blk.responses(off, dtype=numpy.complex64)
    assert single.dtype == numpy.complex64
    assert numpy.abs(single - double).max() < 1e-6 * numpy.abs(double).max()


def test_a_cached_block_is_the_block_it_stored(tmp_path):
    """sky_block fits a block once into its cache directory and loads it
    again: the same rank, coordinates and responses, without refitting."""
    from pycbc.tdi.sky_block_response import sky_block
    args = (ORBIT, TERMS, F, CENTRE, SCALES, 0.03)
    kw = dict(kmax=4, points=200, test_points=40)
    first = sky_block(*args, cache=str(tmp_path), **kw)
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    again = sky_block(*args, cache=str(tmp_path), **kw)
    assert again._geometry is None and again.rank == first.rank
    offsets = numpy.array([[0.3, -0.2], [-1.0, 0.5]]) * 0.03 / numpy.array(SCALES)
    numpy.testing.assert_allclose(again.responses(offsets),
                                  first.responses(offsets))
    # another block, another file
    sky_block(ORBIT, TERMS, F, CENTRE, SCALES, 0.02, cache=str(tmp_path), **kw)
    assert len(list(tmp_path.iterdir())) == 2


def _exact_one_point_at_a_time(blk, offsets):
    """The exact harmonics as frequency_response_factors gives them, one
    sky point per call."""
    from pycbc.tdi.onthefly import frequency_response_factors
    nf, ns = len(blk.frequencies), blk.samples
    f = numpy.tile(blk.frequencies, ns)
    out = []
    for d in numpy.atleast_2d(offsets):
        lon = blk.centre[0] + d[0] * blk.scales[0]
        lat = blk.centre[1] + d[1] * blk.scales[1]
        factors = frequency_response_factors(blk._geometry, f, lon, lat)
        r = numpy.array([[factors[c][a] for c in blk._geometry.channel_names]
                         for a in range(2)]) * blk._shift
        r = r.reshape(2, -1, ns, nf).transpose(2, 0, 1, 3)
        rho = numpy.fft.fft(r, axis=0) / ns
        out.append(rho[blk.orders % ns])
    return numpy.array(out)


def test_the_batched_exact_harmonics_are_the_pointwise_ones(block):
    """exact() works on several sky points at once; chunked or not, it is
    the response computed one point at a time."""
    rng = numpy.random.default_rng(3)
    off = block._offsets(rng.uniform(-1, 1, (7, 2)))
    expected = _exact_one_point_at_a_time(block, off)
    for chunk in (None, 1, 3):
        numpy.testing.assert_allclose(block.exact(off, chunk=chunk), expected,
                                      rtol=1e-12, atol=1e-14)


def test_a_coarsened_block_is_the_block_fitted_at_its_tolerance():
    """One fit gives the fine and the coarse block: the coarse one is the
    block fitted afresh at the looser tolerance."""
    args = (ORBIT, TERMS, F, CENTRE, SCALES, 0.03)
    kw = dict(kmax=4, points=200, test_points=40)
    fine = SkyBlockResponse(*args, tolerance=1e-4, **kw)
    coarse = fine.coarsened(1e-2)
    fresh = SkyBlockResponse(*args, tolerance=1e-2, **kw)
    assert coarse.rank == fresh.rank < fine.rank
    assert coarse.degree == fresh.degree
    assert coarse.error == pytest.approx(fresh.error, rel=1e-8)
    off = numpy.array([[0.3, -0.2], [-1.0, 0.5]]) * 0.03 / numpy.array(SCALES)
    numpy.testing.assert_allclose(coarse.responses(off), fresh.responses(off),
                                  rtol=1e-10, atol=1e-13)
    # the fine block is untouched by it
    assert fine.rank > coarse.rank and fine.tolerance == 1e-4
    with pytest.raises(ValueError):
        fine.coarsened(1e-5)
    fine.release()
    with pytest.raises(ValueError):
        fine.coarsened(1e-2)


def test_sky_blocks_fit_once_and_share_the_cache_of_sky_block(tmp_path):
    """sky_blocks stores each tolerance's block under the file sky_block
    would use, so a cache filled either way serves the other."""
    from pycbc.tdi.sky_block_response import sky_block, sky_blocks
    args = (ORBIT, TERMS, F, CENTRE, SCALES, 0.03)
    kw = dict(kmax=4, points=200, test_points=40)
    fine, coarse = sky_blocks(*args, tolerances=(1e-4, 1e-2),
                              cache=str(tmp_path), **kw)
    assert len(list(tmp_path.iterdir())) == 2
    assert not hasattr(fine, '_fit_state')
    alone = sky_block(*args, cache=str(tmp_path), tolerance=1e-2, **kw)
    assert alone._geometry is None and alone.rank == coarse.rank
    assert len(list(tmp_path.iterdir())) == 2
    again = sky_blocks(*args, tolerances=(1e-4, 1e-2), cache=str(tmp_path),
                       **kw)
    assert [b._geometry for b in again] == [None, None]
    off = numpy.array([[0.3, -0.2], [-1.0, 0.5]]) * 0.03 / numpy.array(SCALES)
    for a, b in zip(again, (fine, coarse)):
        numpy.testing.assert_allclose(a.responses(off), b.responses(off))


def test_the_exact_harmonics_on_the_gpu_are_the_cpus(block):
    cupy = pytest.importorskip('cupy')
    try:
        cupy.zeros(1)
    except Exception:  # noqa: BLE001 -- no device
        pytest.skip('no GPU')
    gpu = SkyBlockResponse.__new__(SkyBlockResponse)
    gpu.__dict__.update(block.__dict__)
    gpu._xp = cupy
    off = block._offsets(numpy.random.default_rng(4).uniform(-1, 1, (5, 2)))
    numpy.testing.assert_allclose(gpu.exact(off), block.exact(off),
                                  rtol=1e-10, atol=1e-13)


def test_exact_frequency_indices_and_the_all_frequency_check():
    from pycbc.tdi.sky_block_response import exact_frequency_count
    kw = dict(kmax=8, samples=32, points=300, test_points=60, tolerance=1e-3)
    by_count = SkyBlockResponse(ORBIT, TERMS, F, CENTRE, SCALES, radius=0.03,
                                exact_frequencies=4, **kw)
    by_index = SkyBlockResponse(ORBIT, TERMS, F, CENTRE, SCALES, radius=0.03,
                                exact_frequencies=[4, 7], **kw)
    numpy.testing.assert_array_equal(by_index.frequencies, F[[0, 4, 7, 11]])
    numpy.testing.assert_array_equal(by_count.frequencies, F[[0, 4, 7, 11]])
    everywhere = SkyBlockResponse(ORBIT, TERMS, F, CENTRE, SCALES, radius=0.03,
                                  exact_frequencies=4,
                                  interpolation_check='all', **kw)
    assert len(everywhere._fit_state['mid']) == len(F) - 4
    assert numpy.isfinite(everywhere.interpolation_error)
    # the same fit, only the check differs
    numpy.testing.assert_allclose(everywhere._basis, by_count._basis)
    with pytest.raises(ValueError):
        SkyBlockResponse(ORBIT, TERMS, F, CENTRE, SCALES, radius=0.03,
                         exact_frequencies=[20], **kw)
    assert exact_frequency_count(F, 4e-5) == len(F)
    assert exact_frequency_count(numpy.linspace(0.0136, 0.01361, 100),
                                 4e-5) == 5
    assert exact_frequency_count(numpy.linspace(0.02, 0.03, 1500),
                                 4e-5) == 251


def test_exact_frequency_indices_add_the_dense_bands():
    from pycbc.tdi.sky_block_response import (exact_frequency_count,
                                              exact_frequency_indices)
    f = numpy.linspace(0.02, 0.03, 1501)
    plain = exact_frequency_indices(f, 4e-5)
    assert len(plain) == exact_frequency_count(f, 4e-5)
    dense = exact_frequency_indices(f, 4e-5, dense=[(0.025, 0.0251)])
    inside = numpy.nonzero((f >= 0.025) & (f <= 0.0251))[0]
    assert set(inside) <= set(dense)
    assert set(plain) <= set(dense)
    assert dense[0] == 0 and dense[-1] == len(f) - 1
