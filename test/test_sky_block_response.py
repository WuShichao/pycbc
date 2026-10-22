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
