"""The CUDA kernels of the FFT block search against numpy; skipped without
cupy and a device."""
import numpy
import pytest

cupy = pytest.importorskip('cupy')
try:
    cupy.cuda.runtime.getDeviceCount()
except Exception:      # pragma: no cover
    pytest.skip('no CUDA device', allow_module_level=True)


def complex_normal(rng, *shape):
    return (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(
        numpy.complex64)


def test_twof_max_is_the_sky_combination_maximum():
    from pycbc.filter.sky_block_kernels import twof_max
    rng = numpy.random.default_rng(0)
    R, P, L, D = 9, 21, 700, 3
    Z, C = complex_normal(rng, R, 2, L), complex_normal(rng, P, R)
    g00 = rng.uniform(1, 2, (P, D)).astype(numpy.float32)
    g11 = rng.uniform(1, 2, (P, D)).astype(numpy.float32)
    g01 = (0.3 * complex_normal(rng, P, D)).astype(numpy.complex64)
    det = (g00 * g11 - numpy.abs(g01) ** 2).astype(numpy.float32)
    day = numpy.minimum(numpy.arange(L) // 250, D - 1).astype(numpy.int32)
    z = numpy.einsum('pr,rcl->pcl', C.astype(complex), Z.astype(complex)) / 2
    g = [a[:, day] for a in (g00, g11, g01, det)]
    want = (g[1] * numpy.abs(z[:, 0]) ** 2 + g[0] * numpy.abs(z[:, 1]) ** 2
            - 2 * numpy.real(numpy.conj(z[:, 0]) * g[2] * z[:, 1])) / g[3]
    best, arg = twof_max(*(cupy.asarray(a) for a in (Z, C, g00, g11, g01,
                                                       det, day)), 0.5)
    best, arg = best.get(), arg.get()
    for t in range(best.shape[1]):
        tile = want[:, t * 128:(t + 1) * 128]
        numpy.testing.assert_allclose(best[:, t], tile.max(axis=1), rtol=1e-4)
        numpy.testing.assert_array_equal(arg[:, t],
                                         t * 128 + tile.argmax(axis=1))


def test_multiply_fold_sums_the_folds_of_the_product():
    from pycbc.filter.sky_block_kernels import multiply_fold
    rng = numpy.random.default_rng(1)
    shared, factor = complex_normal(rng, 3, 2, 1200), complex_normal(rng, 1200)
    got = multiply_fold(cupy.asarray(shared), cupy.asarray(factor), 3).get()
    want = (shared * factor).reshape(3, 2, 3, 400).sum(axis=2)
    numpy.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
