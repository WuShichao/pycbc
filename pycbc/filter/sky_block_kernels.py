# Copyright (C) 2026 Shichao Wu
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
"""CUDA kernels of the FFT block search: the correlation spectra of a
template with every sky-basis vector of a block from one evaluation of the
template's harmonic phases, and the sky combination with 2F and its maximum
in one pass, so the (sky point, lag) array is never stored."""

import numpy

from pycbc.tdi.linear_exponent_surrogate import ORBITAL_ANGULAR_FREQUENCY

_SPECTRA = r"""
#include <cupy/complex.cuh>
extern "C" __global__ void spectra(
    const complex<float>* data, const complex<float>* b, const double* t,
    const int* index, const float* weight, const complex<float>* basis,
    complex<float>* out, const int n, const int ncoarse, const int kmin,
    const int shift, const double turns_per_second, const int r0,
    const int nr) {
  // out[r - r0, a, f] = sum_k sum_c conj(u_rkac(f + k shift) y_k(f + k shift))
  //                      X_c(f + k shift), y_k = b exp(i k Omega t)
  int f = blockDim.x * blockIdx.x + threadIdx.x;
  if (f >= n) return;
  complex<float> e[NK * NCH];
  int ii[NK];
  float ww[NK];
#pragma unroll
  for (int q = 0; q < NK; q++) {
    int k = kmin + q;
    int j = f + k * shift;
    j = j < 0 ? j + n : (j >= n ? j - n : j);
    double turns = t[j] * turns_per_second;
    float c, s;
    sincospif(2.0f * k * (float)(turns - floor(turns)), &s, &c);
    complex<float> ph = conj(b[j] * complex<float>(c, s));
    ii[q] = index[j];
    ww[q] = weight[j];
#pragma unroll
    for (int ch = 0; ch < NCH; ch++) e[q * NCH + ch] = ph * data[ch * n + j];
  }
  for (int r = r0; r < r0 + nr; r++) {
    complex<float> acc0(0, 0), acc1(0, 0);
#pragma unroll
    for (int q = 0; q < NK; q++) {
      int i = ii[q];
      float w = ww[q];
#pragma unroll
      for (int ch = 0; ch < NCH; ch++) {
        const complex<float>* u0 =
            basis + ((((size_t)r * NK + q) * 2 + 0) * NCH + ch) * ncoarse;
        const complex<float>* u1 =
            basis + ((((size_t)r * NK + q) * 2 + 1) * NCH + ch) * ncoarse;
        complex<float> f0 = u0[i] * (1.0f - w) + u0[i + 1] * w;
        complex<float> f1 = u1[i] * (1.0f - w) + u1[i + 1] * w;
        acc0 += conj(f0) * e[q * NCH + ch];
        acc1 += conj(f1) * e[q * NCH + ch];
      }
    }
    out[((size_t)(r - r0) * 2 + 0) * n + f] = acc0;
    out[((size_t)(r - r0) * 2 + 1) * n + f] = acc1;
  }
}
"""

_TWOF = r"""
#include <cupy/complex.cuh>
extern "C" __global__ void twof_max(
    const complex<float>* Z, const complex<float>* C, const float* g00,
    const float* g11, const complex<float>* g01, const float* det,
    const int* lag_day, float* best, int* arg, const int R, const int P,
    const int L, const int D, const float inv_var) {
  // blockIdx.x: a tile of PT sky points; blockIdx.y: a tile of lags
  extern __shared__ complex<float> cs[];
  __shared__ float sv[PT][THREADS];
  __shared__ int si[PT][THREADS];
  int tid = threadIdx.x;
  int p0 = blockIdx.x * PT;
  int l = blockIdx.y * THREADS + tid;
  for (int i = tid; i < PT * R; i += THREADS) {
    int p = p0 + i / R;
    cs[i] = p < P ? C[(size_t)p * R + i % R] : complex<float>(0, 0);
  }
  __syncthreads();
  complex<float> a0[PT], a1[PT];
#pragma unroll
  for (int p = 0; p < PT; p++) { a0[p] = 0; a1[p] = 0; }
  if (l < L) {
    for (int r = 0; r < R; r++) {
      complex<float> z0 = Z[((size_t)r * 2) * L + l];
      complex<float> z1 = Z[((size_t)r * 2 + 1) * L + l];
#pragma unroll
      for (int p = 0; p < PT; p++) {
        complex<float> c = cs[p * R + r];
        a0[p] += c * z0;
        a1[p] += c * z1;
      }
    }
  }
  int d = l < L ? lag_day[l] : 0;
#pragma unroll
  for (int p = 0; p < PT; p++) {
    float v = -1e30f;
    if (l < L && p0 + p < P) {
      size_t g = (size_t)(p0 + p) * D + d;
      complex<float> z0 = a0[p] * inv_var, z1 = a1[p] * inv_var;
      v = (g11[g] * norm(z0) + g00[g] * norm(z1)
           - 2.0f * (conj(z0) * g01[g] * z1).real()) / det[g];
    }
    sv[p][tid] = v;
    si[p][tid] = l;
  }
  __syncthreads();
  for (int s = THREADS / 2; s > 0; s >>= 1) {
    if (tid < s) {
#pragma unroll
      for (int p = 0; p < PT; p++) {
        if (sv[p][tid + s] > sv[p][tid]) {
          sv[p][tid] = sv[p][tid + s];
          si[p][tid] = si[p][tid + s];
        }
      }
    }
    __syncthreads();
  }
  if (tid < PT && p0 + tid < P) {
    best[(size_t)(p0 + tid) * gridDim.y + blockIdx.y] = sv[tid][0];
    arg[(size_t)(p0 + tid) * gridDim.y + blockIdx.y] = si[tid][0];
  }
}
"""

_FOLD = r"""
#include <cupy/complex.cuh>
extern "C" __global__ void multiply_fold(
    const complex<float>* shared, const complex<float>* factor,
    complex<float>* out, const long long n, const long long size,
    const int q) {
  // out[row, m] = sum_j shared[row, m + j size] factor[m + j size]
  long long m = (long long)blockDim.x * blockIdx.x + threadIdx.x;
  long long row = blockIdx.y;
  if (m >= size) return;
  complex<float> acc(0, 0);
  for (int j = 0; j < q; j++) {
    long long i = m + j * size;
    acc += shared[row * n + i] * factor[i];
  }
  out[row * size + m] = acc;
}
"""

_CACHE = {}


def _kernel(source, name, **macros):
    import cupy
    key = (name,) + tuple(sorted(macros.items()))
    if key not in _CACHE:
        head = ''.join('#define {} {}\n'.format(k, v)
                       for k, v in macros.items())
        _CACHE[key] = cupy.RawKernel(head + source, name)
    return _CACHE[key]


def correlation_spectra(family, data, b, t, basis, r0, nr, out=None):
    """The correlation spectra of the template ``(b, t)`` with basis vectors
    ``r0`` to ``r0 + nr`` of ``basis`` (``(R, nk, 2, nch, ncoarse)``), shape
    ``(nr, 2, N)``: each is ``family.correlation_spectrum(data, b, t,
    basis[r])``, the template's harmonic phases evaluated once for all."""
    import cupy
    nk, nch, ncoarse = basis.shape[1], basis.shape[3], basis.shape[4]
    n = family.size
    if out is None:
        out = cupy.empty((nr, 2, n), dtype=cupy.complex64)
    if not hasattr(family, '_index32'):
        family._index32 = family._index.astype(cupy.int32)
        family._weight32 = family._weight.astype(cupy.float32)
    threads = 128
    _kernel(_SPECTRA, 'spectra', NK=nk, NCH=nch)(
        ((n + threads - 1) // threads,), (threads,), (
            cupy.ascontiguousarray(data, dtype=cupy.complex64),
            cupy.ascontiguousarray(b, dtype=cupy.complex64),
            cupy.ascontiguousarray(t, dtype=cupy.float64),
            family._index32, family._weight32,
            cupy.ascontiguousarray(basis, dtype=cupy.complex64), out,
            numpy.int32(n), numpy.int32(ncoarse),
            numpy.int32(family.orders[0]), numpy.int32(family.shift),
            numpy.float64(ORBITAL_ANGULAR_FREQUENCY / (2 * numpy.pi)),
            numpy.int32(r0), numpy.int32(nr)))
    return out


def twof_max(Z, C, g00, g11, g01, det, lag_day, inv_var, tile=16,
             threads=128):
    """The largest 2F of each sky point in each tile of ``threads`` lags and
    its lag: ``z = C Z`` (``C`` ``(P, R)``, ``Z`` ``(R, 2, L)``) times
    ``inv_var``, with the Gram ``g`` ``(P, D)`` of the day ``lag_day`` of
    each lag. Returns ``(P, ntiles)`` maxima and lag indices."""
    import cupy
    R, _, L = Z.shape
    P = C.shape[0]
    D = g00.shape[1]
    ntiles = (L + threads - 1) // threads
    best = cupy.empty((P, ntiles), dtype=cupy.float32)
    arg = cupy.empty((P, ntiles), dtype=cupy.int32)
    _kernel(_TWOF, 'twof_max', PT=tile, THREADS=threads)(
        ((P + tile - 1) // tile, ntiles), (threads,), (
            cupy.ascontiguousarray(Z, dtype=cupy.complex64),
            cupy.ascontiguousarray(C, dtype=cupy.complex64),
            cupy.ascontiguousarray(g00, dtype=cupy.float32),
            cupy.ascontiguousarray(g11, dtype=cupy.float32),
            cupy.ascontiguousarray(g01, dtype=cupy.complex64),
            cupy.ascontiguousarray(det, dtype=cupy.float32),
            cupy.ascontiguousarray(lag_day, dtype=cupy.int32),
            best, arg, numpy.int32(R), numpy.int32(P), numpy.int32(L),
            numpy.int32(D), numpy.float32(inv_var)),
        shared_mem=tile * R * 8)
    return best, arg


def multiply_fold(shared, factor, decimate=1, out=None):
    """``shared`` (rows..., N) times ``factor`` (N), its ``decimate`` folds
    summed: the spectra of a template from the reference's, ready for a
    transform ``decimate`` times shorter."""
    import cupy
    n = shared.shape[-1]
    size = n // decimate
    rows = int(numpy.prod(shared.shape[:-1]))
    if out is None:
        out = cupy.empty(shared.shape[:-1] + (size,), dtype=cupy.complex64)
    threads = 256
    _kernel(_FOLD, 'multiply_fold')(
        ((size + threads - 1) // threads, rows), (threads,), (
            shared, cupy.ascontiguousarray(factor, dtype=cupy.complex64),
            out, numpy.int64(n), numpy.int64(size), numpy.int32(decimate)))
    return out
