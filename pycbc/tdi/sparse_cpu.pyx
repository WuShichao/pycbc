# Copyright (C) 2026  Shichao Wu, Alex Nitz
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
"""One fused pass over the TDI terms of a sparse response.

`pycbc.tdi.onthefly` builds the same quantity in numpy, and at 8,000 grid
points that costs 77 ms: a quarter in the sky projection, a third in the delay
expansion, a third in the complex exponential, across twenty temporary arrays
of shape (terms, grid). None of it is reused.

Here the projection, the expansion, the exponential and the channel sum happen
in one loop, with nothing of that shape ever written. The grid is walked in
blocks with the term index outside and the grid index inside, because the
geometry is stored as (term, grid, component): the other order strides
1.4 MB per step at sixty thousand grid points and misses cache on every read.
Blocks are independent and each holds its own accumulator, so the thread count
does not change the answer.
"""

cimport cython
from cython.parallel cimport parallel, prange, threadid
from libc.stdlib cimport free, malloc

from libc.math cimport cos, sin

ctypedef double complex complex128

DEF MAX_CHANNELS = 16
DEF BLOCK = 256


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef int sparse_brackets(
        double[:, :, ::1] n_hat,
        double[:, :, ::1] r_emit,
        double[:, :, ::1] r_recv,
        double[:, :, ::1] v_emit,
        double[:, :, ::1] v_recv,
        double[:, ::1] n_dot_v_recv,
        double[:, ::1] n_dot_v_mix,
        double[:, ::1] ltt,
        double[:, ::1] shifted,
        double[::1] coefficient,
        long[::1] channel,
        double[::1] anchor,
        double[::1] u_hat,
        double[::1] v_hat,
        double[::1] k_hat,
        complex128[::1] amp_p,
        complex128[::1] slope_p,
        complex128[::1] bend_p,
        complex128[::1] amp_c,
        complex128[::1] slope_c,
        complex128[::1] bend_c,
        double[::1] omega,
        double[::1] rate,
        double[::1] curve,
        int order,
        int velocity_order,
        double light_speed,
        complex128[:, ::1] out,
        int num_threads) except -1:
    """Accumulate every term of every channel into ``out``."""
    cdef Py_ssize_t n_term = n_hat.shape[0]
    cdef Py_ssize_t n_grid = n_hat.shape[1]
    cdef Py_ssize_t n_channel = out.shape[0]
    if n_channel > MAX_CHANNELS:
        raise ValueError("at most 16 output channels")
    if out.shape[1] != n_grid:
        raise ValueError("out must be (channels, grid points)")

    cdef double u0 = u_hat[0], u1 = u_hat[1], u2 = u_hat[2]
    cdef double v0 = v_hat[0], v1 = v_hat[1], v2 = v_hat[2]
    cdef double k0 = k_hat[0], k1 = k_hat[1], k2 = k_hat[2]
    cdef bint cubic = order >= 3
    cdef bint boosted = velocity_order > 0
    cdef Py_ssize_t n_block = (n_grid + BLOCK - 1) // BLOCK
    cdef Py_ssize_t width = 2 * n_channel

    cdef Py_ssize_t b, g, g0, g1, t, side, c, slot, base
    cdef double nu, nv, nk, prefactor_plus, prefactor_cross, inverse
    cdef double gap, transverse
    cdef double tau_emit, tau_recv, weight_emit, weight_recv, coef
    cdef double delay, phase, weight, angle_cos, angle_sin, squared
    cdef double real, imag, cross_real, cross_imag
    # One accumulator per thread, indexed explicitly. Leaving Cython to
    # privatise a pointer assigned inside `parallel` did not hold here: eight
    # threads disagreed with one by 6e+01, which is a shared buffer.
    cdef int n_thread = num_threads if num_threads > 0 else 1
    cdef double *accumulate = <double *> malloc(
        n_thread * BLOCK * width * sizeof(double))
    if accumulate == NULL:
        raise MemoryError("could not allocate the term accumulator")

    with nogil, parallel(num_threads=n_thread):
        for b in prange(n_block, schedule='static'):
            base = threadid() * BLOCK * width
            g0 = b * BLOCK
            g1 = g0 + BLOCK
            if g1 > n_grid:
                g1 = n_grid
            for slot in range((g1 - g0) * width):
                accumulate[base + slot] = 0.0

            for t in range(n_term):
                c = channel[t]
                coef = coefficient[t]
                for g in range(g0, g1):
                    nu = (n_hat[t, g, 0] * u0 + n_hat[t, g, 1] * u1
                          + n_hat[t, g, 2] * u2)
                    nv = (n_hat[t, g, 0] * v0 + n_hat[t, g, 1] * v1
                          + n_hat[t, g, 2] * v2)
                    nk = (n_hat[t, g, 0] * k0 + n_hat[t, g, 1] * k1
                          + n_hat[t, g, 2] * k2)
                    gap = 1.0 - nk
                    if gap < 1e-4:
                        transverse = nu * nu + nv * nv
                        if transverse > 2.0e-28:
                            inverse = 0.5 * (1.0 + nk) / transverse
                            prefactor_plus = (nu * nu - nv * nv) * inverse
                            prefactor_cross = 2.0 * nu * nv * inverse
                        else:
                            prefactor_plus = 0.0
                            prefactor_cross = 0.0
                    else:
                        inverse = 1.0 / (2.0 * gap)
                        prefactor_plus = (nu * nu - nv * nv) * inverse
                        prefactor_cross = 2.0 * nu * nv * inverse
                    tau_emit = ltt[t, g] + (
                        r_emit[t, g, 0] * k0 + r_emit[t, g, 1] * k1
                        + r_emit[t, g, 2] * k2) / light_speed
                    tau_recv = (r_recv[t, g, 0] * k0 + r_recv[t, g, 1] * k1
                                + r_recv[t, g, 2] * k2) / light_speed
                    if boosted:
                        weight_emit = 1.0 + (
                            -(v_emit[t, g, 0] * k0 + v_emit[t, g, 1] * k1
                              + v_emit[t, g, 2] * k2)
                            + n_dot_v_recv[t, g]) / light_speed
                        weight_recv = 1.0 + (
                            -(v_recv[t, g, 0] * k0 + v_recv[t, g, 1] * k1
                              + v_recv[t, g, 2] * k2)
                            + n_dot_v_mix[t, g]) / light_speed
                    else:
                        weight_emit = 1.0
                        weight_recv = 1.0

                    for side in range(2):
                        if side == 0:
                            delay = anchor[g] - (shifted[t, g] - tau_emit)
                            weight = coef * weight_emit
                        else:
                            delay = anchor[g] - (shifted[t, g] - tau_recv)
                            weight = -coef * weight_recv
                        phase = delay * (0.5 * rate[g] * delay - omega[g])
                        real = amp_p[g].real - delay * slope_p[g].real
                        imag = amp_p[g].imag - delay * slope_p[g].imag
                        cross_real = amp_c[g].real - delay * slope_c[g].real
                        cross_imag = amp_c[g].imag - delay * slope_c[g].imag
                        if cubic:
                            squared = delay * delay
                            phase = phase - curve[g] * squared * delay / 6.0
                            real = real + 0.5 * squared * bend_p[g].real
                            imag = imag + 0.5 * squared * bend_p[g].imag
                            cross_real = (cross_real
                                          + 0.5 * squared * bend_c[g].real)
                            cross_imag = (cross_imag
                                          + 0.5 * squared * bend_c[g].imag)
                        real = weight * (prefactor_plus * real
                                         + prefactor_cross * cross_real)
                        imag = weight * (prefactor_plus * imag
                                         + prefactor_cross * cross_imag)
                        # cos and sin, not sincos: taking the address of
                        # an output makes the variable un-privatisable and
                        # OpenMP then shares it between threads. That cost an
                        # afternoon and showed up as eight threads disagreeing
                        # with one by 5e+01. gcc fuses the pair anyway.
                        angle_cos = cos(phase)
                        angle_sin = sin(phase)
                        slot = base + (g - g0) * width + 2 * c
                        accumulate[slot] += real * angle_cos - imag * angle_sin
                        accumulate[slot + 1] += (real * angle_sin
                                                 + imag * angle_cos)

            for g in range(g0, g1):
                for c in range(n_channel):
                    slot = base + (g - g0) * width + 2 * c
                    out[c, g] = accumulate[slot] + 1j * accumulate[slot + 1]
    free(accumulate)
    return 0
