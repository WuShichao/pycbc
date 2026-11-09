# Copyright (C) 2026 Shichao Wu
# This program is free software under the GNU General Public License,
# version 3 or (at your option) any later version.
"""Uniform sub-sample delays via an exact chirp convolution of the spectrum.

This evaluates the same finite Fourier polynomial as direct correlation_at.
It neither crops frequency bins nor interpolates the sampled statistic. All
operations use NumPy/CuPy FFTs; no C/CUDA kernels are added or modified.
"""
import numpy
from scipy.fft import next_fast_len
from pycbc.filter.subsample_arrival_time import correlation_at
from pycbc.filter.eccentric_fast_search import (FusedWindowedModeFFT,
                                               FastEccentricFFTBlockModel)


def uniform_correlation_at(spectrum, delays, duration, xp=numpy):
    """Evaluate a uniform run of delays with Bluestein's chirp identity.

    Nonuniform/singleton targets use the direct reference. Chirps are formed
    from real float64 angles, avoiding roundoff magnification from powers of
    an approximately unit-magnitude complex root at millions of bins.
    """
    delays=numpy.atleast_1d(numpy.asarray(delays,dtype=float))
    if (delays.ndim!=1 or not len(delays) or not numpy.all(numpy.isfinite(delays))
            or not numpy.isfinite(duration) or duration<=0):
        raise ValueError('finite delays and a positive duration are required')
    n=spectrum.shape[-1]
    if n<1:raise ValueError('the spectrum cannot be empty')
    step=delays[1]-delays[0] if len(delays)>1 else 0.
    if len(delays)<2 or not numpy.allclose(numpy.diff(delays),step,rtol=1e-10,atol=1e-12):
        return correlation_at(spectrum,delays,duration,chunk=2,xp=xp)
    m=len(delays)
    length=next_fast_len(n+m-1)
    dtype=xp.result_type(spectrum.dtype,xp.complex64)
    k=xp.arange(n,dtype=xp.float64)
    q=step/duration
    a=xp.asarray(spectrum,dtype=dtype)*xp.exp(1j*numpy.pi*(q*k*k+2*delays[0]/duration*k)).astype(dtype)
    l=xp.arange(1-n,m,dtype=xp.float64)
    b=xp.exp(-1j*numpy.pi*q*l*l).astype(dtype)
    transformed=xp.fft.fft(a,n=length,axis=-1)
    del a,k,l
    transformed*=xp.fft.fft(b,n=length)
    del b
    out=xp.fft.ifft(transformed,axis=-1)[...,n-1:n-1+m]
    j=xp.arange(m,dtype=xp.float64)
    return out*xp.exp(1j*numpy.pi*q*j*j)


class CZTWindowedModeFFT(FusedWindowedModeFFT):
    def fractional_correlations(self,delays,chunk=2):
        xp=self.xp
        values=uniform_correlation_at(self.spectra,delays,self.duration,xp=xp)
        carrier=xp.exp(2j*numpy.pi*float(self.grid.frequencies[0])*xp.asarray(delays))
        return values*carrier[None,None]


class CZTEccentricFFTBlockModel(FastEccentricFFTBlockModel):
    fft_engine_class=CZTWindowedModeFFT
