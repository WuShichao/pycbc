# Copyright (C) 2026 Shichao Wu
# This program is free software under the GNU General Public License,
# version 3 or (at your option) any later version.
"""Reuse existing production correlation kernels for an eccentric mode bank.

No CUDA kernels are added or modified. Each physical source mode is passed
through the existing sky-block spectra kernel with its own clock. Complete
windowed Grams and the physical anomaly combination remain unchanged.
"""
from types import SimpleNamespace
import numpy
from pycbc.filter.eccentric_mode_fft import WindowedModeFFT
from pycbc.filter.eccentric_block_search import EccentricFFTBlockModel
from pycbc.filter.mean_anomaly import mean_anomaly_components


def mean_anomaly_twof_analytic(z,gram,c,n_l0=None,rcond=1e-12,xp=numpy):
    """The same coherent statistic, with a conditioned 2-by-2 direct inverse.

    Ill-conditioned, rank-deficient and zero cases use the Hermitian
    eigensolver/pseudoinverse. Hermitian and PSD checks remain mandatory.
    """
    if not numpy.isfinite(rcond) or not 0<rcond<1:raise ValueError('invalid rcond')
    zl,gl,angles=mean_anomaly_components(z,gram,c,n_l0,xp=xp)
    zz=xp.moveaxis(zl,1,-1)
    gg=xp.moveaxis(gl,(1,2),(-2,-1))
    gg=gg.reshape((len(angles),)+(1,)*(zz.ndim-gg.ndim+1)+gg.shape[1:])
    gg=xp.broadcast_to(gg,zz.shape[:-1]+(2,2))
    scale=xp.max(abs(gg),axis=(-2,-1))
    if bool(xp.any(abs(gg-gg.swapaxes(-1,-2).conj())>
                   max(rcond,32*numpy.finfo(float).eps)*scale[...,None,None])):
        raise ValueError('Gram must be Hermitian')
    divisor=xp.where(scale>0,scale,1.)
    a,d,b=gg[...,0,0].real/divisor,gg[...,1,1].real/divisor,gg[...,0,1]/divisor
    disc=xp.sqrt((a-d)**2+4*abs(b)**2)
    high,low=(a+d+disc)/2,(a+d-disc)/2
    if bool(xp.any(low < -rcond*xp.maximum(abs(low),abs(high)))):
        raise ValueError('Gram must be positive semidefinite')
    good=(scale>0)&(low>max(1e-4,rcond)*high)
    det=a*d-abs(b)**2
    numerator=d*abs(zz[...,0])**2+a*abs(zz[...,1])**2-2*xp.real(zz[...,0].conj()*b*zz[...,1])
    value=xp.where(good,numerator/(divisor*xp.where(good,det,1.)),0.)
    bad=xp.flatnonzero(~good.ravel())
    if bad.size:
        matrix=gg.reshape(-1,2,2)[bad]
        rhs=zz.reshape(-1,2)[bad]
        vals,vec=xp.linalg.eigh(.5*(matrix+matrix.swapaxes(-1,-2).conj()))
        cutoff=rcond*xp.max(abs(vals),axis=-1,keepdims=True)
        if bool(xp.any(vals < -cutoff)):raise ValueError('Gram must be positive semidefinite')
        positive=vals>cutoff
        inverse=xp.where(positive,1./xp.where(positive,vals,1.),0.)
        projection=xp.einsum('...ai,...a->...i',vec.conj(),rhs)
        value.reshape(-1)[bad]=xp.sum(abs(projection)**2*inverse,axis=-1)
    return value,angles


class FusedWindowedModeFFT(WindowedModeFFT):
    """The existing spectra kernel supplies all mode correlation spectra."""
    def __init__(self,grid,waveforms,times,response,orders,data,observation,delay_bounds,
                 xp=numpy,dtype=numpy.complex64,max_cache_bytes=4_000_000_000):
        if xp is numpy or numpy.dtype(dtype)!=numpy.dtype(numpy.complex64):
            super().__init__(grid,waveforms,times,response,orders,data,observation,
                delay_bounds,xp=xp,dtype=dtype,max_cache_bytes=max_cache_bytes)
            return
        from pycbc.filter.sky_block_kernels import correlation_spectra
        self.grid,self.xp,self.dtype=grid,xp,numpy.dtype(dtype)
        self.orders=numpy.asarray(orders,dtype=int)
        if not numpy.array_equal(self.orders,numpy.arange(self.orders[0],self.orders[-1]+1)):
            raise ValueError('the production kernel needs contiguous yearly orders')
        self.size,self.duration,self.dt=int(grid.size),float(grid.duration),float(grid.delta_t)
        self.first,self.last=map(int,observation)
        self.bounds=tuple(map(int,delay_bounds))
        self.crop_first,self.crop_last=self.first-self.bounds[1],self.last-self.bounds[0]
        self.crop_size=self.crop_last-self.crop_first
        if self.crop_size>self.size:raise ValueError('observation plus delays exceeds the FFT period')
        self.b=xp.asarray(waveforms,dtype=self.dtype)
        self.t=xp.asarray(times,dtype=xp.float64)
        self.response=response
        self.max_workspace_bytes=int(max_cache_bytes)
        self.nmode,self.nchannel=self.b.shape[0],data.shape[0]
        self.cache_bytes=self.b.nbytes+self.t.nbytes
        self.spectra=xp.empty((self.nmode,2,self.size),dtype=self.dtype)
        family=SimpleNamespace(size=self.size,orders=self.orders,shift=grid.harmonic_shift,
            _index=response.index,_weight=response.weight)
        basis=response.coarse[None]
        for k in range(self.nmode):
            self.spectra[k]=correlation_spectra(family,data,self.b[k],self.t[k],basis,0,1)[0]


class FastEccentricFFTBlockModel(EccentricFFTBlockModel):
    fft_engine_class=FusedWindowedModeFFT
    statistic=staticmethod(mean_anomaly_twof_analytic)
