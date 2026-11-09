# Copyright (C) 2026 Shichao Wu
# This program is free software under the GNU General Public License,
# version 3 or (at your option) any later version.
"""Evaluate prepared eccentric mode splines on NumPy or CuPy frequency grids.

The offline native-reference cell remains on its compact stencil grid. Each
candidate builds small cubic-spline coefficient arrays there, then evaluates
amplitude, unwrapped phase and each mode's clock on the target device. No
native waveform is generated online. Support gaps are never interpolated.
"""
import numpy
from scipy.interpolate import CubicSpline
from pycbc.tdi.eccentric_mode_source import EccentricModeFamilySource


class DeviceEccentricModeSource:
    """Whitened face-on (2,2,n) mode arrays for a dense FFT grid.

    family must have a global TaylorF2 phase basis and static 'all' support.
    Physical coordinates are chirp mass and eccentricity, with q/spins fixed.
    source_weight includes whitening and any frequency notches/band masks.
    """
    def __init__(self,family,frequencies,keys,source_weight,
                 reference_mchirp,clock_time=0.,xp=numpy):
        if family.circular_phase_basis!='taylorf2' or family.support!='all':
            raise ValueError('device source needs a TaylorF2-basis all-support cell')
        self.family,self.xp=family,xp
        self.f=numpy.asarray(frequencies)
        self.keys=tuple(keys)
        self.weight=xp.asarray(source_weight,dtype=xp.float64)
        self.mc0,self.clock_time=float(reference_mchirp),float(clock_time)
        self._indices={}

    def __call__(self,theta):
        xp=self.xp
        e=float(theta[1]) if len(theta)>1 else self.family.reference['eccentricity']
        d=numpy.log(float(theta[0])/self.mc0)
        de=e-self.family.reference['eccentricity']
        source=EccentricModeFamilySource(self.family,d=d,de=de,clock_time=self.clock_time)
        b=xp.zeros((len(self.keys),len(self.f)),dtype=xp.complex64)
        t=xp.zeros(b.shape,dtype=xp.float64)
        for key,sl,amp,phase,have in self.family._windows(d,de,circular=source.circular):
            if key not in self.keys:continue
            k=self.keys.index(key)
            active=numpy.flatnonzero(have)
            pieces=numpy.split(active,numpy.flatnonzero(numpy.diff(active)!=1)+1)
            for bins in pieces:
                if len(bins)<2:continue
                x=self.family.f[sl][bins]
                token=(key,int(sl.start+bins[0]),int(sl.start+bins[-1]))
                if token not in self._indices:
                    lo=int(numpy.searchsorted(self.f,x[0]))
                    hi=int(numpy.searchsorted(self.f,x[-1],side='right'))
                    index=numpy.clip(numpy.searchsorted(x,self.f[lo:hi],side='right')-1,0,len(x)-2)
                    dx=self.f[lo:hi]-x[index]
                    self._indices[token]=(lo,hi,xp.asarray(index,dtype=xp.int32),
                                          xp.asarray(dx),xp.asarray(self.f[lo:hi]))
                lo,hi,index,dx,fine=self._indices[token]
                if hi<=lo:continue
                pc=xp.asarray(CubicSpline(x,phase[bins]).c)
                ac=xp.asarray(CubicSpline(x,amp[0,bins]).c)
                p=((pc[0,index]*dx+pc[1,index])*dx+pc[2,index])*dx+pc[3,index]
                a=((ac[0,index]*dx+ac[1,index])*dx+ac[2,index])*dx+ac[3,index]
                time=((3*pc[0,index]*dx+2*pc[1,index])*dx+pc[2,index])/(2*numpy.pi)+source.shift
                if bool(xp.any(xp.diff(time)<=0)):
                    raise ValueError('device mode clock must increase')
                angle=-p-2*numpy.pi*fine*source.shift
                b[k,lo:hi]=(a*xp.exp(1j*angle)*self.weight[lo:hi]).astype(xp.complex64)
                t[k,lo:hi]=time
        return b,t
