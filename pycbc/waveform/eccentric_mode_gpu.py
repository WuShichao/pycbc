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


class TemplateDomainError(ValueError):
    """The cell makes no template at this point: a mode's clock (its
    stationary time against frequency) stops increasing, as it does where
    the cell is extrapolated too far (the Yorsh sobhb6 cell at q 1.9 and
    chi_eff +-0.9, stbbh_cell_box_check.py)."""


class DeviceEccentricModeSource:
    """Whitened face-on (2,2,n) mode arrays for a dense FFT grid.

    family must have a global TaylorF2 phase basis and static 'all' support.
    Physical coordinates are chirp mass, eccentricity, and optionally the
    common aligned spin chi (both spin1z and spin2z, a cell with step_chi)
    and the mass ratio q = m1/m2 (a cell with step_q): theta = (Mc, e[, chi[,
    q]]); with a family carrying a ``precession`` main-row model, theta = (Mc,
    e, chi, q, chi1_perp, chi2_perp, dphi) adds its phase correction.
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
        dchi=float(theta[2])-self.family.reference.get('spin1z',0.) if len(theta)>2 else 0.
        dq=numpy.log(float(theta[3])/self.family.q0) if len(theta)>3 else 0.
        source=EccentricModeFamilySource(self.family,d=d,de=de,dq=dq,dchi=dchi,clock_time=self.clock_time)
        b=xp.zeros((len(self.keys),len(self.f)),dtype=xp.complex64)
        t=xp.zeros(b.shape,dtype=xp.float64)
        # a precessing source (theta of 7: ..., chi1_perp, chi2_perp, dphi): the main row's phase correction of the
        # family's pycbc.waveform.precession_main_row.PrecessionMainRow, if it has one
        prec=getattr(self.family,'precession',None) if len(theta)>4 else None
        for key,sl,amp,phase,have in self.family._windows(d,de,dq,circular=source.circular,dchi=dchi):
            if key not in self.keys:continue
            k=self.keys.index(key)
            if prec is not None and tuple(key) in prec.shapes:
                phase=phase+prec.phase(key,numpy.asarray(theta,dtype=float)[None],
                                       frequencies=self.family.f[sl])[0]
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
                    raise TemplateDomainError('device mode clock must increase')
                angle=-p-2*numpy.pi*fine*source.shift
                b[k,lo:hi]=(a*xp.exp(1j*angle)*self.weight[lo:hi]).astype(xp.complex64)
                t[k,lo:hi]=time
        return b,t
