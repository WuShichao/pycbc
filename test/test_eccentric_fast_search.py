"""Optimized eccentric paths retain the reference statistic and mode splines."""
import numpy as np
import pytest

from pycbc.filter.eccentric_fast_search import mean_anomaly_twof_analytic
from pycbc.filter.mean_anomaly import mean_anomaly_twof
from pycbc.waveform.eccentric_mode_gpu import DeviceEccentricModeSource
from pycbc.waveform.eccentric_mode_family import EccentricModeFamily
from pycbc.tdi.eccentric_mode_source import EccentricModeFamilySource


@pytest.mark.parametrize('rank',[0,1,2])
@pytest.mark.parametrize('shared',[True,False])
def test_analytic_inverse_matches_reference_with_broadcast_and_rank_deficiency(rank,shared):
    rng=np.random.default_rng(82)
    h=rng.normal(size=(4,2,47))+1j*rng.normal(size=(4,2,47))
    if rank==1:h[:,1]=1j*h[:,0]
    if rank==0:h*=0
    data=rng.normal(size=(47,2,3))+1j*rng.normal(size=(47,2,3))
    w=np.ones((47,3)) if shared else rng.uniform(.3,2,(47,3))
    z=np.einsum('kaf,frs,fs->kars',h.conj(),data,w)
    g=np.einsum('kaf,lbf,fs->kalbs',h.conj(),h,w)
    if shared:g=g[...,0]
    for cutoff in (1e-12,.8):
        expected,angles=mean_anomaly_twof(z,g,[-1,0,0,2],16,rcond=cutoff)
        value,actual=mean_anomaly_twof_analytic(z,g,[-1,0,0,2],16,rcond=cutoff)
        np.testing.assert_allclose(value,expected,atol=3e-12,rtol=3e-12)
        np.testing.assert_array_equal(actual,angles)


@pytest.mark.parametrize('case',['nonhermitian','indefinite'])
def test_analytic_statistic_rejects_invalid_gram(case):
    g=np.eye(2,dtype=complex).reshape(1,2,1,2)
    if case=='nonhermitian':g[0,0,0,1]=1j
    else:g[0,0,0,0]=-1
    with pytest.raises(ValueError):mean_anomaly_twof_analytic(np.ones((1,2)),g,[0])


def spline_case(xp=np):
    f=np.linspace(.01,.02,61)
    def generator(p):
        out={}
        for n in (1,2,3):
            phase=2*np.pi*(n*1e5*f+2e7*f*f)+p['eccentricity']*f
            amp=np.ones((2,len(f)),complex)*(1+n*p['eccentricity'])
            have=np.ones(len(f),bool)
            if n==3:have[22:28]=False
            out[2,2,n]=amp,phase,have
        return out
    ref=dict(mass1=36.,mass2=31.,eccentricity=.05)
    family=EccentricModeFamily(ref,f,.015,step_mass=0.,step_e=.002,step_q=0.,
        generator=generator,circular_phase_basis='taylorf2')
    fine=np.linspace(.009,.021,317)
    weight=np.linspace(.7,1.3,len(fine))
    keys=sorted(family.keys)
    builder=DeviceEccentricModeSource(family,fine,keys,weight,30.,clock_time=14.,xp=xp)
    b,t=builder([30.,.0513])
    source=EccentricModeFamilySource(family,de=.0513-.05,clock_time=14.)
    records=source.frequency_harmonics(fine)
    for k,key in enumerate(keys):
        expected=np.zeros(len(fine),complex)
        clock=np.zeros(len(fine))
        rec=records[key];idx=rec['indices']
        expected[idx]=rec['plus']*weight[idx]
        clock[idx]=rec['time']
        actual=b[k].get() if hasattr(b[k],'get') else b[k]
        actual_t=t[k].get() if hasattr(t[k],'get') else t[k]
        np.testing.assert_allclose(actual,expected,atol=5e-7,rtol=5e-7)
        np.testing.assert_allclose(actual_t,clock,atol=2e-7,rtol=1e-12)
    return b,t


def test_device_spline_numpy_matches_existing_source_without_bridging_gaps():
    spline_case()


def test_device_spline_and_analytic_statistic_cupy():
    cp=pytest.importorskip('cupy')
    try:available=cp.cuda.runtime.getDeviceCount()>0
    except cp.cuda.runtime.CUDARuntimeError:available=False
    if not available:pytest.skip('CUDA unavailable')
    spline_case(cp)
    z=np.array([[1+2j,2-3j],[.4+.8j,1-.2j]])
    g=np.eye(4,dtype=complex).reshape(2,2,2,2)
    expected,_=mean_anomaly_twof(z,g,[-1,0],16)
    value,_=mean_anomaly_twof_analytic(cp.asarray(z),cp.asarray(g),[-1,0],16,xp=cp)
    np.testing.assert_allclose(value.get(),expected,atol=2e-12,rtol=2e-12)


def test_fused_existing_gpu_kernel_matches_complete_reference_fft():
    from types import SimpleNamespace
    from pycbc.filter.eccentric_fast_search import FusedWindowedModeFFT
    from pycbc.filter.eccentric_mode_fft import WindowedModeFFT
    from pycbc.tdi.ssb_yearly_harmonics import ORBITAL_PERIOD
    cp=pytest.importorskip('cupy')
    try:available=cp.cuda.runtime.getDeviceCount()>0
    except cp.cuda.runtime.CUDARuntimeError:available=False
    if not available:pytest.skip('CUDA unavailable')
    rng=np.random.default_rng(19)
    n=256
    grid=SimpleNamespace(size=n,duration=ORBITAL_PERIOD,delta_t=ORBITAL_PERIOD/n,
        frequencies=.01+np.arange(n)/ORBITAL_PERIOD,harmonic_shift=1)
    b=rng.normal(size=(3,n))+1j*rng.normal(size=(3,n))
    b[:,:4]=0;b[:,-4:]=0
    t=rng.uniform(-ORBITAL_PERIOD,ORBITAL_PERIOD,(3,n))
    rho=rng.normal(size=(3,2,2,n))+1j*rng.normal(size=(3,2,2,n))
    dt=np.zeros((2,n),complex)
    dt[:,20:190]=rng.normal(size=(2,170))+1j*rng.normal(size=(2,170))
    data=np.fft.fft(dt,axis=-1)/np.sqrt(n)
    def response(r):return cp.asarray(rho[r],dtype=cp.complex64)
    response.coarse=cp.asarray(rho,dtype=cp.complex64)
    response.index=cp.minimum(cp.arange(n,dtype=cp.int32),n-2)
    response.weight=cp.asarray(np.arange(n)==n-1,dtype=cp.float32)
    expected=WindowedModeFFT(grid,b,t,lambda r:rho[r],[-1,0,1],data,(20,190),(-5,5),dtype=np.complex128)
    actual=FusedWindowedModeFFT(grid,b,t,response,[-1,0,1],cp.asarray(data),
        (20,190),(-5,5),xp=cp,dtype=np.complex64)
    for a,e in ((actual.correlations([-3,0,4]).get(),expected.correlations([-3,0,4])),
                (actual.gram([-3,0,4]).get(),expected.gram([-3,0,4])),
                (actual.fractional_correlations(np.array([-.4,1.7])*grid.delta_t).get(),
                 expected.fractional_correlations(np.array([-.4,1.7])*grid.delta_t))):
        assert np.linalg.norm(a-e)/np.linalg.norm(e)<3e-6
