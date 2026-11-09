"""The FFT mode path equals explicit windowed time-domain templates."""
from types import SimpleNamespace
import numpy as np
import pytest

from pycbc.filter.eccentric_mode_fft import WindowedModeFFT
from pycbc.tdi.ssb_yearly_harmonics import ORBITAL_PERIOD, ORBITAL_ANGULAR_FREQUENCY as OMEGA


def case(xp=np):
    rng=np.random.default_rng(52)
    n=128
    grid=SimpleNamespace(size=n,duration=ORBITAL_PERIOD,delta_t=ORBITAL_PERIOD/n,
        frequencies=.01+np.arange(n)/ORBITAL_PERIOD,harmonic_shift=1)
    b=rng.normal(size=(3,n))+1j*rng.normal(size=(3,n))
    b[:,:4]=0;b[:,-4:]=0
    t=rng.uniform(-ORBITAL_PERIOD,ORBITAL_PERIOD,(3,n))
    rho=rng.normal(size=(3,2,2,n))+1j*rng.normal(size=(3,2,2,n))
    orders=np.array([-1,0,1])
    data_time=np.zeros((2,n),complex)
    data_time[:,20:70]=rng.normal(size=(2,50))+1j*rng.normal(size=(2,50))
    data=np.fft.fft(data_time,axis=-1)/np.sqrt(n)
    engine=WindowedModeFFT(grid,b,t,lambda r:rho[r],orders,data,(20,70),(-5,5),
                           xp=xp,dtype=np.complex128)
    return engine,grid,b,t,rho,orders,data_time


def direct(grid,b,t,rho,orders,data_time,lags):
    hs=[]
    for lag in lags:
        delay=lag*grid.delta_t
        h=np.zeros((3,2,2,128),complex)
        for r,order in enumerate(orders):
            source=rho[r][None]*b[:,None,None]*np.exp(1j*order*OMEGA*t[:,None,None])
            source*=np.exp((-2j*np.pi*delay)*grid.frequencies)
            source*=np.exp(1j*order*OMEGA*delay)
            h+=np.fft.ifft(source,axis=-1)*np.sqrt(grid.size)
        hs.append(h[...,20:70])
    hs=np.array(hs)
    return (np.einsum('lkacf,cf->kal',hs.conj(),data_time[:,20:70]),
            np.einsum('lkacf,lmbcf->kambl',hs.conj(),hs))


def test_fft_correlations_and_complete_gram_equal_direct_windowed_templates():
    engine,grid,b,t,rho,orders,data=case()
    lags=np.array([-5,0,3,5])
    z,g=direct(grid,b,t,rho,orders,data,lags)
    np.testing.assert_allclose(engine.correlations(lags),z,atol=2e-7,rtol=2e-8)
    np.testing.assert_allclose(engine.gram(lags),g,atol=2e-7,rtol=2e-8)
    fractional=np.array([-.2,.4,2.3])
    z,_=direct(grid,b,t,rho,orders,data,fractional)
    np.testing.assert_allclose(engine.fractional_correlations(fractional*grid.delta_t),z,
                               atol=2e-7,rtol=2e-8)


def test_full_matrix_interpolation_is_psd_and_rejects_extrapolation():
    engine,*_=case()
    anchors=np.array([-5,0,5])*engine.dt
    g=engine.gram([-5,0,5])
    got=engine.interpolated_gram(np.array([-2.,2.])*engine.dt,anchors,g)
    for j in range(2):
        assert np.linalg.eigvalsh(got[...,j].reshape(6,6)).min()>-1e-10
    with pytest.raises(ValueError):engine.interpolated_gram([6*engine.dt],anchors,g)


def test_gpu_matches_cpu():
    cp=pytest.importorskip('cupy')
    try:available=cp.cuda.runtime.getDeviceCount()>0
    except cp.cuda.runtime.CUDARuntimeError:available=False
    if not available:pytest.skip('CUDA unavailable')
    cpu,*_=case()
    gpu,*_=case(cp)
    np.testing.assert_allclose(gpu.correlations([-3,0,4]).get(),cpu.correlations([-3,0,4]),atol=1e-8,rtol=1e-8)
    np.testing.assert_allclose(gpu.gram([-3,0,4]).get(),cpu.gram([-3,0,4]),atol=1e-8,rtol=1e-8)



def test_large_single_precision_gpu_gram_reduction_matches_double_cpu():
    cp=pytest.importorskip('cupy')
    try:available=cp.cuda.runtime.getDeviceCount()>0
    except cp.cuda.runtime.CUDARuntimeError:available=False
    if not available:pytest.skip('CUDA unavailable')
    rng=np.random.default_rng(71)
    n=131072
    grid=SimpleNamespace(size=n,duration=ORBITAL_PERIOD,delta_t=ORBITAL_PERIOD/n,
        frequencies=.01+np.arange(n)/ORBITAL_PERIOD,harmonic_shift=1)
    b=rng.normal(size=(2,n))+1j*rng.normal(size=(2,n))
    b[:,:4]=0;b[:,-4:]=0
    t=rng.uniform(-ORBITAL_PERIOD,ORBITAL_PERIOD,(2,n))
    rho=rng.normal(size=(3,2,2,n))+1j*rng.normal(size=(3,2,2,n))
    data=np.zeros((2,n),complex)
    for xp,dtype in ((np,np.complex128),(cp,np.complex64)):
        engine=WindowedModeFFT(grid,b,t,lambda r:rho[r],[-1,0,1],data,
                               (1000,n-1000),(-5,5),xp=xp,dtype=dtype)
        value=engine.gram([-3,0,4])
        if xp is np:reference=value
        else:
            relative=np.linalg.norm(value.get()-reference)/np.linalg.norm(reference)
            assert relative<2e-6
