"""Chirp convolution preserves the exact finite correlation polynomial."""
import numpy as np
import pytest
from pycbc.filter.eccentric_mode_czt import uniform_correlation_at
from pycbc.filter.subsample_arrival_time import correlation_at


@pytest.mark.parametrize('n,m,step',[(257,17,.125),(100003,17,.125),(257,9,-.25),(257,1,0.)])
def test_chirp_convolution_matches_direct_sum(n,m,step):
    rng=np.random.default_rng(27)
    a=rng.normal(size=(3,2,n))+1j*rng.normal(size=(3,2,n))
    duration=221e6
    delays=(-311.37+step*np.arange(m))*duration/n
    expected=correlation_at(a,delays,duration)
    actual=uniform_correlation_at(a,delays,duration)
    assert np.linalg.norm(actual-expected)/np.linalg.norm(expected)<2e-9


def test_nonuniform_delays_use_reference_and_invalid_targets_raise():
    a=np.arange(100,dtype=complex)
    delays=[-.45,.9,.17]
    np.testing.assert_allclose(uniform_correlation_at(a,delays,1.),correlation_at(a,delays,1.))
    with pytest.raises(ValueError):uniform_correlation_at(a,[],1.)
    with pytest.raises(ValueError):uniform_correlation_at(a,[0.],0.)


def test_uniform_czt_gpu_single_precision_matches_direct_double_cpu():
    cp=pytest.importorskip('cupy')
    try:available=cp.cuda.runtime.getDeviceCount()>0
    except cp.cuda.runtime.CUDARuntimeError:available=False
    if not available:pytest.skip('CUDA unavailable')
    n=131072
    rng=np.random.default_rng(29)
    a=(rng.normal(size=(4,2,n))+1j*rng.normal(size=(4,2,n))).astype(np.complex64)
    delays=(-72.4+np.arange(17)/8)*221e6/n
    expected=correlation_at(a,delays,221e6)
    actual=uniform_correlation_at(cp.asarray(a),delays,221e6,xp=cp)
    assert np.linalg.norm(actual.get()-expected)/np.linalg.norm(expected)<3e-6


def test_czt_mode_engine_refinement_equals_direct_windowed_model():
    from types import SimpleNamespace
    from pycbc.filter.eccentric_mode_czt import CZTEccentricFFTBlockModel
    from pycbc.filter.eccentric_block_search import EccentricFFTBlockModel, MassEccentricCoordinates
    from pycbc.tdi.ssb_yearly_harmonics import ORBITAL_PERIOD
    rng=np.random.default_rng(39)
    n=256
    grid=SimpleNamespace(size=n,duration=ORBITAL_PERIOD,delta_t=ORBITAL_PERIOD/n,
        frequencies=.01+np.arange(n)/ORBITAL_PERIOD,harmonic_shift=1)
    b=rng.normal(size=(2,n))+1j*rng.normal(size=(2,n))
    b[:,:4]=0;b[:,-4:]=0
    t=rng.uniform(-ORBITAL_PERIOD,ORBITAL_PERIOD,(2,n))
    rho=rng.normal(size=(3,2,2,n))+1j*rng.normal(size=(3,2,2,n))
    dt=np.zeros((2,n),complex)
    dt[:,20:190]=rng.normal(size=(2,170))+1j*rng.normal(size=(2,170))
    data=np.fft.fft(dt,axis=-1)/np.sqrt(n)
    coord=MassEccentricCoordinates(30.,.05,[.01,.001])
    candidates=[]
    for cls in (EccentricFFTBlockModel,CZTEccentricFFTBlockModel):
        model=cls(grid,data,lambda theta:(b,t),lambda sky:rho,grid.frequencies,
            [-1,0,1],[0,1],coord,np.eye(2),
            dict(eclipticlongitude=1.,eclipticlatitude=.2),
            dict(eclipticlongitude=.01,eclipticlatitude=.01),(20,190),
            (-5*grid.delta_t,5*grid.delta_t),dtype=np.complex128,n_l0=32)
        candidate=dict(theta=[30.,.05],sky=[0.,0.],time=.37*grid.delta_t,twof=0.)
        model.refine_candidate(candidate)
        candidates.append(candidate)
    for key in ('twof','time','mean_anomaly','z','gram'):
        np.testing.assert_allclose(candidates[0][key],candidates[1][key],rtol=2e-9,atol=2e-9)
