"""The production multilevel driver uses eccentric coordinates and mode FFTs."""
from types import SimpleNamespace
import numpy as np

from pycbc.filter.eccentric_block_search import EccentricFFTBlockModel, MassEccentricCoordinates
from pycbc.filter.sky_block_search import multilevel_block_search, aligned_lattice, fft_search_sky_block
from pycbc.tdi.ssb_yearly_harmonics import ORBITAL_PERIOD


def test_multilevel_eccentric_backend_recovers_injection_and_matches_exhaustive_grid():
    n=128
    grid=SimpleNamespace(size=n,duration=ORBITAL_PERIOD,delta_t=ORBITAL_PERIOD/n,
        frequencies=.01+np.arange(n)/ORBITAL_PERIOD,harmonic_shift=1)
    coordinates=MassEccentricCoordinates(30.,.05,[.01,.001])
    rng=np.random.default_rng(19)
    carrier=rng.normal(size=(2,n))+1j*rng.normal(size=(2,n))
    carrier[:,:4]=0;carrier[:,-4:]=0
    def source(theta):
        s=coordinates.encode(theta)[0]
        phase=s[0]*np.linspace(-1,1,n)+s[1]*np.linspace(-1,1,n)**2
        return carrier*np.exp(1j*phase),np.zeros((2,n))
    response=np.array([[[1.,.3j],[.2+.4j,1.]]],complex)[...,None]*np.ones(n)
    truth=np.array([30.,.05])
    b,t=source(truth)
    h=np.sum(b*np.exp(1j*np.array([0,1])[:,None]*.7),axis=0)
    projected=np.einsum('a,acf,f->cf',np.array([.7,-.5j]),response[0],h)
    y=np.fft.ifft(projected,axis=-1)*np.sqrt(n)
    y[:,:20]=0;y[:,70:]=0
    y*=15/np.linalg.norm(y)
    data=np.fft.fft(y,axis=-1)/np.sqrt(n)
    ref=dict(eclipticlongitude=1.,eclipticlatitude=.2)
    scales=dict(eclipticlongitude=.01,eclipticlatitude=.01)
    model=EccentricFFTBlockModel(grid,data,source,lambda sky:response,
        grid.frequencies,[0],[0,1],coordinates,np.eye(2),ref,scales,
        (20,70),(-5*grid.delta_t,5*grid.delta_t),dtype=np.complex128,n_l0=32)
    region=coordinates.decode(np.array([[-.6,-.6],[-.6,.6],[.6,-.6],[.6,.6]]))
    levels=[dict(mismatch=.1,sky_step=1.,decimate=1,threshold=20.),
            dict(mismatch=.02,sky_step=.5,decimate=1)]
    output=multilevel_block_search(model,model,region,levels,radius=0.,
        merge_time=10*grid.delta_t,refine_threshold=np.inf,followup_lags=2,branch=2)
    s=coordinates.encode(region)
    nodes,near,_=aligned_lattice(s,np.eye(2),.02)
    baseline=fft_search_sky_block(model,region[near],np.zeros((1,2)),coefficients=nodes)
    assert output['followups']>0
    np.testing.assert_allclose(output['candidates'][0]['twof'],baseline['candidates'][0]['twof'],rtol=1e-12)
    assert output['candidates'][0]['twof']>200
    assert len(output['candidates'][0]['theta'])==2
