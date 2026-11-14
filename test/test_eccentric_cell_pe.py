"""EccentricCellPE: its adaptive arrival-time quadrature matches a dense direct integral, and reconstruct draws
finite marginalised parameters."""
from types import SimpleNamespace
import numpy as np
from pycbc.filter.bstatistic import DistancePowerLaw
from pycbc.filter.eccentric_block_search import EccentricFFTBlockModel, MassEccentricCoordinates
from pycbc.filter.mean_anomaly import mean_anomaly_log_bayes
from pycbc.inference.models.eccentric_bstatistic import EccentricCellPE
from pycbc.tdi.ssb_yearly_harmonics import ORBITAL_PERIOD


def toy():
    n=256
    grid=SimpleNamespace(size=n,duration=ORBITAL_PERIOD,delta_t=ORBITAL_PERIOD/n,
        frequencies=.01+np.arange(n)/ORBITAL_PERIOD,harmonic_shift=1)
    coordinates=MassEccentricCoordinates(30.,.05,[.01,.001,.01])
    rng=np.random.default_rng(41)
    carrier=rng.normal(size=(3,n))+1j*rng.normal(size=(3,n))
    carrier[:,:6]=0;carrier[:,-6:]=0
    def source(theta):
        x=np.linspace(-1,1,n)
        s=coordinates.encode(theta[:3])[0]
        phase=s[0]*x+s[1]*x**2+s[2]*x**3+5*np.log(theta[3]/1.5)*x**4
        return carrier*np.exp(1j*phase),np.zeros((3,n))
    response=np.array([[[1.,.3j],[.2+.4j,1.]]],complex)[...,None]*np.ones(n)
    truth=np.array([30.,.05,0.,1.5])
    b,_=source(truth)
    signal=np.einsum('kf,cf->cf',b,np.ones((2,n)))*0+np.array([b[0]+.3*b[1]+.1*b[2],.2j*b[0]+b[1]])
    data=(rng.normal(size=(2,n))+1j*rng.normal(size=(2,n)))*.5+3*signal/np.linalg.norm(signal)
    ref=dict(eclipticlongitude=1.,eclipticlatitude=.2)
    scales=dict(eclipticlongitude=.01,eclipticlatitude=.01)
    block=EccentricFFTBlockModel(grid,data,source,lambda sky:response,grid.frequencies,[0],[-1,0,1],coordinates,
        np.eye(3),ref,scales,(20,200),(-8*grid.delta_t,8*grid.delta_t),dtype=np.complex128,n_l0=16)
    model=EccentricCellPE(['mchirp','q','eccentricity','chi_eff','eclipticlongitude','eclipticlatitude'],block,
        (1.,.2,.01,.01),(-6*grid.delta_t,6*grid.delta_t),DistancePowerLaw(.1,1e3,scale=10.),n_l0=16,
        orientation_nodes=(8,4),fine_subsamples=16,fine_drop=60.)
    return model,block,grid


def test_time_quadrature_matches_a_dense_integral_and_reconstruct_draws():
    model,block,grid=toy()
    params=dict(mchirp=30.,q=1.5,eccentricity=.05,chi_eff=0.,eclipticlongitude=1.,eclipticlatitude=.2)
    model.update(**params)
    lnl=model.loglikelihood
    ev=model.evaluate()
    # dense direct integral over the whole window, 64 points per lag interval
    dt=grid.delta_t
    lo,hi=model.delay_window
    j0,j1=int(np.ceil(lo/dt)),int(np.floor(hi/dt))
    d=(np.arange(j0,j1)[:,None]+(np.arange(64)+.5)[None,:]/64).ravel()*dt
    z=ev['engine'].fractional_correlations(d)
    g=ev['engine'].interpolated_gram(d,ev['anchor_times'],ev['anchor_grams'])
    lnb=mean_anomaly_log_bayes(z,g,block.offsets,model.distance_prior,model.nodes,n_l0=16)[0]
    top=lnb.max()
    dense=top+np.log(np.sum(np.exp(lnb-top))*dt/64)-np.log((j1-j0)*dt)
    assert abs(lnl-dense)<2e-2*max(1.,abs(dense)), (lnl,dense)
    rec=model.reconstruct(seed=3)
    for k in ('delta_tc','mean_anomaly','inclination','polarization','distance','coa_phase','loglr'):
        assert np.isfinite(rec[k]), k
    assert lo<=rec['delta_tc']<=hi and 0<=rec['mean_anomaly']<2*np.pi and rec['distance']>0
