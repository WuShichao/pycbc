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
        orientation_nodes=(8,4),oversampling=16,screen_margin=80.,bayes_margin=60.)
    model.keep_engine=True
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


def test_oversampled_correlations_equal_the_direct_sum():
    model,block,grid=toy()
    model.update(mchirp=30.,q=1.5,eccentricity=.05,chi_eff=0.,eclipticlongitude=1.,eclipticlatitude=.2)
    engine=model.evaluate()['engine']
    z,d=engine.oversampled_correlations(np.array([-3,0,2]),5)
    direct=engine.fractional_correlations(d.ravel())
    np.testing.assert_allclose(z.reshape(z.shape[0],2,-1),direct,rtol=1e-6,atol=1e-6*np.abs(direct).max())


def test_shared_region_gram_interpolation_matches_the_engine_rule():
    from types import SimpleNamespace as NS
    from pycbc.filter.eccentric_mode_fft import WindowedModeFFT
    from pycbc.filter.eccentric_shared import EccentricSharedRegion
    rng=np.random.default_rng(5)
    region=EccentricSharedRegion.__new__(EccentricSharedRegion)
    region.xp=np
    region._anchor_times=np.array([-10.,0.,25.])
    stack=rng.normal(size=(3,4,4,3))+1j*rng.normal(size=(3,4,4,3))
    rows=np.array([0,2,1,1,0])
    delays=np.array([-10.,-3.,0.,24.,25.])
    got=region._gram_at(stack,rows,delays)
    for s,(r,d) in enumerate(zip(rows,delays)):
        want=WindowedModeFFT.interpolated_gram(NS(xp=np),[d],region._anchor_times,stack[r])[...,0]
        np.testing.assert_allclose(got[...,s],want,rtol=1e-12)


def test_reconstruct_many_draws_what_reconstruct_draws():
    model,block,grid=toy()
    points=[dict(mchirp=30.*np.exp(d),q=1.5,eccentricity=.05+de,chi_eff=0.,eclipticlongitude=1.,eclipticlatitude=.2)
            for d,de in ((0.,0.),(1e-4,0.),(-1e-4,1e-4),(0.,-1e-4))]
    seeds=[3,4,5,6]
    many=model.reconstruct_many(points,seeds,chunk=3)
    for p,seed,got in zip(points,seeds,many):
        model.update(**p)
        want=model.reconstruct(seed=seed)
        for k,v in want.items():
            np.testing.assert_allclose(got[k],v,rtol=1e-9,atol=1e-12,err_msg=k)


def test_a_point_without_a_template_has_no_likelihood_and_is_counted():
    """Where the cell makes no template (TemplateDomainError, a mode clock that stops increasing) ln B is -inf,
    directly and in a batch, the count is kept, and the next point is evaluated as before."""
    from pycbc.waveform.eccentric_mode_gpu import TemplateDomainError
    model,block,grid=toy()
    good=dict(mchirp=30.,q=1.5,eccentricity=.05,chi_eff=0.,eclipticlongitude=1.,eclipticlatitude=.2)
    bad=dict(good,q=1.9)
    model.update(**good)
    expected=model.loglikelihood
    inner=block.source_builder

    def source(theta):
        if theta[3]>1.8:
            raise TemplateDomainError('device mode clock must increase')
        return inner(theta)
    block.source_builder=source
    model._last=None
    model.update(**bad)
    assert model.loglikelihood==-np.inf
    out=model.loglikelihoods([bad,good])
    assert out[0]==-np.inf and abs(out[1]-expected)<1e-9
    assert model.shared_statistics['outside_template_domain']==2
