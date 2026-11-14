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
        (20,70),(-5*grid.delta_t,5*grid.delta_t),dtype=np.complex128,n_l0=32,reuse_grams=True)
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

    model.set_data(data*2)
    doubled=fft_search_sky_block(model,region[near],np.zeros((1,2)),coefficients=nodes)
    np.testing.assert_allclose(doubled['candidates'][0]['twof'],
                               4*baseline['candidates'][0]['twof'],rtol=1e-12)
    assert model.work['gram_cache_hits']>0


def test_mc_interpolated_gram_matches_anchor_and_stays_close_between():
    n=128
    grid=SimpleNamespace(size=n,duration=ORBITAL_PERIOD,delta_t=ORBITAL_PERIOD/n,
        frequencies=.01+np.arange(n)/ORBITAL_PERIOD,harmonic_shift=1)
    coordinates=MassEccentricCoordinates(30.,.05,[.01,.001])
    rng=np.random.default_rng(23)
    carrier=rng.normal(size=(2,n))+1j*rng.normal(size=(2,n))
    carrier[:,:4]=0;carrier[:,-4:]=0
    def source(theta):
        s=coordinates.encode(theta)[0]
        phase=s[0]*np.linspace(-1,1,n)+s[1]*np.linspace(-1,1,n)**2
        return carrier*np.exp(1j*phase)*(theta[0]/30.)**(5/6),np.zeros((2,n))
    response=np.array([[[1.,.3j],[.2+.4j,1.]]],complex)[...,None]*np.ones(n)
    data=(rng.normal(size=(2,n))+1j*rng.normal(size=(2,n)))
    ref=dict(eclipticlongitude=1.,eclipticlatitude=.2)
    scales=dict(eclipticlongitude=.01,eclipticlatitude=.01)
    def build(**kw):
        return EccentricFFTBlockModel(grid,data,source,lambda sky:response,
            grid.frequencies,[0],[0,1],coordinates,np.eye(2),ref,scales,
            (20,70),(-5*grid.delta_t,5*grid.delta_t),dtype=np.complex128,n_l0=32,**kw)
    exact,interp=build(),build(gram_mc_anchors=[-.01,0.,.01])
    sky=np.zeros(2)
    # at an anchor's chirp mass and the reference eccentricity the Gram is the anchor's, exactly
    _,_,g_exact=exact._prepare(np.array([30.*np.exp(.01),.05]),sky)
    _,_,g_interp=interp._prepare(np.array([30.*np.exp(.01),.05]),sky)
    np.testing.assert_allclose(g_interp,g_exact,rtol=1e-12,atol=1e-12)
    # between anchors: a convex combination, Hermitian and close to the exact Gram
    theta=np.array([30.*np.exp(.004),.0505])
    _,_,g_exact=exact._prepare(theta,sky)
    _,_,g_interp=interp._prepare(theta,sky)
    k=g_exact.shape[0]*2
    for j in range(g_exact.shape[-1]):
        a=g_exact[...,j].reshape(k,k);b=g_interp[...,j].reshape(k,k)
        np.testing.assert_allclose(b,b.conj().T,atol=1e-10)
        assert np.min(np.linalg.eigvalsh(.5*(b+b.conj().T)))>-1e-10
        assert np.linalg.norm(b-a)/np.linalg.norm(a)<2e-2
    assert interp.work['gram_cache_hits']>0
    # outside the anchors: refused, not extrapolated
    try:
        interp._prepare(np.array([30.*np.exp(.02),.05]),sky)
    except ValueError:
        pass
    else:
        raise AssertionError('a template outside the Gram anchors must be refused')


def test_sky_interpolated_gram_is_the_anchor_gram_at_anchors_and_psd_between():
    n=128
    grid=SimpleNamespace(size=n,duration=ORBITAL_PERIOD,delta_t=ORBITAL_PERIOD/n,
        frequencies=.01+np.arange(n)/ORBITAL_PERIOD,harmonic_shift=1)
    coordinates=MassEccentricCoordinates(30.,.05,[.01,.001])
    rng=np.random.default_rng(29)
    carrier=rng.normal(size=(2,n))+1j*rng.normal(size=(2,n))
    carrier[:,:4]=0;carrier[:,-4:]=0
    def source(theta):
        s=coordinates.encode(theta)[0]
        return carrier*np.exp(1j*s[0]*np.linspace(-1,1,n)),np.zeros((2,n))
    def response(sky):
        a=np.array([[[1.,.3j],[.2+.4j,1.]]],complex)[...,None]*np.ones(n)
        return a*np.exp(1j*(.3*sky[0]-.2*sky[1]))*(1+.05*sky[0])
    data=(rng.normal(size=(2,n))+1j*rng.normal(size=(2,n)))
    ref=dict(eclipticlongitude=1.,eclipticlatitude=.2)
    scales=dict(eclipticlongitude=.01,eclipticlatitude=.01)
    def build(**kw):
        return EccentricFFTBlockModel(grid,data,source,response,grid.frequencies,[0],[0,1],coordinates,np.eye(2),
            ref,scales,(20,70),(-5*grid.delta_t,5*grid.delta_t),dtype=np.complex128,n_l0=32,**kw)
    exact=build()
    both=build(gram_sky_anchors=(3,2.),gram_mc_anchors=[-.01,0.,.01])
    sky_only=build(gram_sky_anchors=(3,2.))
    theta=np.array([30.,.05])
    for model in (both,sky_only):
        _,_,g=model._prepare(theta,np.array([2.,-2.]))
        _,_,ge=exact._prepare(theta,np.array([2.,-2.]))
        np.testing.assert_allclose(g,ge,rtol=1e-12,atol=1e-12)
        _,_,g=model._prepare(theta,np.array([.7,-1.3]))
        k=g.shape[0]*2
        for j in range(g.shape[-1]):
            m=g[...,j].reshape(k,k)
            np.testing.assert_allclose(m,m.conj().T,atol=1e-10)
            assert np.min(np.linalg.eigvalsh(.5*(m+m.conj().T)))>-1e-10
    try:
        sky_only._prepare(theta,np.array([2.5,0.]))
    except ValueError:
        pass
    else:
        raise AssertionError('a sky point outside the Gram sky anchors must be refused')


def test_merge_sources_whole_block_radius_and_branch_separation():
    from pycbc.filter.sky_block_search import _merge_sources
    sky=np.array([[0.,0.],[2.,0.],[6.,0.],[-6.,2.],[0.,2.]])
    delays=np.array([0.,100.,200000.])
    # rows: template, sky point, lag, 2F, spare
    surv=np.array([[0,0,0,100.,0],[1,1,1,90.,0],[0,2,0,70.,0],[0,3,1,60.,0],[0,4,0,50.,0],[0,0,2,40.,0]])
    # default: two sky steps of 2 -> the sidelobes at 6 and -6 are separate sources
    default=_merge_sources(surv,sky,delays,2.,86400.,2)
    assert [len(k) for k in default]==[2,1,1,1]
    # whole block: one source per day, the far-in-time row stays its own
    whole=_merge_sources(surv,sky,delays,2.,86400.,2,merge_sky=np.inf)
    assert [[x[1] for x in k] for k in whole]==[[0,1],[0]]
    # separation of 1.5 steps (3 units): the second kept row must be far from the first
    sep=_merge_sources(surv,sky,delays,2.,86400.,3,merge_sky=np.inf,separation=1.5)
    assert [[x[1] for x in k] for k in sep]==[[0,2,3],[0]]


def test_coordinates_clip_below_zero_eccentricity_only_when_asked():
    import pytest
    from pycbc.filter.eccentric_block_search import MassEccentricCoordinates
    c=MassEccentricCoordinates(30.,.005,[1e-6,1e-8])
    s=np.array([[0.,-5000.]])
    with pytest.raises(ValueError):
        c.decode(s)
    clipped=MassEccentricCoordinates(30.,.005,[1e-6,1e-8],clip_at_zero=True).decode(s)
    assert clipped[0,1]==0.
    np.testing.assert_allclose(clipped[0,0],30.)


def test_clip_projects_onto_the_circular_edge_in_the_metric():
    from pycbc.filter.eccentric_block_search import MassEccentricCoordinates
    g=np.array([[4.,1.9],[1.9,1.]])
    c=MassEccentricCoordinates(30.,.005,[1e-6,1e-8],clip_at_zero=True,metric=g)
    edge=-.005**2/1e-8
    s=np.array([[3.,edge-50.]])
    x=c.encode(c.decode(s))[0]
    np.testing.assert_allclose(x[1],edge,atol=1e-6)
    # the metric-nearest point of the edge: the gradient along the edge vanishes
    d=x-s[0]
    assert abs((g@d)[0])<1e-6*np.abs(g@d).max()
    # non-expansive: closer in the metric to a physical point than the unprojected node
    p=np.array([2.,edge+30.])
    dist=lambda a,b:(a-b)@g@(a-b)
    assert dist(x,p)<=dist(s[0],p)


def test_three_dimensional_coordinates_roundtrip_and_project_in_the_metric():
    from pycbc.filter.eccentric_block_search import MassEccentricCoordinates
    g=np.array([[4.,1.9,.5],[1.9,1.,.3],[.5,.3,2.]])
    c=MassEccentricCoordinates(30.,.005,[1e-6,1e-8,1e-2],clip_at_zero=True,metric=g,spin=.1)
    theta=np.array([[30.01,.006,.13]])
    np.testing.assert_allclose(c.decode(c.encode(theta)),theta)
    edge=-.005**2/1e-8
    s=np.array([3.,edge-50.,-2.])
    x=c.encode(c.decode(s[None]))[0]
    np.testing.assert_allclose(x[1],edge,atol=1e-6)
    grad=g@(x-s)
    assert abs(grad[0])<1e-6*abs(grad).max() and abs(grad[2])<1e-6*abs(grad).max()


def test_chi_interpolated_gram_is_the_anchor_gram_at_anchors_and_keeps_e_at_the_reference():
    n=128
    grid=SimpleNamespace(size=n,duration=ORBITAL_PERIOD,delta_t=ORBITAL_PERIOD/n,
        frequencies=.01+np.arange(n)/ORBITAL_PERIOD,harmonic_shift=1)
    coordinates=MassEccentricCoordinates(30.,.05,[.01,.001,.01])
    rng=np.random.default_rng(29)
    carrier=rng.normal(size=(2,n))+1j*rng.normal(size=(2,n))
    carrier[:,:4]=0;carrier[:,-4:]=0
    def source(theta):
        s=coordinates.encode(theta)[0]
        x=np.linspace(-1,1,n)
        phase=s[0]*x+s[1]*x**2+s[2]*x**3
        return carrier*np.exp(1j*phase)*(1+.1*theta[2]),np.zeros((2,n))
    response=np.array([[[1.,.3j],[.2+.4j,1.]]],complex)[...,None]*np.ones(n)
    data=(rng.normal(size=(2,n))+1j*rng.normal(size=(2,n)))
    ref=dict(eclipticlongitude=1.,eclipticlatitude=.2)
    scales=dict(eclipticlongitude=.01,eclipticlatitude=.01)
    def build(**kw):
        return EccentricFFTBlockModel(grid,data,source,lambda sky:response,
            grid.frequencies,[0],[0,1],coordinates,np.eye(3),ref,scales,
            (20,70),(-5*grid.delta_t,5*grid.delta_t),dtype=np.complex128,n_l0=32,**kw)
    mc=build(gram_mc_anchors=[-.01,0.,.01])
    chi=build(gram_mc_anchors=[-.01,0.,.01],gram_chi_anchors=[-.1,0.,.1])
    sky=np.zeros(2)
    # at a spin anchor the Gram is the ln-Mc-interpolated one at that spin, with e at the cell's reference
    theta=np.array([30.*np.exp(.004),.052,.1])
    _,_,a=mc._prepare(theta,sky)
    _,_,b=chi._prepare(theta,sky)
    np.testing.assert_allclose(b,a,rtol=1e-12,atol=1e-12)
    # a fresh model, so no cache is shared: the anchors sit at the cell's e whatever the template's
    ref_e=build(gram_mc_anchors=[-.01,0.,.01])._prepare(np.array([30.*np.exp(.004),.05,.1]),sky)[2]
    np.testing.assert_allclose(a,ref_e,rtol=1e-12,atol=1e-12)
    # between spin anchors: convex in chi
    t=np.array([30.*np.exp(.004),.05,.03])
    g=chi._prepare(t,sky)[2]
    g0=mc._prepare(np.array([30.*np.exp(.004),.05,0.]),sky)[2]
    g1=mc._prepare(np.array([30.*np.exp(.004),.05,.1]),sky)[2]
    np.testing.assert_allclose(g,.7*g0+.3*g1,rtol=1e-10,atol=1e-12)
    try:
        chi._prepare(np.array([30.,.05,.2]),sky)
    except ValueError:
        pass
    else:
        raise AssertionError('a template outside the spin anchors must be refused')
