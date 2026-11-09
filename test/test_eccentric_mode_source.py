"""The eccentric native-FD bridge preserves mode clocks, phase and support."""
import numpy as np
import pytest

from pycbc.waveform.eccentric_mode_family import EccentricModeFamily
from pycbc.tdi.eccentric_mode_source import EccentricModeFamilySource


def family():
    f = np.linspace(.01,.02,61)
    def generator(p):
        out = {}
        for n in (1,2,3):
            phase = 2*np.pi*(n*1e5*f + 2e7*f**2)
            amp = np.array([1+.5j*f, .3-.7j+0*f]) * n
            have = np.ones(len(f),bool)
            out[2,2,n] = amp,phase,have
        return out
    ref = dict(mass1=36.,mass2=31.,eccentricity=.05)
    return EccentricModeFamily(ref,f,.015,step_mass=0,step_e=0,step_q=0,
                               generator=generator, circular_phase_basis='taylorf2')


def test_translation_and_anomaly_preserve_native_contract():
    fam = family()
    original = EccentricModeFamilySource(fam,clock_time=100.)
    shifted = EccentricModeFamilySource(fam,clock_time=100.,time_shift=723.,mean_anomaly=.4)
    f = np.linspace(.011,.019,47)
    a,b = original.frequency_harmonics(f),shifted.frequency_harmonics(f)
    for key in a:
        np.testing.assert_allclose(b[key]['time']-a[key]['time'],723.,atol=1e-8)
        ratio = np.exp(-2j*np.pi*f*723. + 1j*(key[2]-key[1])*.4)
        for pol in ('plus','cross'):
            np.testing.assert_allclose(b[key][pol],a[key][pol]*ratio,atol=1e-9)
    anchor = original.frequency_harmonics([fam.f_ref])[2,2,2]['time'][0]
    assert anchor == pytest.approx(100.,abs=1e-7)
    # The distinct modes must not be assigned the dominant harmonic clock.
    np.testing.assert_allclose(a[2,2,3]['time']-a[2,2,2]['time'],1e5,atol=1e-6)
    got = shifted.frequency_polarizations(f)
    np.testing.assert_allclose(got,[sum(r[p] for r in b.values()) for p in ('plus','cross')])


def test_spa_observation_window_clips_each_mode_separately():
    fam = family()
    source = EccentricModeFamilySource(fam,time_window=(-1e4,1e4))
    records = source.frequency_harmonics(np.linspace(.01,.02,301))
    assert records
    for item in records.values():
        assert np.all((item['time'] >= -1e4) & (item['time'] < 1e4))


@pytest.mark.parametrize('kwargs',[dict(clock_frequency=.03),dict(time_window=(1,0)),
                                   dict(mean_anomaly=np.nan)])
def test_invalid_anchor_and_coordinates_are_refused(kwargs):
    with pytest.raises(ValueError):
        EccentricModeFamilySource(family(),**kwargs)


def test_source_projects_through_existing_tdi_frequency_interface():
    from pycbc.coordinates.space_orbit import LisaEqualArmOrbit
    from pycbc.tdi.inference import channel_terms
    from pycbc.tdi.onthefly import project_frequency_tdi
    source = EccentricModeFamilySource(family())
    f = np.linspace(.011,.019,13)
    orbit, terms = LisaEqualArmOrbit(),channel_terms(('A','E'),generation=2,delta_t=5.)
    full = project_frequency_tdi(source,orbit,terms,f,1.,.3,chunk_size=13)
    chunked = project_frequency_tdi(source,orbit,terms,f,1.,.3,chunk_size=4)
    for key in full:
        assert np.isfinite(full[key]).all()
        np.testing.assert_allclose(chunked[key],full[key],rtol=1e-8,atol=1e-10)


@pytest.mark.parametrize('d,dq', [(.01, 0.), (0., .01), (-.01, -.01)])
def test_default_online_source_never_calls_native_generator(d, dq):
    fam = family()
    def forbidden(*args, **kwargs):
        raise AssertionError("online source must use the analytic PN ratio")
    fam.generator = fam._circ_generator = forbidden
    source = EccentricModeFamilySource(fam, d=d, dq=dq)
    # This artificial family has no physical mass-dependent curvature.
    # Test the native-call guard at the clock anchor; physical off-reference
    # clocks require independent native waveform validation.
    records = source.frequency_harmonics([fam.f_ref])
    assert records
    for record in records.values():
        assert np.isfinite(record['plus']).all()


def test_unknown_circular_method_is_refused():
    with pytest.raises(ValueError, match='circular_method'):
        EccentricModeFamilySource(family(), circular_method='typo')


def test_mismatched_online_and_construction_bases_are_refused():
    fam = family()
    fam.circular_phase_basis = 'pyefpe'
    with pytest.raises(ValueError, match='match the cell phase basis'):
        EccentricModeFamilySource(fam, d=.001)
