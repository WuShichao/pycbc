"""The galactic-binary source and its ``tdi_source = gb`` registration."""
import numpy as np
import pytest

from pycbc.tdi.sources import GalacticBinarySource

DAY = 86400.0
GB = dict(amplitude=2e-22, frequency=3.1e-3, fdot=2e-17, inclination=0.9,
          polarization=0.6, initial_phase=1.3)


def test_polarizations_follow_ldc_gb_fdot():
    """h_+ = -A(1+cos^2 i) cos Phi, h_x = -2A cos i sin Phi at psi = 0, and
    psi rotates the pair by 2 psi."""
    t = np.linspace(0.0, 3 * DAY, 101)
    a, f, fd, inc, phi0 = 2e-22, 3.1e-3, 2e-17, 0.9, 1.3
    phase = -phi0 + np.pi * (2 * f * t + fd * t ** 2)
    plain = GalacticBinarySource(a, f, fd, inc, 0.0, phi0, 0.0, 3 * DAY)
    hp, hc = plain.polarizations(t)
    assert np.allclose(hp, -a * (1 + np.cos(inc) ** 2) * np.cos(phase),
                       rtol=0, atol=1e-12 * a)
    assert np.allclose(hc, -2 * a * np.cos(inc) * np.sin(phase),
                       rtol=0, atol=1e-12 * a)
    psi = 0.6
    rotated = GalacticBinarySource(a, f, fd, inc, psi, phi0, 0.0, 3 * DAY)
    rp, rc = rotated.polarizations(t)
    c, s = np.cos(2 * psi), np.sin(2 * psi)
    assert np.allclose(rp, c * hp - s * hc, rtol=0, atol=1e-12 * a)
    assert np.allclose(rc, s * hp + c * hc, rtol=0, atol=1e-12 * a)


def test_gb_is_a_registered_tdi_source():
    from pycbc.tdi.inference import _SOURCES
    source = _SOURCES['gb'](t_obs_start=10 * DAY, t_obs_end=20 * DAY, **GB)
    assert isinstance(source, GalacticBinarySource)
    assert source.support('gb') == (9 * DAY, 21 * DAY)
    padded = _SOURCES['gb'](t_obs_start=10 * DAY, t_obs_end=20 * DAY,
                            tdi_support_padding=600.0, **GB)
    assert padded.support('gb') == (10 * DAY - 600.0, 20 * DAY + 600.0)


def test_time_domain_injection_is_the_direct_projection():
    """``TDISparse`` with ``tdi_source = gb`` gives, at the strain's sample
    times, what projecting the source directly gives."""
    from pycbc.coordinates.space_orbit import LisaEqualArmOrbit
    from pycbc.tdi import delay_padding, sparse_tdi_response
    from pycbc.tdi.inference import channel_terms, clear_cache
    from pycbc.waveform import get_td_det_waveform_sequence

    clear_cache()
    span, dt = 4 * DAY, 10.0
    lam, beta = 4.4, -0.2
    times = np.arange(int(span / dt)) * dt
    out = get_td_det_waveform_sequence(
        approximant='TDISparse', tdi_source='gb', sample_times=times,
        ifos=('LISA_A', 'LISA_E'), tdi_band_edges=[0.0030, 0.0032],
        tdi_delta_t=dt, eclipticlongitude=lam, eclipticlatitude=beta, **GB)
    orbit = LisaEqualArmOrbit(armlength=2.5e9, t0=0.0)
    terms = channel_terms(('A', 'E'), generation=2, delta_t=dt)
    pad = delay_padding(orbit, np.array([0.0, span]),
                        tuple(t for g in terms.values() for t in g))
    source = GalacticBinarySource(GB['amplitude'], GB['frequency'],
                                  GB['fdot'], GB['inclination'],
                                  GB['polarization'], GB['initial_phase'],
                                  -DAY, span + DAY)
    direct = sparse_tdi_response(source, orbit, terms, lam, beta,
                                 t_start=0.0, t_end=span, initial_step=DAY,
                                 relative_tolerance=1e-4,
                                 support_padding=pad, delay_expansion=2,
                                 reference_delay=True, threads=1)
    ref = direct.sample(times, channels=('A', 'E'))
    for name in ('A', 'E'):
        x = np.asarray(out['LISA_' + name])
        r = np.real(ref[name])
        assert np.max(np.abs(x - r)) <= 1e-3 * np.max(np.abs(r))
        overlap = abs(np.dot(x, r)) / np.sqrt(np.dot(x, x) * np.dot(r, r))
        assert overlap == pytest.approx(1.0, abs=1e-8)
