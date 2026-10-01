"""The inverse-SPA clock must not amplify native phase roundoff."""
import numpy as np
import pytest

from pycbc.tdi.sources import LALFDSource


def _parameters():
    return dict(mass1=35.0, mass2=30.0, f_lower=0.015, f_upper=0.019,
                spin1z=0.1, spin2z=-0.05, distance=100.0,
                inclination=0.8, coa_phase=0.2)


@pytest.mark.parametrize("approximant", ["IMRPhenomD", "IMRPhenomXAS",
                                         "TaylorF2"])
def test_tiny_mass_changes_do_not_jump_the_source_clock(approximant):
    pytest.importorskip("lalsimulation")
    params = _parameters()
    source = LALFDSource(approximant=approximant, **params)
    times = np.linspace(0.1 * source.t_end, 0.9 * source.t_end, 301)
    phase = source.carrier_phase(2, times)
    for mass1 in (np.nextafter(params["mass1"], 0.0),
                  params["mass1"] + 1e-12,
                  params["mass1"] + 1e-10):
        candidate = LALFDSource(approximant=approximant,
                                **dict(params, mass1=mass1))
        assert abs(candidate.stationary_times[0]
                   - source.stationary_times[0]) < 0.005
        assert abs(candidate.t_end - source.t_end) < 0.005
        assert np.max(np.abs(candidate.carrier_phase(2, times)
                             - phase)) < 0.001


def test_stationary_time_keeps_the_physical_mass_derivative():
    pytest.importorskip("lalsimulation")
    params = _parameters()
    derivatives = []
    for step in (1e-5, 1e-4):
        low = LALFDSource(**dict(params, mass1=params["mass1"] - step))
        high = LALFDSource(**dict(params, mass1=params["mass1"] + step))
        derivatives.append((high.stationary_times - low.stationary_times)
                           / (2 * step))
    np.testing.assert_allclose(derivatives[0], derivatives[1], rtol=1e-4)
    # An independently generated candidate must retain its own time origin;
    # neither rounding parameters nor sharing the fiducial clock is allowed.
    assert np.min(np.abs(derivatives[1])) > 1e6


def test_local_fit_measures_the_waveform_not_the_newtonian_surrogate():
    # A quadratic FD phase has an analytic stationary time and curvature.
    # Its large linear slope represents an inspiral decades from merger.
    source = object.__new__(LALFDSource)
    source._parameters = dict(mass1=35.0, mass2=30.0)
    centre = 0.015
    origin, rate = -1e8, 1e10

    def waveform(frequency):
        dx = np.asarray(frequency) - centre
        phase = -2 * np.pi * (origin * dx + 0.5 * rate * dx ** 2)
        strain = np.exp(1j * phase)
        return strain, -1j * strain

    source._fd_waveform = waveform
    frequency = np.array([centre, centre + 1e-5])
    stationary, curvature = source._local_phase_derivatives(frequency)
    np.testing.assert_allclose(stationary, origin + rate * (frequency - centre),
                               rtol=0, atol=0.005)
    np.testing.assert_allclose(curvature, rate, rtol=2e-6)
