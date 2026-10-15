"""Injections from generators that carry the detector response in time."""
import numpy as np
import pytest

from pycbc.inject.inject import CBCHDFInjectionSet
from pycbc.io import FieldArray
from pycbc.types import TimeSeries
from pycbc.waveform import get_td_det_waveform_sequence, td_det_sequence
from pycbc.waveform.plugin import add_custom_waveform

APPROXIMANT = 'TestTimeDomainDetector'


def _channels(**params):
    """A different smooth signal per channel, so a swapped key would show."""
    names = params['ifos']
    if isinstance(names, str):
        names = (names,)
    t = np.asarray(params['sample_times'], dtype=float) - float(params['tc'])
    out = {}
    for k, name in enumerate(names):
        offset = {'A': 0.0, 'E': 0.3, 'T': 0.7}[name.rsplit('_', 1)[-1]]
        out[name] = 1e-21 * np.cos(2 * np.pi * 3e-3 * t + offset) \
            * np.exp(-(t / 4e3) ** 2)
    return out


_channels.required = ('approximant', 'sample_times', 'ifos', 'tc')


@pytest.fixture(autouse=True)
def registered():
    add_custom_waveform(APPROXIMANT, _channels, 'time', sequence=True,
                        has_det_response=True, force=True)
    yield
    td_det_sequence.pop(APPROXIMANT, None)


def test_registration_lands_in_the_time_domain_detector_registry():
    from pycbc.waveform.waveform import cpu_td
    assert td_det_sequence[APPROXIMANT] is _channels
    assert APPROXIMANT not in cpu_td
    times = np.arange(10) * 5.0
    out = get_td_det_waveform_sequence(approximant=APPROXIMANT,
                                       sample_times=times, ifos='LISA_E',
                                       tc=0.0)
    assert np.allclose(out['LISA_E'], _channels(sample_times=times,
                                               ifos='LISA_E', tc=0.0)['LISA_E'])


def test_missing_sample_times_is_refused():
    with pytest.raises(ValueError, match='sample_times'):
        get_td_det_waveform_sequence(approximant=APPROXIMANT, ifos='LISA_A',
                                     tc=0.0)


def test_injection_is_the_generator_on_the_strain_times(tmp_path):
    """The strain receives exactly the generator's channel at its own sample
    times: no transform in between, so nothing wraps and nothing is padded."""
    tc = 1e7 + 6e3
    samples = FieldArray.from_kwargs(tc=np.array([tc]), f_lower=np.array([1e-4]),
                                     mass1=np.array([30.0]),
                                     mass2=np.array([25.0]))
    path = str(tmp_path / 'inj.hdf')
    CBCHDFInjectionSet.write(path, samples,
                             static_args={'approximant': APPROXIMANT})
    injections = CBCHDFInjectionSet(path)
    for name in ('LISA_A', 'LISA_E'):
        strain = TimeSeries(np.zeros(4096), delta_t=5.0, epoch=1e7)
        injections.apply(strain, name)
        expected = _channels(sample_times=np.asarray(strain.sample_times),
                             ifos=name, tc=tc)[name]
        assert np.max(np.abs(strain.numpy() - expected)) <= \
            1e-12 * np.max(np.abs(expected))


def test_sparse_tdi_is_known_without_importing_its_backend():
    """Tools that never call `register` still know TDISparse, and asking for
    it is what imports the module."""
    import subprocess
    import sys
    code = ("import sys; from pycbc.waveform import fd_det_sequence, "
            "td_det_sequence; "
            "assert 'TDISparse' in fd_det_sequence; "
            "assert 'TDISparse' in td_det_sequence; "
            "assert 'pycbc.tdi.inference' not in sys.modules")
    subprocess.run([sys.executable, '-c', code], check=True)


def test_register_replaces_the_lazy_entry():
    from pycbc.tdi.inference import (register, sparse_tdi_fd_det_sequence,
                                     sparse_tdi_td_det_sequence)
    from pycbc.waveform.waveform import fd_det_sequence, td_det_sequence
    saved = fd_det_sequence['TDISparse'], td_det_sequence['TDISparse']
    try:
        register()
        assert fd_det_sequence['TDISparse'] is sparse_tdi_fd_det_sequence
        assert td_det_sequence['TDISparse'] is sparse_tdi_td_det_sequence
    finally:
        fd_det_sequence['TDISparse'], td_det_sequence['TDISparse'] = saved
