"""Padding is removed as whole samples, so it has to be whole samples."""
import argparse

import pycbc.strain


def _options(rate, pad, end):
    parser = argparse.ArgumentParser()
    pycbc.strain.insert_strain_option_group(parser)
    return parser.parse_args([
        '--gps-start-time', '0', '--gps-end-time', str(end),
        '--sample-rate', str(rate), '--fake-strain', 'zeroNoise',
        '--fake-strain-flow', '1e-4', '--channel-name', 'LISA_A:STRAIN',
        '--pad-data', str(pad)])


def test_low_rate_padding_lands_on_the_start_time():
    """At 0.2 Hz the default 8 s is 1.6 samples; the series used to start
    3 s early."""
    opt = _options(0.2, 8, 2000)
    strain = pycbc.strain.from_cli(opt)
    assert opt.pad_data == 10
    assert float(strain.start_time) == 0.0
    assert float(strain.end_time) == 2000.0


def test_whole_sample_padding_is_untouched():
    opt = _options(256, 8, 64)
    strain = pycbc.strain.from_cli(opt)
    assert opt.pad_data == 8
    assert float(strain.start_time) == 0.0
    assert float(strain.end_time) == 64.0
