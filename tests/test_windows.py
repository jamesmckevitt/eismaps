"""Tests of window title parsing and the choice of fit."""

import numpy as np
import pytest

from eismaps.windows import choose_fit, parse_title


@pytest.mark.parametrize('title, ions, wavelength', [
    ('Fe XII 195.120', [('FE', 12)], 195.12),
    ('FE XII 195.120', [('FE', 12)], 195.12),
    ('S X 264.220', [('S', 10)], 264.22),
    ('FeXIV MgVI 270.510', [('FE', 14), ('MG', 6)], 270.51),
    ('Fe XXIII/SX 264.010', [('FE', 23), ('S', 10)], 264.01),
    ('FEVIII XII 186.850', [('FE', 8), ('FE', 12)], 186.85),
    ('Fe IX_XI 188.350', [('FE', 9), ('FE', 11)], 188.35),
    ('OVI FEX 184.020', [('O', 6), ('FE', 10)], 184.02),
    ('SI_XI DBLD 191.270', [('SI', 11)], 191.27),
    ('CCDBSHRT 0 177.540', [], 177.54),
    ('ccdbsmod 0 181.750', [], 181.75),
    ('LINE 3  ', [], None),
    ('Line 10 ', [], None),
])
def test_parse_title(title, ions, wavelength):
    assert parse_title(title) == (ions, wavelength)


def spectrum(lines, low, high, background=100.0, step=0.0223):
    wave = np.arange(low, high, step)
    inten = np.full_like(wave, background)
    for peak, centre, width in lines:
        inten = inten + peak * np.exp(-0.5 * ((wave - centre) / width) ** 2)
    return wave, inten, np.sqrt(inten) / 10


def test_title_ion_is_fitted_not_the_first_template():
    wave, inten, errs = spectrum([(1000.0, 203.826, 0.03), (300.0, 203.72, 0.03)], 203.55, 204.06)
    fit = choose_fit('FE XIII 203.820', 203.55, 204.06, wave, inten, errs)
    assert fit.line == 'Fe XIII 203.826'


def test_single_gaussian_is_preferred():
    wave, inten, errs = spectrum([(5000.0, 195.119, 0.03)], 194.86, 195.37)
    fit = choose_fit('Fe XII 195.120', 194.86, 195.37, wave, inten, errs)
    assert fit.single and int(fit.template['n_gauss']) == 1


def test_neighbouring_line_forces_more_components():
    # Fe XI 188.299 sits 0.08 A from the stronger 188.216, inside a single Gaussian's fit range.
    wave, inten, errs = spectrum([(5000.0, 188.216, 0.03), (1500.0, 188.299, 0.03)], 187.99, 188.69)
    fit = choose_fit('Fe XI 188.300', 187.99, 188.69, wave, inten, errs)
    assert fit.line == 'Fe XI 188.299'
    assert not fit.single
    assert int(fit.template['n_gauss']) >= 2


def test_full_ccd_window_is_not_fitted():
    wave = 170.0 + 0.0223 * np.arange(1024)
    fit = choose_fit('CCDBSHRT 0 177.540', wave[0], wave[-1], wave, np.ones(1024), np.ones(1024))
    assert fit is None


def test_unnamed_window_uses_its_strongest_line():
    wave, inten, errs = spectrum([(3000.0, 195.119, 0.03)], 194.86, 195.37)
    fit = choose_fit('LINE 3  ', 194.86, 195.37, wave, inten, errs)
    assert fit.line == 'Fe XII 195.119'


def test_ion_with_no_template_gets_a_generic_fit():
    wave, inten, errs = spectrum([(3000.0, 278.402, 0.03)], 278.14, 278.66)
    fit = choose_fit('Mg VII 278.400', 278.14, 278.66, wave, inten, errs)
    assert fit.source == 'generic'
    assert fit.line == 'Mg VII 278.400'
    assert fit.template['wmin'] < 278.402 < fit.template['wmax']


def test_generic_fit_of_another_line_is_not_named_after_the_title():
    wave, inten, errs = spectrum([(3000.0, 278.62, 0.03)], 278.14, 278.70)
    fit = choose_fit('Mg VII 278.400', 278.14, 278.70, wave, inten, errs)
    assert fit.line.startswith('Line at 278.6')


def test_empty_window_has_no_fit():
    wave, inten, errs = spectrum([], 278.14, 278.66)
    assert choose_fit('LINE 1  ', 278.14, 278.66, wave, inten, errs) is None


def test_wide_window_named_after_a_line_is_fitted():
    # A 272-pixel window of the slit_hefe_fast3 study, with Fe XII 195.119 among other lines.
    wave, inten, errs = spectrum([(1000.0, 195.119, 0.03), (2000.0, 193.509, 0.03), (800.0, 192.394, 0.03)],
                                 191.82, 197.88)
    assert wave.size == 272
    fit = choose_fit('Fe XII 195.120', wave[0], wave[-1], wave, inten, errs)
    assert fit.line == 'Fe XII 195.119'


def test_wide_window_of_an_ion_with_no_template_is_not_fitted():
    # S X has no template near 180 A, and the strongest line, Fe XI 180.401, belongs to another ion.
    wave, inten, errs = spectrum([(5000.0, 180.401, 0.03), (1000.0, 188.216, 0.03)], 174.5, 190.8)
    assert choose_fit('S X 180.730', wave[0], wave[-1], wave, inten, errs) is None
