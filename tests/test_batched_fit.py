"""Tests of the batched fit on synthetic spectra."""

import numpy as np
import pytest

from eismaps.batched_fit import fit_spectra


def template_and_parinfo(n_gauss=1, centre=195.119, wmin=194.9, wmax=195.35):
    fit = []
    parinfo = []
    for n in range(n_gauss):
        fit += [1000.0, centre + 0.1 * n, 0.03]
        parinfo += [{'limited': [1, 0], 'limits': [0.0, 0.0], 'tied': '', 'fixed': 0},
                    {'limited': [1, 1], 'limits': [centre + 0.1 * n - 0.1, centre + 0.1 * n + 0.1], 'tied': '', 'fixed': 0},
                    {'limited': [1, 1], 'limits': [0.019, 0.051], 'tied': '', 'fixed': 0}]
    fit.append(50.0)
    parinfo.append({'limited': [0, 0], 'limits': [0.0, 0.0], 'tied': '', 'fixed': 0})
    template = {'fit': np.array(fit), 'n_gauss': n_gauss, 'n_poly': 1, 'wmin': wmin, 'wmax': wmax}
    return template, parinfo


def spectra(peaks, centres, widths, background=50.0, noise=0.0, seed=1, centre=195.119):
    wave = np.tile(centre + 0.0223 * np.arange(-10, 11), (len(peaks), 1))
    inten = background + np.asarray(peaks)[:, None] * np.exp(
        -0.5 * ((wave - np.asarray(centres)[:, None]) / np.asarray(widths)[:, None]) ** 2)
    if noise:
        inten = inten + np.random.default_rng(seed).normal(scale=noise, size=inten.shape)
    return wave, inten, np.full_like(wave, noise if noise else 1.0)


def test_recovers_noise_free_line():
    template, parinfo = template_and_parinfo()
    peaks, centres, widths = [800.0, 2000.0], [195.10, 195.14], [0.028, 0.035]
    wave, inten, errs = spectra(peaks, centres, widths)
    result = fit_spectra(wave, inten, errs, template, parinfo, tolerance=1e-10)
    assert (result['status'] > 0).all()
    assert np.allclose(result['params'][:, 0], peaks, rtol=1e-4)
    assert np.allclose(result['params'][:, 1], centres, atol=1e-5)
    assert np.allclose(result['params'][:, 2], widths, rtol=1e-4)


def test_errors_match_the_scatter_of_noisy_fits():
    # With unit-weighted noise the formal error should match the scatter of many fits to within 10%.
    template, parinfo = template_and_parinfo()
    n = 3000
    wave, inten, errs = spectra([800.0] * n, [195.119] * n, [0.03] * n, noise=20.0)
    result = fit_spectra(wave, inten, errs, template, parinfo)
    good = result['status'] > 0
    assert good.mean() > 0.99
    for index in (0, 1, 2):
        scatter = result['params'][good, index].std()
        assert np.median(result['perror'][good, index]) == pytest.approx(scatter, rel=0.1)
    assert np.median(result['chi2'][good]) == pytest.approx(1.0, abs=0.1)


def test_too_few_points_are_not_fitted():
    template, parinfo = template_and_parinfo()
    wave, inten, errs = spectra([800.0], [195.119], [0.03])
    errs[:, 5:] = 0.0
    result = fit_spectra(wave, inten, errs, template, parinfo)
    assert result['status'][0] == -1
    assert (result['params'][0] == 0).all()


def test_no_line_gives_zero_peak():
    template, parinfo = template_and_parinfo()
    wave, inten, errs = spectra([0.0], [195.119], [0.03], noise=1e-3)
    inten -= 1.0 * np.exp(-0.5 * ((wave - 195.119) / 0.03) ** 2)
    result = fit_spectra(wave, inten, errs, template, parinfo)
    assert result['params'][0, 0] == 0


def test_line_off_the_guess_is_not_lost():
    # The data at the template centroid is at background level, so the start peak is zero,
    # and a fit that stays at a peak of zero would lose the line 0.07 A away.
    template, parinfo = template_and_parinfo()
    wave, inten, errs = spectra([500.0], [195.189], [0.022], noise=5.0)
    result = fit_spectra(wave, inten, errs, template, parinfo)
    assert result['params'][0, 0] == pytest.approx(500.0, rel=0.1)
    assert result['params'][0, 1] == pytest.approx(195.189, abs=0.005)


def test_tied_widths_stay_equal():
    template, parinfo = template_and_parinfo(n_gauss=2)
    parinfo[5]['tied'] = 'p[2]'
    wave = np.tile(195.119 + 0.0223 * np.arange(-12, 16), (1, 1))
    inten = (40.0 + 900.0 * np.exp(-0.5 * ((wave - 195.119) / 0.03) ** 2)
             + 400.0 * np.exp(-0.5 * ((wave - 195.219) / 0.03) ** 2))
    template['wmax'] = 195.5
    result = fit_spectra(wave, inten, np.ones_like(wave), template, parinfo, tolerance=1e-10)
    params = result['params'][0]
    assert params[2] == params[5]
    assert params[2] == pytest.approx(0.03, rel=1e-3)
    assert result['perror'][0, 5] == 0


def test_blocks_do_not_change_the_result():
    template, parinfo = template_and_parinfo()
    n = 50
    wave, inten, errs = spectra([800.0] * n, [195.119] * n, [0.03] * n, noise=20.0)
    whole = fit_spectra(wave, inten, errs, template, parinfo)
    blocks = fit_spectra(wave, inten, errs, template, parinfo, block=7)
    assert np.array_equal(whole['params'], blocks['params'])
