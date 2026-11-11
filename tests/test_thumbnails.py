"""Tests of thumbnail names, exposure choice and rendering."""

import json
import os

import h5py
import numpy as np
import pytest
from PIL import Image

import eismaps.thumbnails as thumbnails

from eismaps.thumbnails import (MAP_AREA, _scale, fitted_maps, MAX_MAP_RATIO, _arcsec, _column_zero_point, _map_rectangle,
                                _one_exposure_copy, _exposure_choice, _exposure_factor, _is_done,
                                _axes, _count_rates, _detector_norm, _is_full_ccd, _process, _record_path, _slit_width,
                                _summed_intensity, make_thumbnails, process_raster, render_full_ccd, render_thumbnail, RasterUnavailable, thumbnail_path)


def test_path_follows_the_archive():
    path = thumbnail_path('root', '20230429_105920', 8, 'Fe XII 195.120', 'vel')
    assert path == ('root/2023/04/29/eis_l0_20230429_105920.fits/'
                    'eis_l0_20230429_105920.fits_line_08_Fe_XII_195.120.vel.gif')


def test_path_keeps_trailing_spaces_as_underscores():
    assert thumbnail_path('r', '20061122_224822', 2, 'LINE 2  ', 'int').endswith('_line_02_LINE_2__.int.gif')


def test_path_has_no_slash_from_the_title():
    assert thumbnail_path('r', '20251002_200728', 13, 'Fe XXIII/SX 264.010', 'int').endswith(
        '_line_13_Fe_XXIII_SX_264.010.int.gif')


def write_durations(tmp_path, durations):
    with h5py.File(tmp_path / 'eis_20090310_122513.head.h5', 'w') as head:
        head['exposure_times/duration'] = np.asarray(durations, dtype=float)
    return str(tmp_path / 'eis_20090310_122513.data.h5')


def test_one_exposure_needs_no_choice(tmp_path):
    assert _exposure_choice(write_durations(tmp_path, [[30.0] * 5])) == (None, 30.0, False)


def test_longest_exposure_is_chosen_and_the_median_is_the_reference(tmp_path):
    # The calibration in the header is for 50 s here, the median of 50, 5 and 655 s.
    assert _exposure_choice(write_durations(tmp_path, [[50.0] * 4, [5.0] * 4, [655.0] * 4])) == (2, 50.0, False)


def test_two_exposures_use_the_longer_as_reference(tmp_path):
    # The calibration in the header is for the longer of two exposures.
    assert _exposure_choice(write_durations(tmp_path, [[30.0] * 4, [90.0] * 4])) == (1, 90.0, False)


def test_no_durations_give_no_reference(tmp_path):
    assert _exposure_choice(write_durations(tmp_path, [[0.0] * 4]))[1] is None


def test_single_position_is_flagged(tmp_path):
    assert _exposure_choice(write_durations(tmp_path, [[30.0], [90.0]])) == (1, 90.0, True)


def test_one_exposure_copy_changes_only_the_copy(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    data = source / 'eis_20070315_111949.data.h5'
    with h5py.File(data, 'w') as f:
        f['x'] = [1]
    head = source / 'eis_20070315_111949.head.h5'
    with h5py.File(head, 'w') as f:
        f['index/nexp'] = [2]
    head.chmod(0o444)
    copy = tmp_path / 'copy'
    copy.mkdir()
    path = _one_exposure_copy(str(data), str(copy))
    assert path.endswith('.head.h5')
    with h5py.File(path, 'r') as f:
        assert f['index/nexp'][0] == 1
    with h5py.File(head, 'r') as f:
        assert f['index/nexp'][0] == 2
    assert (copy / 'eis_20070315_111949.data.h5').is_symlink()


class FakeCube:
    def __init__(self, duration):
        self.data = np.zeros((2, len(duration), 3))
        self.meta = {'duration': np.asarray(duration, dtype=float)}


def test_columns_are_scaled_for_their_own_exposure():
    # Automatic exposure control can shorten exposures part way through a raster.
    factor = _exposure_factor(FakeCube([60.0, 60.0, 10.0, 0.0]), 60.0)
    assert factor[:3].tolist() == [1.0, 1.0, 6.0]
    assert np.isnan(factor[3])


def test_summed_intensity_integrates_over_wavelength():
    wave = np.arange(10.0)[None, None, :] * 0.0223 + np.zeros((2, 3, 1))
    inten = np.full((2, 3, 10), 5.0)
    errs = np.ones_like(inten)
    errs[1, 2, :] = 0.0
    summed = _summed_intensity(wave, inten, errs)
    assert np.allclose(summed[0, 0], 5.0 * 10 * 0.0223)
    assert np.isnan(summed[1, 2])


def test_summed_intensity_allows_for_missing_points():
    wave = np.arange(10.0)[None, None, :] * 0.0223 + np.zeros((1, 1, 1))
    inten = np.full((1, 1, 10), 5.0)
    errs = np.ones_like(inten)
    errs[..., :3] = 0.0
    assert np.allclose(_summed_intensity(wave, inten, errs), 5.0 * 10 * 0.0223)


def test_column_with_few_pixels_uses_its_neighbours():
    # One bright pixel in the middle column must keep its velocity.
    velocity = np.full((20, 3), np.nan)
    velocity[:, 0] = 2.0
    velocity[:, 2] = 4.0
    velocity[3, 1] = -50.0
    assert _column_zero_point(velocity).tolist() == [2.0, 3.0, 4.0]


def test_long_column_needs_more_than_a_few_pixels():
    # Five pixels of a flare kernel in a 512-pixel column are not a zero point.
    velocity = np.full((512, 3), np.nan)
    velocity[:, 0] = 2.0
    velocity[:, 2] = 4.0
    velocity[:5, 1] = -100.0
    assert _column_zero_point(velocity)[1] == 3.0


def test_short_window_still_removes_column_drift():
    velocity = np.array([[1.0, 5.0], [1.0, 5.0], [1.0, 5.0], [1.0, 5.0]])
    assert _column_zero_point(velocity).tolist() == [1.0, 5.0]


def test_arcsec_wraps_both_ways():
    # The WCS returned -359.9938 degrees for a raster crossing Solar-X = 0.
    assert np.allclose(_arcsec([-0.0102, -359.9938, 359.99, 0.01]), [-36.72, 22.32, -36.0, 36.0])


def test_single_row_keeps_differences_between_columns():
    velocity = np.array([[1.0, 2.0, 9.0]])
    assert _column_zero_point(velocity).tolist() == [2.0, 2.0, 2.0]


def test_blank_map_has_a_zero_point():
    assert _column_zero_point(np.full((4, 3), np.nan)).tolist() == [0.0, 0.0, 0.0]


def test_rasters_with_errors_are_not_done(tmp_path):
    root = str(tmp_path)
    timestamp = '20130116_093720'
    assert not _is_done(root, timestamp)
    path = _record_path(root, timestamp)
    os.makedirs(os.path.dirname(path))
    with open(path, 'w') as output:
        json.dump({'timestamp': timestamp, 'windows': [{'iwin': 0, 'fit_error': 'bad template'}]}, output)
    assert not _is_done(root, timestamp)
    with open(path, 'w') as output:
        json.dump({'timestamp': timestamp, 'windows': [{'iwin': 0}]}, output)
    assert _is_done(root, timestamp)


def test_failure_does_not_replace_a_good_record(tmp_path, monkeypatch):
    root = str(tmp_path / 'out')
    timestamp = '20130116_093720'
    path = _record_path(root, timestamp)
    os.makedirs(os.path.dirname(path))
    with open(path, 'w') as output:
        json.dump({'timestamp': timestamp, 'windows': []}, output)

    def unavailable(*args):
        raise RasterUnavailable('server down')

    monkeypatch.setattr('eismaps.thumbnails.fetch_raster', unavailable)
    record = process_raster(timestamp, root, str(tmp_path))
    assert record['error'] == 'server down'
    assert _is_done(root, timestamp)
    assert os.path.exists(path + '.error')


def test_one_failed_raster_does_not_stop_a_batch():
    # A name with no time raises in process_raster, and must come back as a record.
    record = _process(('nonsense', 'out', None, 'del_zanna_2025'))
    assert 'error' in record


def test_thumbnail_is_a_768_pixel_gif(tmp_path):
    image = np.random.default_rng(1).normal(size=(40, 30))
    image[5, 5] = np.nan
    path = tmp_path / 'a' / 'b.gif'
    render_thumbnail(image, (0, 10, 0, 20), ('x', 'y'), 'title', 'subtitle', 'vel', str(path))
    with Image.open(path) as gif:
        assert gif.format == 'GIF'
        assert gif.size == (768, 768)


def test_equal_scales_in_x_and_y():
    # In a square image, the sides of the map must have the ratio of the extent.
    left, bottom, width, height = _map_rectangle((0, 200, 0, 400), True)
    assert width / height == 0.5
    assert height == MAP_AREA[3]


def test_wide_map_fills_the_width():
    left, bottom, width, height = _map_rectangle((0, 400, 0, 100), True)
    assert width == MAP_AREA[2]
    assert width / height == 4.0


def test_very_narrow_map_is_stretched_to_the_limit():
    left, bottom, width, height = _map_rectangle((0, 10, 0, 512), True)
    assert width / height == 1 / MAX_MAP_RATIO


def test_very_wide_map_is_stretched_to_the_limit():
    left, bottom, width, height = _map_rectangle((0, 512, 0, 10), True)
    assert width / height == MAX_MAP_RATIO


def test_map_stays_in_its_area():
    for extent in ((0, 1, 0, 1), (0, 10, 0, 512), (0, 512, 0, 10), (0, 491, 0, 512)):
        left, bottom, width, height = _map_rectangle(extent, True)
        assert left >= MAP_AREA[0] and bottom >= MAP_AREA[1]
        assert left + width <= MAP_AREA[0] + MAP_AREA[2] + 1e-12
        assert bottom + height <= MAP_AREA[1] + MAP_AREA[3] + 1e-12


def test_degenerate_extent_fills_the_area():
    for extent in ((5, 5, 3, 3), (0, 10, 5, 5), (0, float('nan'), 0, 1)):
        assert _map_rectangle(extent, True) == MAP_AREA


def test_stretched_and_wide_maps_render(tmp_path):
    image = np.random.default_rng(1).normal(size=(10, 10))
    for name, extent in (('wide', (0, 512, 0, 10)), ('narrow', (0, 10, 0, 512))):
        path = tmp_path / f'{name}.gif'
        render_thumbnail(image, extent, ('x', 'y'), 't', 's', 'int', str(path), equal=True)
        assert path.exists()


def test_time_axis_fills_the_area():
    assert _map_rectangle((0, 8, -100, 200), False) == MAP_AREA


CENTROID_LIMITS = {'value': 195.119, 'limited': [1, 1], 'limits': [195.04, 195.2]}
WIDTH_LIMITS = {'limited': [1, 1], 'limits': [0.019, 0.05]}


class FakeFit:
    """Fit of `rows` spectra in one column, with a line at 195.13 A."""

    def __init__(self, rows=40):
        params = np.tile([100.0, 195.13, 0.03, 10.0], (rows, 1, 1))
        perror = np.tile([5.0, 0.002, 0.002, 1.0], (rows, 1, 1))
        self.fit = {'params': params, 'perror': perror, 'status': np.ones((rows, 1)), 'chi2': np.ones((rows, 1))}
        self.parinfo = [{}, dict(CENTROID_LIMITS), dict(WIDTH_LIMITS), {}]


def test_values_on_template_limits_are_shown():
    fit = FakeFit()
    fit.fit['params'][0, 0, 2] = 0.05
    fit.fit['perror'][0, 0, 2] = 0.0
    fit.fit['params'][1, 0, 1] = 195.2
    fit.fit['perror'][1, 0, 1] = 0.0
    velocity, width, velocity_off_limit, width_off_limit = fitted_maps(fit, 0)
    assert width[0, 0] == 0.05
    assert velocity[1, 0] == pytest.approx(107.6, abs=0.2)
    assert not width_off_limit[0, 0] and not velocity_off_limit[1, 0]


def test_weak_peak_is_blank():
    fit = FakeFit()
    fit.fit['params'][2, 0, 0] = 1.0
    velocity, width = fitted_maps(fit, 0)[:2]
    assert np.isnan(velocity[2, 0]) and np.isnan(width[2, 0])


def test_unconstrained_parameter_is_blank():
    # A zero error off the limits means the fit did not constrain the parameter.
    fit = FakeFit()
    fit.fit['perror'][3, 0, 1] = 0.0
    fit.fit['perror'][3, 0, 2] = 0.0
    velocity, width = fitted_maps(fit, 0)[:2]
    assert np.isnan(velocity[3, 0]) and np.isnan(width[3, 0])


def test_zero_point_ignores_pixels_on_a_limit():
    # Most of the column on the limit must not set its zero point.
    fit = FakeFit()
    fit.fit['params'][:25, 0, 1] = 195.2
    fit.fit['perror'][:25, 0, 1] = 0.0
    velocity = fitted_maps(fit, 0)[0]
    assert velocity[30, 0] == pytest.approx(0.0, abs=1e-6)
    assert velocity[0, 0] == pytest.approx((195.2 - 195.13) / 195.119 * 2.9979e5, rel=1e-3)


def test_tied_parameters_of_a_second_component():
    # The second Gaussian has its centroid and width tied to the first, as in fe_12_195_179.2c.
    rows = 40
    params = np.tile([100.0, 195.13, 0.03, 50.0, 195.19, 0.03, 10.0], (rows, 1, 1))
    perror = np.tile([5.0, 0.002, 0.002, 4.0, 0.0, 0.0, 1.0], (rows, 1, 1))
    fit = FakeFit()
    fit.fit['params'], fit.fit['perror'] = params, perror
    fit.parinfo = [{}, dict(CENTROID_LIMITS), dict(WIDTH_LIMITS), {},
                   {'value': 195.179, 'tied': 'p[1]+0.06'}, {'tied': 'p[2]'}, {}]
    velocity, width, velocity_off_limit, width_off_limit = fitted_maps(fit, 1)
    assert np.isfinite(velocity).all()
    assert (width == 0.03).all()
    assert velocity_off_limit.all() and width_off_limit.all()


def test_colour_scale_can_ignore_pixels(tmp_path):
    image = np.full((10, 10), 0.03)
    image[0, :] = 0.05
    assert _scale(image[image < 0.04], 'wid')[1].vmax < 0.04
    assert _scale(image, 'wid')[1].vmax > 0.04
    path = tmp_path / 'c.gif'
    render_thumbnail(image, (0, 1, 0, 1), ('x', 'y'), 't', 's', 'wid', str(path), scale_pixels=image < 0.04)
    assert path.exists()


def test_all_blank_map_still_renders(tmp_path):
    path = tmp_path / 'b.gif'
    render_thumbnail(np.full((4, 4), np.nan), (0, 1, 0, 1), ('x', 'y'), 't', 's', 'int', str(path))
    assert path.exists()


def test_peak_cut_can_be_lowered_or_removed():
    fit = FakeFit()
    fit.fit['params'][2, 0, 0] = 8.0
    fit.fit['params'][3, 0, 0] = 2.0
    assert np.isnan(fitted_maps(fit, 0)[0][2, 0])
    velocity = fitted_maps(fit, 0, min_peak_snr=1.5)[0]
    assert np.isfinite(velocity[2, 0]) and np.isnan(velocity[3, 0])
    assert np.isfinite(fitted_maps(fit, 0, min_peak_snr=0)[0][3, 0])


def test_good_fraction_can_be_lowered():
    wave = np.arange(10.0)[None, None, :] * 0.0223 + np.zeros((1, 1, 1))
    inten = np.full((1, 1, 10), 5.0)
    errs = np.ones_like(inten)
    errs[..., :7] = 0.0
    assert np.isnan(_summed_intensity(wave, inten, errs)[0, 0])
    assert np.allclose(_summed_intensity(wave, inten, errs, 0.25), 5.0 * 10 * 0.0223)
    errs[...] = 0.0
    assert np.isnan(_summed_intensity(wave, inten, errs, 1e-9)[0, 0])


def test_nonpositive_intensity_can_take_the_darkest_colour(tmp_path):
    image = np.full((20, 20), 100.0)
    image[:5] = -3.0
    image[5:8] = np.nan
    colours = {}
    for choice in ('grey', 'darkest'):
        path = tmp_path / f'{choice}.gif'
        render_thumbnail(image, (0, 1, 0, 1), ('x', 'y'), 't', 's', 'int', str(path), nonpositive=choice)
        with Image.open(path) as gif:
            pixels = np.asarray(gif.convert('RGB')).astype(int)
        # Centre of the bottom rows of the map, and of the missing rows.
        left, bottom, width, height = MAP_AREA
        x = int((left + 0.5 * width) * 768)
        colours[choice] = (tuple(pixels[int((1 - bottom - 0.1 * height) * 768), x]),
                           tuple(pixels[int((1 - bottom - 0.32 * height) * 768), x]))
    assert colours['grey'][0] == colours['grey'][1]
    assert colours['darkest'][0] != colours['darkest'][1]
    assert sum(colours['darkest'][0]) < sum(colours['darkest'][1])


def test_bad_options_are_refused():
    for options in ({'min_peak_snr': -1}, {'min_good_fraction': 0}, {'min_good_fraction': 1.5},
                    {'nonpositive': 'black'}):
        with pytest.raises(ValueError):
            make_thumbnails('eis_20170407_000335.data.h5', 'out', **options)


class FakeSlitCube:
    def __init__(self, slit_id):
        self.meta = {'mod_index': {'slit_id': slit_id}}


def test_slit_width_comes_from_the_header():
    assert _slit_width(FakeSlitCube('1"')) == 1.0
    assert _slit_width(FakeSlitCube('266"')) == 266.0
    assert _slit_width(FakeSlitCube('')) is None


class FakeWCS:
    world_axis_physical_types = ['em.wl', 'custom:pos.helioprojective.lon', 'custom:pos.helioprojective.lat']

    def pixel_to_world_values(self, wave, column, row):
        return np.zeros(3), (-0.44 + 1.0 * np.asarray(column)) / 3600, (-1401.2 + np.asarray(row)) / 3600


class FakeColumnCube:
    """One exposure at one slit position, as in a full-CCD study."""

    def __init__(self, columns=1):
        self.data = np.zeros((512, columns, 1024))
        self.wcs = FakeWCS()
        self.meta = {'mod_index': {'slit_id': '1"', 'obs_type': 'sit-and-stare'},
                     'date_obs': np.array(['2017-04-07T00:03:35'] * columns)}


def test_one_column_is_a_strip_as_wide_as_the_slit():
    extent, x_label, y_label, reverse, equal = _axes(FakeColumnCube())
    assert extent[:2] == pytest.approx((-0.94, 0.06))
    assert extent[2:] == pytest.approx((-1401.7, -889.7))
    assert x_label == 'Solar-X (arcsec)' and equal and not reverse
    left, bottom, width, height = _map_rectangle(extent, equal)
    assert width / height == pytest.approx(1 / MAX_MAP_RATIO)


def test_full_ccd_windows():
    assert _is_full_ccd('CCDBSHRT 0 177.540', 1024)
    assert _is_full_ccd('CCDBSHRT 0 178.880', 904)
    assert _is_full_ccd('FE XII 195.120', 1024)
    assert _is_full_ccd('ccdbsmod 0 181.750', 672)
    assert not _is_full_ccd('Fe XII 195.120', 272)
    assert not _is_full_ccd('Fe XII 195.120', 32)
    assert not _is_full_ccd('LINE 3  ', 200)


class FakeCounts:
    def __init__(self, duration):
        self.data = np.full((2, len(duration), 3), 30.0)
        self.meta = {'duration': np.asarray(duration, dtype=float)}


def test_count_rates_use_the_exposure_of_each_column():
    errs = np.ones((2, 3, 3))
    errs[0, 0, 1] = 0.0
    rate = _count_rates(FakeCounts([10.0, 30.0, 0.0]), errs)
    assert rate[1, 0, 0] == 3.0 and rate[1, 1, 0] == 1.0
    assert np.isnan(rate[0, 0, 1])
    assert np.isnan(rate[:, 2]).all()


def test_count_rates_need_an_exposure_for_each_column():
    with pytest.raises(ValueError):
        _count_rates(FakeCounts([10.0, 30.0]), np.ones((2, 3, 3)))


def test_detector_scale_ignores_hot_pixels():
    image = np.random.default_rng(1).uniform(0.5, 1.5, size=(100, 100))
    # 9 of 10000 pixels, below the 0.1% that the scale leaves out.
    image[:3, :3] = 1e5
    norm = _detector_norm(image)
    assert norm.vmax < 1e5 and 0.5 < norm.vmin < 1.5


def test_full_ccd_thumbnails_render(tmp_path):
    rate = np.random.default_rng(1).normal(1.0, 1.0, size=(50, 1, 64))
    rate[3, 0, 3] = np.nan
    wavelength = 170.0 + 0.0223 * np.arange(64)
    for name, data, extent, equal in (('one', rate, (-1, 1, 0, 50), True),
                                      ('many', np.repeat(rate, 6, axis=1), (0, 12, 0, 50), True),
                                      ('blank', np.full((50, 6, 64), np.nan), (0, 12, 0, 50), False)):
        path = tmp_path / f'{name}.gif'
        render_full_ccd(data, wavelength, extent, ('x', 'y'), 't', 's', str(path), equal)
        with Image.open(path) as gif:
            assert gif.size == (768, 768)


def test_full_ccd_window_uses_uncalibrated_rates_in_time_order(tmp_path, monkeypatch):
    # Columns of 10 s and 30 s with 30 photons give 3 and 1 photons/s, reversed to run forwards in time.
    counts = FakeCounts([10.0, 30.0])
    counts.data = np.full((2, 2, 1024), 30.0)
    wave = np.tile(170.0 + 0.0223 * np.arange(1024), (2, 2, 1))
    wininfo = np.array([(0, 'CCDBSHRT 0 177.540', 166.0, 189.0)],
                       dtype=[('iwin', int), ('line_id', 'U32'), ('wvl_min', float), ('wvl_max', float)])
    radcal, drawn = [], {}

    def read(data_file, iwin, exposure, apply_radcal=True):
        radcal.append(apply_radcal)
        return counts

    def draw(rate, wavelength, *args):
        drawn.update(rate=rate, wavelength=wavelength)

    monkeypatch.setattr(thumbnails.eispac, 'read_wininfo', lambda data_file: wininfo)
    monkeypatch.setattr(thumbnails, '_read_cube', read)
    monkeypatch.setattr(thumbnails, 'cube_arrays', lambda cube: (wave, cube.data, np.ones_like(cube.data)))
    monkeypatch.setattr(thumbnails, '_axes', lambda cube: ((0, 1, 0, 1), 'x', 'y', True, False))
    monkeypatch.setattr(thumbnails, 'render_full_ccd', draw)
    options = {'min_peak_snr': 3.0, 'min_good_fraction': 0.5, 'nonpositive': 'darkest'}
    record = thumbnails._raster_thumbnails('eis_20170407_000335.data.h5', str(tmp_path), 'del_zanna_2025', None, 300.0,
                                            options, str(tmp_path))
    assert radcal == [True, False]
    assert drawn['rate'][0, :, 0].tolist() == [1.0, 3.0]
    assert drawn['wavelength'].shape == (1024,)
    assert record['windows'][0]['units'] == 'photons/s' and record['windows'][0]['fit'] is None


def test_centred_pointing_copy_changes_only_the_copy(tmp_path):
    data = tmp_path / 'eis_20090623_214004.data.h5'
    with h5py.File(data, 'w') as f:
        f['x'] = [1]
    head = tmp_path / 'eis_20090623_214004.head.h5'
    with h5py.File(head, 'w') as f:
        for name, value in (('xcen', 307.5), ('ycen', 1.4745546e7), ('offset_x', -2.5), ('offset_y', 37.2)):
            f[f'pointing/{name}'] = np.array([value], dtype=np.float32)
    path = thumbnails._centred_pointing_copy(str(data), str(tmp_path / 'centred'))
    with h5py.File(path, 'r') as f:
        assert [f[f'pointing/{n}'][0] for n in ('xcen', 'ycen', 'offset_x', 'offset_y')] == [0, 0, 0, 0]
    with h5py.File(head, 'r') as f:
        assert f['pointing/ycen'][0] == np.float32(1.4745546e7)
    assert (tmp_path / 'centred' / 'eis_20090623_214004.data.h5').resolve() == data.resolve()


def test_header_copy_of_a_header_copy(tmp_path):
    # A raster with one position and impossible pointing gets a copy of the one-exposure copy.
    data = tmp_path / 'eis_20090623_214004.data.h5'
    with h5py.File(data, 'w') as f:
        f['x'] = [1]
    with h5py.File(tmp_path / 'eis_20090623_214004.head.h5', 'w') as f:
        f['index/nexp'] = [2]
        f['pointing/ycen'] = np.array([1e7], dtype=np.float32)
    first = _one_exposure_copy(str(data), str(tmp_path / 'one'))
    second = thumbnails._header_copy(first, str(tmp_path / 'two'), {'pointing/ycen': 0.0})
    with h5py.File(second, 'r') as f:
        assert f['index/nexp'][0] == 1 and f['pointing/ycen'][0] == 0
    assert (tmp_path / 'two' / 'eis_20090623_214004.data.h5').resolve() == data.resolve()


def test_raster_with_impossible_pointing_is_drawn_in_pixels(tmp_path, monkeypatch):
    wininfo = np.array([(0, 'Fe XII 195.120', 194.8, 195.4)],
                       dtype=[('iwin', int), ('line_id', 'U32'), ('wvl_min', float), ('wvl_max', float)])
    cube = FakeColumnCube(columns=4)
    cube.meta['mod_index']['obs_type'] = 'scan'
    cube.data = np.ones((512, 4, 24))
    cube.meta['duration'] = np.full(4, 30.0)
    wave = np.tile(194.85 + 0.0223 * np.arange(24), (512, 4, 1))
    read, drawn = [], {}

    def fake_read(data_file, iwin, exposure, apply_radcal=True):
        read.append(data_file)
        if 'centred' not in data_file:
            raise ValueError('EISPAC could not read window 0')
        return cube

    def fake_render(image, extent, labels, *args, **kwargs):
        drawn.update(extent=extent, labels=labels)

    monkeypatch.setattr(thumbnails.eispac, 'read_wininfo', lambda data_file: wininfo)
    monkeypatch.setattr(thumbnails, '_read_cube', fake_read)
    monkeypatch.setattr(thumbnails, '_centred_pointing_copy', lambda data_file, directory: directory + '/x.head.h5')
    monkeypatch.setattr(thumbnails, 'cube_arrays', lambda c: (wave, c.data, np.ones_like(c.data)))
    monkeypatch.setattr(thumbnails, 'choose_fit', lambda *args: None)
    monkeypatch.setattr(thumbnails, 'render_thumbnail', fake_render)
    options = {'min_peak_snr': 3.0, 'min_good_fraction': 0.5, 'nonpositive': 'darkest'}
    record = thumbnails._raster_thumbnails('eis_20090623_214004.data.h5', str(tmp_path), 'del_zanna_2025', None, 30.0,
                                            options, str(tmp_path))
    assert record['windows'][0]['pixel_axes'] and 'error' not in record['windows'][0]
    assert read == ['eis_20090623_214004.data.h5', str(tmp_path) + '/centred/x.head.h5']
    assert drawn['labels'] == ('Raster position (pixels)', 'Slit position (pixels)')
    assert drawn['extent'] == (-0.5, 3.5, -0.5, 511.5)
