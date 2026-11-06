"""Tests of thumbnail names, exposure choice and rendering."""

import json
import os

import h5py
import numpy as np
import pytest
from PIL import Image

from eismaps.thumbnails import (MAP_AREA, fitted_maps, MAX_MAP_RATIO, _arcsec, _column_zero_point, _map_rectangle,
                                _one_exposure_copy, _exposure_choice, _exposure_factor, _is_done,
                                _process, _record_path, _summed_intensity, process_raster,
                                render_thumbnail, RasterUnavailable, thumbnail_path)


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
    image[0, 0] = 0.05
    path = tmp_path / 'c.gif'
    render_thumbnail(image, (0, 1, 0, 1), ('x', 'y'), 't', 's', 'wid', str(path), scale_pixels=image < 0.04)
    assert path.exists()


def test_all_blank_map_still_renders(tmp_path):
    path = tmp_path / 'b.gif'
    render_thumbnail(np.full((4, 4), np.nan), (0, 1, 0, 1), ('x', 'y'), 't', 's', 'int', str(path))
    assert path.exists()
