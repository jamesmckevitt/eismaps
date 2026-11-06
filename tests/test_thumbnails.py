"""Tests of thumbnail names, exposure choice and rendering."""

import json
import os

import h5py
import numpy as np
from PIL import Image

from eismaps.thumbnails import (_arcsec, _column_zero_point, _exposure_choice, _exposure_factor, _is_done,
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
    assert _exposure_choice(write_durations(tmp_path, [[30.0] * 5])) == (None, 30.0)


def test_longest_exposure_is_chosen_and_the_median_is_the_reference(tmp_path):
    # The calibration in the header is for 50 s here, the median of 50, 5 and 655 s.
    assert _exposure_choice(write_durations(tmp_path, [[50.0] * 4, [5.0] * 4, [655.0] * 4])) == (2, 50.0)


def test_two_exposures_use_the_longer_as_reference(tmp_path):
    # The calibration in the header is for the longer of two exposures.
    assert _exposure_choice(write_durations(tmp_path, [[30.0] * 4, [90.0] * 4])) == (1, 90.0)


def test_no_durations_give_no_reference(tmp_path):
    assert _exposure_choice(write_durations(tmp_path, [[0.0] * 4]))[1] is None


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


def test_all_blank_map_still_renders(tmp_path):
    path = tmp_path / 'b.gif'
    render_thumbnail(np.full((4, 4), np.nan), (0, 1, 0, 1), ('x', 'y'), 't', 's', 'int', str(path))
    assert path.exists()
