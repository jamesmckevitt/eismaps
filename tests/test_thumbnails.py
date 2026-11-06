"""Tests of thumbnail names, exposure choice and rendering."""

import h5py
import numpy as np
from PIL import Image

from eismaps.thumbnails import _exposure_choice, _summed_intensity, render_thumbnail, thumbnail_path


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
    assert _exposure_choice(write_durations(tmp_path, [[30.0] * 5])) == (None, 1.0)


def test_longest_exposure_is_scaled_to_the_median(tmp_path):
    # The calibration in the header is for 50 s here, the median of 50, 5 and 655 s.
    choice, factor = _exposure_choice(write_durations(tmp_path, [[50.0] * 4, [5.0] * 4, [655.0] * 4]))
    assert choice == 2
    assert factor == 50.0 / 655.0


def test_two_exposures_use_the_longer_as_reference(tmp_path):
    # The calibration in the header is for the longer of two exposures.
    choice, factor = _exposure_choice(write_durations(tmp_path, [[30.0] * 4, [90.0] * 4]))
    assert choice == 1
    assert factor == 1.0


def test_summed_intensity_integrates_over_wavelength():
    wave = np.arange(10.0)[None, None, :] * 0.0223 + np.zeros((2, 3, 1))
    inten = np.full((2, 3, 10), 5.0)
    errs = np.ones_like(inten)
    errs[1, 2, :] = 0.0
    summed = _summed_intensity(wave, inten, errs)
    assert np.allclose(summed[0, 0], 5.0 * 10 * 0.0223)
    assert np.isnan(summed[1, 2])


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
