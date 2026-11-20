"""Tests of reading level 0 files, on small files written here."""

import numpy as np
import pytest
from astropy.io import fits

from eismaps.level0 import CLOCK_WRAP, SATURATED, TICKS_PER_SECOND, read_level0


def write_level0(path, counts, durations, n_raster, n_exposures=1, step_x=-2.0, ticks=None, slit='2"', mirror=None):
    """Write a one-window level 0 file. `counts` has shape (n_rows, n_y, n_wave)."""
    n_rows, n_y, n_wave = counts.shape
    primary = fits.PrimaryHDU()
    primary.header.update(NWIN=1, NRASTER=n_raster, NEXP=n_rows, NEXP_PRP=n_exposures, CDELT1=step_x, XCEN=100.0,
                          YCEN=-50.0, FOVX=abs(step_x) * n_raster, FOVY=float(n_y), SLIT_ID=slit)
    columns = [fits.Column(name='Fe XII 195.120', format=f'{n_y * n_wave}I', dim=f'({n_wave},{n_y})',
                           array=counts.astype(np.int16)),
               fits.Column(name='mhc_dur', format='J', array=np.asarray(durations, dtype=np.int32)),
               fits.Column(name='TI_1', format='D', array=np.zeros(n_rows) if ticks is None else np.asarray(ticks)),
               fits.Column(name='FMIRR', format='I', array=np.repeat(np.arange(n_rows // n_exposures) * 4 + 1000, n_exposures)
                           if mirror is None else np.asarray(mirror))]
    table = fits.BinTableHDU.from_columns(columns)
    table.header.update(TWMIN1=194.8, TWMAX1=195.4, TDELT1='(0.022318068,1.00000,1.00000)')
    fits.HDUList([primary, table]).writeto(path)
    return str(path)


def test_scan_columns_run_east_to_west(tmp_path):
    # Rows are in time order from west to east, so the first row becomes the last column.
    counts = np.stack([np.full((3, 4), 100 * (row + 1)) for row in range(5)])
    raster = read_level0(write_level0(tmp_path / 'a.fits.gz', counts, [1_000_000] * 5, n_raster=5))
    rate = raster.windows[0].rate
    assert rate.shape == (3, 5, 4)
    assert rate[0, :, 0].tolist() == [500.0, 400.0, 300.0, 200.0, 100.0]
    assert raster.extent == (95.0, 105.0, -51.5, -48.5)
    assert raster.equal


def test_longest_exposure_and_missing_rows(tmp_path):
    # Two exposures per position, 1 s and 3 s; the second position's 3 s exposure is missing.
    counts = np.full((6, 3, 4), 300)
    counts[0::2] = 100
    counts[3] = 0
    durations = [1_000_000, 3_000_000, 1_000_000, 0, 1_000_000, 3_000_000]
    rate = read_level0(write_level0(tmp_path / 'b.fits.gz', counts, durations, n_raster=3, n_exposures=2)).windows[0].rate
    assert rate[0, :, 0].tolist()[0] == 100.0 and rate[0, :, 0].tolist()[2] == 100.0
    assert np.isnan(rate[:, 1]).all()


def test_zero_and_saturated_counts_are_missing(tmp_path):
    counts = np.full((2, 3, 4), 500)
    counts[0, 1, 2] = 0
    counts[1, 2, 3] = SATURATED
    rate = read_level0(write_level0(tmp_path / 'c.fits.gz', counts, [2_000_000] * 2, n_raster=2)).windows[0].rate
    assert np.isnan(rate).sum() == 2
    assert np.nanmax(rate) == 250.0


def test_sit_and_stare_has_time_along_x(tmp_path):
    # The clock wraps between the second and third exposures.
    ticks = [CLOCK_WRAP - TICKS_PER_SECOND * 60, 0.0, TICKS_PER_SECOND * 60]
    counts = np.full((3, 3, 4), 500)
    raster = read_level0(write_level0(tmp_path / 'd.fits.gz', counts, [1_000_000] * 3, n_raster=1, ticks=ticks))
    assert raster.labels[0] == 'Time since first exposure (minutes)'
    assert raster.extent[:2] == pytest.approx((-0.5, 2.5))
    assert not raster.equal


def test_one_exposure_is_a_strip_as_wide_as_the_slit(tmp_path):
    counts = np.full((1, 3, 4), 500)
    raster = read_level0(write_level0(tmp_path / 'e.fits.gz', counts, [1_000_000], n_raster=1, slit='1"'))
    assert raster.extent[:2] == (99.5, 100.5)
    assert raster.equal


def test_rows_must_hold_whole_positions(tmp_path):
    counts = np.full((5, 3, 4), 500)
    with pytest.raises(ValueError):
        read_level0(write_level0(tmp_path / 'f.fits.gz', counts, [1_000_000] * 5, n_raster=3, n_exposures=2))


def test_scan_order_comes_from_the_mirror_not_cdelt1(tmp_path):
    # Early files give a positive CDELT1 for scans that still step west to east.
    counts = np.stack([np.full((3, 4), 100 * (row + 1)) for row in range(3)])
    raster = read_level0(write_level0(tmp_path / 'g.fits.gz', counts, [1_000_000] * 3, n_raster=3, step_x=1.0))
    assert raster.windows[0].rate[0, :, 0].tolist() == [300.0, 200.0, 100.0]
    # A scan with the mirror stepping the other way is kept in time order.
    raster = read_level0(write_level0(tmp_path / 'h.fits.gz', counts, [1_000_000] * 3, n_raster=3, mirror=[1008, 1004, 1000]))
    assert raster.windows[0].rate[0, :, 0].tolist() == [100.0, 200.0, 300.0]
