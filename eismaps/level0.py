"""Reading EIS level 0 FITS files for quicklook images.

Level 0 data are raw counts (DN), with no dark subtraction, calibration,
cleaning or wavelength correction. The binary table has one row per exposure,
`NEXP` in all, with the `NEXP_PRP` exposures of each position in turn, and one
column per window. A sit-and-stare observation has `NRASTER` 1 and many
positions in time.
"""

import re
from dataclasses import dataclass

import numpy as np
from astropy.io import fits

# The spacecraft clock in the TI_1 column counts 1/512 s and wraps at 2^32.
TICKS_PER_SECOND = 512
CLOCK_WRAP = 2 ** 32

# Counts at or above this are saturated (14-bit converter).
SATURATED = 16383


@dataclass
class Level0Window:
    """One window of a level 0 raster.

    Attributes
    ----------
    title : str
        Window title.
    low, high : float
        Wavelength range in Angstrom.
    wavelength : np.ndarray
        Wavelength of each pixel in Angstrom, shape (n_wave,).
    rate : np.ndarray
        Counts per second, shape (n_y, n_x, n_wave), NaN for missing and
        saturated pixels. Raster columns run from east to west.
    """

    title: str
    low: float
    high: float
    wavelength: np.ndarray
    rate: np.ndarray


@dataclass
class Level0Raster:
    """A level 0 raster read for quicklook images.

    Attributes
    ----------
    windows : list[Level0Window]
        Windows in their order in the file.
    extent : tuple
        Left, right, bottom and top of the maps.
    labels : tuple of str
        X and Y axis labels.
    equal : bool
        True if X and Y are both in arcsec.
    slit : str
        Slit or slot, e.g. '2"'.
    """

    windows: list
    extent: tuple
    labels: tuple
    equal: bool
    slit: str


def _column(table, name):
    """Return a table column by name, whatever its case."""
    names = {n.lower(): n for n in table.columns.names}
    return np.asarray(table.data.field(names[name]), dtype=float)


def _first_number(text):
    """Return the first number in a keyword value such as '(0.0223,1.0,1.0)'."""
    return float(re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', str(text)).group())


def _upper_median(values):
    return float(np.sort(values)[np.size(values) // 2])


def read_level0(path):
    """Read a level 0 file.

    For several exposures per position, the longest is used. Maps use the
    pointing in the header.

    Parameters
    ----------
    path : str
        Level 0 FITS file, which may be gzipped.

    Returns
    -------
    Level0Raster

    Raises
    ------
    ValueError
        If the number of rows does not match the exposures.
    """
    with fits.open(path) as hdul:
        primary, table = hdul[0].header, hdul[1]
        n_raster = int(primary['NRASTER'])
        n_exposures = int(primary.get('NEXP_PRP') or 1)
        n_rows = len(table.data)
        if n_rows != int(primary.get('NEXP', n_rows)) or n_rows % n_exposures:
            raise ValueError(f'{n_rows} rows for {primary.get("NEXP")} exposures, {n_exposures} per position')
        n_positions = n_rows // n_exposures

        duration = _column(table, 'mhc_dur').reshape(n_positions, n_exposures) / 1e6
        medians = [_upper_median(d[d > 0]) if (d > 0).any() else 0.0 for d in duration.T]
        exposure = int(np.argmax(medians))
        rows = np.arange(n_rows).reshape(n_positions, n_exposures)[:, exposure]
        seconds = duration[:, exposure]
        per_second = np.where(seconds > 0, 1.0 / np.where(seconds > 0, seconds, 1.0), np.nan)

        step_x = float(primary.get('CDELT1') or 0.0)
        # Scans step from west to east, so their columns are reversed to run east to west.
        order = slice(None, None, -1) if n_raster > 1 and step_x < 0 else slice(None)

        windows = []
        header = table.header
        for iwin in range(int(primary['NWIN'])):
            counts = np.asarray(table.data.field(iwin), dtype=float)[rows]
            counts = np.where((counts > 0) & (counts < SATURATED), counts, np.nan)
            rate = (counts * per_second[:, None, None]).transpose(1, 0, 2)[:, order]
            low = float(header[f'TWMIN{iwin + 1}'])
            step = _first_number(header[f'TDELT{iwin + 1}'])
            wavelength = low + step * np.arange(rate.shape[-1])
            windows.append(Level0Window(table.columns.names[iwin], low, float(header[f'TWMAX{iwin + 1}']),
                                        wavelength, rate))

        x_centre, y_centre = float(primary['XCEN']), float(primary['YCEN'])
        height = float(primary['FOVY'])
        bottom, top = y_centre - 0.5 * height, y_centre + 0.5 * height
        slit = str(primary.get('SLIT_ID', '')).strip()
        if n_raster > 1:
            width = float(primary['FOVX'])
            extent = (x_centre - 0.5 * width, x_centre + 0.5 * width, bottom, top)
            labels, equal = ('Solar-X (arcsec)', 'Solar-Y (arcsec)'), True
        elif n_positions == 1:
            width = _first_number(slit) if re.match(r'\s*\d', slit) else abs(step_x)
            extent = (x_centre - 0.5 * width, x_centre + 0.5 * width, bottom, top)
            labels, equal = ('Solar-X (arcsec)', 'Solar-Y (arcsec)'), True
        else:
            ticks = _column(table, 'ti_1')[rows]
            minutes = ((ticks - ticks[0]) % CLOCK_WRAP) / TICKS_PER_SECOND / 60.0
            span = float(minutes.max())
            half = 0.5 * span / (minutes.size - 1) if span > 0 else 0.5
            extent = (-half, span + half, bottom, top)
            labels, equal = ('Time since first exposure (minutes)', 'Solar-Y (arcsec)'), False
    return Level0Raster(windows, extent, labels, equal, slit)
