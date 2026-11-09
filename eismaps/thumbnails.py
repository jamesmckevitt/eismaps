"""Quicklook thumbnails of EIS rasters for the MSSL SolarB archive.

:func:`make_thumbnails` writes one GIF per window and quantity, in the
directory layout and with the file names of the archive::

    YYYY/MM/DD/eis_l0_<time>.fits/eis_l0_<time>.fits_line_NN_<title>.<qty>.gif

`int` is the window summed over wavelength, so it exists for every window. For
a full-CCD window it shows the detector image and mean spectrum in DN/s instead.
`vel` and `wid` come from Gaussian fits of the window's line. Velocities are
relative to the median of each raster column, which removes the drift left by
the wavelength correction. A JSON file records what was done for each window.
"""

import argparse
import contextlib
import io
import json
import multiprocessing
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
from importlib.metadata import version

import eispac
import h5py
import matplotlib.colors as mcolor
import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter
from PIL import Image

from eismaps.archive import RasterUnavailable, fetch_raster, list_rasters
from eismaps.batched_fit import cube_arrays, fit_cube
from eismaps.calibration import CALIBRATION_METHODS, calibration_factor
from eismaps.windows import MAX_FIT_PIXELS, choose_fit, parse_title

CC = 2.9979E+5  # speed of light in km/s
THUMBNAIL_PIXELS = 768

# Fit tolerance. Fits with three or more Gaussians are often degenerate and
# have no single minimum to converge to, so they use the EISPAC value.
TOLERANCE = 1e-8
DEGENERATE_TOLERANCE = 1e-6

# Velocities and widths are shown where the peak exceeds this many errors.
MIN_PEAK_SNR = 3.0

# Pixels need this fraction of good points to have a summed intensity.
MIN_GOOD_FRACTION = 0.5

# How summed intensities at or below zero are drawn: 'grey' like missing
# pixels, or 'darkest' in the lowest colour of the scale.
NONPOSITIVE = 'darkest'
NONPOSITIVE_CHOICES = ('grey', 'darkest')

# Windows at least this wide are full-CCD readouts, even if named after a line.
# The widest window named after a line is 744 pixels and the narrowest full-CCD
# readout is 904.
FULL_CCD_PIXELS = 900

# Fewest pixels of a raster column for its own median velocity, and the
# fraction of the column they must be.
MIN_COLUMN_PIXELS = 3
MIN_COLUMN_FRACTION = 0.02

# Range of the symmetric velocity colour scale in km/s.
VELOCITY_SCALE_LIMITS = (10.0, 60.0)

# Largest area of a thumbnail for the map, as left, bottom, width and height.
MAP_AREA = (0.11, 0.09, 0.70, 0.80)

# Maps are drawn with equal arcsec scales in X and Y, up to this ratio of their
# sides. Longer maps are stretched to this ratio, to stay readable.
MAX_MAP_RATIO = 8.0

# Shortest colour bar, as a fraction of the thumbnail height.
MIN_BAR_HEIGHT = 0.4

# Panels of a full-CCD thumbnail, as left, bottom, width and height.
FULL_CCD_MAP_PANEL = (0.11, 0.70, 0.70, 0.20)
FULL_CCD_PANELS = {True: ((0.11, 0.41, 0.70, 0.21), (0.11, 0.09, 0.70, 0.23)),
                   False: ((0.11, 0.47, 0.70, 0.41), (0.11, 0.09, 0.70, 0.29))}

# Percentiles of the positive pixels that set the colour scale of a detector
# image. Most pixels hold no line, and a few are hot or hit by cosmic rays.
DETECTOR_PERCENTILES = (50.0, 99.9)


def _timestamp(name):
    """Return the 'YYYYMMDD_HHMMSS' in a file name or string."""
    match = re.search(r'(\d{8}_\d{6})', os.path.basename(name))
    if match is None:
        raise ValueError(f'No raster time in {name}')
    return match.group(1)


def thumbnail_path(root, timestamp, iwin, title, measurement):
    """Return the path of a thumbnail in the archive layout.

    Parameters
    ----------
    root : str
        Top directory of the thumbnails.
    timestamp : str
        Raster start time as 'YYYYMMDD_HHMMSS'.
    iwin : int
        Window number, from 0.
    title : str
        Window title. Spaces become underscores and the case is kept.
    measurement : str
        'int', 'vel' or 'wid'.
    """
    level0 = f'eis_l0_{timestamp}.fits'
    name = title.replace(' ', '_').replace('/', '_')
    return os.path.join(root, timestamp[:4], timestamp[4:6], timestamp[6:8], level0,
                        f'{level0}_line_{iwin:02d}_{name}.{measurement}.gif')


def _exposure_choice(data_file):
    """Return the exposure set to use and the exposure time of the calibration.

    The calibration in the header holds one exposure time, the median of the
    exposures of a position. The longest exposure of a position is used, as
    it has the most signal.

    Returns
    -------
    tuple
        Index of the exposure set, or None for one exposure per position,
        the exposure time of the calibration in seconds, or None if no
        exposure has a duration, and True for several exposures at a single
        position.
    """
    with h5py.File(data_file.replace('.data.h5', '.head.h5'), 'r') as head:
        durations = np.atleast_2d(np.array(head['exposure_times/duration'], dtype=float))
    medians = np.array([np.median(d[d > 0]) if (d > 0).any() else 0.0 for d in durations])
    reference = float(np.sort(medians)[medians.size // 2])
    several = durations.shape[0] > 1
    longest = int(np.argmax(medians)) if several else None
    return longest, reference if reference > 0 else None, several and durations.shape[1] == 1


def _exposure_factor(cube, reference):
    """Return the intensity factor of each raster column for its own exposure time.

    Exposure times can change within a raster, for example during a flare.
    """
    duration = np.asarray(cube.meta['duration'], dtype=float)
    if reference is None or duration.size != cube.data.shape[1]:
        return np.ones(cube.data.shape[1])
    return np.where(duration > 0, reference / np.where(duration > 0, duration, 1.0), np.nan)


def _read_cube(data_file, iwin, exposure, apply_radcal=True):
    """Read a window without EISPAC's messages.

    Raises
    ------
    ValueError
        If EISPAC cannot read the window.
    """
    messages = io.StringIO()
    with contextlib.redirect_stdout(messages), contextlib.redirect_stderr(messages):
        if exposure is None:
            cube = eispac.read_cube(data_file, iwin, apply_radcal=apply_radcal)
        else:
            cube = eispac.read_cube(data_file, iwin, exp_set=exposure, apply_radcal=apply_radcal)
    if cube is None:
        raise ValueError(f'EISPAC could not read window {iwin}: {messages.getvalue()[-200:]}')
    return cube


def _one_exposure_copy(data_file, directory):
    """Return a file to read a raster with several exposures at a single position.

    EISPAC computes a cadence for these from one time, which gives NaN and
    fails. Reading a copy of the header that has one exposure avoids this. The
    data file is linked beside it, and EISPAC is given the header, as it
    would follow a link given to it.
    """
    stem = os.path.basename(data_file).replace('.data.h5', '')
    head = os.path.join(directory, f'{stem}.head.h5')
    shutil.copyfile(data_file.replace('.data.h5', '.head.h5'), head)
    os.symlink(os.path.abspath(data_file), os.path.join(directory, f'{stem}.data.h5'))
    with h5py.File(head, 'r+') as header:
        header['index/nexp'][...] = 1
    return head


def _window_sum(values, good, min_good_fraction):
    """Return each spectrum summed over wavelength, NaN with too few good points.

    Missing points are taken to have the mean of the others.
    """
    fraction = good.mean(axis=-1)
    summed = np.where(good, values, 0.0).sum(axis=-1)
    enough = (fraction >= min_good_fraction) & (fraction > 0)
    return np.where(enough, summed / np.where(enough, fraction, 1.0), np.nan)


def _summed_intensity(wave, inten, errs, min_good_fraction=MIN_GOOD_FRACTION):
    """Return the intensity of each pixel summed over the window, in erg / (cm2 s sr).

    Pixels with less than `min_good_fraction` of their points good are NaN.
    """
    step = float(np.median(np.abs(np.diff(wave, axis=-1))))
    return _window_sum(inten, errs > 0, min_good_fraction) * step


def _is_full_ccd(title, n_wave):
    """Return True for a window drawn as a detector image and spectrum.

    These are full-CCD readouts, and wide windows named after no line.
    """
    return n_wave >= FULL_CCD_PIXELS or (n_wave > MAX_FIT_PIXELS and not parse_title(title)[0])


def _count_rates(counts, errs):
    """Return a window in DN/s, with NaN for bad points.

    Parameters
    ----------
    counts : eispac.EISCube
        Window read without the radiometric calibration.
    errs : np.ndarray
        Errors of the window, zero or negative for bad points.

    Returns
    -------
    np.ndarray
        DN/s, shape (n_y, n_x, n_wave).

    Raises
    ------
    ValueError
        If the header has no exposure time for each raster column.
    """
    duration = np.asarray(counts.meta['duration'], dtype=float)
    if duration.size != counts.data.shape[1]:
        raise ValueError(f'{duration.size} exposure times for {counts.data.shape[1]} raster columns')
    per_second = np.where(duration > 0, 1.0 / np.where(duration > 0, duration, 1.0), np.nan)
    return np.where(errs > 0, np.asarray(counts.data, dtype=float), np.nan) * per_second[None, :, None]


def _mean_spectrum(wave, inten, errs):
    """Return the mean spectrum of a window, with the error of the mean."""
    good = errs > 0
    count = good.sum(axis=(0, 1))
    used = np.maximum(count, 1)
    mean = np.where(good, inten, 0.0).sum(axis=(0, 1)) / used
    error = np.sqrt(np.where(good, errs ** 2, 0.0).sum(axis=(0, 1))) / used
    return wave.mean(axis=(0, 1)), mean, np.where(count > 0, error, 0.0)


def _parameter_state(fit_res, index):
    """Return where a parameter has an error and where it is on a template limit.

    A tied parameter has no error of its own, so the one it is tied to is used.
    """
    tied = str(fit_res.parinfo[index].get('tied', '') or '')
    match = re.search(r'p\[(\d+)\]', tied)
    source = int(match.group(1)) if match else index
    values = fit_res.fit['params'][:, :, source]
    has_error = fit_res.fit['perror'][:, :, source] > 0
    on_limit = np.zeros(values.shape, dtype=bool)
    info = fit_res.parinfo[source]
    for side in (0, 1):
        if info.get('limited', (0, 0))[side]:
            on_limit |= values == info['limits'][side]
    return has_error, on_limit


def _column_zero_point(velocity):
    """Return the median velocity of each raster column.

    Columns with few pixels take their value from the neighbouring columns,
    as the median of a few pixels would remove their real velocity.
    """
    needed = max(MIN_COLUMN_PIXELS, MIN_COLUMN_FRACTION * velocity.shape[0])
    enough = np.isfinite(velocity).sum(axis=0) >= needed
    if not enough.any():
        values = velocity[np.isfinite(velocity)]
        return np.full(velocity.shape[1], np.median(values) if values.size else 0.0)
    medians = np.nanmedian(velocity[:, enough], axis=0)
    columns = np.arange(velocity.shape[1])
    return np.interp(columns, columns[enough], medians)


def fitted_maps(fit_res, component, min_peak_snr=MIN_PEAK_SNR):
    """Return velocity and width maps of one component.

    Pixels are left out where the fit is not usable, the peak is below
    `min_peak_snr` errors, or the parameter is not constrained. Values on a
    template limit are kept, as these are often the fastest and broadest
    pixels of a flare, and the true value is then at or beyond the limit.
    :func:`eismaps.maps.measurement_mask` leaves them out for measurements.
    The column zero point and the colour scales use the other pixels.

    Parameters
    ----------
    fit_res : eispac.EISFitResult
        Fit of a window.
    component : int
        Gaussian to use.
    min_peak_snr : float, default=3
        Smallest peak in units of its error. 0 keeps every positive peak.

    Returns
    -------
    tuple of np.ndarray
        Velocity in km/s, relative to the median of each raster column,
        Gaussian width in Angstrom, and True where each of them is off its
        limits, all of shape (n_pixels, n_steps).
    """
    params, perror = fit_res.fit['params'], fit_res.fit['perror']
    peak, peak_error = params[:, :, 3 * component], perror[:, :, 3 * component]
    usable = (fit_res.fit['status'] > 0) & (peak > 0) & (peak_error > 0) & np.isfinite(fit_res.fit['chi2'])
    usable &= peak >= min_peak_snr * peak_error

    centroid_index = 3 * component + 1
    has_error, on_limit = _parameter_state(fit_res, centroid_index)
    rest = float(fit_res.parinfo[centroid_index]['value'])
    velocity = CC * (params[:, :, centroid_index] - rest) / rest
    velocity_off_limit = usable & has_error & ~on_limit
    zero_point = _column_zero_point(np.where(velocity_off_limit, velocity, np.nan))
    velocity = np.where(usable & (has_error | on_limit), velocity - zero_point[None, :], np.nan)

    width_index = 3 * component + 2
    has_error, on_limit = _parameter_state(fit_res, width_index)
    width = np.where(usable & (has_error | on_limit), params[:, :, width_index], np.nan)
    return velocity, width, velocity_off_limit, usable & has_error & ~on_limit


def _arcsec(degrees):
    """Return angles in degrees as arcsec from -180 to 180 degrees, as the WCS can wrap them."""
    return (np.asarray(degrees) * 3600.0 + 648000.0) % 1296000.0 - 648000.0


def _slit_width(cube):
    """Return the slit width in arcsec, or None if the header does not give it."""
    match = re.match(r'\s*(\d+(?:\.\d+)?)', str(cube.meta['mod_index'].get('slit_id', '')))
    return float(match.group(1)) if match else None


def _axes(cube):
    """Return the image extent and axis labels of a window.

    Solar-X and Solar-Y in arcsec come from the cube WCS, which holds the
    pointing correction. Sit-and-stare observations have time along X. A map
    of one column is drawn as a strip as wide as the slit.

    Returns
    -------
    tuple
        Extent as (left, right, bottom, top), X label, Y label, True if the
        columns run backwards in time and must be reversed, and True if X and
        Y are both in arcsec.
    """
    n_y, n_x = cube.data.shape[:2]
    names = cube.wcs.world_axis_physical_types
    longitude = [i for i, n in enumerate(names) if 'lon' in str(n)][0]
    latitude = [i for i, n in enumerate(names) if 'lat' in str(n)][0]

    # Pixel 1 gives the pixel size, and exists in the WCS for one column or row.
    columns = np.array([0, 1, n_x - 1])
    rows = np.array([0, 1, n_y - 1])
    world_x = _arcsec(cube.wcs.pixel_to_world_values(np.zeros(3), columns, np.zeros(3))[longitude])
    world_y = _arcsec(cube.wcs.pixel_to_world_values(np.zeros(3), np.zeros(3), rows)[latitude])
    half_x = 0.5 * (world_x[1] - world_x[0])
    half_y = 0.5 * (world_y[1] - world_y[0])
    bottom, top = world_y[0] - half_y, world_y[2] + half_y
    y_label = 'Solar-Y (arcsec)'

    slit = _slit_width(cube)
    if n_x == 1 and slit:
        return (world_x[0] - 0.5 * slit, world_x[0] + 0.5 * slit, bottom, top), 'Solar-X (arcsec)', y_label, False, True

    if cube.meta['mod_index'].get('obs_type') == 'sit-and-stare':
        times = np.array(cube.meta['date_obs'], dtype='datetime64[ms]')
        minutes = (times - times.min()) / np.timedelta64(60, 's')
        reverse = minutes[0] > minutes[-1]
        span = float(minutes.max())
        half_t = 0.5 * span / (n_x - 1) if n_x > 1 and span > 0 else 0.5
        return (-half_t, span + half_t, bottom, top), 'Time since first exposure (minutes)', y_label, bool(reverse), False

    return (world_x[0] - half_x, world_x[2] + half_x, bottom, top), 'Solar-X (arcsec)', y_label, False, True


def _scale(image, measurement):
    """Return the colour map, normalisation, label and colour bar extension of a thumbnail."""
    values = image[np.isfinite(image)]
    if measurement == 'int':
        positive = values[values > 0]
        low = np.percentile(positive, 5) if positive.size else 1.0
        high = np.percentile(positive, 99.5) if positive.size else 10.0
        norm = mcolor.LogNorm(vmin=low, vmax=max(high, low * 10.0))
        return 'gist_heat', norm, 'Intensity (erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$)', 'both'
    if measurement == 'vel':
        limit = np.percentile(np.abs(values), 98) if values.size else VELOCITY_SCALE_LIMITS[0]
        limit = float(np.clip(limit, *VELOCITY_SCALE_LIMITS))
        return 'RdBu_r', mcolor.Normalize(-limit, limit), 'Doppler velocity (km/s)', 'both'
    low, high = (np.percentile(values, (1, 99)) if values.size else (0.02, 0.06))
    return 'viridis', mcolor.Normalize(low, max(high, low + 1e-3)), 'Gaussian width ($\\AA$)', 'both'


def _map_ratio(extent):
    """Return the width over height of a map, or None if a side is zero or not finite."""
    width, height = abs(extent[1] - extent[0]), abs(extent[3] - extent[2])
    if not (np.isfinite(width) and np.isfinite(height)) or width == 0 or height == 0:
        return None
    return float(width / height)


def _map_rectangle(extent, equal, area=MAP_AREA):
    """Return the left, bottom, width and height of a map in a square thumbnail.

    With `equal`, the sides follow the extent, centred in `area`, so that an
    arcsec has the same length in X and Y. Otherwise the map fills the area.
    """
    left, bottom, width, height = area
    ratio = _map_ratio(extent)
    if not equal or ratio is None:
        return left, bottom, width, height
    ratio = min(max(ratio, 1.0 / MAX_MAP_RATIO), MAX_MAP_RATIO)
    if ratio > width / height:
        new_height = width / ratio
        return left, bottom + 0.5 * (height - new_height), width, new_height
    new_width = height * ratio
    return left + 0.5 * (width - new_width), bottom, new_width, height


def _apply_nonpositive(image, norm, nonpositive):
    """Return an image for a logarithmic scale with values at or below zero drawn as chosen."""
    if nonpositive != 'darkest':
        return image
    # Values below the scale take the colour of its low end.
    return np.where(np.isfinite(image) & (image <= 0), 0.5 * norm.vmin, image)


def _add_version_note(figure):
    """Write the versions of the software in the corner of a thumbnail."""
    figure.text(0.995, 0.005, f"eismaps {version('eismaps')}, eispac {version('eispac')}",
                ha='right', va='bottom', fontsize=7, color='0.55')


def _write_gif(canvas, path):
    """Write a figure as a GIF."""
    canvas.draw()
    pixels = np.asarray(canvas.buffer_rgba())[:, :, :3]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(pixels).quantize(colors=256, dither=Image.Dither.NONE).save(path, format='GIF')


def render_thumbnail(image, extent, labels, title, subtitle, measurement, path, equal=False, scale_pixels=None,
                     nonpositive=NONPOSITIVE):
    """Write one thumbnail as a GIF.

    Parameters
    ----------
    image : np.ndarray
        Map, shape (n_y, n_x). NaN pixels are grey.
    extent : tuple
        Left, right, bottom and top of the image.
    labels : tuple of str
        X and Y axis labels.
    title, subtitle : str
        First and second lines of the title.
    measurement : str
        'int', 'vel' or 'wid'.
    path : str
        Output file.
    equal : bool, default=False
        Draw X and Y to the same scale, for axes in the same unit.
    scale_pixels : np.ndarray, optional
        True for the pixels that set the colour scale. All pixels by default.
    nonpositive : str, default='darkest'
        For 'int', draw values at or below zero 'grey' like missing pixels,
        or in the 'darkest' colour of the scale.
    """
    cmap_name, norm, label, extend = _scale(image if scale_pixels is None else image[scale_pixels], measurement)
    cmap = colormaps[cmap_name].with_extremes(bad='0.7')
    if measurement == 'int':
        image = _apply_nonpositive(image, norm, nonpositive)

    size = THUMBNAIL_PIXELS / 100.0
    figure = Figure(figsize=(size, size), dpi=100)
    canvas = FigureCanvasAgg(figure)
    left, bottom, width, height = _map_rectangle(extent, equal)
    axes = figure.add_axes([left, bottom, width, height])
    bar_height = max(height, MIN_BAR_HEIGHT)
    bar = figure.add_axes([left + width + 0.03, bottom + 0.5 * (height - bar_height), 0.03, bar_height])
    shown = axes.imshow(image, origin='lower', aspect='auto', extent=extent, cmap=cmap, norm=norm,
                        interpolation='nearest')
    figure.colorbar(shown, cax=bar, label=label, extend=extend)
    axes.set_xlabel(labels[0])
    axes.set_ylabel(labels[1])
    figure.suptitle(f'{title}\n{subtitle}', fontsize=11)
    _add_version_note(figure)
    ratio = _map_ratio(extent)
    if equal and ratio is not None and not 1.0 / MAX_MAP_RATIO <= ratio <= MAX_MAP_RATIO:
        figure.text(0.005, 0.005, 'X and Y scales differ', ha='left', va='bottom', fontsize=7, color='0.55')
    _write_gif(canvas, path)


def _detector_norm(image):
    """Return the logarithmic colour normalisation of a detector image."""
    positive = image[np.isfinite(image) & (image > 0)]
    low, high = np.percentile(positive, DETECTOR_PERCENTILES) if positive.size else (1.0, 10.0)
    return mcolor.LogNorm(vmin=low, vmax=max(high, low * 10.0))


def _mean_of_good(values, axis):
    """Return the mean over an axis of the finite values, NaN where there are none."""
    good = np.isfinite(values)
    count = good.sum(axis=axis)
    return np.where(count > 0, np.where(good, values, 0.0).sum(axis=axis) / np.maximum(count, 1), np.nan)


def render_full_ccd(rate, wavelength, extent, labels, title, subtitle, path, equal=False,
                    min_good_fraction=MIN_GOOD_FRACTION, nonpositive=NONPOSITIVE):
    """Write the thumbnail of a full-CCD window as a GIF.

    It shows the detector image and the spectrum, both averaged over the
    raster columns, and above them a map of the window sum if there is more
    than one column.

    Parameters
    ----------
    rate : np.ndarray
        Window in DN/s, shape (n_y, n_x, n_wave), with NaN for bad points.
    wavelength : np.ndarray
        Wavelength in Angstrom, shape (n_wave,).
    extent : tuple
        Left, right, bottom and top of the map.
    labels : tuple of str
        X and Y axis labels of the map.
    title, subtitle : str
        First and second lines of the title.
    path : str
        Output file.
    equal : bool, default=False
        Draw the map with X and Y to the same scale.
    min_good_fraction : float, default=0.5
        Fraction of good points a pixel of the map needs.
    nonpositive : str, default='darkest'
        Draw values at or below zero 'grey' like missing pixels, or in the
        'darkest' colour of the scale.
    """
    size = THUMBNAIL_PIXELS / 100.0
    figure = Figure(figsize=(size, size), dpi=100)
    canvas = FigureCanvasAgg(figure)
    cmap = colormaps['gist_heat'].with_extremes(bad='0.7')

    def show(image, norm, rectangle, image_extent, bar_label):
        left, bottom, width, height = rectangle
        axes = figure.add_axes(rectangle)
        bar = figure.add_axes([left + width + 0.03, bottom, 0.03, height])
        shown = axes.imshow(_apply_nonpositive(image, norm, nonpositive), origin='lower', aspect='auto',
                            extent=image_extent, cmap=cmap, norm=norm, interpolation='nearest')
        colorbar = figure.colorbar(shown, cax=bar, label=bar_label, extend='both')
        # The bars are short, so only values of 1 and 3 times a power of ten are labelled.
        colorbar.ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 3.0), numticks=8))
        colorbar.ax.yaxis.set_minor_formatter(NullFormatter())
        return axes

    has_map = rate.shape[1] > 1
    if has_map:
        window_sum = _window_sum(rate, np.isfinite(rate), min_good_fraction)
        rectangle = _map_rectangle(extent, equal, FULL_CCD_MAP_PANEL)
        axes = show(window_sum, _scale(window_sum, 'int')[1], rectangle, extent, 'DN/s')
        axes.xaxis.set_major_locator(MaxNLocator(nbins=3))
        axes.set_xlabel(labels[0], labelpad=1)
        axes.set_ylabel(labels[1])

    step = float(np.median(np.abs(np.diff(wavelength))))
    edges = (wavelength[0] - 0.5 * step, wavelength[-1] + 0.5 * step)
    detector_panel, spectrum_panel = FULL_CCD_PANELS[has_map]
    detector = _mean_of_good(rate, 1)
    axes = show(detector, _detector_norm(detector), detector_panel, (*edges, extent[2], extent[3]), 'DN/s per pixel')
    axes.set_xlabel('Wavelength ($\\AA$)', labelpad=1)
    axes.set_ylabel(labels[1])

    axes = figure.add_axes(spectrum_panel)
    axes.plot(wavelength, _mean_of_good(rate, (0, 1)), color='k', linewidth=0.6)
    axes.set_xlim(*edges)
    axes.set_xlabel('Wavelength ($\\AA$)')
    axes.set_ylabel('Mean DN/s per pixel')

    figure.suptitle(f'{title}\n{subtitle}', fontsize=11)
    _add_version_note(figure)
    _write_gif(canvas, path)


def _calibration(date, wavelengths, method):
    """Return the calibration factor at the first wavelength that has one, or None."""
    for wavelength in wavelengths:
        if wavelength is None:
            continue
        try:
            factor = calibration_factor(date, wavelength, method)
        except (ValueError, IndexError):
            continue
        if np.isfinite(factor):
            return factor
    return None


def _check_options(min_peak_snr, min_good_fraction, nonpositive):
    """Raise ValueError if an option of :func:`make_thumbnails` is out of range."""
    if not min_peak_snr >= 0:
        raise ValueError(f'min_peak_snr must be 0 or more, not {min_peak_snr}')
    if not 0 < min_good_fraction <= 1:
        raise ValueError(f'min_good_fraction must be above 0 and at most 1, not {min_good_fraction}')
    if nonpositive not in NONPOSITIVE_CHOICES:
        raise ValueError(f'nonpositive must be one of {NONPOSITIVE_CHOICES}, not {nonpositive!r}')


def make_thumbnails(data_file, output_root, calibration='del_zanna_2025', min_peak_snr=MIN_PEAK_SNR,
                    min_good_fraction=MIN_GOOD_FRACTION, nonpositive=NONPOSITIVE):
    """Make the thumbnails of a raster.

    Parameters
    ----------
    data_file : str
        Path of the level 1 data file, with the header file beside it.
    output_root : str
        Top directory of the thumbnails.
    calibration : str, default='del_zanna_2025'
        Calibration of the intensities, one of
        :data:`eismaps.calibration.CALIBRATION_METHODS`.
    min_peak_snr : float, default=3
        Velocities and widths are shown where the fitted peak is at least
        this many errors. 0 shows every fit with a positive peak.
    min_good_fraction : float, default=0.5
        Fraction of good points a pixel needs for its summed intensity.
    nonpositive : str, default='darkest'
        Summed intensities at or below zero are drawn 'grey' like missing
        pixels, or in the 'darkest' colour of the scale.

    Returns
    -------
    dict
        Record of the raster: exposure used, options, and for each window its
        title, fit and thumbnails, or the error if a window failed.
    """
    if calibration not in CALIBRATION_METHODS:
        raise ValueError(f'Unknown calibration method: {calibration}. Choose from {sorted(CALIBRATION_METHODS)}')
    _check_options(min_peak_snr, min_good_fraction, nonpositive)
    options = {'min_peak_snr': min_peak_snr, 'min_good_fraction': min_good_fraction, 'nonpositive': nonpositive}
    exposure, reference_exposure, single_position = _exposure_choice(data_file)
    if single_position:
        with tempfile.TemporaryDirectory() as temporary:
            copy = _one_exposure_copy(data_file, temporary)
            return _raster_thumbnails(copy, output_root, calibration, exposure, reference_exposure, options)
    return _raster_thumbnails(data_file, output_root, calibration, exposure, reference_exposure, options)


def _raster_thumbnails(data_file, output_root, calibration, exposure, reference_exposure, options):
    """Make the thumbnails of a raster with a chosen exposure set, returning its record."""
    timestamp = _timestamp(data_file)
    wininfo = eispac.read_wininfo(data_file)
    date = f'{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}T{timestamp[9:11]}:{timestamp[11:13]}:{timestamp[13:15]}'
    record = {'timestamp': timestamp, 'exposure_set': exposure, 'reference_exposure': reference_exposure,
              'calibration': calibration, 'options': options, 'windows': []}

    for row in wininfo:
        iwin, title = int(row['iwin']), str(row['line_id'])
        low, high = float(row['wvl_min']), float(row['wvl_max'])
        entry = {'iwin': iwin, 'title': title, 'files': []}
        record['windows'].append(entry)
        try:
            cube = _read_cube(data_file, iwin, exposure)
            wave, inten, errs = cube_arrays(cube)
            extent, x_label, y_label, reverse, equal = _axes(cube)
            order = slice(None, None, -1) if reverse else slice(None)
            labels = (x_label, y_label)
            heading = f'{title.strip()} [{low:.2f}-{high:.2f} $\\AA$]'

            if _is_full_ccd(title, wave.shape[-1]):
                rate = _count_rates(_read_cube(data_file, iwin, exposure, apply_radcal=False), errs)
                path = thumbnail_path(output_root, timestamp, iwin, title, 'int')
                render_full_ccd(rate[:, order], wave.mean(axis=(0, 1)), extent, labels, f'Spectrum - {heading}', date, path,
                                equal, options['min_good_fraction'], options['nonpositive'])
                entry.update(fit=None, layout='full_ccd', units='DN/s')
                entry['files'].append(os.path.basename(path))
                continue

            fit = None
            try:
                fit = choose_fit(title, low, high, *_mean_spectrum(wave, inten, errs))
            except Exception:
                entry['fit_error'] = traceback.format_exc(limit=3)
            line_wave = parse_title(fit.line)[1] if fit else parse_title(title)[1]
            factor = _calibration(date, (line_wave, 0.5 * (low + high)), calibration)
            entry['calibration_factor'] = factor

            intensity = _summed_intensity(wave, inten, errs, options['min_good_fraction'])
            intensity = intensity * _exposure_factor(cube, reference_exposure)[None, :]
            intensity = intensity * (1.0 if factor is None else factor)
            path = thumbnail_path(output_root, timestamp, iwin, title, 'int')
            render_thumbnail(intensity[:, order], extent, labels, f'Intensity - {heading}', date, 'int', path, equal,
                             nonpositive=options['nonpositive'])
            entry['files'].append(os.path.basename(path))

            if fit is None:
                entry['fit'] = None
                continue
            tolerance = TOLERANCE if int(fit.template['n_gauss']) <= 2 else DEGENERATE_TOLERANCE
            result = fit_cube(cube, fit.template, fit.parinfo, component=fit.component, tolerance=tolerance)
            velocity, width, velocity_off_limit, width_off_limit = fitted_maps(result, fit.component, options['min_peak_snr'])
            entry['fit'] = {'line': fit.line, 'source': fit.source, 'n_gauss': int(fit.template['n_gauss']),
                            'single': fit.single, 'intensity_ratio': fit.intensity_ratio,
                            'velocity_offset': fit.velocity_offset,
                            'fitted_fraction': float((result.fit['status'] > 0).mean()),
                            'velocity_fraction': float(np.isfinite(velocity).mean())}
            detail = f'{fit.line}, {int(fit.template["n_gauss"])} Gaussian' + ('s' if not fit.single else '')
            for name, image, off_limit, label in (('vel', velocity, velocity_off_limit, 'Velocity'),
                                                  ('wid', width, width_off_limit, 'Width')):
                path = thumbnail_path(output_root, timestamp, iwin, title, name)
                render_thumbnail(image[:, order], extent, labels, f'{label} - {heading}', f'{date}  ({detail})', name, path,
                                 equal, off_limit[:, order])
                entry['files'].append(os.path.basename(path))
        except Exception:
            entry['error'] = traceback.format_exc(limit=3)
    return record


def process_raster(timestamp_or_file, output_root, work_dir=None, calibration='del_zanna_2025',
                   min_peak_snr=MIN_PEAK_SNR, min_good_fraction=MIN_GOOD_FRACTION, nonpositive=NONPOSITIVE):
    """Fetch a raster if needed, make its thumbnails and write its record.

    Parameters
    ----------
    timestamp_or_file : str
        Raster start time as 'YYYYMMDD_HHMMSS', or the path of a data file.
    output_root : str
        Top directory of the thumbnails.
    work_dir : str, optional
        Directory for downloads. A temporary directory by default.
    calibration : str, default='del_zanna_2025'
        Calibration of the intensities.
    min_peak_snr, min_good_fraction, nonpositive
        As for :func:`make_thumbnails`.

    Returns
    -------
    dict
        Record of the raster, written as `<level 0 file>.json` beside the
        thumbnails. A raster that could not be processed has `error` set, and
        is written with `.error` added to the name.
    """
    started = time.perf_counter()
    local = os.path.exists(timestamp_or_file)
    timestamp = _timestamp(timestamp_or_file)
    with tempfile.TemporaryDirectory(dir=work_dir) as temporary:
        try:
            data_file = timestamp_or_file if local else fetch_raster(timestamp, temporary)
            record = make_thumbnails(data_file, output_root, calibration, min_peak_snr, min_good_fraction,
                                     nonpositive)
        except RasterUnavailable as error:
            record = {'timestamp': timestamp, 'error': str(error)}
        except Exception:
            record = {'timestamp': timestamp, 'error': traceback.format_exc(limit=3)}
    record['seconds'] = round(time.perf_counter() - started, 1)
    # A failure goes in its own file, so it cannot replace a good record.
    path = _record_path(output_root, timestamp) + ('.error' if 'error' in record else '')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as output:
        json.dump(record, output, indent=1)
    return record


def _window_failed(window):
    """Return True if a window of a record has an error."""
    return 'error' in window or 'fit_error' in window


def _record_path(output_root, timestamp):
    """Return the path of a raster's record."""
    folder = os.path.dirname(thumbnail_path(output_root, timestamp, 0, 'x', 'int'))
    return os.path.join(folder, f'eis_l0_{timestamp}.fits.json')


def _is_done(output_root, raster):
    """Return True if a raster has a record with no errors."""
    try:
        with open(_record_path(output_root, _timestamp(raster))) as stored:
            record = json.load(stored)
    except (OSError, ValueError):
        return False
    return not any(_window_failed(w) for w in record['windows'])


def _process(arguments):
    """Process a raster in a worker, so that one failure does not stop the others."""
    try:
        return process_raster(*arguments)
    except Exception:
        return {'timestamp': str(arguments[0]), 'error': traceback.format_exc(limit=3), 'seconds': 0.0}


def main(argv=None):
    """Make thumbnails from the command line."""
    parser = argparse.ArgumentParser(description='Make EIS thumbnails for the SolarB archive.')
    parser.add_argument('rasters', nargs='*', help="Timestamps as YYYYMMDD_HHMMSS, or paths of data files")
    parser.add_argument('--date', action='append', default=[], help='Add every raster of a day (YYYY-MM-DD)')
    parser.add_argument('--output', required=True, help='Top directory of the thumbnails')
    parser.add_argument('--work', default=None, help='Directory for downloads')
    parser.add_argument('--processes', type=int, default=1, help='Rasters to process at once')
    parser.add_argument('--calibration', default='del_zanna_2025', choices=CALIBRATION_METHODS,
                        help='Intensity calibration')
    parser.add_argument('--min-peak-snr', type=float, default=MIN_PEAK_SNR,
                        help='Smallest fitted peak, in units of its error, for velocity and width (0 for no cut)')
    parser.add_argument('--min-good-fraction', type=float, default=MIN_GOOD_FRACTION,
                        help='Fraction of good points a pixel needs for its summed intensity')
    parser.add_argument('--nonpositive', default=NONPOSITIVE, choices=NONPOSITIVE_CHOICES,
                        help='Colour of summed intensities at or below zero')
    parser.add_argument('--overwrite', action='store_true', help='Remake rasters done without errors')
    args = parser.parse_args(argv)
    try:
        _check_options(args.min_peak_snr, args.min_good_fraction, args.nonpositive)
    except ValueError as error:
        parser.error(str(error))

    os.makedirs(args.output, exist_ok=True)
    if args.work is not None:
        os.makedirs(args.work, exist_ok=True)
    rasters = list(args.rasters)
    failed_dates = []
    for date in args.date:
        try:
            rasters.extend(list_rasters(date))
        except ConnectionError as error:
            print(error, flush=True)
            failed_dates.append(date)
    if not args.overwrite:
        rasters = [r for r in rasters if not _is_done(args.output, r)]

    jobs = [(r, args.output, args.work, args.calibration, args.min_peak_snr, args.min_good_fraction, args.nonpositive)
            for r in rasters]
    with multiprocessing.Pool(args.processes) as pool:
        for record in pool.imap_unordered(_process, jobs):
            failed = sum(_window_failed(w) for w in record.get('windows', []))
            print(record['timestamp'], f"{record['seconds']} s", record.get('error', f'{failed} windows failed'), flush=True)
    if failed_dates:
        sys.exit(f'No listing for {failed_dates}')


if __name__ == '__main__':
    main()
