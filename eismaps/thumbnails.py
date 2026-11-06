"""Quicklook thumbnails of EIS rasters for the MSSL SolarB archive.

:func:`make_thumbnails` writes one GIF per window and quantity, in the
directory layout and with the file names of the archive::

    YYYY/MM/DD/eis_l0_<time>.fits/eis_l0_<time>.fits_line_NN_<title>.<qty>.gif

`int` is the window summed over wavelength, so it exists for every window.
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
from PIL import Image

from eismaps.archive import RasterUnavailable, fetch_raster, list_rasters
from eismaps.batched_fit import cube_arrays, fit_cube
from eismaps.calibration import CALIBRATION_METHODS, calibration_factor
from eismaps.windows import choose_fit, parse_title

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

# Fewest pixels of a raster column for its own median velocity, and the
# fraction of the column they must be.
MIN_COLUMN_PIXELS = 3
MIN_COLUMN_FRACTION = 0.02

# Range of the symmetric velocity colour scale in km/s.
VELOCITY_SCALE_LIMITS = (10.0, 60.0)


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
        and the exposure time of the calibration in seconds, or None if no
        exposure has a duration.
    """
    with h5py.File(data_file.replace('.data.h5', '.head.h5'), 'r') as head:
        durations = np.atleast_2d(np.array(head['exposure_times/duration'], dtype=float))
    medians = np.array([np.median(d[d > 0]) if (d > 0).any() else 0.0 for d in durations])
    reference = float(np.sort(medians)[medians.size // 2])
    longest = int(np.argmax(medians)) if durations.shape[0] > 1 else None
    return longest, reference if reference > 0 else None


def _exposure_factor(cube, reference):
    """Return the intensity factor of each raster column for its own exposure time.

    Exposure times can change within a raster, for example during a flare.
    """
    duration = np.asarray(cube.meta['duration'], dtype=float)
    if reference is None or duration.size != cube.data.shape[1]:
        return np.ones(cube.data.shape[1])
    return np.where(duration > 0, reference / np.where(duration > 0, duration, 1.0), np.nan)


def _read_cube(data_file, iwin, exposure):
    """Read a window without EISPAC's messages.

    Raises
    ------
    ValueError
        If EISPAC cannot read the window, as for one position with several
        exposures.
    """
    messages = io.StringIO()
    with contextlib.redirect_stdout(messages):
        if exposure is None:
            cube = eispac.read_cube(data_file, iwin)
        else:
            cube = eispac.read_cube(data_file, iwin, exp_set=exposure)
    if cube is None:
        raise ValueError(f'EISPAC could not read window {iwin}: {messages.getvalue()[-200:]}')
    return cube


def _summed_intensity(wave, inten, errs):
    """Return the intensity of each pixel summed over the window, in erg / (cm2 s sr)."""
    good = errs > 0
    step = float(np.median(np.abs(np.diff(wave, axis=-1))))
    fraction = good.mean(axis=-1)
    summed = np.where(good, inten, 0.0).sum(axis=-1) * step
    # Missing points are taken to have the mean of the others.
    return np.where(fraction >= MIN_GOOD_FRACTION, summed / np.maximum(fraction, MIN_GOOD_FRACTION), np.nan)


def _mean_spectrum(wave, inten, errs):
    """Return the mean spectrum of a window, with the error of the mean."""
    good = errs > 0
    count = good.sum(axis=(0, 1))
    used = np.maximum(count, 1)
    mean = np.where(good, inten, 0.0).sum(axis=(0, 1)) / used
    error = np.sqrt(np.where(good, errs ** 2, 0.0).sum(axis=(0, 1))) / used
    return wave.mean(axis=(0, 1)), mean, np.where(count > 0, error, 0.0)


def _parameter_source(fit_res, index):
    """Return the parameter holding the value and error of a parameter, following any tie."""
    tied = str(fit_res.parinfo[index].get('tied', '') or '')
    match = re.search(r'p\[(\d+)\]', tied)
    return int(match.group(1)) if match else index


def _measured(fit_res, index):
    """Return True where a parameter of a usable fit is off its limits and has an error."""
    source = _parameter_source(fit_res, index)
    values = fit_res.fit['params'][:, :, source]
    errors = fit_res.fit['perror'][:, :, source]
    measured = errors > 0
    info = fit_res.parinfo[source]
    for side in (0, 1):
        if info['limited'][side]:
            measured &= values != info['limits'][side]
    return measured


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


def fitted_maps(fit_res, component):
    """Return velocity and width maps of one component.

    Pixels are left out where the fit is not usable, the peak is below
    `MIN_PEAK_SNR` errors, or the parameter is on a template limit.

    Parameters
    ----------
    fit_res : eispac.EISFitResult
        Fit of a window.
    component : int
        Gaussian to use.

    Returns
    -------
    tuple of np.ndarray
        Velocity in km/s, relative to the median of each raster column, and
        Gaussian width in Angstrom, shape (n_pixels, n_steps).
    """
    params, perror = fit_res.fit['params'], fit_res.fit['perror']
    peak, peak_error = params[:, :, 3 * component], perror[:, :, 3 * component]
    usable = (fit_res.fit['status'] > 0) & (peak > 0) & (peak_error > 0) & np.isfinite(fit_res.fit['chi2'])
    usable &= peak >= MIN_PEAK_SNR * peak_error

    centroid_index = 3 * component + 1
    rest = float(fit_res.parinfo[centroid_index]['value'])
    velocity = CC * (params[:, :, centroid_index] - rest) / rest
    velocity = np.where(usable & _measured(fit_res, centroid_index), velocity, np.nan)
    velocity = velocity - _column_zero_point(velocity)[None, :]

    width_index = 3 * component + 2
    width = np.where(usable & _measured(fit_res, width_index), params[:, :, width_index], np.nan)
    return velocity, width


def _arcsec(degrees):
    """Return angles in degrees as arcsec from -180 to 180 degrees, as the WCS can wrap them."""
    return (np.asarray(degrees) * 3600.0 + 648000.0) % 1296000.0 - 648000.0


def _axes(cube):
    """Return the image extent and axis labels of a window.

    Solar-X and Solar-Y in arcsec come from the cube WCS, which holds the
    pointing correction. Sit-and-stare observations have time along X.

    Returns
    -------
    tuple
        Extent as (left, right, bottom, top), X label, Y label, and True if
        the columns run backwards in time and must be reversed.
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

    if cube.meta['mod_index'].get('obs_type') == 'sit-and-stare':
        times = np.array(cube.meta['date_obs'], dtype='datetime64[ms]')
        minutes = (times - times.min()) / np.timedelta64(60, 's')
        reverse = minutes[0] > minutes[-1]
        span = float(minutes.max())
        half_t = 0.5 * span / (n_x - 1) if n_x > 1 and span > 0 else 0.5
        return (-half_t, span + half_t, bottom, top), 'Time since first exposure (minutes)', y_label, bool(reverse)

    return (world_x[0] - half_x, world_x[2] + half_x, bottom, top), 'Solar-X (arcsec)', y_label, False


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


def render_thumbnail(image, extent, labels, title, subtitle, measurement, path):
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
    """
    cmap_name, norm, label, extend = _scale(image, measurement)
    cmap = colormaps[cmap_name].with_extremes(bad='0.7')

    size = THUMBNAIL_PIXELS / 100.0
    figure = Figure(figsize=(size, size), dpi=100)
    canvas = FigureCanvasAgg(figure)
    axes = figure.add_axes([0.11, 0.09, 0.70, 0.80])
    bar = figure.add_axes([0.84, 0.09, 0.03, 0.80])
    shown = axes.imshow(image, origin='lower', aspect='auto', extent=extent, cmap=cmap, norm=norm,
                        interpolation='nearest')
    figure.colorbar(shown, cax=bar, label=label, extend=extend)
    axes.set_xlabel(labels[0])
    axes.set_ylabel(labels[1])
    figure.suptitle(f'{title}\n{subtitle}', fontsize=11)
    figure.text(0.995, 0.005, f"eismaps {version('eismaps')}, eispac {version('eispac')}",
                ha='right', va='bottom', fontsize=5, color='0.4')

    canvas.draw()
    pixels = np.asarray(canvas.buffer_rgba())[:, :, :3]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(pixels).quantize(colors=256, dither=Image.Dither.NONE).save(path, format='GIF')


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


def make_thumbnails(data_file, output_root, calibration='del_zanna_2025'):
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

    Returns
    -------
    dict
        Record of the raster: exposure used, and for each window its title,
        fit and thumbnails, or the error if a window failed.
    """
    if calibration not in CALIBRATION_METHODS:
        raise ValueError(f'Unknown calibration method: {calibration}. Choose from {sorted(CALIBRATION_METHODS)}')
    timestamp = _timestamp(data_file)
    wininfo = eispac.read_wininfo(data_file)
    exposure, reference_exposure = _exposure_choice(data_file)
    date = f'{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}T{timestamp[9:11]}:{timestamp[11:13]}:{timestamp[13:15]}'
    record = {'timestamp': timestamp, 'exposure_set': exposure, 'reference_exposure': reference_exposure,
              'calibration': calibration, 'windows': []}

    for row in wininfo:
        iwin, title = int(row['iwin']), str(row['line_id'])
        low, high = float(row['wvl_min']), float(row['wvl_max'])
        entry = {'iwin': iwin, 'title': title, 'files': []}
        record['windows'].append(entry)
        try:
            cube = _read_cube(data_file, iwin, exposure)
            wave, inten, errs = cube_arrays(cube)
            extent, x_label, y_label, reverse = _axes(cube)
            order = slice(None, None, -1) if reverse else slice(None)
            labels = (x_label, y_label)
            heading = f'{title.strip()} [{low:.2f}-{high:.2f} $\\AA$]'

            fit = None
            try:
                fit = choose_fit(title, low, high, *_mean_spectrum(wave, inten, errs))
            except Exception:
                entry['fit_error'] = traceback.format_exc(limit=3)
            line_wave = parse_title(fit.line)[1] if fit else parse_title(title)[1]
            factor = _calibration(date, (line_wave, 0.5 * (low + high)), calibration)
            entry['calibration_factor'] = factor

            intensity = _summed_intensity(wave, inten, errs) * _exposure_factor(cube, reference_exposure)[None, :]
            intensity = intensity * (1.0 if factor is None else factor)
            path = thumbnail_path(output_root, timestamp, iwin, title, 'int')
            render_thumbnail(intensity[:, order], extent, labels, f'Intensity - {heading}', date, 'int', path)
            entry['files'].append(os.path.basename(path))

            if fit is None:
                entry['fit'] = None
                continue
            tolerance = TOLERANCE if int(fit.template['n_gauss']) <= 2 else DEGENERATE_TOLERANCE
            result = fit_cube(cube, fit.template, fit.parinfo, component=fit.component, tolerance=tolerance)
            velocity, width = fitted_maps(result, fit.component)
            entry['fit'] = {'line': fit.line, 'source': fit.source, 'n_gauss': int(fit.template['n_gauss']),
                            'single': fit.single, 'intensity_ratio': fit.intensity_ratio,
                            'velocity_offset': fit.velocity_offset,
                            'fitted_fraction': float((result.fit['status'] > 0).mean()),
                            'velocity_fraction': float(np.isfinite(velocity).mean())}
            detail = f'{fit.line}, {int(fit.template["n_gauss"])} Gaussian' + ('s' if not fit.single else '')
            for name, image, label in (('vel', velocity, 'Velocity'), ('wid', width, 'Width')):
                path = thumbnail_path(output_root, timestamp, iwin, title, name)
                render_thumbnail(image[:, order], extent, labels, f'{label} - {heading}', f'{date}  ({detail})', name, path)
                entry['files'].append(os.path.basename(path))
        except Exception:
            entry['error'] = traceback.format_exc(limit=3)
    return record


def process_raster(timestamp_or_file, output_root, work_dir=None, calibration='del_zanna_2025'):
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
            record = make_thumbnails(data_file, output_root, calibration)
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
    parser.add_argument('--overwrite', action='store_true', help='Remake rasters done without errors')
    args = parser.parse_args(argv)

    os.makedirs(args.output, exist_ok=True)
    if args.work is not None:
        os.makedirs(args.work, exist_ok=True)
    rasters = list(args.rasters)
    for date in args.date:
        rasters.extend(list_rasters(date))
    if not args.overwrite:
        rasters = [r for r in rasters if not _is_done(args.output, r)]

    jobs = [(r, args.output, args.work, args.calibration) for r in rasters]
    with multiprocessing.Pool(args.processes) as pool:
        for record in pool.imap_unordered(_process, jobs):
            failed = sum(_window_failed(w) for w in record.get('windows', []))
            print(record['timestamp'], f"{record['seconds']} s", record.get('error', f'{failed} windows failed'), flush=True)


if __name__ == '__main__':
    main()
