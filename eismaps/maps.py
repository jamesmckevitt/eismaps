"""Convert EISPAC fit results into raster maps and plots."""

import eispac
import os
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolor
import sunpy.map
from sunpy.coordinates import frames
import astropy.units as u

from eismaps.utils import change_line_format
from eismaps.utils import apply_default_plot_settings

CC = 2.9979E+5 # speed of light
VALID_MEASUREMENTS = ['int', 'vel', 'wid', 'ntv', 'chi2']

def map_sd_filter(map,stds=3,log=False):
    """Mask outliers in a map using a standard-deviation threshold.

    Parameters
    ----------
    map : sunpy.map.Map
        Input map to filter.
    stds : float, default=3
        Number of standard deviations to keep.
    log : bool, default=False
        Apply the filter in log10 space.

    Returns
    -------
    sunpy.map.Map
        Filtered map copy.
    """

    data = np.array(map.data, copy=True)
    if log:
        data[data == 0] = np.nan
        data = np.log10(data)
    data_avg = np.nanmean(data)
    data_std = np.nanstd(data)

    filtered_data = np.array(map.data, copy=True)
    filtered_data[np.abs(data-data_avg) > stds*data_std] = np.nan

    filtered_map = sunpy.map.Map(filtered_data, map.meta)
    return apply_default_plot_settings(filtered_map)


def _normalise_measurements(measurement):
    """Return a validated list of requested measurement names."""
    if measurement is None:
        raise ValueError(f"A measurement must be specified. Valid options are {VALID_MEASUREMENTS}.")
    if isinstance(measurement, str):
        measurement = [measurement]

    invalid_measurements = [m for m in measurement if m not in VALID_MEASUREMENTS]
    if invalid_measurements:
        raise ValueError(f"Invalid measurement(s) specified: {invalid_measurements}. Must be chosen from {VALID_MEASUREMENTS}.")

    return measurement


def _coerce_fit_input(fit_input):
    """Normalise a fit input into an EISPAC fit result and source path.

    Parameters
    ----------
    fit_input : str or object
        Saved fit file path or raw EISPAC fit object.

    Returns
    -------
    tuple
        Two-element tuple ``(fit_result, source_file)``.
    """
    if hasattr(fit_input, 'fit') and hasattr(fit_input, 'get_map'):
        return fit_input, None
    return eispac.core.read_fit(fit_input), str(fit_input)


def _meta_scalar(value):
    """Return a plain Python scalar from EISPAC metadata values."""
    if isinstance(value, np.ndarray):
        if value.size == 1:
            value = value.reshape(-1)[0]
        else:
            value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return str(value)


def _default_fit_filename(fit_res):
    """Build a stable synthetic fit filename for in-memory fit results."""
    date_obs = _meta_scalar(fit_res.meta['date_obs'])
    line_id = fit_res.meta.get('line_id')
    if line_id is None:
        line_ids = fit_res.fit.get('line_ids')
        if line_ids is None or len(line_ids) == 0:
            line_id = 'unknown_line'
        else:
            line_id = change_line_format(_meta_scalar(line_ids[0]))
    else:
        line_id = _meta_scalar(line_id).replace(' ', '_').replace('.', '_').lower()
    file_date = date_obs.split('T')[0].replace('-', '')
    file_time = date_obs.split('T')[1].split('.')[0].replace(':', '')
    return f"eis_{file_date}_{file_time}.{line_id}.fit.h5"


def _package_maps(measurements, maps_by_measurement):
    """Return generated maps as SunPy objects instead of custom wrappers."""
    def package_one(map_list):
        if not map_list:
            return None
        if len(map_list) == 1:
            return map_list[0]
        return sunpy.map.Map(map_list, sequence=True)

    if len(measurements) == 1:
        return package_one(maps_by_measurement[measurements[0]])

    packaged = {}
    for current_measurement in measurements:
        packaged[current_measurement] = package_one(maps_by_measurement[current_measurement])
    return packaged
    
def batch(files, measurement=None, clip=False, save_fit=True, save_plot=False, output_dir=None, output_dir_tree=False, vel_los_correct=False, skip_done=True, mssl_solarb_file_format=False):
    """Create raster maps from saved or in-memory fit results.

    Parameters
    ----------
    files : list[str or object]
        Saved fit file paths or raw EISPAC fit results.
    measurement : list[str] or str
        Requested measurements from ``['int', 'vel', 'wid', 'ntv', 'chi2']``.
    clip : bool, default=False
        Apply outlier clipping to each generated map.
    save_fit : bool, default=True
        Save generated maps as FITS files.
    save_plot : bool, default=False
        Save generated quicklook plots.
    output_dir : str, optional
        Output directory used when saving in-memory products.
    output_dir_tree : bool, default=False
        If ``True``, create ``YYYY/MM/DD`` subdirectories beneath ``output_dir``.
    vel_los_correct : bool, default=False
        Apply a line-of-sight correction to velocity maps.
    skip_done : bool, default=True
        Skip files whose outputs already exist.
    mssl_solarb_file_format : bool, default=False
        Emit Solar-B style GIF names when saving plots.

    Returns
    -------
    sunpy.map.Map or sunpy.map.MapSequence or dict[str, sunpy.map.Map | sunpy.map.MapSequence]
        Generated maps packaged as standard SunPy objects.
    """

    measurement = _normalise_measurements(measurement)
    maps_by_measurement = {name: [] for name in measurement}

    for file in files:
        fit_res, fit_source_file = _coerce_fit_input(file)

        if fit_source_file is None and (save_fit or save_plot) and output_dir is None:
            raise ValueError('output_dir must be provided when saving maps generated from in-memory fit results.')

        if fit_source_file is not None:
            file_name = os.path.basename(fit_source_file)
            if file_name.endswith('.fit.h5'):
                file_date = os.path.basename(fit_source_file).split('.')[0].split('_')[1]
            else:
                file_name = _default_fit_filename(fit_res)
                file_date = file_name.split('.')[0].split('_')[1]
        else:
            file_name = _default_fit_filename(fit_res)
            file_date = file_name.split('.')[0].split('_')[1]

        if output_dir is not None:
            file_output_dir = output_dir
        elif fit_source_file is not None:
            file_output_dir = os.path.dirname(fit_source_file)
        else:
            file_output_dir = None

        if output_dir_tree and file_output_dir is not None:
            file_output_dir = os.path.join(file_output_dir, file_date[:4], file_date[4:6], file_date[6:8])
        if (save_fit or save_plot) and file_output_dir is not None:
            os.makedirs(file_output_dir, exist_ok=True)

        main_component = fit_res.fit['main_component']

        for m in measurement:
            output_file_fit = None
            if file_output_dir is not None:
                output_file_fit = os.path.join(file_output_dir, f"{file_name.replace('.fit.h5', f'.{m}.fits')}")

            if save_fit and skip_done and os.path.exists(output_file_fit):
                print(f"Skipping {m} map for {file_name} because it already exists.")
                continue

            output_file_png = None
            output_file_gif = None
            if save_plot:
                output_file_png = os.path.join(file_output_dir, f"{file_name.replace('.fit.h5', f'.{m}.png')}")
                if mssl_solarb_file_format:
                    output_file_png_filename = os.path.basename(output_file_png)
                    output_file_datetime = f"{output_file_png_filename.split('.')[0].split('_')[1]}_{output_file_png_filename.split('.')[0].split('_')[2]}"
                    output_file_iwin = fit_res.meta['iwin']
                    output_file_lineid = fit_res.meta['line_id']
                    output_file_gif_filename = f"eis_l0_{output_file_datetime}.fits_line_{output_file_iwin}_{output_file_lineid.replace(' ', '_').upper()}.{m}.gif"
                    output_file_gif = os.path.join(file_output_dir, output_file_gif_filename)
                if skip_done and os.path.exists(output_file_png):
                    print(f"Skipping {m} map for {file_name} because it already exists.")
                    continue

            if m == 'ntv':
                m_map = get_ntv_map(fit_res, component=main_component)
            elif m == 'chi2':
                m_map = get_chi2_map(fit_res, component=main_component)
            else:
                m_map = fit_res.get_map(component=main_component, measurement=m)

            if m_map is None:
                line_id = fit_res.meta.get('line_id', 'unknown')
                print(f"Skipping {m} map for {file_name} (line {line_id}, component {main_component}): get_map returned None.")
                continue

            if clip:
                if m == 'int':
                    m_map = map_sd_filter(m_map, stds=6, log=True)
                if m == 'vel':
                    m_map = map_sd_filter(m_map, stds=6)
                if m == 'wid':
                    m_map = map_sd_filter(m_map, stds=3)
                if m == 'ntv':
                    m_map = map_sd_filter(m_map, stds=3)

            if vel_los_correct and m == 'vel':
                helioprojective_coords = sunpy.map.all_coordinates_from_map(m_map)
                heliographic_coords = helioprojective_coords.transform_to(frames.HeliographicStonyhurst)
                latitude = heliographic_coords.lat.to(u.deg).value
                longitude = heliographic_coords.lon.to(u.deg).value

                observer_lon = m_map.observer_coordinate.lon.to(u.deg).value
                observer_lat = m_map.observer_coordinate.lat.to(u.deg).value

                latitude_rad = np.deg2rad(latitude)
                longitude_rad = np.deg2rad(longitude)
                observer_lon_rad = np.deg2rad(observer_lon)
                observer_lat_rad = np.deg2rad(observer_lat)

                los_factor = np.sin(latitude_rad) * np.sin(observer_lat_rad) + np.cos(latitude_rad) * np.cos(observer_lat_rad) * np.cos(longitude_rad - observer_lon_rad)

                corrected_data = np.full(m_map.data.shape, np.nan, dtype=float)
                np.divide(m_map.data, los_factor, out=corrected_data, where=np.abs(los_factor) > 1e-6)
                m_map = sunpy.map.Map(corrected_data, m_map.meta)

            m_map = apply_default_plot_settings(m_map, measurement=m)

            if save_fit:
                m_map.save(output_file_fit, overwrite=True)

            if save_plot:
                plt.figure()
                m_map.plot()
                extend = 'both' if m in {'vel', 'wid', 'ntv'} else 'max'
                cb_label = str(m_map.meta.get('measrmnt') or '').strip()
                bunit = str(m_map.meta.get('bunit') or '').strip()
                if cb_label and bunit:
                    cb_label = f'{cb_label} [{bunit}]'
                plt.colorbar(label=cb_label or None, extend=extend)

                if mssl_solarb_file_format:
                    plt.savefig(output_file_gif, dpi=100)
                else:
                    plt.savefig(output_file_png)
                plt.close()

            maps_by_measurement[m].append(m_map)

    return _package_maps(measurement, maps_by_measurement)

def get_ntv_map(fit_res, component=None):
    """Create a non-thermal velocity map from an EISPAC fit result.

    Parameters
    ----------
    fit_res : object
        EISPAC fit result.
    component : int, optional
        Fit component to extract. Defaults to the main component.

    Returns
    -------
    sunpy.map.Map
        Non-thermal velocity map in km/s.
    """
    from eismaps.utils import roman_to_int
    from eismaps.utils import width2velocity

    if component is None: component = fit_res.fit['main_component']

    map_wid = fit_res.get_map(component=component,measurement='wid')

    yy = map_wid.data.shape[0]
    xx = map_wid.data.shape[1]

    wavelength_array = np.full((yy,xx),map_wid.wavelength.to_value()) # fixed reference wavelength

    line = map_wid.meta['line_id'] # format e.g. ('line_id': 'Fe XII 195.119')
    line = line.replace('  ',' ') # protecting against e.g. 'S  XXI 135.8'

    element = line.split(' ')[0]
    ion = int(roman_to_int(line.split(' ')[1]))

    t_max, v_therm = width2velocity(element, ion)

    obser_fwhm = (map_wid.data*2*np.sqrt(2*np.log(2))) # set the observed fwhm (use the fwhm formula)

    if 'slit_width' in fit_res.meta and fit_res.meta['slit_width'] is not None:
        instr_fwhm = np.asarray(fit_res.meta['slit_width'], dtype=float)
        if instr_fwhm.ndim == 0:
            instr_fwhm = np.full((yy, xx), instr_fwhm)
        else:
            instr_fwhm = np.broadcast_to(instr_fwhm.reshape(yy, 1), (yy, xx))
    else:
        print("WARNING: Instrumental width not in metadata and so not used")
        instr_fwhm = np.zeros((yy, xx), dtype=float)

    therm_fwhm = np.sqrt(4*np.log(2))*wavelength_array*v_therm/CC # calculate the thermal fwhm

    dl_o = obser_fwhm # set the observed fwhm
    dl_i = instr_fwhm # set the instrument fwhm
    dl_t = therm_fwhm # set the thermal fwhm

    dl_nt_2 = (np.square(dl_o) - np.square(dl_i) - np.square(dl_t)) # calculate non-thermal velocity fwhm**2

    dl_nt_2 = np.where( dl_nt_2>0, dl_nt_2, 0 ) # replace negative values with 0

    v_nt = np.sqrt( dl_nt_2 * np.divide(CC**2, (4*np.log(2)*np.square(wavelength_array)), out=np.zeros_like(wavelength_array), where=wavelength_array!=0) ) # calculate non-thermal velocity using v_nt = np.sqrt(dl_nt_2 * (cc**2/(4*np.log(2)*np.square(wavelength)))) but covering the fact that sometimes the wavelength is 0

    map_meta = map_wid.meta.copy()
    map_meta['measrmnt'] = 'non-thermal velocity'
    map_meta['bunit'] = 'km/s'
    map_ntv = sunpy.map.Map(v_nt, map_meta) # create a non-thermal velocity map

    return apply_default_plot_settings(map_ntv, measurement='ntv')

# def get_bwa_map(fit_res):
#     # blue wing asymmetry
#     return None

def get_chi2_map(fit_res, component=None):
    """Create a chi-squared map from an EISPAC fit result.

    Parameters
    ----------
    fit_res : object
        EISPAC fit result.
    component : int, optional
        Fit component whose metadata will be used for the output map.

    Returns
    -------
    sunpy.map.Map
        Chi-squared map.
    """
    if component is None: component = fit_res.fit['main_component']

    map_int = fit_res.get_map(component=component,measurement='int')

    map_meta = map_int.meta.copy()
    map_meta['measrmnt'] = 'chi2'
    map_meta['bunit'] = 'chi2'
    map_chi2 = sunpy.map.Map(fit_res.fit['chi2'], map_meta)

    return apply_default_plot_settings(map_chi2, measurement='chi2')


def _resolve_ncpu(ncpu):
    """Return a validated worker count from ``ncpu`` input."""
    if ncpu in (None, 'max'):
        return max(1, os.cpu_count() or 1)
    if isinstance(ncpu, int) and ncpu > 0:
        return ncpu
    raise ValueError("ncpu must be a positive integer or 'max'.")


def _flatten_map_collection(map_collection):
    """Return a flat list of SunPy maps from a map or map-sequence container."""
    if map_collection is None:
        return []
    if hasattr(map_collection, 'maps'):
        return list(map_collection.maps)
    if hasattr(map_collection, 'data') and hasattr(map_collection, 'meta'):
        return [map_collection]
    return list(map_collection)


def _package_map_collection(map_list):
    """Package a flat list into a single map or a MapSequence."""
    if not map_list:
        return None
    if len(map_list) == 1:
        return map_list[0]
    return sunpy.map.Map(map_list, sequence=True)


def make_maps(
    fits,
    *,
    measurement='int',
    ncpu='max',
    clip=False,
    save=None,
    save_fit=None,
    save_plot=False,
    output_dir=None,
    output_dir_tree=False,
    vel_los_correct=False,
    skip_done=True,
    mssl_solarb_file_format=False,
):
    """Create SunPy map objects from saved or in-memory EISPAC fit results.

    Multiple fit items are processed in parallel when ``ncpu`` allows it.
    Results for each requested measurement are merged into a single map or
    MapSequence.

    Parameters
    ----------
    fits : object or list
        A single EISPAC fit result, a list of fit results, or a list of saved
        fit file paths.
    measurement : str or list[str], default='int'
        Measurement(s) to extract.  One or more of ``'int'``, ``'vel'``,
        ``'wid'``, ``'ntv'``, ``'chi2'``.
    ncpu : int or str, default='max'
        Number of parallel workers.  ``'max'`` uses all available CPUs.
    clip : bool, default=False
        Apply outlier clipping to each generated map.
    save : bool or None, default=None
        Whether to save map FITS products.  Defaults to ``True`` when
        ``output_dir`` is given, ``False`` otherwise.
    save_fit : bool or None, default=None
        Override for ``save`` that applies only to map FITS files.
    save_plot : bool, default=False
        Save quicklook plots alongside FITS outputs.
    output_dir : str, optional
        Directory for saved products.
    output_dir_tree : bool, default=False
        Append ``YYYY/MM/DD`` sub-directories under ``output_dir``.
    vel_los_correct : bool, default=False
        Apply a line-of-sight correction to velocity maps.
    skip_done : bool, default=True
        Skip maps whose output files already exist.
    mssl_solarb_file_format : bool, default=False
        Use Solar-B style filenames for saved plots.

    Returns
    -------
    sunpy.map.Map or sunpy.map.MapSequence or dict
        When a single measurement is requested the return value is a map or
        MapSequence.  When multiple measurements are requested the return
        value is a ``dict`` keyed by measurement name.
    """
    if hasattr(fits, 'fit') and hasattr(fits, 'get_map'):
        fits = [fits]
    fit_items = list(fits)

    resolved_save_fit = (
        (output_dir is not None)
        if save is None and save_fit is None
        else (save if save is not None else save_fit)
    )
    workers = _resolve_ncpu(ncpu)

    if workers == 1 or len(fit_items) <= 1 or measurement is None:
        return batch(
            fit_items,
            measurement=measurement,
            clip=clip,
            save_fit=resolved_save_fit,
            save_plot=save_plot,
            output_dir=output_dir,
            output_dir_tree=output_dir_tree,
            vel_los_correct=vel_los_correct,
            skip_done=skip_done,
            mssl_solarb_file_format=mssl_solarb_file_format,
        )

    def _make_maps_for_single_fit(fit_item):
        return batch(
            [fit_item],
            measurement=measurement,
            clip=clip,
            save_fit=resolved_save_fit,
            save_plot=save_plot,
            output_dir=output_dir,
            output_dir_tree=output_dir_tree,
            vel_los_correct=vel_los_correct,
            skip_done=skip_done,
            mssl_solarb_file_format=mssl_solarb_file_format,
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        batched_results = list(executor.map(_make_maps_for_single_fit, fit_items))

    measurement_names = [measurement] if isinstance(measurement, str) else list(measurement)

    if len(measurement_names) == 1:
        merged = []
        for result in batched_results:
            merged.extend(_flatten_map_collection(result))
        return _package_map_collection(merged)

    merged_by_measurement = {name: [] for name in measurement_names}
    for result in batched_results:
        for name in measurement_names:
            merged_by_measurement[name].extend(_flatten_map_collection(result.get(name)))
    return {name: _package_map_collection(merged_by_measurement[name]) for name in measurement_names}