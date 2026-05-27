"""Calibration helpers for Hinode/EIS rasters.

The module keeps the Python calibration logic close to the SolarSoft EIS
reference implementation.  Raw calibration assets can be synced from an SSW
tree into ``eismaps/calibration_data`` with
``eismaps-sync-calibration``.
"""

import argparse
import datetime
import json
import os
import re
import shutil
import warnings
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.io import readsav
import sunpy.map

from eismaps.utils import apply_default_plot_settings

CALIBRATION_DATA_PATH = Path(__file__).with_name('calibration_data')
CALIBRATION_MANIFEST_FILE = CALIBRATION_DATA_PATH / 'sources.json'

FIT_EA_PATTERN = 'fit_eis_ea_*.sav'
NRL_EA_PATTERN = 'nrl_ea_coeff_v*.genx'
LEGACY_WARREN_2014_FILE = CALIBRATION_DATA_PATH / 'eis_calib_warren_2014.sav'
LEGACY_PREFLIGHT_SHORT_FILE = CALIBRATION_DATA_PATH / 'preflight_calib_short.sav'
LEGACY_PREFLIGHT_LONG_FILE = CALIBRATION_DATA_PATH / 'preflight_calib_long.sav'
RAW_PREFLIGHT_SHORT_PATTERN = 'EIS_EffArea_B.*'
RAW_PREFLIGHT_LONG_PATTERN = 'EIS_EffArea_A.*'
GDZ_2013_SHORT_RESPONSE_FILE = CALIBRATION_DATA_PATH / 'EIS_EffArea_B.004'
GDZ_2013_LONG_RESPONSE_FILE = CALIBRATION_DATA_PATH / 'EIS_EffArea_A.004'
SSW_FIT_EA_DIR = Path('hinode/eis/idl/atest/hwarren/calibration/new')
SSW_NRL_EA_DIR = Path('hinode/eis/idl/atest/hwarren/calibration')
SSW_RESPONSE_DIR = Path('hinode/eis/response')

_WARNED_WARREN_CACHE = False

__all__ = [
    'apply_calibration',
    'calib_2014',
    'calib_2023',
    'calib_del_zanna_2013',
    'calib_del_zanna_2025',
    'calibrate_map',
    'eis_ltds',
    'interpol_eis_ea',
    'eis_ea',
    'sync_solarsoft_calibration_data',
    'sync_solarsoft_calibration_data_cli',
]

CALIBRATION_LABELS = {
    'ground': 'Ground',
    'ground_cal': 'Ground',
    'preflight': 'Preflight',
    'del_zanna_2013': 'Del Zanna 2013',
    'warren_2014': 'Warren 2014',
    'del_zanna_2025': 'Del Zanna 2025',
    'del_zanna_2023': 'Del Zanna 2025',
}

DEFAULT_INTENSITY_BUNIT = 'erg / (s sr cm2)'

GDZ_2013_SW_WAVELENGTHS = np.array([
    165.0, 171.0, 174.5, 177.2, 178.1, 180.4, 182.2, 184.5, 185.2,
    186.9, 188.3, 190.0, 192.4, 192.8, 193.5, 194.7, 195.1, 196.6,
    197.4, 200.0, 201.1, 202.0, 202.7, 204.9, 208.0, 209.9, 211.3,
], dtype=float)
GDZ_2013_SW_EFFECTIVE_AREA = np.array([
    0.000174973 / 1.5,
    0.000255772 / 1.5,
    0.00158207 / 1.5,
    0.00476608 / 1.55,
    0.00705735 / 1.5,
    0.0168637 / 1.45,
    0.0316499 / 1.4,
    0.0647319 / 1.35,
    0.0779082 / 1.35,
    0.115240 / 1.4,
    0.150199 / 1.45,
    0.194897 / 1.25,
    0.255993 / 1.13,
    0.264945 / 1.1,
    0.279607 / 1.05,
    0.298884 / 1.02,
    0.302737,
    0.301859 / 1.05,
    0.287675 / 1.15,
    0.174608 * 1.05,
    0.119586,
    0.0838537,
    0.0635698,
    0.0332376,
    0.0189209,
    0.0133581,
    0.0105513,
], dtype=float)
GDZ_2013_LW_WAVELENGTHS = np.array([
    245.0, 252.0, 255.0, 257.0, 259.0, 263.0, 265.0, 268.0, 270.0,
    272.0, 274.0, 277.0, 281.0, 286.0, 292.0,
], dtype=float)
GDZ_2013_LW_EFFECTIVE_AREA = np.array([
    0.022673 * 0.8,
    0.03908 * 0.75,
    0.05065 * 0.78,
    0.0588 * 0.8,
    0.06738 * 0.85,
    0.0861 * 0.9,
    0.09551 * 0.95,
    0.106984,
    0.110764 * 1.02,
    0.10944 * 1.03,
    0.1026 * 1.03,
    0.084775 * 0.9,
    0.05718 * 0.87,
    0.0333 * 0.85,
    0.01679 * 0.85,
], dtype=float) / 1.1
GDZ_2013_REFERENCE_DATE = '2006-09-22T21:36:00.000'
GDZ_2013_LAST_DATE = '2012-09-14T00:00:00.000'
GDZ_2013_LW_COEFFICIENTS = np.array([1.0326230, -5.2495791e-09, 1.2055185e-17], dtype=float)


def _latest_matching_file(directory, pattern):
    """Return the lexicographically latest file matching ``pattern`` in ``directory``."""
    matches = sorted(Path(directory).glob(pattern))
    return matches[-1] if matches else None


def _resolve_fit_ea_file(ea_file=None):
    """Return the fit-based Del Zanna calibration file to use."""
    if ea_file is not None:
        return Path(ea_file)

    resolved_file = _latest_matching_file(CALIBRATION_DATA_PATH, FIT_EA_PATTERN)
    if resolved_file is None:
        raise FileNotFoundError(
            'No SolarSoft fit_eis_ea_*.sav file is available in calibration_data. '
            'Run eismaps-sync-calibration to copy the latest SSW calibration assets.'
        )
    return resolved_file


def _resolve_response_file(short=False, long=False):
    """Return the packaged pre-flight effective-area file for one channel."""
    if short == long:
        raise ValueError('Exactly one of short=True or long=True must be set.')

    pattern = RAW_PREFLIGHT_SHORT_PATTERN if short else RAW_PREFLIGHT_LONG_PATTERN
    legacy_file = LEGACY_PREFLIGHT_SHORT_FILE if short else LEGACY_PREFLIGHT_LONG_FILE
    resolved_file = _latest_matching_file(CALIBRATION_DATA_PATH, pattern)
    if resolved_file is not None:
        return resolved_file
    if legacy_file.exists():
        return legacy_file

    channel = 'short-wave' if short else 'long-wave'
    raise FileNotFoundError(
        f'No {channel} effective-area file is available in calibration_data. '
        'Run eismaps-sync-calibration to copy the latest SSW response files.'
    )


def _resolve_del_zanna_2013_response_file(short=False, long=False):
    """Return the response file used by the Del Zanna 2013 degradation model."""
    if short == long:
        raise ValueError('Exactly one of short=True or long=True must be set.')

    preferred_file = GDZ_2013_SHORT_RESPONSE_FILE if short else GDZ_2013_LONG_RESPONSE_FILE
    if preferred_file.exists():
        return preferred_file
    return _resolve_response_file(short=short, long=long)


def _warn_legacy_warren_cache():
    """Warn once when the cached Warren 2014 coefficients are used."""
    global _WARNED_WARREN_CACHE
    if _WARNED_WARREN_CACHE:
        return

    _WARNED_WARREN_CACHE = True
    warnings.warn(
        'Using legacy Warren 2014 coefficients from eis_calib_warren_2014.sav. '
        'The raw SolarSoft nrl_ea_coeff_v*.genx file is copied for provenance, '
        'but direct Python decoding is not implemented yet.',
        RuntimeWarning,
        stacklevel=2,
    )


def _load_warren_2014_coefficients(file_path=None):
    """Load Warren 2014 calibration coefficients from the packaged cache."""
    resolved_file = Path(file_path) if file_path is not None else LEGACY_WARREN_2014_FILE
    if resolved_file.suffix.lower() == '.genx':
        raise ValueError(
            'Direct reading of SolarSoft .genx Warren coefficient files is not implemented. '
            'Use the cached eis_calib_warren_2014.sav file or provide a converted .sav file.'
        )
    if not resolved_file.exists():
        raw_file = _latest_matching_file(CALIBRATION_DATA_PATH, NRL_EA_PATTERN)
        if raw_file is not None:
            raise FileNotFoundError(
                'Found a SolarSoft nrl_ea_coeff_v*.genx file, but no converted '
                'eis_calib_warren_2014.sav cache is available. Keep the cache in '
                'calibration_data until a direct .genx reader is added.'
            )
        raise FileNotFoundError(
            'No Warren 2014 calibration coefficients are available in calibration_data.'
        )

    _warn_legacy_warren_cache()
    return readsav(resolved_file)['eis']


def _read_effective_area_table(file_path):
    """Read a pre-flight effective-area table from text or IDL save format."""
    resolved_file = Path(file_path)
    if resolved_file.suffix.lower() == '.sav':
        preflight = readsav(resolved_file)
        return np.asarray(preflight['wave'], dtype=float), np.asarray(preflight['ea'], dtype=float)

    wave, area = np.loadtxt(resolved_file, comments='#', unpack=True)
    return np.asarray(wave, dtype=float), np.asarray(area, dtype=float)


def _resolve_ssw_root(ssw_root=None):
    """Resolve and validate the local SolarSoft root directory."""
    if ssw_root is not None:
        resolved_root = Path(ssw_root).expanduser().resolve()
    else:
        ssw_env = Path(os.environ['SSW']).expanduser() if 'SSW' in os.environ else None
        if ssw_env is None:
            raise ValueError('An SSW root was not provided and the SSW environment variable is not set.')
        resolved_root = ssw_env.resolve()

    if not resolved_root.exists():
        raise FileNotFoundError(f'SSW root does not exist: {resolved_root}')

    return resolved_root


def _spline_interpolate(x_values, y_values, target_values):
    """Sample a calibration curve using cubic interpolation where possible."""
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    target_values = np.atleast_1d(np.asarray(target_values, dtype=float))
    kind = 'cubic' if x_values.size >= 4 else 'linear'
    sampled = interp1d(x_values, y_values, kind=kind, bounds_error=False, fill_value='extrapolate')(target_values)
    return sampled


def _calibrated_meta(map_obj, method):
    """Return updated metadata for a calibrated intensity map."""
    calibration_label = CALIBRATION_LABELS.get(method, method.replace('_', ' ').title())
    base_measurement = str(map_obj.meta.get('measrmnt', '')).strip()
    if not base_measurement or 'intensity' not in base_measurement.lower():
        base_measurement = 'Intensity'
    elif ' (' in base_measurement:
        base_measurement = base_measurement.split(' (', 1)[0]
    else:
        base_measurement = 'Intensity'

    calibrated_meta = map_obj.meta.copy()
    calibrated_meta['measrmnt'] = f'{base_measurement} ({calibration_label})'

    current_bunit = str(calibrated_meta.get('bunit', '')).strip()
    if not current_bunit:
        calibrated_meta['bunit'] = DEFAULT_INTENSITY_BUNIT

    return calibrated_meta


def sync_solarsoft_calibration_data(ssw_root=None, destination_dir=CALIBRATION_DATA_PATH, overwrite=True):
    """Copy the SolarSoft EIS calibration assets used by this package.

    Parameters
    ----------
    ssw_root : str or pathlib.Path, optional
        Root of the local SolarSoft checkout. Defaults to the ``SSW``
        environment variable.
    destination_dir : str or pathlib.Path, optional
        Directory that will receive the copied calibration assets.
    overwrite : bool, default=True
        Whether to replace existing files in ``destination_dir``.

    Returns
    -------
    dict
        Manifest describing the copied files.
    """
    resolved_root = _resolve_ssw_root(ssw_root)
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    source_specs = {
        'fit_ea': (resolved_root / SSW_FIT_EA_DIR, FIT_EA_PATTERN),
        'nrl_coefficients': (resolved_root / SSW_NRL_EA_DIR, NRL_EA_PATTERN),
        'preflight_long': (resolved_root / SSW_RESPONSE_DIR, RAW_PREFLIGHT_LONG_PATTERN),
        'preflight_short': (resolved_root / SSW_RESPONSE_DIR, RAW_PREFLIGHT_SHORT_PATTERN),
        'del_zanna_2013_preflight_short': (resolved_root / SSW_RESPONSE_DIR, 'EIS_EffArea_B.004'),
    }

    copied = {}
    for key, (directory, pattern) in source_specs.items():
        source_file = _latest_matching_file(directory, pattern)
        if source_file is None:
            raise FileNotFoundError(f'Could not find {pattern} under {directory}')

        destination_file = destination_dir / source_file.name
        if overwrite or not destination_file.exists():
            shutil.copy2(source_file, destination_file)

        copied[key] = {
            'source': str(source_file),
            'destination': str(destination_file),
        }

    manifest = {
        'synced_at_utc': datetime.datetime.utcnow().replace(microsecond=0).isoformat() + 'Z',
        'ssw_root': str(resolved_root),
        'copied': copied,
        'notes': [
            'fit_eis_ea_*.sav is used directly for the Del Zanna 2025-style interpolation path.',
            'EIS_EffArea_A.* and EIS_EffArea_B.* are used directly for the pre-flight effective areas.',
            'nrl_ea_coeff_v*.genx is copied for provenance; the Warren 2014 Python path still uses the converted eis_calib_warren_2014.sav cache.',
        ],
    }
    CALIBRATION_MANIFEST_FILE.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def sync_solarsoft_calibration_data_cli():
    """CLI wrapper for copying SolarSoft calibration assets into the package."""
    parser = argparse.ArgumentParser(description='Sync EIS calibration assets from SolarSoft into eismaps.')
    parser.add_argument('--ssw-root', help='Path to the local SolarSoft checkout. Defaults to the SSW environment variable.')
    parser.add_argument('--destination', default=str(CALIBRATION_DATA_PATH), help='Destination directory for copied assets.')
    parser.add_argument('--no-overwrite', action='store_true', help='Do not overwrite destination files that already exist.')
    args = parser.parse_args()

    manifest = sync_solarsoft_calibration_data(
        ssw_root=args.ssw_root,
        destination_dir=args.destination,
        overwrite=not args.no_overwrite,
    )
    print(json.dumps(manifest, indent=2))

def anytim2tai(time_str):
    """Convert a compact ISO-like timestamp string into TAI seconds."""
    time_str = re.sub(r'[^\w\s.]', '', time_str)
    time_str, _, fractional_part = time_str.partition('.')
    if 'T' in time_str:
        date_str, time_str = time_str.split('T')
        time_str = f"{date_str} {time_str}"
    else:
        time_str = re.sub(r'[^\w\s]', '', time_str)
    dt = datetime.datetime.strptime(time_str, '%Y%m%d %H%M%S')
    seconds_since_epoch = (dt - datetime.datetime(1970, 1, 1)).total_seconds()
    tai_offset = (datetime.datetime(1970, 1, 1) - datetime.datetime(1958, 1, 1)).total_seconds()
    tai_time = seconds_since_epoch + tai_offset + 37
    return tai_time

def interpol_eis_ea(date, wavelength, short=False, long=False, radcal=False, ea_file=None, quiet=False):
    """Interpolate Del Zanna fit-based effective areas for one observation date."""
    if np.size(date) != 1:
        raise ValueError('ERROR: please input a single date')
    
    # Ensure wavelength is always treated as an array
    wavelength = np.atleast_1d(wavelength)
    
    in_tai = anytim2tai(date)

    if in_tai < anytim2tai('2006-10-20T10:20:00.000'):
        print('WARNING: Selected date is before the start of normal EIS science operations. Output values may be inaccurate.')

    if not short and not long:
        n_input_wave = np.size(wavelength)
        loc_short = np.where((wavelength >= 165) & (wavelength <= 213))[0]
        loc_long = np.where((wavelength >= 245) & (wavelength <= 292))[0]
        if (len(loc_short) + len(loc_long) < n_input_wave) or (len(loc_short) > 0 and len(loc_long) > 0):
            raise ValueError('ERROR: Invalid wavelength(s). Please only select values in either the short (165 - 213) or long (245 - 292) wavelength bands.')

    if short:
        wavelength = 1
    elif long:
        wavelength = 1000

    ea_file = _resolve_fit_ea_file(ea_file)
    fit_ea = readsav(ea_file)['fit_ea']
    fit_dates = fit_ea.date_obs[0].astype(str)
    fit_easw = fit_ea.sw_ea[0]
    fit_ealw = fit_ea.lw_ea[0]
    sw_wave = fit_ea.sw_wave[0]
    lw_wave = fit_ea.lw_wave[0]

    ref_tai = np.array([anytim2tai(date) for date in fit_dates])

    if in_tai < ref_tai[0]:
        if not quiet:
            print(f"WARNING: Selected date is before the first calibrated date on {fit_ea.date_obs[0]}. Returning first fit calibration")
        in_tai = ref_tai[0]

    if in_tai > ref_tai[-1]:
        if not quiet:
            print(f"WARNING: Selected date is after the last calibrated date on {fit_ea.date_obs[-1][-1]}. Returning last fit calibration")
        in_tai = ref_tai[-1]

    if short or (np.size(wavelength) > 0 and np.max(wavelength) < 220):
        ref_ea = fit_easw
        ref_wave = sw_wave
    else:
        ref_ea = fit_ealw
        ref_wave = lw_wave

    n_ref_waves = len(ref_wave)
    new_ea = np.zeros(n_ref_waves)
    for w in range(n_ref_waves):
        ea_values = ref_ea[w, :]
        new_ea[w] = np.interp(in_tai, ref_tai, ea_values)

    if not short and not long:
        out_ea = interp1d(ref_wave, new_ea, kind='cubic')(wavelength)
    else:
        wavelength = ref_wave
        out_ea = new_ea

    if radcal:
        sr_factor = (725.0 / 1.496e8) ** 2
        ergs_to_photons = 6.626e-27 * 2.998e10 * 1.e8
        gain = 6.3
        phot_to_elec = 12398.5 / 3.65
        tau_sensitivity = 1894.0

        print('Returning radcal values for converting [DN/s] to [ergs/(sr cm^2 s)]')
        print('   Note: You may still need to adjust for exposure time and slitsize.')

        radcal = (wavelength * gain) / (out_ea * phot_to_elec)
        radcal = radcal * ergs_to_photons / wavelength / sr_factor
        out_ea = radcal

    return out_ea


def eis_ltds(date, wavelengths, quiet=False, undecay=False):
    """Return Del Zanna 2013 correction factors for ground-calibrated intensities.

    Parameters
    ----------
    date : str
        Observation timestamp.
    wavelengths : float or array-like
        EIS wavelengths in Angstrom.
    quiet : bool, default=False
        Suppress informational messages.
    undecay : bool, default=False
        Undo the older global 1894-day exponential decay correction.

    Returns
    -------
    float or numpy.ndarray
        Multiplicative correction factors for the supplied wavelengths.
    """
    wavelength_array = np.atleast_1d(np.asarray(wavelengths, dtype=float))
    short_mask = (wavelength_array > 165.0) & (wavelength_array < 212.0)
    long_mask = (wavelength_array > 245.0) & (wavelength_array < 292.0)

    if not np.any(short_mask | long_mask):
        raise ValueError('No input wavelengths fall within the EIS spectral bands.')

    eff_lw_wave, eff_lw_area = _read_effective_area_table(_resolve_del_zanna_2013_response_file(long=True))
    eff_sw_wave, eff_sw_area = _read_effective_area_table(_resolve_del_zanna_2013_response_file(short=True))
    effa_sw = _spline_interpolate(GDZ_2013_SW_WAVELENGTHS, GDZ_2013_SW_EFFECTIVE_AREA, eff_sw_wave)
    effa_lw = _spline_interpolate(GDZ_2013_LW_WAVELENGTHS, GDZ_2013_LW_EFFECTIVE_AREA, eff_lw_wave)

    xtime = anytim2tai(date)
    xtime_save = xtime
    xtime_ref = anytim2tai(GDZ_2013_REFERENCE_DATE)
    xtime_last = anytim2tai(GDZ_2013_LAST_DATE)
    corrections = np.ones(wavelength_array.size, dtype=float)

    if xtime < xtime_ref:
        if not quiet:
            print('% EIS_LTDS: input time before launch date. Returning unity correction factors.')
        return corrections[0] if np.isscalar(wavelengths) else corrections

    if xtime > xtime_last:
        if not quiet:
            print('% EIS_LTDS: input time is after 14-Sep-2012. Freezing the LW degradation at that date.')
        xtime = xtime_last

    lw_corr = np.polyval(GDZ_2013_LW_COEFFICIENTS[::-1], xtime - xtime_ref)
    effa_lw = lw_corr * effa_lw

    if np.any(short_mask):
        corrections[short_mask] = (
            _spline_interpolate(eff_sw_wave, eff_sw_area, wavelength_array[short_mask])
            / _spline_interpolate(eff_sw_wave, effa_sw, wavelength_array[short_mask])
        )

    if np.any(long_mask):
        corrections[long_mask] = (
            _spline_interpolate(eff_lw_wave, eff_lw_area, wavelength_array[long_mask])
            / _spline_interpolate(eff_lw_wave, effa_lw, wavelength_array[long_mask])
        )

    if undecay:
        dt_days = (xtime_save - xtime_ref) / 86400.0
        decay_factor = np.exp(-dt_days / 1894.0)
        if not quiet:
            print('% EIS_LTDS: undoing the historic 1894-day universal exponential decay correction.')
        corrections = corrections * decay_factor

    if np.isscalar(wavelengths):
        return corrections[0]
    return corrections


def calib_del_zanna_2013(map):
    """Calibrate a map using the Del Zanna 2013 time-dependent sensitivity model."""
    match = re.search(r'\d+\.\d+', map.meta['line_id'])
    if match is None:
        raise ValueError('The map metadata does not contain a parsable line_id wavelength.')
    wavelength = float(match.group())
    calib_ratio = eis_ltds(map.date.value, wavelength)
    print(f'Calibration ratio for {wavelength} A: {calib_ratio} using map date {map.date.value} was applied.')
    calibrated_map = sunpy.map.Map(map.data * calib_ratio, _calibrated_meta(map, 'del_zanna_2013'))
    return apply_default_plot_settings(calibrated_map)


def calib_del_zanna_2025(map):
    """Calibrate a map using the fit-based Del Zanna 2025 sensitivity model."""
    match = re.search(r'\d+\.\d+', map.meta['line_id'])
    wvl_value = float(match.group())
    calib_ratio = eis_ea(wvl_value) / interpol_eis_ea(map.date.value, wvl_value)
    print(f'Calibration ratio for {wvl_value} A: {calib_ratio} using map date {map.date.value} was applied.')
    calibrated_map = sunpy.map.Map(map.data * calib_ratio, _calibrated_meta(map, 'del_zanna_2025'))
    return apply_default_plot_settings(calibrated_map)


def calibrate_map(map, method='del_zanna_2025'):
    """Apply one of the supported intensity calibration methods to a map.

    Supported methods are ``ground``, ``ground_cal``, ``preflight``,
    ``del_zanna_2013``, ``warren_2014``, ``del_zanna_2025``, and the legacy
    alias ``del_zanna_2023``.
    """
    methods = {
        'ground': lambda current_map: apply_default_plot_settings(sunpy.map.Map(np.array(current_map.data, copy=True), _calibrated_meta(current_map, 'ground'))),
        'ground_cal': lambda current_map: apply_default_plot_settings(sunpy.map.Map(np.array(current_map.data, copy=True), _calibrated_meta(current_map, 'ground_cal'))),
        'preflight': lambda current_map: apply_default_plot_settings(sunpy.map.Map(np.array(current_map.data, copy=True), _calibrated_meta(current_map, 'preflight'))),
        'del_zanna_2013': calib_del_zanna_2013,
        'warren_2014': calib_2014,
        'del_zanna_2025': calib_del_zanna_2025,
        'del_zanna_2023': calib_del_zanna_2025,
    }
    if method not in methods:
        raise ValueError(f'Unknown calibration method: {method}. Choose from {sorted(methods)}')
    return methods[method](map)

def calib_2023(map):
    """Backward-compatible alias for the Del Zanna 2025 fit-based calibration."""
    return calib_del_zanna_2025(map)

def calib_2014(map):
    """Calibrate a map using the Warren 2014 sensitivity model."""
    match = re.search(r'\d+\.\d+', map.meta['line_id'])
    wvl_value = float(match.group())
    calib_ratio = eis_ea(wvl_value) / eis_ea_nrl(map.date.value, wvl_value)
    print(f'Calibration ratio for {wvl_value} A: {calib_ratio} using map date {map.date.value} was applied.')
    new_map = sunpy.map.Map(map.data * calib_ratio, _calibrated_meta(map, 'warren_2014'))
    return apply_default_plot_settings(new_map)

def eis_ea(input_wave, short=False, long=False):
    """Return the ground-calibration effective area for one or more wavelengths."""
    if short:
        wave, ea = eis_effective_area_read(short=True)
        input_wave = wave
        return ea

    if long:
        wave, ea = eis_effective_area_read(long=True)
        input_wave = wave
        return ea

    if isinstance(input_wave, (int, float)):
        input_wave = np.array([input_wave])

    nWave = len(input_wave)
    ea = np.zeros(nWave)

    for i in range(nWave):
        short, long = is_eis_wavelength(input_wave[i])

        if not short and not long:
            ea[i] = 0.0
        else:
            wave, area = eis_effective_area_read(long=long, short=short)
            ea[i] = np.exp(np.interp(input_wave[i], wave, np.log(area)))

    if nWave == 1:
        ea = ea[0]

    return ea

def eis_ea_nrl(date, wave, short=False, long=False):
    """Return Warren 2014 effective areas for one date and one or more wavelengths."""
    eis = read_calib_file()
    t = (get_time_tai(date) - get_time_tai(eis['t0'][0].decode('utf-8'))) / (86400 * 365.25)
    ea_knots_SW = eis['a0_sw'][0] * np.exp(-t / eis['tau_sw'][0])
    ea_knots_LW = eis['a0_lw'][0] * np.exp(-t / eis['tau_lw'][0])

    if short:
        wave = eis['wave_area_sw'][0]
    elif long:
        wave = eis['wave_area_lw'][0]

    if isinstance(wave, (int, float)):
        wave = np.array([wave])

    nWave = len(wave)
    ea_out = np.zeros(nWave)

    for i in range(nWave):
        band = eis_get_band(wave[i])
        if band == 'SW':
            w = eis['wave_knots_sw'][0]
            e = np.log(ea_knots_SW)
        elif band == 'LW':
            w = eis['wave_knots_lw'][0]
            e = np.log(ea_knots_LW)
        else:
            print(f"WAVELENGTH OUT OF BOUNDS {wave[i]}")
            continue

        interp_func = interp1d(w, e, kind='linear')
        ea_out[i] = np.exp(interp_func(wave[i]))

    if nWave == 1:
        ea_out = ea_out[0]

    return ea_out

def get_time_tai(date_string):
    """Convert an ISO timestamp into the IDL-style TAI reference used by Warren 2014."""
    idl_ref_epoch = datetime.datetime(1979, 1, 1)
    unix_epoch = datetime.datetime(1970, 1, 1)
    epoch_diff = (idl_ref_epoch - unix_epoch).total_seconds()
    date_object = datetime.datetime.fromisoformat(date_string)
    unix_timestamp = date_object.timestamp()
    idl_timestamp = unix_timestamp - epoch_diff + 3600
    return idl_timestamp

def eis_get_band(wave):
    """Return the EIS channel label for a wavelength."""
    if 165 <= wave <= 213:
        return 'SW'
    elif 245 <= wave <= 292:
        return 'LW'
    else:
        return ''

def read_calib_file(file_path=LEGACY_WARREN_2014_FILE):
    """Read the Warren 2014 calibration coefficient cache."""
    return _load_warren_2014_coefficients(file_path)

def eis_effective_area_read(short=False, long=False):
    """Read a pre-flight effective-area curve for the selected EIS channel."""
    if short:
        response_file = _resolve_response_file(short=True)
    elif long:
        response_file = _resolve_response_file(long=True)
    else:
        raise ValueError('Either short=True or long=True must be set.')

    return _read_effective_area_table(response_file)

def is_eis_wavelength(input_wave):
    """Return whether a wavelength falls within the EIS short or long bands."""
    wave_sw_min = 165
    wave_sw_max = 213
    wave_lw_min = 245
    wave_lw_max = 292

    short = long = False

    if wave_sw_min <= input_wave <= wave_sw_max:
        short = True
    if wave_lw_min <= input_wave <= wave_lw_max:
        long = True

    return short, long


def _resolve_ncpu(ncpu):
    """Return a validated worker count from ``ncpu`` input."""
    if ncpu in (None, 'max'):
        return max(1, os.cpu_count() or 1)
    if isinstance(ncpu, int) and ncpu > 0:
        return ncpu
    raise ValueError("ncpu must be a positive integer or 'max'.")


def apply_calibration(maps, *, method='del_zanna_2025', ncpu='max'):
    """Apply a named calibration model to a map, list of maps, or dict of maps.

    Parameters
    ----------
    maps : sunpy.map.Map, list, dict, or path-like
        Input to calibrate.  A ``dict`` value causes each entry to be
        calibrated recursively.  A path string is loaded as a SunPy map
        before calibration.
    method : str, default='del_zanna_2025'
        Calibration method name passed to :func:`calibrate_map`.
    ncpu : int or str, default='max'
        Number of parallel workers when calibrating a list.

    Returns
    -------
    Same type as ``maps`` with calibration applied.
    """
    if isinstance(maps, Mapping):
        return {
            name: apply_calibration(map_collection, method=method, ncpu=ncpu)
            for name, map_collection in maps.items()
        }
    if isinstance(maps, (str, os.PathLike)):
        return calibrate_map(sunpy.map.Map(maps), method=method)
    if hasattr(maps, 'data') and hasattr(maps, 'meta'):
        return calibrate_map(maps, method=method)

    map_items = list(maps)
    workers = _resolve_ncpu(ncpu)
    if workers == 1 or len(map_items) <= 1:
        return [calibrate_map(m, method=method) for m in map_items]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(lambda m: calibrate_map(m, method=method), map_items))


if __name__ == '__main__':
    sync_solarsoft_calibration_data_cli()