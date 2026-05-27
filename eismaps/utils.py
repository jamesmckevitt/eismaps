"""Utility helpers for EISMaps.

Consolidates Roman numeral conversion, EIS line label formatting, width-to-
velocity lookup, and EIS-specific plot defaults in a single module.
"""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import astropy.units as u
import matplotlib.colors as mcolor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Roman numerals
# ---------------------------------------------------------------------------

def int_to_roman(input_integer):
    """Convert an integer to a Roman numeral."""
    if isinstance(input_integer, bool) or not isinstance(input_integer, int):
        raise TypeError("expected integer, got %s" % type(input_integer))
    if not 0 < input_integer < 4000:
        raise ValueError("Argument must be between 1 and 3999")
    ints = (1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1)
    nums = ('M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I')
    result = []
    for i in range(len(ints)):
        count = int(input_integer / ints[i])
        result.append(nums[i] * count)
        input_integer -= ints[i] * count
    return ''.join(result)


def roman_to_int(roman_str):
    """Convert a Roman numeral to an integer."""
    if not isinstance(roman_str, str) or not roman_str:
        raise ValueError("expected a non-empty Roman numeral string")

    roman_map = {
        'I': 1, 'V': 5, 'X': 10, 'L': 50,
        'C': 100, 'D': 500, 'M': 1000
    }
    integer_value = 0
    prev_value = 0

    for char in reversed(roman_str.upper()):
        if char not in roman_map:
            raise ValueError(f"invalid Roman numeral character: {char}")
        value = roman_map[char]
        if value < prev_value:
            integer_value -= value
        else:
            integer_value += value
        prev_value = value

    return integer_value


# ---------------------------------------------------------------------------
# Line label formatting
# ---------------------------------------------------------------------------

def change_line_format(line):
    """Convert a human-readable line label into the package filename format."""

    elements, ion, wavelength = line.split()
    formatted_element = elements.lower()

    if len(formatted_element) == 1:
        formatted_element += '_'

    formatted_ion = roman_to_int(ion)
    formatted_wavelength = wavelength.replace('.', '_')

    # If the formatted ion integer is only one digit, add a zero before
    if formatted_ion < 10:
        formatted_ion = f'0{formatted_ion}'

    return f"{formatted_element}_{formatted_ion}_{formatted_wavelength}"


# ---------------------------------------------------------------------------
# Width to velocity
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_ion_equilibriums():
    """Load the packaged ion equilibrium lookup table."""
    dat_path = files('eismaps').joinpath('eis_width2velocity.dat')
    ion_equilibriums = pd.read_fwf(dat_path, widths=[7, 10, 10, 10], skiprows=9, header=None)
    ion_equilibriums.columns = ['ELEM', 'ION', 'T_MAX', 'V_THERM']
    return ion_equilibriums


def width2velocity(element, ion):
    """Return the equilibrium temperature and thermal velocity for an ion."""
    element = element.capitalize()  # Ensure the element is in title case
    ion = int_to_roman(ion)  # Convert the ion number to roman numerals

    ion_equilibriums = _load_ion_equilibriums()

    t_max = ion_equilibriums.loc[(ion_equilibriums['ELEM'] == element) & (ion_equilibriums['ION'] == ion), 'T_MAX'].values
    v_therm = ion_equilibriums.loc[(ion_equilibriums['ELEM'] == element) & (ion_equilibriums['ION'] == ion), 'V_THERM'].values

    if len(t_max) == 0 or len(v_therm) == 0:
        raise ValueError(f"No data found for element {element} and ion {ion}")

    t_max = float(t_max[0])
    v_therm = float(v_therm[0])

    return t_max, v_therm


# ---------------------------------------------------------------------------
# Plot defaults
# ---------------------------------------------------------------------------

def _copy_cmap(name: str, bad_color: str):
    """Return a mutable colormap copy with a configured NaN color."""
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(color=bad_color)
    return cmap


def infer_measurement(map_obj, measurement: str | None = None) -> str | None:
    """Infer an eismaps measurement name from explicit input or map metadata."""
    if measurement is not None:
        return measurement

    measrmnt = str(map_obj.meta.get('measrmnt', '')).strip().lower()
    if measrmnt == 'intensity' or 'intensity' in measrmnt:
        return 'int'
    if measrmnt == 'doppler velocity' or measrmnt == 'velocity':
        return 'vel'
    if measrmnt == 'non-thermal velocity':
        return 'ntv'
    if measrmnt == 'width':
        return 'wid'
    if measrmnt == 'chi2':
        return 'chi2'
    if measrmnt == 'overlap count':
        return 'overlap'
    return None


def format_bunit(bunit) -> str | None:
    """Return a readable unit string suitable for plot labels."""
    if bunit is None:
        return None

    unit_text = str(bunit).strip()
    if not unit_text or unit_text.lower() in {'none', 'unknown'}:
        return None

    try:
        unit = u.Unit(unit_text)
    except Exception:
        return unit_text

    numerator = []
    denominator = []
    for base, power in zip(unit.bases, unit.powers):
        base_text = base.to_string()
        magnitude = abs(int(power)) if float(power).is_integer() else abs(power)
        if magnitude == 1:
            rendered = base_text
        else:
            rendered = f'{base_text}{magnitude}'

        if power > 0:
            numerator.append(rendered)
        elif power < 0:
            denominator.append(rendered)

    if denominator:
        def _denominator_order(item):
            if item == 's':
                return (0, item)
            if item.startswith('cm'):
                return (1, item)
            if item == 'sr':
                return (2, item)
            return (3, item)

        denominator.sort(key=_denominator_order)

    if not numerator:
        numerator = ['1']
    if not denominator:
        return '*'.join(numerator)
    return '/'.join([*numerator, *denominator])


def _positive_values(data):
    return np.asarray(data)[np.isfinite(data) & (data > 0)]


def _finite_values(data):
    return np.asarray(data)[np.isfinite(data)]


def apply_default_plot_settings(map_obj, measurement: str | None = None):
    """Mutate a map's plot settings with EIS-specific cmap, norm, and aspect defaults.

    SunPy's ``GenericMap.plot`` defaults to ``aspect=1`` (equal screen pixels),
    which distorts non-square data pixels such as EIS rasters (CDELT1 != CDELT2).
    We set ``aspect = |CDELT2/CDELT1|`` so each arcsec on the screen is equal in
    X and Y, producing a correctly proportioned image.
    """
    try:
        cdelt = map_obj.wcs.wcs.cdelt
        if cdelt[0] != 0:
            map_obj.plot_settings['aspect'] = float(abs(cdelt[1] / cdelt[0]))
    except Exception:
        pass

    measurement = infer_measurement(map_obj, measurement=measurement)

    if measurement == 'int':
        positive = _positive_values(map_obj.data)
        vmin = max(np.nanpercentile(positive, 5), 1e-1) if positive.size else 1e-1
        vmax = max(np.nanpercentile(positive, 99.5), vmin * 10.0) if positive.size else 1.0
        map_obj.plot_settings['cmap'] = _copy_cmap('gist_heat', 'gray')
        map_obj.plot_settings['norm'] = mcolor.LogNorm(vmin=vmin, vmax=vmax)
    elif measurement == 'vel':
        finite = _finite_values(map_obj.data)
        limit = np.nanpercentile(np.abs(finite), 98) if finite.size else 10.0
        limit = max(float(limit), 5.0)
        map_obj.plot_settings['cmap'] = _copy_cmap('RdBu_r', 'gray')
        map_obj.plot_settings['norm'] = mcolor.Normalize(vmin=-limit, vmax=limit)
    elif measurement == 'wid':
        finite = _finite_values(map_obj.data)
        vmin = np.nanpercentile(finite, 1) if finite.size else 0.0
        vmax = np.nanpercentile(finite, 99) if finite.size else 0.1
        if not np.isfinite(vmin):
            vmin = 0.0
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 0.1
        map_obj.plot_settings['cmap'] = _copy_cmap('viridis', 'gray')
        map_obj.plot_settings['norm'] = mcolor.Normalize(vmin=vmin, vmax=vmax)
    elif measurement == 'ntv':
        finite = _finite_values(map_obj.data)
        vmax = np.nanpercentile(finite, 99) if finite.size else 40.0
        vmax = max(float(vmax), 10.0)
        map_obj.plot_settings['cmap'] = _copy_cmap('inferno', 'gray')
        map_obj.plot_settings['norm'] = mcolor.Normalize(vmin=0.0, vmax=vmax)
    elif measurement == 'chi2':
        map_obj.plot_settings['cmap'] = _copy_cmap('gray', 'red')
        map_obj.plot_settings['norm'] = mcolor.Normalize(vmin=0.0, vmax=4.0)
    elif measurement == 'overlap':
        finite = _finite_values(map_obj.data)
        vmax = np.nanmax(finite) if finite.size else 1.0
        vmax = max(float(vmax), 1.0)
        map_obj.plot_settings['cmap'] = _copy_cmap('viridis', 'black')
        map_obj.plot_settings['norm'] = mcolor.Normalize(vmin=0.0, vmax=vmax)

    return map_obj


__all__ = [
    'int_to_roman',
    'roman_to_int',
    'change_line_format',
    'width2velocity',
    'infer_measurement',
    'format_bunit',
    'apply_default_plot_settings',
]
