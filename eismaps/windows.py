"""Choosing a fit for an EIS spectral window from its title.

Window titles are free text from the observing study, such as 'Fe XII 195.120',
'FeXIV MgVI 270.510' or 'LINE 3', and the wavelength in a title is often the
window centre and not the line. :func:`choose_fit` finds the line a window is
named after among the EISPAC templates and fits it with a single Gaussian,
unless a check on the window's mean spectrum shows that a neighbouring line
would spoil the result.
"""

import copy
import glob
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import eispac
import numpy as np

from eismaps.batched_fit import fit_spectra
from eismaps.utils import int_to_roman, roman_to_int

CC = 2.9979E+5  # speed of light in km/s

ELEMENTS = ('HE', 'NE', 'NA', 'MG', 'AL', 'SI', 'AR', 'CA', 'CR', 'MN', 'FE', 'CO', 'NI',
            'C', 'N', 'O', 'S', 'K')

# Wider windows hold many lines and get an intensity thumbnail only, unless
# they are named after an ion.
MAX_UNNAMED_FIT_PIXELS = 128

# Windows wider than this are not fitted at all.
MAX_FIT_PIXELS = 256

# Lines this close to the end of a window in Angstrom are not fitted.
WINDOW_MARGIN = 0.05

# Half width in Angstrom of the fit range of a single Gaussian taken from a
# multi-component template, about 5 pixels on each side of the line.
SINGLE_GAUSSIAN_HALF_WIDTH = 0.12

# Centroid limits in Angstrom around the line for such a Gaussian.
CENTROID_RANGE = 0.08

# Width limits in Angstrom, as in the EISPAC 1-component templates.
WIDTH_LIMITS = (0.019108, 0.050955)
WIDTH_GUESS = 0.03

# A single Gaussian represents a line if, on the mean spectrum, its intensity
# is within this fraction of the multi-component fit and its centroid within
# this velocity in km/s.
MAX_INTENSITY_DIFFERENCE = 0.3
MAX_VELOCITY_DIFFERENCE = 30.0

# Lowest mean-spectrum peak, in units of its error, worth fitting in a window
# with no named line.
MIN_MEAN_PEAK_SNR = 5.0

# A template line this close in Angstrom to a peak is taken to be that peak.
LINE_MATCH = 0.05


@dataclass
class WindowFit:
    """How to fit a window.

    Attributes
    ----------
    template : dict
        EISPAC template, or a single-Gaussian one built from it.
    parinfo : list[dict]
        MPFIT parameter settings.
    component : int
        Gaussian of the template that is the window's line.
    line : str
        Line fitted, e.g. 'Fe XII 195.119'.
    source : str
        Name of the EISPAC template, or 'generic'.
    single : bool
        True for a single Gaussian.
    intensity_ratio, velocity_offset : float or None
        Single Gaussian against the multi-component fit on the mean spectrum:
        intensity ratio and centroid difference in km/s. None if not compared.
    """

    template: dict
    parinfo: list
    component: int
    line: str
    source: str
    single: bool
    intensity_ratio: Optional[float] = None
    velocity_offset: Optional[float] = None


def _is_roman(text):
    return re.fullmatch(r'[IVX]+', text) is not None


def parse_title(title):
    """Return the ions named in a window title and its wavelength.

    Parameters
    ----------
    title : str
        Window title, e.g. 'Fe XII 195.120', 'FEVIII XII 186.850',
        'Fe IX_XI 188.350' or 'CCDBSHRT 0 177.540'.

    Returns
    -------
    tuple
        List of ions as (element, charge) such as ('FE', 12), which may be
        empty, and the wavelength in Angstrom, or None.
    """
    wavelength = None
    match = re.search(r'(?<![\d.])(\d{3}(?:\.\d*)?)\s*$', title)
    if match:
        wavelength = float(match.group(1))
        title = title[:match.start()]

    ions = []
    element = None
    for token in re.split(r'[\s/_]+', title.strip().upper()):
        if token in ELEMENTS:
            element = token
        elif _is_roman(token) and element is not None:
            ions.append((element, roman_to_int(token)))
        else:
            for size in (2, 1):
                if token[:size] in ELEMENTS and _is_roman(token[size:]):
                    element = token[:size]
                    ions.append((element, roman_to_int(token[size:])))
                    break
    return ions, wavelength


@lru_cache(maxsize=1)
def template_library():
    """Return the EISPAC templates as a list of (path, n_gauss, lines).

    Each entry of `lines` is (element, charge, wavelength), or None for a
    component that is not an identified line.
    """
    directory = os.path.join(os.path.dirname(eispac.__file__), 'data', 'templates')
    library = []
    for path in sorted(glob.glob(os.path.join(directory, '*.template.h5'))):
        template = eispac.read_template(path).template
        lines = []
        for line_id in np.atleast_1d(template['line_ids']):
            line_id = line_id.decode() if isinstance(line_id, bytes) else str(line_id)
            ions, wavelength = parse_title(line_id)
            lines.append((*ions[0], wavelength) if ions and wavelength else None)
        library.append((path, int(template['n_gauss']), lines))
    return library


def _single_gaussian(path, component):
    """Return a single-Gaussian template and parinfo from one component of a template."""
    source = eispac.read_template(path)
    template = copy.deepcopy(source.template)
    parinfo = copy.deepcopy(source.parinfo)
    n_gauss, n_poly = int(template['n_gauss']), int(template['n_poly'])
    fit = np.asarray(template['fit'], dtype=float)
    keep = [3 * component, 3 * component + 1, 3 * component + 2]
    keep += [3 * n_gauss + k for k in range(n_poly)]
    new = [parinfo[k] for k in keep]
    centre = fit[3 * component + 1]
    for entry in new:
        entry['tied'] = ''
        entry['fixed'] = 0
    new[1]['limited'] = [1, 1]
    new[1]['limits'] = [centre - CENTROID_RANGE, centre + CENTROID_RANGE]
    new[2]['limited'] = [1, 1]
    new[2]['limits'] = list(WIDTH_LIMITS)
    fit = fit[keep]
    fit[2] = np.clip(fit[2], *WIDTH_LIMITS)
    template.update(n_gauss=1, n_poly=n_poly, fit=fit, component=0,
                    wmin=centre - SINGLE_GAUSSIAN_HALF_WIDTH, wmax=centre + SINGLE_GAUSSIAN_HALF_WIDTH)
    return template, new


def _generic_gaussian(centre):
    """Return a single-Gaussian template and parinfo around a wavelength with no template."""
    template = {'fit': np.array([1.0, centre, WIDTH_GUESS, 1.0]), 'n_gauss': 1, 'n_poly': 1, 'component': 0,
                'wmin': centre - SINGLE_GAUSSIAN_HALF_WIDTH, 'wmax': centre + SINGLE_GAUSSIAN_HALF_WIDTH,
                'line_ids': np.array([f'Unknown {centre:.3f}'])}
    parinfo = [{'value': 1.0, 'fixed': 0, 'limited': [1, 0], 'limits': [0.0, 0.0], 'tied': ''},
               {'value': centre, 'fixed': 0, 'limited': [1, 1],
                'limits': [centre - CENTROID_RANGE, centre + CENTROID_RANGE], 'tied': ''},
               {'value': WIDTH_GUESS, 'fixed': 0, 'limited': [1, 1], 'limits': list(WIDTH_LIMITS), 'tied': ''},
               {'value': 1.0, 'fixed': 0, 'limited': [0, 0], 'limits': [0.0, 0.0], 'tied': ''}]
    return template, parinfo


def _fit_mean(wave, inten, errs, template, parinfo, component):
    """Fit a mean spectrum, returning status, centroid and intensity of a component."""
    result = fit_spectra(wave[None], inten[None], errs[None], template, parinfo, tolerance=1e-10)
    params = result['params'][0]
    peak, centroid, width = params[3 * component:3 * component + 3]
    return int(result['status'][0]), centroid, np.sqrt(2 * np.pi) * peak * width


def _strongest_peak(wave, inten, errs):
    """Return the wavelength of the strongest point above background, or None if not significant."""
    good = errs > 0
    if good.sum() < 7:
        return None
    above = inten - np.median(inten[good])
    snr = np.where(good, above / np.where(good, errs, 1.0), -np.inf)
    best = int(np.argmax(snr))
    return float(wave[best]) if snr[best] >= MIN_MEAN_PEAK_SNR else None


def _format_line(line):
    element, charge, wavelength = line
    return f'{element.capitalize()} {int_to_roman(charge)} {wavelength:.3f}'


def _target_line(title, low, high, wave, inten, errs, n_wave):
    """Return the line a window should be fitted with, or a peak wavelength, or None.

    Returns
    -------
    tuple or float or None
        (element, charge, wavelength) of a template line, the wavelength of a
        mean-spectrum peak with no template, or None if the window is not to
        be fitted.
    """
    ions, title_wave = parse_title(title)
    lines = {line for _, _, entries in template_library() for line in entries
             if line and low + WINDOW_MARGIN < line[2] < high - WINDOW_MARGIN}

    named = [line for line in lines if line[:2] in ions]
    if named:
        reference = title_wave if title_wave is not None else _strongest_peak(wave, inten, errs)
        if reference is None:
            reference = 0.5 * (low + high)
        return min(named, key=lambda line: abs(line[2] - reference))

    if not ions and n_wave > MAX_UNNAMED_FIT_PIXELS:
        return None
    peak = _strongest_peak(wave, inten, errs)
    if peak is None:
        return None
    nearby = [line for line in lines if abs(line[2] - peak) < LINE_MATCH]
    return min(nearby, key=lambda line: abs(line[2] - peak)) if nearby else peak


def choose_fit(title, low, high, wave, inten, errs):
    """Choose the fit for a window.

    Parameters
    ----------
    title : str
        Window title.
    low, high : float
        Wavelength range of the window in Angstrom.
    wave, inten, errs : np.ndarray
        Mean spectrum of the window, shape (n_wave,). Points with `errs` <= 0
        are not used.

    Returns
    -------
    WindowFit or None
        None for a window with no line to fit, such as a full-CCD window.
    """
    n_wave = wave.size
    if n_wave > MAX_FIT_PIXELS:
        return None
    target = _target_line(title, low, high, wave, inten, errs, n_wave)
    if target is None:
        return None
    if not isinstance(target, tuple):
        template, parinfo = _generic_gaussian(target)
        name = title.strip() if parse_title(title)[0] else f'Line at {target:.3f}'
        return WindowFit(template, parinfo, 0, name, 'generic', True)

    holders = [(path, n_gauss, lines.index(target)) for path, n_gauss, lines in template_library()
               if target in lines]
    ones = [h for h in holders if h[1] == 1]
    multi = sorted((h for h in holders if h[1] > 1), key=lambda h: h[1])
    name = _format_line(target)

    if ones:
        loaded = eispac.read_template(ones[0][0])
        template, parinfo = loaded.template, loaded.parinfo
        source = os.path.basename(ones[0][0])
    else:
        template, parinfo = _single_gaussian(multi[0][0], multi[0][2])
        source = 'single from ' + os.path.basename(multi[0][0])
    if not multi:
        return WindowFit(template, parinfo, 0, name, source, True)

    full = eispac.read_template(multi[0][0])
    status_1, centroid_1, intensity_1 = _fit_mean(wave, inten, errs, template, parinfo, 0)
    status_m, centroid_m, intensity_m = _fit_mean(wave, inten, errs, full.template, full.parinfo, multi[0][2])
    if status_1 <= 0 or status_m <= 0 or intensity_m <= 0:
        return WindowFit(template, parinfo, 0, name, source, True)

    ratio = intensity_1 / intensity_m
    offset = (centroid_1 - centroid_m) / target[2] * CC
    if abs(ratio - 1) <= MAX_INTENSITY_DIFFERENCE and abs(offset) <= MAX_VELOCITY_DIFFERENCE:
        return WindowFit(template, parinfo, 0, name, source, True, float(ratio), float(offset))
    return WindowFit(full.template, full.parinfo, multi[0][2], name, os.path.basename(multi[0][0]), False,
                     float(ratio), float(offset))
