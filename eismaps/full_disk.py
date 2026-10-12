"""Full-disk mosaics of EIS raster maps.

:func:`make_helioprojective_map` combines raster maps into one helioprojective
map, rotated to the time of a reference map. It can carry uncertainties and
other quantities through the mosaic with the pixel weights of reproject's
exact algorithm. :func:`make_carrington_map` makes a Carrington map.
"""

import contextlib
import copy
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from datetime import datetime
from sunpy.coordinates import frames
import sunpy.map
from matplotlib.colors import LogNorm
from tqdm import tqdm
from sunpy.coordinates.frames import Helioprojective
import sunpy.sun.constants
from sunpy.physics.differential_rotation import solar_rotate_coordinate
from sunpy.coordinates import Helioprojective, propagate_with_solar_surface, RotatedSunFrame
from sunpy.coordinates import SphericalScreen
from typing import List, Optional, Literal, Union, TYPE_CHECKING
import logging

from eismaps.utils import apply_default_plot_settings

if TYPE_CHECKING:
    import sunpy.map


def _normalise_map_collection(map_files):
    """Return a list of map-like inputs for full-disk assembly."""
    if isinstance(map_files, (str, os.PathLike)):
        return [map_files]
    if hasattr(map_files, 'maps'):
        return list(map_files.maps)
    if hasattr(map_files, 'data') and hasattr(map_files, 'meta'):
        return [map_files]
    return list(map_files)


def _coerce_map_input(map_input):
    """Return a SunPy map and a readable source label for file or object inputs."""
    if hasattr(map_input, 'data') and hasattr(map_input, 'meta'):
        return map_input, '<in-memory-map>'
    return sunpy.map.Map(map_input), str(map_input)


def _resolve_ncpu(ncpu):
    """Return a validated worker count from ``ncpu`` input."""
    if ncpu in (None, 'max'):
        return max(1, os.cpu_count() or 1)
    if isinstance(ncpu, int) and ncpu > 0:
        return ncpu
    raise ValueError("ncpu must be a positive integer or 'max'.")

def _combine_data_max_abs(combined_data: np.ndarray, new_data: np.ndarray) -> np.ndarray:
    """
    Combine two arrays, keeping the value of larger magnitude in overlaps.

    NaN counts as no data. The sign is kept, as needed for Doppler velocities.
    """
    combined_has_data = ~np.isnan(combined_data)
    new_has_data = ~np.isnan(new_data)

    result = np.where(
        ~combined_has_data, new_data,
        np.where(
            ~new_has_data, combined_data,
            np.where(
                np.abs(combined_data) >= np.abs(new_data), 
                combined_data, 
                new_data
            )
        )
    )
    return result


def _combine_data_mean(combined_data: np.ndarray, new_data: np.ndarray) -> np.ndarray:
    """
    Add two arrays for a later average.

    Where only one of them has data (is not NaN), its value is used.
    """
    combined_has_data = ~np.isnan(combined_data)
    new_has_data = ~np.isnan(new_data)
    
    return np.where(
        ~combined_has_data, new_data,
        np.where(
            ~new_has_data, combined_data,
            combined_data + new_data
        )
    )


def _combine_data_with_fill(combined_data: np.ndarray, new_data: np.ndarray, fill_value: float, covered: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Combine two arrays, setting pixels where both have data to `fill_value`.

    Elsewhere the valid value is used. `covered` marks pixels that earlier
    maps already supplied; pass it when the fill value can be NaN, which
    would otherwise look like a gap.
    """
    combined_has_data = ~np.isnan(combined_data) if covered is None else covered
    new_has_data = ~np.isnan(new_data)
    
    return np.where(
        ~combined_has_data, new_data,
        np.where(
            ~new_has_data, combined_data,
            fill_value
        )
    )


def _update_overlap_mask(overlap_mask: np.ndarray, new_data: np.ndarray) -> np.ndarray:
    """Add 1 to the overlap count where the new data are valid."""
    return np.where(~np.isnan(new_data), overlap_mask + 1, overlap_mask)


# Keywords copied from the reference raster map into full-disk products, so a
# saved mosaic says what it is a map of. INSTRUME is left out: with it, sunpy
# would make the mosaic an eispac EISMap, which needs raster metadata that a
# mosaic does not have.
FULL_DISK_INHERITED_KEYS = ('measrmnt', 'bunit', 'line_id', 'wavelnth', 'waveunit')

_C_INT_MIN = -2147483648


def _c_int_cast(values):
    """C's ``(int)`` cast of doubles on x86.

    Values are truncated toward zero. NaN, infinities and values outside the
    int range give INT_MIN.
    """
    values = np.asarray(values, dtype=float)
    in_range = np.isfinite(values) & (np.abs(values) < 2147483648.0)
    out = np.full(values.shape, _C_INT_MIN, dtype=np.int64)
    out[in_range] = np.trunc(values[in_range]).astype(np.int64)
    return out


def _c_min4(corners):
    """Minimum over the last axis, computed as reproject's C ``min_4`` does.

    It is a chain of comparisons, so NaNs are skipped unless the first value
    is NaN.
    """
    result = corners[..., 0].copy()
    for k in range(1, 4):
        result = np.where(corners[..., k] < result, corners[..., k], result)
    return result


def _c_max4(corners):
    """Maximum over the last axis, computed as reproject's C ``max_4`` does."""
    result = corners[..., 0].copy()
    for k in range(1, 4):
        result = np.where(corners[..., k] > result, corners[..., k], result)
    return result


def _pixel_corners(grid):
    """Return the four corners of every pixel from a (ny+1, nx+1) corner grid.

    They are in the order reproject passes them to its overlap function:
    (j+1, i), (j+1, i+1), (j, i+1), (j, i).
    """
    return np.stack([grid[1:, :-1], grid[1:, 1:], grid[:-1, 1:], grid[:-1, :-1]], axis=-1)


def _output_corner_coordinates(wcs_out, shape_out):
    """World coordinates of the output pixel corners, as reproject computes them.

    Returns the corner SkyCoord grid and its longitude and latitude in degrees.
    They depend only on the output grid, so a mosaic computes them once.
    """
    ny_out, nx_out = shape_out
    x = np.arange(nx_out + 1.0) - 0.5
    y = np.arange(ny_out + 1.0) - 0.5
    xp_out, yp_out = np.meshgrid(x, y, indexing='xy', sparse=False, copy=False)
    world_out = wcs_out.pixel_to_world(xp_out, yp_out)
    world_out_unitsph = world_out.represent_as('unitspherical')
    return world_out, world_out_unitsph.lon.to_value(u.deg), world_out_unitsph.lat.to_value(u.deg)


def _exact_overlap_weights(map_in, wcs_out, shape_out, output_corners):
    """Return the pixel weights of reproject's exact algorithm for one map.

    ``reproject.reproject_exact`` sets every output pixel to
    ``sum(f * x) / sum(f)`` over input pixels ``x``, where ``f`` is the
    fraction of the output pixel's solid angle that the input pixel covers,
    and returns only those sums. This function returns each ``f`` instead,
    so values, uncertainties and companion quantities can be resampled with
    the same weights. It follows reproject's C kernel step for step: the
    same corner transforms, the same candidate output pixels (the rounded
    bounding box of each input pixel) and the same spherical overlap
    function. Call it inside the same coordinate contexts
    (``propagate_with_solar_surface``, ``SphericalScreen``) as the
    reprojection it replaces.

    Returns
    -------
    out_index, in_index : numpy.ndarray of int64
        Flattened output and input pixel indices of every weighted pair.
    frac : numpy.ndarray of float64
        Weight of each pair.
    """
    from reproject.spherical_intersect._overlap import _compute_overlap

    world_out, xw_out, yw_out = output_corners
    ny_in, nx_in = map_in.data.shape
    ny_out, nx_out = shape_out

    x = np.arange(nx_in + 1.0) - 0.5
    y = np.arange(ny_in + 1.0) - 0.5
    xp_in, yp_in = np.meshgrid(x, y, indexing='xy', sparse=False, copy=False)
    world_in = map_in.wcs.pixel_to_world(xp_in, yp_in)
    world_in = world_in.transform_to(world_out.frame)
    xp_inout, yp_inout = wcs_out.world_to_pixel(world_in)
    world_in_unitsph = world_in.represent_as('unitspherical')
    xw_in = world_in_unitsph.lon.to_value(u.deg)
    yw_in = world_in_unitsph.lat.to_value(u.deg)

    # Candidate output pixels: the bounding box of the input pixel's corners
    # in output pixel coordinates, rounded and clipped as in the C kernel.
    # Its min_4/max_4 read the corners in the order (j,i), (j,i+1),
    # (j+1,i+1), (j+1,i), which is the reverse of _pixel_corners.
    corners_x = _pixel_corners(np.asarray(xp_inout, dtype=float))[..., ::-1]
    corners_y = _pixel_corners(np.asarray(yp_inout, dtype=float))[..., ::-1]
    xmin = np.maximum(_c_int_cast(_c_min4(corners_x) + 0.5), 0).ravel()
    xmax = np.minimum(_c_int_cast(_c_max4(corners_x) + 0.5), nx_out - 1).ravel()
    ymin = np.maximum(_c_int_cast(_c_min4(corners_y) + 0.5), 0).ravel()
    ymax = np.minimum(_c_int_cast(_c_max4(corners_y) + 0.5), ny_out - 1).ravel()

    n_x = np.maximum(xmax - xmin + 1, 0)
    n_y = np.maximum(ymax - ymin + 1, 0)
    counts = n_x * n_y
    total = int(counts.sum())
    if total == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros(0, dtype=float)

    in_index = np.repeat(np.arange(ny_in * nx_in, dtype=np.int64), counts)
    local = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
    n_y_rep = np.repeat(n_y, counts)
    ii = np.repeat(xmin, counts) + local // n_y_rep
    jj = np.repeat(ymin, counts) + local % n_y_rep

    deg_to_rad = 0.017453292519943295  # the constant the C kernel uses
    ilon = (_pixel_corners(xw_in).reshape(-1, 4) * deg_to_rad)[in_index]
    ilat = (_pixel_corners(yw_in).reshape(-1, 4) * deg_to_rad)[in_index]
    # Corners of the candidate output pixels only, in _pixel_corners order.
    olon = np.stack([xw_out[jj + 1, ii], xw_out[jj + 1, ii + 1], xw_out[jj, ii + 1], xw_out[jj, ii]], axis=-1) * deg_to_rad
    olat = np.stack([yw_out[jj + 1, ii], yw_out[jj + 1, ii + 1], yw_out[jj, ii + 1], yw_out[jj, ii]], axis=-1) * deg_to_rad

    overlap, _ = _compute_overlap(np.ascontiguousarray(ilon), np.ascontiguousarray(ilat),
                                  np.ascontiguousarray(olon), np.ascontiguousarray(olat))
    original, _ = _compute_overlap(np.ascontiguousarray(olon), np.ascontiguousarray(olat),
                                   np.ascontiguousarray(olon), np.ascontiguousarray(olat))

    keep = (original > 0.0) & ~np.isnan(overlap)
    frac = overlap[keep] / original[keep]
    out_index = (jj * nx_out + ii)[keep]
    return out_index, in_index[keep], frac


def _weighted_resample(weights, data, n_out, uncertainty=False, use=None):
    """Resample one raster array onto the output grid with exact weights.

    With ``uncertainty=False`` the result is ``sum(f*x)/sum(f)``, as from
    ``reproject_exact``, including its NaN behaviour: a NaN input pixel makes
    every output pixel it is paired with NaN. With ``uncertainty=True`` the
    input is a 1-sigma uncertainty and the result is
    ``sqrt(sum(f**2 * sigma**2)) / sum(f)``, the uncertainty of that
    weighted mean for independent input pixels.

    `use`, a boolean array on the raster grid, restricts the sums to those
    input pixels. An output pixel is then the weighted mean of the selected
    pixels it overlaps, and NaN only if it overlaps none.
    """
    out_index, in_index, frac = weights
    values = np.asarray(data, dtype=float).ravel()[in_index]
    if use is not None:
        selected = np.asarray(use, dtype=bool).ravel()[in_index]
        frac = np.where(selected, frac, 0.0)
        values = np.where(selected, values, 0.0)
    denominator = np.bincount(out_index, weights=frac, minlength=n_out)
    if uncertainty:
        numerator = np.sqrt(np.bincount(out_index, weights=(values * frac) ** 2, minlength=n_out))
    else:
        numerator = np.bincount(out_index, weights=values * frac, minlength=n_out)
    with np.errstate(invalid='ignore', divide='ignore'):
        return numerator / denominator


def _reprojection_context(map_obj, apply_rotation, drag_rotate, preserve_limb):
    """Coordinate contexts for reprojecting one raster in make_helioprojective_map.

    ``propagate_with_solar_surface`` (Howard model) when rotating without
    `drag_rotate`, and ``SphericalScreen`` for off-disk pixels when
    `preserve_limb` is True.
    """
    stack = contextlib.ExitStack()
    if apply_rotation and not drag_rotate:
        stack.enter_context(propagate_with_solar_surface(rotation_model='howard'))
    if preserve_limb:
        stack.enter_context(SphericalScreen(map_obj.observer_coordinate, only_off_disk=True))
    return stack


def _drag_rotate_map(map_obj, first_map):
    """Shift a map by the differential rotation of one point since the reference map.

    The point is the map centre or, if that is 950 arcsec or more from disk
    centre, the point 950 arcsec from disk centre in the same direction. The
    rotation uses the Howard model (the RotatedSunFrame default).
    """
    SAFE_LIMB_DIST_ARCSEC = 950
    map_center_radial_distance = np.sqrt(map_obj.center.Tx.value ** 2 + map_obj.center.Ty.value ** 2)

    if map_center_radial_distance < SAFE_LIMB_DIST_ARCSEC:
        point = map_obj.center
    else:
        angle = np.arctan2(map_obj.center.Tx, map_obj.center.Ty)
        point = SkyCoord(SAFE_LIMB_DIST_ARCSEC * np.sin(angle) * u.arcsec, SAFE_LIMB_DIST_ARCSEC * np.cos(angle) * u.arcsec, obstime=map_obj.date, observer=map_obj.observer_coordinate, frame=Helioprojective)

    map_time = datetime.strptime(map_obj.meta['date_obs'], '%Y-%m-%dT%H:%M:%S.%f')
    first_map_time = datetime.strptime(first_map.meta['date_obs'], '%Y-%m-%dT%H:%M:%S.%f')
    duration = map_time - first_map_time
    # Signed seconds: timedelta.seconds is never negative.
    duration = duration.total_seconds() * u.second
    diffrot_point = RotatedSunFrame(base=point, duration=duration)
    transformed_diffrot_point = diffrot_point.transform_to(map_obj.coordinate_frame)
    shift_x = transformed_diffrot_point.Tx - point.Tx
    shift_y = transformed_diffrot_point.Ty - point.Ty
    return map_obj.shift_reference_coord(shift_x, shift_y)


def _los_correction_factor(map_obj):
    """Return 1/cos(viewing angle) on disk and 1 off disk, for one raster map."""
    map_coords = sunpy.map.all_coordinates_from_map(map_obj)
    r_arcsec = np.sqrt(map_coords.Tx.value**2 + map_coords.Ty.value**2)
    solar_radius_arcsec = map_obj.rsun_obs.to(u.arcsec).value
    # Viewing angle: 0 at disk centre, pi/2 at the limb.
    on_disk = sunpy.map.coordinate_is_on_solar_disk(map_coords)
    viewing_angle = np.zeros_like(r_arcsec)
    viewing_angle[on_disk] = np.arcsin(r_arcsec[on_disk] / solar_radius_arcsec)
    cos_correction = np.ones_like(map_obj.data)
    cos_correction[on_disk] = 1.0 / np.cos(viewing_angle[on_disk])
    return cos_correction


def _companion_array(companion_input, primary_map, label):
    """Return a companion map's data and the map, checking it is on the primary grid.

    The companion is resampled with the primary map's weights, so it must
    have the same shape and the same pixel-to-sky mapping. Keywords that do
    not change that mapping (dates, observer) are not compared.
    """
    from astropy.wcs import WCSCOMPARE_ANCILLARY

    companion_map, _ = _coerce_map_input(companion_input)
    data = np.asarray(companion_map.data, dtype=float)
    if data.shape != primary_map.data.shape:
        raise ValueError(f"{label} map has shape {data.shape} but its primary map has shape {primary_map.data.shape}.")
    if not primary_map.wcs.wcs.compare(companion_map.wcs.wcs, cmp=WCSCOMPARE_ANCILLARY, tolerance=1e-10):
        raise ValueError(f"{label} map is not on the same pixel grid as its primary map.")
    return data, companion_map


def _mosaic_with_exact_weights(
    map_files, uncertainty_inputs, companion_inputs, first_map, fd_map, overlap,
    apply_rotation, preserve_limb, drag_rotate, remove_off_disk, apply_los_correction,
    skip_nan=False,
):
    """Assemble a full-disk mosaic and its uncertainty and companion maps.

    Each raster is resampled with the weights of reproject's exact algorithm
    (see _exact_overlap_weights), computed once per raster and applied to
    the value, its uncertainty and every companion. With `skip_nan`, raster
    pixels whose value is NaN are left out of the sums for all of these.
    Overlapping rasters are then combined per pixel:

    - 'max': the value of largest magnitude, as in make_helioprojective_map.
      The uncertainty and companions come from the same raster.
    - 'mean': values and companions are averaged. The uncertainty is that of
      the mean, ``sqrt(sum(sigma**2)) / N``.
    - a number: pixels covered more than once are set to that number in
      every product.
    """
    shape_out = fd_map.data.shape
    n_out = shape_out[0] * shape_out[1]
    wcs_out = fd_map.wcs
    output_corners = _output_corner_coordinates(wcs_out, shape_out)

    combined = np.full(n_out, np.nan)
    count = np.zeros(n_out)
    combined_sigma = np.full(n_out, np.nan) if uncertainty_inputs is not None else None
    combined_companions = {name: np.full(n_out, np.nan) for name in companion_inputs}
    companion_meta = {}
    if overlap == 'mean':
        combined_variance = np.zeros(n_out)
        companion_sums = {name: np.zeros(n_out) for name in companion_inputs}
        companion_counts = {name: np.zeros(n_out) for name in companion_inputs}

    for k, map_file in enumerate(tqdm(map_files, desc="Processing maps", leave=False)):
        try:
            map_obj, _ = _coerce_map_input(map_file)
        except Exception as e:
            print(f"Error loading map {map_file}: {e}. Skipping.")
            continue

        data = np.array(map_obj.data, dtype=float, copy=True)
        sigma = None
        if uncertainty_inputs is not None:
            sigma, _ = _companion_array(uncertainty_inputs[k], map_obj, 'Uncertainty')
            sigma = sigma.copy()
        companions = {}
        for name, inputs in companion_inputs.items():
            companions[name], companion_map = _companion_array(inputs[k], map_obj, f"Companion '{name}'")
            companions[name] = companions[name].copy()
            companion_meta.setdefault(name, companion_map.meta)

        if remove_off_disk is True or remove_off_disk == 'before':
            on_disk_mask = sunpy.map.coordinate_is_on_solar_disk(sunpy.map.all_coordinates_from_map(map_obj))
            data[~on_disk_mask] = np.nan
            if sigma is not None:
                sigma[~on_disk_mask] = np.nan
            for name in companions:
                companions[name][~on_disk_mask] = np.nan

        if apply_los_correction:
            factor = _los_correction_factor(map_obj)
            data = data * factor
            if sigma is not None:
                sigma = sigma * factor

        geometry_map = map_obj
        if apply_rotation and drag_rotate:
            geometry_map = _drag_rotate_map(map_obj, first_map)

        with _reprojection_context(geometry_map, apply_rotation, drag_rotate, preserve_limb):
            weights = _exact_overlap_weights(geometry_map, wcs_out, shape_out, output_corners)

        use = ~np.isnan(data) if skip_nan else None
        new = _weighted_resample(weights, data, n_out, use=use)
        new_valid = ~np.isnan(new)
        new_sigma = None
        if sigma is not None:
            new_sigma = _weighted_resample(weights, sigma, n_out, uncertainty=True, use=use)
            new_sigma[~new_valid] = np.nan
        new_companions = {name: _weighted_resample(weights, arr, n_out, use=use) for name, arr in companions.items()}

        if overlap == 'max':
            take = new_valid & (np.isnan(combined) | (np.abs(new) > np.abs(combined)))
            combined[take] = new[take]
            if new_sigma is not None:
                combined_sigma[take] = new_sigma[take]
            for name, arr in new_companions.items():
                combined_companions[name][take] = arr[take]
        elif overlap == 'mean':
            combined = np.where(new_valid, np.where(np.isnan(combined), new, combined + new), combined)
            if new_sigma is not None:
                combined_variance = np.where(new_valid, combined_variance + new_sigma ** 2, combined_variance)
            for name, arr in new_companions.items():
                arr_valid = new_valid & ~np.isnan(arr)
                companion_sums[name] = np.where(arr_valid, companion_sums[name] + arr, companion_sums[name])
                companion_counts[name] = companion_counts[name] + arr_valid
        else:
            # Coverage comes from the count, since the fill value may be NaN.
            already = count > 0
            if new_sigma is not None:
                combined_sigma = np.where(already & new_valid, overlap, np.where(already, combined_sigma, new_sigma))
            for name, arr in new_companions.items():
                combined_companions[name] = np.where(already & new_valid, overlap, np.where(already, combined_companions[name], arr))
            combined = _combine_data_with_fill(combined, new, overlap, covered=already)

        count = count + new_valid

    if overlap == 'mean':
        with np.errstate(invalid='ignore', divide='ignore'):
            combined = np.where(count > 0, combined / count, np.nan)
            if combined_sigma is not None:
                combined_sigma = np.where(count > 0, np.sqrt(combined_variance) / count, np.nan)
            for name in companion_inputs:
                combined_companions[name] = np.where(companion_counts[name] > 0, companion_sums[name] / companion_counts[name], np.nan)

    def _as_map(flat, meta_source, measurement=None, bunit=None):
        meta = fd_map.meta.copy()
        for key in FULL_DISK_INHERITED_KEYS:
            if key in meta_source:
                meta[key] = meta_source[key]
        if measurement is not None:
            meta['measrmnt'] = measurement
        if bunit is not None:
            meta['bunit'] = bunit
        return sunpy.map.Map(flat.reshape(shape_out), meta)

    fd_out = _as_map(combined, first_map.meta)
    overlap_meta = fd_out.meta.copy()
    overlap_meta['measrmnt'] = 'overlap count'
    overlap_meta['bunit'] = 'count'
    overlap_out = sunpy.map.Map(count.reshape(shape_out), overlap_meta)

    extras = {}
    if combined_sigma is not None:
        primary_measurement = str(first_map.meta.get('measrmnt', '')).strip()
        label = f'{primary_measurement} uncertainty' if primary_measurement else 'uncertainty'
        extras['uncertainty'] = _as_map(combined_sigma, first_map.meta, measurement=label)
    for name in companion_inputs:
        extras[name] = _as_map(combined_companions[name], companion_meta.get(name, {}))

    def _mask(map_obj, mask):
        return sunpy.map.Map(np.where(mask, map_obj.data, np.nan), map_obj.meta)

    if remove_off_disk == 'after' or not preserve_limb:
        on_disk_mask = sunpy.map.coordinate_is_on_solar_disk(sunpy.map.all_coordinates_from_map(fd_out))
        fd_out = _mask(fd_out, on_disk_mask)
        extras = {name: _mask(m, on_disk_mask) for name, m in extras.items()}

    fd_out = apply_default_plot_settings(fd_out)
    overlap_out = apply_default_plot_settings(overlap_out, measurement='overlap')
    extras = {name: apply_default_plot_settings(m) for name, m in extras.items()}
    return fd_out, overlap_out, extras


def make_helioprojective_map(
    map_files: List[object], 
    overlap: Union[Literal['max', 'mean'], float],
    apply_rotation: bool = True, 
    preserve_limb: bool = True, 
    drag_rotate: bool = False,
    algorithm: Literal['exact', 'interpolation', 'adaptive'] = 'exact',
    remove_off_disk: Union[bool, Literal['before', 'after']] = False,
    apply_los_correction: bool = False,
    ncpu: Union[int, Literal['max']] = 'max',
    reference_map: Literal['first', 'last'] = 'first',
    uncertainty: Optional[List[object]] = None,
    companions: Optional[dict] = None,
    skip_nan: bool = False,
):
    """
    Combine EIS raster maps into a helioprojective full-disk map.

    The output grid is 3500 arcsec across in each direction, centred on disk
    centre, with the reference map's pixel scale and a TAN projection.

    Parameters
    ----------
    map_files : List[object]
        One map, a map sequence, or a collection of map file paths or
        ``sunpy.map.Map`` objects. Must not be empty. The reference map (see
        `reference_map`) sets the reference time and coordinate frame.
    overlap : {'max', 'mean'} or float
        How overlapping rasters are combined:
        - 'max': the value of largest magnitude (for Doppler velocities)
        - 'mean': the mean
        - a number (e.g. np.nan, 0, -999): pixels covered more than once are
          set to this value
    apply_rotation : bool, default=True
        Differentially rotate the maps to the reference time.
    preserve_limb : bool, default=True
        Keep off-limb data. If False, off-disk pixels of the result are NaN.
    drag_rotate : bool, default=False
        Instead of rotating each pixel, shift each map as a whole by the
        rotation of one point: its centre or, if that is 950 arcsec or more
        from disk centre, the point 950 arcsec out in the same direction.
        Both use the Howard rotation model.
    algorithm : {'exact', 'interpolation', 'adaptive'}, default='exact'
        Reprojection algorithm:
        - 'exact': exact reprojection (slower but more accurate)
        - 'interpolation': interpolation (faster but less accurate)
        - 'adaptive': adaptive, anti-aliased resampling, with optional flux
          conservation
    remove_off_disk : bool or {'before', 'after'}, default=False
        When off-disk pixels are set to NaN:
        - False: never
        - True or 'before': in each map, before combining
        - 'after': in the combined map
    apply_los_correction : bool, default=False
        Divide on-disk pixel values by the cosine of the viewing angle, to
        correct for foreshortening. Useful for radial quantities, such as
        velocities, of which only the line-of-sight component is observed.
    ncpu : int or {'max'}, default='max'
        Number of threads used to prepare and reproject the maps. 'max' uses
        all available CPUs. The maps are processed one at a time when
        `preserve_limb` is True, and with `uncertainty`, `companions` or
        `skip_nan`.
    reference_map : {'first', 'last'}, default='first'
        Map that sets the reference time and the coordinate frame of the
        full-disk WCS. With ``apply_rotation=True`` all other maps are
        differentially rotated to its observation time:
        - 'first': ``map_files[0]``
        - 'last': ``map_files[-1]``
    uncertainty : list, optional
        1-sigma uncertainty maps, one per entry of `map_files` and on the
        same raster grid. They are resampled with the same pixel weights as
        the values and propagated exactly: a resampled pixel
        ``sum(f*x)/sum(f)`` gets ``sqrt(sum(f**2*sigma**2))/sum(f)``. With
        ``overlap='max'`` each pixel's uncertainty comes from the raster
        that supplied its value; with ``overlap='mean'`` it is the
        uncertainty of the mean. Requires ``algorithm='exact'``.
    companions : dict, optional
        Other quantities to carry with the values, for example a chi-squared
        map or a map of observation times. Keys are names (any but
        ``'uncertainty'``), and values are lists of maps, one per entry of
        `map_files` and on the same raster grid. They are resampled with the
        same weights as the values. With ``overlap='max'`` each pixel comes
        from the raster that supplied the value; with ``overlap='mean'``
        they are averaged. Requires ``algorithm='exact'``.
    skip_nan : bool, default=False
        By default a NaN raster pixel (for example a spectrum that could not
        be fitted) makes every output pixel it overlaps NaN, as in
        ``reproject_exact``. With ``skip_nan=True`` NaN pixels are left out:
        each output pixel is the area-weighted mean of the valid raster
        pixels it overlaps, and NaN only if it overlaps none. Uncertainties
        and companions use the same pixels as the values. Requires
        ``algorithm='exact'``.

    Returns
    -------
    tuple of (sunpy.map.Map, sunpy.map.Map) or (None, None)
        The full-disk map and a map of how many rasters contributed to each
        pixel, or (None, None) if the reference map cannot be loaded. With
        `uncertainty` or `companions`, a third element is returned: a dict
        holding the full-disk uncertainty map (key ``'uncertainty'``) and
        one full-disk map per companion name.

    Notes
    -----
    For Doppler velocities, 'max' keeps the velocity of largest magnitude
    with its sign, so the direction of the flow is kept.

    Examples
    --------
    >>> map_files = ['map1.fits', 'map2.fits', 'map3.fits']
    >>> fd_map, overlap_map = make_helioprojective_map(map_files, 'max')
    >>>
    >>> # Interpolation, for faster processing
    >>> fd_map, overlap_map = make_helioprojective_map(map_files, 'max', algorithm='interpolation')
    >>>
    >>> # Set overlapping regions to NaN
    >>> fd_map, overlap_map = make_helioprojective_map(map_files, np.nan)
    >>>
    >>> # Set overlapping regions to a specific value
    >>> fd_map, overlap_map = make_helioprojective_map(map_files, -999.0)
    >>>
    >>> # Remove off-disk data from each map before combining
    >>> fd_map, overlap_map = make_helioprojective_map(map_files, 'max', remove_off_disk=True)
    >>>
    >>> # The same, given explicitly
    >>> fd_map, overlap_map = make_helioprojective_map(map_files, 'max', remove_off_disk='before')
    >>>
    >>> # Remove off-disk data from the combined map
    >>> fd_map, overlap_map = make_helioprojective_map(map_files, 'max', remove_off_disk='after')
    >>>
    >>> # Line-of-sight correction for radial quantities
    >>> fd_map, overlap_map = make_helioprojective_map(map_files, 'max', apply_los_correction=True)
    """
    map_files = _normalise_map_collection(map_files)

    if not map_files:
        raise ValueError("map_files cannot be empty")

    if reference_map not in ('first', 'last'):
        raise ValueError("reference_map must be 'first' or 'last'.")

    # Suppress SunPy INFO messages about missing solar radius metadata
    sunpy_logger = logging.getLogger('sunpy')
    sunpy_logger.setLevel(logging.WARNING)

    reference_index = 0 if reference_map == 'first' else -1
    try:
        first_map, first_map_label = _coerce_map_input(map_files[reference_index])
    except Exception as e:
        print(f"Error loading reference map {map_files[reference_index]}: {e}")
        return None, None

    fd_size_arcsec = 3500  # fixed, so a bad raster cannot make a huge map and exhaust memory

    # Pixel scale of the reference map in arcsec; the WCS usually holds it in degrees.
    cdelt_wcs = first_map.wcs.wcs.cdelt
    cunit_wcs = first_map.wcs.wcs.cunit

    map_dx = (cdelt_wcs[0] * u.Unit(cunit_wcs[0])).to(u.arcsec).value
    map_dy = (cdelt_wcs[1] * u.Unit(cunit_wcs[1])).to(u.arcsec).value

    fd_width = round(fd_size_arcsec / map_dx)
    fd_height = round(fd_size_arcsec / map_dy)

    # Full-disk grid centred on disk centre, in the reference map's frame.
    fd_coord = SkyCoord(0*u.arcsec, 0*u.arcsec, frame=first_map.coordinate_frame)

    fd_data = np.full((fd_height, fd_width), np.nan)

    fd_header = sunpy.map.make_fitswcs_header(
        fd_data, 
        fd_coord,
        scale=[map_dx, map_dy] * u.arcsec/u.pix,
        projection_code="TAN"
    )

    # Copy the labels and units of the reference map.
    for key in FULL_DISK_INHERITED_KEYS:
        if key in first_map.meta:
            fd_header[key] = first_map.meta[key]

    fd_map = apply_default_plot_settings(sunpy.map.Map(fd_data, fd_header))

    if uncertainty is not None or companions is not None or skip_nan:
        if algorithm != 'exact':
            raise ValueError("uncertainty, companions and skip_nan use the exact reprojection weights, so they need algorithm='exact'.")
        uncertainty_inputs = None
        if uncertainty is not None:
            uncertainty_inputs = _normalise_map_collection(uncertainty)
            if len(uncertainty_inputs) != len(map_files):
                raise ValueError(f"Got {len(uncertainty_inputs)} uncertainty maps for {len(map_files)} value maps.")
        companion_inputs = {}
        if 'uncertainty' in (companions or {}):
            raise ValueError("'uncertainty' is reserved for the propagated uncertainty; give the companion another name.")
        for name, companion in (companions or {}).items():
            companion_inputs[name] = _normalise_map_collection(companion)
            if len(companion_inputs[name]) != len(map_files):
                raise ValueError(f"Got {len(companion_inputs[name])} '{name}' maps for {len(map_files)} value maps.")
        fd_map, overlap_map, extras = _mosaic_with_exact_weights(
            map_files, uncertainty_inputs, companion_inputs, first_map, fd_map, overlap,
            apply_rotation, preserve_limb, drag_rotate, remove_off_disk, apply_los_correction,
            skip_nan=skip_nan,
        )
        if uncertainty is None and companions is None:
            return fd_map, overlap_map
        return fd_map, overlap_map, extras

    overlap_mask = np.zeros((fd_height, fd_width))
    combined_data = np.full((fd_height, fd_width), np.nan)

    workers = _resolve_ncpu(ncpu)
    # SphericalScreen (used when preserve_limb=True) sets the class attribute
    # Helioprojective._assumed_screen, which all threads share, and resets it
    # to None on exit. That breaks reprojections running in other threads with
    # `AttributeError: 'NoneType' object has no attribute 'only_off_disk'` in
    # Helioprojective.make_3d, so these maps are processed one at a time.
    if preserve_limb and workers > 1:
        workers = 1

    def _prepare_reprojected_map(map_file):
        try:
            map, map_label = _coerce_map_input(map_file)
        except Exception as e:
            print(f"Error loading map {map_file}: {e}. Skipping.")
            return None

        # Values are changed in place on a copy, as rebuilding the map with
        # sunpy.map.Map makes an eispac EISMap, whose velocity colour scaling
        # fails on NaN values.
        if remove_off_disk is True or remove_off_disk == 'before' or apply_los_correction:
            map = copy.deepcopy(map)
            if not np.issubdtype(map.data.dtype, np.floating):
                raise TypeError('Raster maps must hold floating-point data.')

        if remove_off_disk is True or remove_off_disk == 'before':
            map_coords = sunpy.map.all_coordinates_from_map(map)
            on_disk_mask = sunpy.map.coordinate_is_on_solar_disk(map_coords)
            map.data[~on_disk_mask] = np.nan

        if apply_los_correction:
            map.data[...] = map.data * _los_correction_factor(map)

        if apply_rotation and drag_rotate:
            map = _drag_rotate_map(map, first_map)

        with _reprojection_context(map, apply_rotation, drag_rotate, preserve_limb):
            map = map.reproject_to(fd_map.wcs, algorithm=algorithm)

        return map

    reprojected_maps = []
    if workers == 1 or len(map_files) == 1:
        for map_file in tqdm(map_files, desc="Processing maps", leave=False):
            reprojected_map = _prepare_reprojected_map(map_file)
            if reprojected_map is not None:
                reprojected_maps.append(reprojected_map)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_prepare_reprojected_map, map_file) for map_file in map_files]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing maps", leave=False):
                try:
                    reprojected_map = future.result()
                except Exception as e:
                    print(f"Error processing map in worker: {e}. Skipping.")
                    continue
                if reprojected_map is not None:
                    reprojected_maps.append(reprojected_map)

    for map in reprojected_maps:

        if overlap == 'max':
            combined_data = _combine_data_max_abs(combined_data, map.data)
        elif overlap == 'mean':
            combined_data = _combine_data_mean(combined_data, map.data)
        else:
            # overlap is a number (e.g. np.nan, 0, -999)
            combined_data = _combine_data_with_fill(combined_data, map.data, overlap, covered=overlap_mask > 0)

        overlap_mask = _update_overlap_mask(overlap_mask, map.data)

    if overlap == 'mean':
        # Pixels without data stay NaN.
        averaged_data = np.where(overlap_mask > 0, combined_data / overlap_mask, np.nan)
        fd_map = sunpy.map.Map(averaged_data, fd_map.meta)
    else:
        fd_map = sunpy.map.Map(combined_data, fd_map.meta)

    fd_map = apply_default_plot_settings(fd_map)

    overlap_meta = fd_map.meta.copy()
    overlap_meta['measrmnt'] = 'overlap count'
    overlap_meta['bunit'] = 'count'
    overlap_map = apply_default_plot_settings(sunpy.map.Map(overlap_mask, overlap_meta), measurement='overlap')

    if remove_off_disk == 'after':
        pixel_coords = sunpy.map.all_coordinates_from_map(fd_map)
        on_disk_mask = sunpy.map.coordinate_is_on_solar_disk(pixel_coords)
        fd_map_data = np.where(on_disk_mask, fd_map.data, np.nan)
        fd_map = apply_default_plot_settings(sunpy.map.Map(fd_map_data, fd_map.meta))

    # Without preserve_limb, make sure no off-disk data remain.
    if not preserve_limb:
        pixel_coords = sunpy.map.all_coordinates_from_map(fd_map)
        limb_mask = sunpy.map.coordinate_is_on_solar_disk(pixel_coords)
        fd_map_data = np.where(limb_mask, fd_map.data, np.nan)
        fd_map = apply_default_plot_settings(sunpy.map.Map(fd_map_data, fd_map.meta))

    return fd_map, overlap_map

def make_carrington_map(map_files, save_dir, wavelength, measurement, overlap, apply_rotation=True, deg_per_pix=0.1, save_fit=False, save_plot=False, plot_ext='png', plot_dpi=300, skip_done=True, reference_map='first'):
    """
    Make a Carrington full-disk map from map files or in-memory maps.

    Parameters
    ----------
    reference_map : {'first', 'last'}, default='first'
        Input map that sets the reference observation time and observer
        metadata of the Carrington WCS: the first or the last.
    """

    if reference_map not in ('first', 'last'):
        raise ValueError("reference_map must be 'first' or 'last'.")

    map_files = _normalise_map_collection(map_files)

    reference_index = 0 if reference_map == 'first' else -1
    try:
        first_map, _ = _coerce_map_input(map_files[reference_index])
    except Exception as e:
        print(f"Error loading reference map {map_files[reference_index]}: {e}")
        return None

    # Output filename from the date of the reference input, which may be a
    # file path or an in-memory map.
    first_input = map_files[reference_index]
    if hasattr(first_input, 'meta') and hasattr(first_input, 'data'):
        date_obs = str(first_input.meta.get('date_obs', '')).strip()
        if not date_obs:
            raise ValueError('First input map has no date_obs metadata; cannot derive output filename.')
        compact_date = date_obs.split('T')[0].replace('-', '')
        compact_time = date_obs.split('T')[1].split('.')[0].replace(':', '') if 'T' in date_obs else '000000'
        map_file_datetime = f"{compact_date}_{compact_time}"
    else:
        map_file_datetime = os.path.basename(str(first_input)).split('.')[0].replace('eis_', '')
    output_filename = f"eis_{map_file_datetime}.{wavelength}.{measurement}.fd_ca"

    if skip_done:
        if os.path.exists(os.path.join(save_dir, f"{output_filename}.fits")):
            fit_exists = True
        else:
            fit_exists = False
        if save_plot and os.path.exists(os.path.join(save_dir, f"{output_filename}.{plot_ext}")):
            plot_exists = True
        else:
            plot_exists = False
        if fit_exists and save_fit and not save_plot:
            print(f"Skipping {output_filename}.fits as it already exists.")
            return
        if plot_exists and save_plot and not save_fit:
            print(f"Skipping {output_filename}.{plot_ext} as it already exists.")
            return
        if fit_exists and plot_exists and save_fit and save_plot:
            print(f"Skipping {output_filename} as both the fits file and plot already exist.")
            return

    lon_pixels = int(360 / deg_per_pix)
    lat_pixels = int(180 / deg_per_pix)
    fd_lon = np.linspace(0, 360, lon_pixels) * u.deg
    fd_lat = np.linspace(-90, 90, lat_pixels) * u.deg
    fd_lon, fd_lat = np.meshgrid(fd_lon, fd_lat)

    fd_header = {
        'cunit1': 'deg',
        'cunit2': 'deg',
        'crpix1': lon_pixels/2,
        'crpix2': lat_pixels/2,
        'crval1': 0,
        'crval2': 0,
        'cdelt1': deg_per_pix,
        'cdelt2': deg_per_pix,
        'ctype1': 'CRLN-CEA',
        'ctype2': 'CRLT-CEA',
        'date_obs': first_map.meta['date_obs'],
        'hgln_obs': first_map.meta['hgln_obs'],
        'dsun_obs': first_map.meta['dsun_obs'],
        'hglt_obs': first_map.meta['hglt_obs'],
    }

    fd_data = np.full((lat_pixels, lon_pixels), np.nan)
    fd_map = sunpy.map.Map(fd_data, fd_header)

    overlap_mask = np.zeros((lat_pixels, lon_pixels))

    for map_file in map_files:

        try:
            map, map_label = _coerce_map_input(map_file)
        except Exception as e:
            print(f"Error loading map {map_file}: {e}. Skipping.")
            continue

        # Set the time of the target map (not a copy) to the raster's time.
        fd_map_temp = fd_map
        if apply_rotation:
            fd_map_temp.meta['date_obs'] = map.meta['date_obs']

        target_wcs = fd_map_temp.wcs

        map_carrington = map.reproject_to(target_wcs)

        if overlap == 'max':
            combined_data = np.where(np.isnan(fd_map.data), map_carrington.data, np.nanmax([fd_map.data, map_carrington.data], axis=0))
        elif overlap == 'mean':
            combined_data = np.where(np.isnan(fd_map.data), map_carrington.data, (fd_map.data + map_carrington.data))
            overlap_mask = np.where(np.isnan(fd_map.data), overlap_mask + 1, overlap_mask)
        elif overlap == 'nan':
            combined_data = np.where(np.isnan(fd_map.data), map_carrington.data, np.nan)

        fd_map = sunpy.map.Map(combined_data, fd_map.meta)

    if overlap == 'mean':
        fd_map.data = fd_map.data / overlap_mask

    if save_fit:

        fd_map.save(os.path.join(save_dir, f"{output_filename}.fits"), overwrite=True)

    if save_plot:

        fig = plt.figure()
        ax = plt.subplot(projection=fd_map)

        if measurement == 'int':
            fd_map.plot_settings['norm'] = LogNorm(vmin=1e1, vmax=5e3)
            im = fd_map.plot(cmap='gist_heat')
        elif measurement == 'vel':
            im = fd_map.plot(cmap='RdBu_r')
            im.set_norm(plt.Normalize(vmin=-10, vmax=10))
        elif measurement == 'ntv':
            im = fd_map.plot(cmap='inferno')
            im.set_norm(plt.Normalize(vmin=0, vmax=40))
        elif measurement == 'mag':
            im = fd_map.plot(cmap='gray')
            im.set_norm(plt.Normalize(vmin=-300, vmax=300))
        elif measurement == 'fip':
            im = fd_map.plot(cmap='CMRmap')
            im.set_norm(plt.Normalize(vmin=0, vmax=3))
        elif measurement == 'chi2':
            im = fd_map.plot(cmap='gray')
            im.set_norm(plt.Normalize(vmin=0, vmax=4))
        else:
            print(f"Error: plotting information for this measurement is not defined in eismaps. Full disk fits file was saved, but can't plot.")
            return

        im = ax.get_images()
        im_lims = im[0].get_extent()
        ax.set_aspect(abs((im_lims[1]-im_lims[0])/(im_lims[3]-im_lims[2])))
        plt.colorbar(extend='both')
        plt.savefig(os.path.join(save_dir, f"{output_filename}.{plot_ext}"), dpi=plot_dpi)
        plt.close()

    return fd_map