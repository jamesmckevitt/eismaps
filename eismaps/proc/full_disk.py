import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from datetime import datetime
from sunpy.coordinates import frames
import sunpy.map
import eismaps.utils.find
from matplotlib.colors import LogNorm
from tqdm import tqdm
from sunpy.coordinates.frames import Helioprojective
import sunpy.sun.constants
from sunpy.physics.differential_rotation import solar_rotate_coordinate
from sunpy.coordinates import Helioprojective, propagate_with_solar_surface, RotatedSunFrame
from sunpy.coordinates import SphericalScreen
from typing import List, Optional, Literal, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import sunpy.map

def _combine_data_max_abs(combined_data: np.ndarray, new_data: np.ndarray) -> np.ndarray:
    """
    Combine data using maximum absolute value method.
    
    For overlapping pixels, keeps the value with the largest absolute magnitude.
    This preserves the sign of the strongest signal (useful for Doppler velocities).
    """
    # Handle NaN cases first
    combined_has_data = ~np.isnan(combined_data)
    new_has_data = ~np.isnan(new_data)
    
    # Where combined is NaN, use new data
    # Where new is NaN, keep combined data  
    # Where both have data, use the one with larger absolute value
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
    Combine data by accumulating for later averaging.
    
    Sums the values where both exist, or uses the single value where only one exists.
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


def _combine_data_with_fill(combined_data: np.ndarray, new_data: np.ndarray, fill_value: float) -> np.ndarray:
    """
    Combine data by setting overlapping regions to a specified fill value.
    
    Where both have data, set to fill_value. Otherwise use the single valid value.
    """
    combined_has_data = ~np.isnan(combined_data)
    new_has_data = ~np.isnan(new_data)
    
    return np.where(
        ~combined_has_data, new_data,
        np.where(
            ~new_has_data, combined_data,
            fill_value  # Both have data - set to fill_value
        )
    )


def _update_overlap_mask(overlap_mask: np.ndarray, new_data: np.ndarray) -> np.ndarray:
    """Update overlap mask by incrementing where new data is valid."""
    return np.where(~np.isnan(new_data), overlap_mask + 1, overlap_mask)


def make_helioprojective_map(
    map_files: List[str], 
    overlap: Union[Literal['max', 'mean'], float],
    apply_rotation: bool = True, 
    preserve_limb: bool = True, 
    drag_rotate: bool = False,
    algorithm: Literal['exact', 'interpolation', 'adaptive'] = 'exact',
    remove_off_disk: bool = False,
    apply_los_correction: bool = False
):
    """
    Create a helioprojective full disk map from multiple EIS raster maps.
    
    This function combines multiple EIS raster maps into a single full disk map 
    in helioprojective coordinates. Different methods are available for handling 
    overlapping regions between maps.
    
    Parameters
    ----------
    map_files : List[str]
        List of file paths to EIS maps to be combined. Must not be empty.
        The first map determines the reference time and coordinate system.
    overlap : {'max', 'mean'} or float
        Method for handling overlapping regions:
        - 'max': Use value with maximum absolute magnitude (for Doppler velocities)
        - 'mean': Average overlapping values
        - Any numeric value (e.g., np.nan, 0, -999): Set overlapping regions to this value
    apply_rotation : bool, default=True
        Whether to apply differential rotation correction to align maps in time.
    preserve_limb : bool, default=True
        Whether to preserve off-limb data or crop to solar disk.
    drag_rotate : bool, default=False
        If True, use drag-based rotation model instead of Howard model.
    algorithm : {'exact', 'interpolation'}, default='exact'
        Reprojection algorithm to use:
        - 'exact': Exact reprojection (slower but more accurate)
        - 'interpolation': Interpolation-based reprojection (faster but less accurate)
        - 'adaptive': Adaptive reprojection (adaptive, anti-aliased resampling algorithm, with optional flux conservation)
    remove_off_disk : bool, default=False
        If True, set off-disk (off-limb) pixels to NaN for each individual map 
        before combining. This is applied after reprojection but before data combination.
    apply_los_correction : bool, default=False
        If True, apply line-of-sight correction to account for viewing angle effects.
        Divides pixel values by cos(viewing_angle) to correct for foreshortening.
        Useful for radial measurements (e.g., velocities) where only the line-of-sight
        component is observed.
        
    Returns
    -------
    tuple of (sunpy.map.Map, sunpy.map.Map) or (None, None)
        A tuple containing:
        - The combined full disk map (or None if processing failed)
        - The overlap count map showing how many maps contributed to each pixel
        
    Notes
    -----
    For Doppler velocity measurements, the 'max' overlap method selects the 
    velocity with the strongest magnitude (largest |velocity|), preserving
    the sign to maintain the direction information.
    
    Examples
    --------
    >>> map_files = ['map1.fits', 'map2.fits', 'map3.fits']
    >>> fd_map, overlap_map = make_helioprojective_map(map_files, 'max')
    >>> 
    >>> # Using interpolation for faster processing
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
    >>> # Apply line-of-sight correction for radial measurements
    >>> fd_map, overlap_map = make_helioprojective_map(map_files, 'max', apply_los_correction=True)
    """
    # Input validation
    if not map_files:
        raise ValueError("map_files cannot be empty")
    
    # Load the first map to establish the reference frame
    try:
        first_map = sunpy.map.Map(map_files[0])
    except Exception as e:
        print(f"Error loading first map {map_files[0]}: {e}")
        return None, None

    fd_size_arcsec = 3500  # Hardcoded to avoid anomolous rasters generating incorrect huge full disk maps and crashing with memory errors

    # Get pixel scale from WCS with proper units
    cdelt_wcs = first_map.wcs.wcs.cdelt  # This is in degrees
    cunit_wcs = first_map.wcs.wcs.cunit  # Units (typically ['deg', 'deg'])
    
    # Convert to arcseconds using astropy units
    map_dx = (cdelt_wcs[0] * u.Unit(cunit_wcs[0])).to(u.arcsec).value
    map_dy = (cdelt_wcs[1] * u.Unit(cunit_wcs[1])).to(u.arcsec).value
    
    fd_width = round(fd_size_arcsec / map_dx)
    fd_height = round(fd_size_arcsec / map_dy)

    # Create full disk coordinate at disk center using first map's coordinate frame
    fd_coord = SkyCoord(0*u.arcsec, 0*u.arcsec, frame=first_map.coordinate_frame)
    
    # Create the full disk data array
    fd_data = np.full((fd_height, fd_width), np.nan)
    
    # Use SunPy's make_fitswcs_header function to create proper header
    fd_header = sunpy.map.make_fitswcs_header(
        fd_data, 
        fd_coord,
        scale=[map_dx, map_dy] * u.arcsec/u.pix,
        projection_code="TAN"
    )
    
    fd_map = sunpy.map.Map(fd_data, fd_header)

    overlap_mask = np.zeros((fd_height, fd_width))
    combined_data = np.full((fd_height, fd_width), np.nan)

    for map_file in map_files:

        try:
            map = sunpy.map.Map(map_file)
        except Exception as e:
            print(f"Error loading map {map_file}: {e}. Skipping.")
            continue

        # Remove off-disk data if requested (before any reprojection)
        if remove_off_disk:
            map_coords = sunpy.map.all_coordinates_from_map(map)
            on_disk_mask = sunpy.map.coordinate_is_on_solar_disk(map_coords)
            map_data_masked = np.where(on_disk_mask, map.data, np.nan)
            map = sunpy.map.Map(map_data_masked, map.meta)

        # Apply line-of-sight correction if requested (before any reprojection)
        if apply_los_correction:
            map_coords = sunpy.map.all_coordinates_from_map(map)
            # Calculate distance from disk center in arcsec
            r_arcsec = np.sqrt(map_coords.Tx.value**2 + map_coords.Ty.value**2)
            # Get solar radius in arcsec for this observation
            solar_radius_arcsec = map.rsun_obs.to(u.arcsec).value
            # Calculate viewing angle (0 at disk center, pi/2 at limb)
            # Only apply correction to on-disk pixels
            on_disk = sunpy.map.coordinate_is_on_solar_disk(map_coords)
            viewing_angle = np.zeros_like(r_arcsec)
            viewing_angle[on_disk] = np.arcsin(r_arcsec[on_disk] / solar_radius_arcsec)
            # Apply correction: divide by cos(viewing_angle) to correct for foreshortening
            cos_correction = np.ones_like(map.data)
            cos_correction[on_disk] = 1.0 / np.cos(viewing_angle[on_disk])
            # # Avoid division by very small numbers near the limb
            # cos_correction = np.where(cos_correction > 10, np.nan, cos_correction)
            corrected_data = map.data * cos_correction
            map = sunpy.map.Map(corrected_data, map.meta)

        if apply_rotation:

            if drag_rotate:

                SAFE_LIMB_DIST_ARCSEC = 950
                map_center_radial_distance = np.sqrt(map.center.Tx.value ** 2 + map.center.Ty.value ** 2)

                def differental_rotate_map_by_drag(map, point):
                    map_time = datetime.strptime(map.meta['date_obs'], '%Y-%m-%dT%H:%M:%S.%f')
                    first_map_time = datetime.strptime(first_map.meta['date_obs'], '%Y-%m-%dT%H:%M:%S.%f')
                    duration = map_time - first_map_time  # Calculate the time difference between the map and the first map
                    duration = duration.seconds * u.second  # Convert the duration to an astropy time object
                    diffrot_point = RotatedSunFrame(base=point, duration=duration)  # Rotate the point by the differential rotation
                    transformed_diffrot_point = diffrot_point.transform_to(map.coordinate_frame)
                    shift_x = transformed_diffrot_point.Tx - point.Tx  # Calculate the difference between the original and transformed points
                    shift_y = transformed_diffrot_point.Ty - point.Ty
                    map = map.shift_reference_coord(shift_x, shift_y)  # Shift the map by the difference
                    return map

                if map_center_radial_distance < SAFE_LIMB_DIST_ARCSEC:
                    point = map.center  # This map center is on the disk so can be used to calculate the differential rotation

                else:  # Choose a point along the line between the map center and the centre of the disk
                    angle = np.arctan2(map.center.Tx, map.center.Ty)  # Calculate the angle between the map center and the centre of the disk
                    point = SkyCoord(SAFE_LIMB_DIST_ARCSEC * np.sin(angle) * u.arcsec, SAFE_LIMB_DIST_ARCSEC * np.cos(angle) * u.arcsec, obstime=map.date, observer=map.observer_coordinate, frame=Helioprojective)  # Get the point at the edge of the disk along this line

                map = differental_rotate_map_by_drag(map, point)  # Perform the differential rotation

                if preserve_limb:
                    with SphericalScreen(map.observer_coordinate, only_off_disk=True):
                        map = map.reproject_to(fd_map.wcs, algorithm=algorithm)  # Add map data to array of same size as full disk data array, for combination below
                else:
                    map = map.reproject_to(fd_map.wcs, algorithm=algorithm)

            else:

                with propagate_with_solar_surface(rotation_model='howard'):
                    if preserve_limb:
                        with SphericalScreen(map.observer_coordinate, only_off_disk=True):
                            map = map.reproject_to(fd_map.wcs, algorithm=algorithm)
                    else:
                        map = map.reproject_to(fd_map.wcs, algorithm=algorithm)

        else:

            if preserve_limb:
                with SphericalScreen(map.observer_coordinate, only_off_disk=True):
                    map = map.reproject_to(fd_map.wcs, algorithm=algorithm)
            else:
                map = map.reproject_to(fd_map.wcs, algorithm=algorithm)

        # Combine data based on overlap method
        if overlap == 'max':
            combined_data = _combine_data_max_abs(combined_data, map.data)
        elif overlap == 'mean':
            combined_data = _combine_data_mean(combined_data, map.data)
        else:
            # overlap is a numeric value (e.g., np.nan, 0, -999)
            combined_data = _combine_data_with_fill(combined_data, map.data, overlap)
        
        # Update overlap mask for all methods
        overlap_mask = _update_overlap_mask(overlap_mask, map.data)

    # Create final map based on overlap method
    if overlap == 'mean':
        # Average accumulated values using overlap count
        # Avoid division by zero by preserving NaN where no data exists
        averaged_data = np.where(overlap_mask > 0, combined_data / overlap_mask, np.nan)
        fd_map = sunpy.map.Map(averaged_data, fd_map.meta)
    else:
        # For 'max' or any numeric fill value
        fd_map = sunpy.map.Map(combined_data, fd_map.meta)
    
    # Create overlap mask map
    overlap_map = sunpy.map.Map(overlap_mask, fd_map.meta)

    # Tidy up the off limb data if the limb should be cropped, to make sure limb is excluded
    if not preserve_limb:
        pixel_coords = sunpy.map.all_coordinates_from_map(fd_map)
        limb_mask = sunpy.map.coordinate_is_on_solar_disk(pixel_coords)
        fd_map_data = np.where(limb_mask, fd_map.data, np.nan)
        fd_map = sunpy.map.Map(fd_map_data, fd_map.meta)

    return fd_map, overlap_map

def make_carrington_map(map_files, save_dir, wavelength, measurement, overlap, apply_rotation=True, deg_per_pix=0.1, save_fit=False, save_plot=False, plot_ext='png', plot_dpi=300, skip_done=True):
    """
    Make a Carrington full disk map from a list of maps.
    """
    
    # Load the first map to establish the reference frame
    try:
        first_map = sunpy.map.Map(map_files[0])
    except Exception as e:
        print(f"Error loading first map {map_files[0]}: {e}")
        return None

    map_file_datetime = os.path.basename(map_files[0]).split('.')[0].replace('eis_', '')
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
            map = sunpy.map.Map(map_file)
        except Exception as e:
            print(f"Error loading map {map_file}: {e}. Skipping.")
            continue

        # Change the time of the fd_map to the same as the time of the raster
        fd_map_temp = fd_map
        if apply_rotation:
            fd_map_temp.meta['date_obs'] = map.meta['date_obs']

        # Create a WCS object for the target map
        target_wcs = fd_map_temp.wcs

        # Reproject the raster map to the Carrington map
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