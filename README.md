# EISMaps

EISMaps is a Python package for processing data from the EIS (Extreme-ultraviolet Imaging Spectrometer) instrument on the Hinode spacecraft. It assembles Hinode/EIS slit full-disk maps with various options. It also provides useful wrapping functionality to the fantastic [eispac](https://github.com/USNavalResearchLaboratory/eispac) code for batch processing of large EIS datasets.

Contact: James McKevitt (jm2@mssl.ucl.ac.uk).

Licence: CC BY-NC-SA 4.0 (see [license file](LICENSE) for details).

If using this code, please cite the paper:
- McKevitt, J., et al. (2026). Coronal non-thermal and Doppler plasma flows driven by photospheric flux in 28 active regions. Publications of the Astronomical Society of Japan. https://doi.org/10.1093/pasj/psag024

and the code using either:
- DOI: [10.5281/zenodo.17640972](https://doi.org/10.5281/zenodo.17640972)  
- GitHub: [github.com/jamesmckevitt/eismaps](https://github.com/jamesmckevitt/eismaps)

## Installation

```bash
python -m pip install git+https://github.com/jamesmckevitt/eismaps.git
```

## Quick Start: Minimal Full Disk Pipeline

If you want to stay in memory instead of writing intermediate `.fit.h5` and map files, use the function-oriented API exported from `eismaps`:

```python
from datetime import datetime
from eismaps import apply_calibration, download, fit, make_helioprojective_map, make_maps, search

file_urls = search(
    datetime(2013, 1, 13, 7, 48, 0),
    datetime(2013, 1, 13, 7, 50, 0),
)
local_files = download(file_urls)
fit_results = fit(
    local_files,
    lines_to_fit=['fe_12_195_119'],
    ncpu=4,
)
map_products = make_maps(
    fit_results,
    measurement=['int', 'vel'],
)
calibrated_maps = apply_calibration(
    map_products,
    method='del_zanna_2025',
)
fd_map, overlap_map = make_helioprojective_map(
    calibrated_maps['int'],
    overlap='mean',
    apply_rotation=True,
)
```

The function API is in-memory by default. To write outputs as part of the same call, pass `output_dir='...'`:

```python
fit_results = fit(
    local_files,
    lines_to_fit=['fe_12_195_119'],
    output_dir='fits',
)

map_products = make_maps(
    fit_results,
    measurement=['int', 'vel'],
    output_dir='maps',
)
```

With one requested measurement, `make_maps()` returns a SunPy map or `MapSequence`. With multiple requested measurements, it returns a dictionary keyed by measurement name whose values are SunPy maps or `MapSequence` objects.

### Step 1: Spectral Line Fitting

First, fit spectral lines in EIS data using the `fit.batch()` function:

```python
import eismaps.utils.find as find
from eismaps import fit

# Batch fit Fe XII 195.119 Å line (wrapping eispac fitting)
fit.batch(
    data_files,                         # List of .data.h5 files to process
    lines_to_fit=['fe_12_195_119'],     # Specific lines to fit (or 'all' for all available)
    ncpu='max',                         # Number of CPUs ('max' uses all available cores)
    filter_chi2=None,                   # Chi-squared filter threshold (None = no filtering)
    save=True,                          # Save fitted results
    output_dir=None,                    # Output directory (None = same as input files)
    output_dir_tree=False,              # Create date-based directory structure
    lock_to_window=False,               # Fit one component per window (True) or all possible (False)
    list_lines_only=False               # Only list available lines without fitting
)
```

### Step 2: Create Individual Raster Maps

Convert fitted data into intensity, Doppler velocity, and non-thermal velocity maps:

```python
from eismaps import maps

# Create individual maps with all available options
maps.batch(
    fit_files,                          # List of .fit.h5 files
    measurement=['int', 'vel', 'ntv'],  # Parameters to map: 'int', 'vel', 'wid', 'ntv', 'chi2'
    clip=False,                         # Apply statistical clipping to remove outliers
    save_fit=True,                      # Save FITS files of the maps
    save_plot=False,                    # Save PNG plots of the maps
    output_dir=None,                    # Output directory (None = same as input)
    output_dir_tree=False,              # Create date-based directory structure  
    vel_los_correct=False,              # Apply line-of-sight velocity correction
    skip_done=True,                     # Skip files that already exist
)
```

### Step 3: Create Full-Disk Maps

Combine individual raster maps into full-disk observations:

```python
from eismaps import full_disk
import numpy as np

# Create intensity full-disk map
intensity_fd = full_disk.make_helioprojective_map(
    int_files,                          # List of individual raster map files
    overlap='mean',                     # How to handle overlaps: 'mean', 'max', or fixed value
    apply_rotation=True,                # Apply differential rotation correction
    preserve_limb=True,                 # Keep off-limb data  
    drag_rotate=False,                  # Use Howard rotation model (False) or drag-based (True)
    algorithm='exact',                  # Reprojection: 'exact', 'interpolation', 'adaptive'
    remove_off_disk='after',            # Remove off-disk pixels: False, True/'before', 'after' disk assembly
    apply_los_correction=False          # Apply line-of-sight viewing angle correction (if not applied earlier)
)

# Create velocity full-disk map
velocity_fd = full_disk.make_helioprojective_map(
    vel_files,
    overlap='max',
    apply_rotation=True,
    preserve_limb=True, 
    drag_rotate=False,
    algorithm='exact',
    remove_off_disk='after',
    apply_los_correction=True
)

# Create non-thermal velocity full-disk map
ntv_fd = full_disk.make_helioprojective_map(
    ntv_files,
    overlap='max',
    apply_rotation=True,
    preserve_limb=True,
    drag_rotate=False, 
    algorithm='exact',
    remove_off_disk='after',
    apply_los_correction=False
)

# Save full-disk maps
intensity_fd.save('./int_fd.fits')
velocity_fd.save('./vel_fd.fits')
ntv_fd.save('./ntv_fd.fits')
```

## Calibration Assets and Release Sync

The calibration helpers in `eismaps.calibration` now expect their reference assets to be traced back to a local SolarSoft checkout.

Before building a release or pushing updated calibration assets, sync the latest upstream files from SolarSoft into the package:

```bash
eismaps-sync-calibration --ssw-root "$SSW"
```

This copies the following upstream files into `eismaps/calibration_data/` and writes a `sources.json` manifest recording where they came from:

- `fit_eis_ea_*.sav` from `hinode/eis/idl/atest/hwarren/calibration/new/`
- `nrl_ea_coeff_v*.genx` from `hinode/eis/idl/atest/hwarren/calibration/`
- `EIS_EffArea_A.*` and `EIS_EffArea_B.*` from `hinode/eis/response/`

The Del Zanna 2025-style interpolation path uses the synced `fit_eis_ea_*.sav` file directly. The pre-flight effective area path uses the synced `EIS_EffArea_*` response files directly. The Warren 2014 path still uses the existing converted `eis_calib_warren_2014.sav` cache, while the raw `nrl_ea_coeff_v*.genx` file is copied alongside it for provenance.

You can also call the sync helper from Python:

```python
from eismaps.calibration import sync_solarsoft_calibration_data

manifest = sync_solarsoft_calibration_data(ssw_root='/path/to/ssw')
print(manifest['copied'])
```

## Data Search and Download

The `eismaps.data` module provides a lightweight discovery and download layer for EIS HDF5 products.

Use `search.main()` to list matching `.h5` files in a time range:

```python
from datetime import datetime
from eismaps.data.search import main as search_eis

file_urls = search_eis(
    datetime(2013, 1, 13, 7, 48, 0),
    datetime(2013, 1, 13, 7, 50, 0),
    source='nrl',
)
```

`search.main(start_datetime, end_datetime, source='nrl', base_url=None)` supports these options:

- `source='nrl'` uses `https://eis.nrl.navy.mil/level1/hdf5`
- `source='mssl'` uses `https://vsolar.mssl.ucl.ac.uk/eispac/hdf5`
- `source='custom'` requires `base_url='https://...'`

Use `search.check_source()` as a quick connectivity check before a larger batch:

```python
from eismaps.data.search import check_source

if check_source('nrl'):
    print('NRL is reachable')
```

There are two download helpers:

```python
from eismaps.data.download import eispac as eispac_download
from eismaps.data.download import eismaps as direct_download

eispac_download(file_urls, source='nrl', local_top='data_eis', datetree=True)
direct_download(file_urls, source='same', local_top='data_eis', datatree=False)
```

- `download.eispac(...)` delegates to `eispac.download.download_hdf5_data()` and returns the expected local file paths
- `download.eismaps(...)` does a plain HTTP download and returns the local file paths it wrote
- `source='same'` leaves each URL untouched
- `source='nrl'` or `source='mssl'` rewrites the input filenames against one of the known mirror roots
- `datetree=True` or `datatree=False` controls whether files are written into `YYYY/MM/DD` directories

## Fitting API

The fitting layer wraps EISPAC but keeps file selection, output naming, and post-fit cleanup in one place.

List all candidate lines in one or more rasters without fitting anything:

```python
from eismaps import fit

available_lines = fit.batch(
    ['eis_20130113_074850.data.h5'],
    list_lines_only=True,
)
print(available_lines)
```

Fit a small set of lines and save the resulting `.fit.h5` files:

```python
fit.batch(
    files=['eis_20130113_074850.data.h5'],
    lines_to_fit=['fe_12_195_119', 'fe_15_284_160'],
    ncpu=4,
    filter_chi2=5,
    save=True,
    output_dir='fits',
    output_dir_tree=False,
    lock_to_window=False,
)
```

`fit.batch()` now returns the in-memory EISPAC fit result objects directly, so file-based and object-based workflows can share the same fitting call without an extra wrapper layer.

Important `fit.batch()` options:

- `lines_to_fit='all'` keeps every matched template line
- `ncpu='max'` passes all available cores through to EISPAC
- `filter_chi2=<float>` masks poor fits before saving
- `output_dir_tree=True` writes outputs under `YYYY/MM/DD`
- `lock_to_window=True` keeps one fit product per spectral window instead of all matching lines
- `list_lines_only=True` returns the discovered line labels instead of fitting

For advanced use, `fit.fit_specific_line()` lets you target a single window and template explicitly:

```python
fit.fit_specific_line(
    file='eis_20130113_074850.data.h5',
    iwin=3,
    template='fe_12_195_119.2c.template.h5',
    lines_to_fit=['fe_12_195_119'],
    ncpu=1,
    output_dir='fits',
)
```

## Map-Making API

The `eismaps.maps` module converts saved fit products into raster maps.

```python
from eismaps import maps

maps.batch(
    files=['fits/eis_20130113_074850.fe_12_195_119.fit.h5'],
    measurement=['int', 'vel', 'wid', 'ntv', 'chi2'],
    clip=False,
    save_fit=True,
    save_plot=True,
    output_dir='maps',
    output_dir_tree=False,
    vel_los_correct=True,
    skip_done=True,
    mssl_solarb_file_format=False,
)
```

`maps.batch()` accepts saved fit filenames or raw EISPAC fit results from `fit.batch()`. It returns standard SunPy objects: a single map, a `MapSequence`, or a dictionary of those when you request multiple measurements.

The `measurement` list accepts:

- `int` for intensity
- `vel` for Doppler velocity
- `wid` for fitted width
- `ntv` for non-thermal velocity
- `chi2` for the fit quality map

Other useful options:

- `clip=True` applies a standard-deviation outlier filter before saving
- `save_fit=True` writes FITS maps
- `save_plot=True` writes PNG quicklook plots
- `vel_los_correct=True` applies a line-of-sight geometric correction to the velocity map
- `mssl_solarb_file_format=True` switches plot filenames to the legacy MSSL/Solar-B convention

If you already have an EISPAC `fit_res` object, you can derive the custom products directly:

```python
ntv_map = maps.get_ntv_map(fit_res)
chi2_map = maps.get_chi2_map(fit_res)
```

## Full-Disk Assembly Options

Two map-combination helpers are available in `eismaps.full_disk`.

Use `make_helioprojective_map()` for solar-disk products in helioprojective coordinates:

```python
from eismaps.full_disk import make_helioprojective_map

fd_map, overlap_map = make_helioprojective_map(
    map_files=int_files,
    overlap='mean',
    apply_rotation=True,
    preserve_limb=True,
    drag_rotate=False,
    algorithm='exact',
    remove_off_disk='after',
    apply_los_correction=False,
)
```

Key options:

- `overlap='mean'` averages overlapping rasters
- `overlap='max'` keeps the value with the largest absolute magnitude
- `overlap=<number>` fills overlapping regions with a fixed numeric value
- `apply_rotation=True` applies differential rotation before reprojection
- `drag_rotate=True` switches to the drag-based rotation path
- `algorithm='exact'`, `'interpolation'`, or `'adaptive'` selects the reprojection method
- `remove_off_disk=False`, `True`, `'before'`, or `'after'` controls when off-disk pixels are masked
- `apply_los_correction=True` applies a viewing-angle correction during assembly

Use `make_carrington_map()` for Carrington projections:

```python
from eismaps.full_disk import make_carrington_map

fd_map = make_carrington_map(
    map_files=int_files,
    save_dir='fd',
    wavelength='fe_12_195_119',
    measurement='int',
    overlap='mean',
    apply_rotation=True,
    deg_per_pix=0.1,
    save_fit=True,
    save_plot=True,
)
```

Both full-disk helpers accept raster map filenames, single SunPy map objects, or SunPy map sequences.

## Calibration API

The calibration helpers operate on SunPy map objects and return a new calibrated map with the same metadata.

```python
from eismaps.calibration import calibrate_map, calib_2014, calib_del_zanna_2013, calib_del_zanna_2025

map_2014 = calib_2014(input_map)
map_2013 = calib_del_zanna_2013(input_map)
map_2025 = calib_del_zanna_2025(input_map)
map_generic = calibrate_map(input_map, method='del_zanna_2025')
```

Supported calibration methods for `calibrate_map()` are:

- `ground`
- `ground_cal`
- `preflight`
- `del_zanna_2013`
- `warren_2014`
- `del_zanna_2025`
- `del_zanna_2023` as a backward-compatible alias

The intended user-facing names are:

- `ground_cal`
- `del_zanna_2013`
- `warren_2014`
- `del_zanna_2025`

For lower-level calculations, the module also exposes the effective-area functions used internally:

```python
from eismaps.calibration import eis_ea, interpol_eis_ea

preflight_ea = eis_ea(195.119)
time_dependent_ea = interpol_eis_ea('2023-05-01T00:00:00', 195.119)
```

## Utility Helpers

The `eismaps.utils` namespace contains a few small helpers that are useful when scripting larger batches.

Collect files from a directory tree:

```python
from eismaps.utils.find import main as find_files

fit_files = find_files('.fit.h5', 'fits', unique_times=False)
```

- `unique_times=True` keeps only one file per `eis_YYYYMMDD_HHMMSS` timestamp

Convert an EIS line label into the lowercase filename style used across the package:

```python
from eismaps.utils.format import change_line_format

print(change_line_format('Fe XIV 264.700'))
# fe_14_264_700
```

Convert line-width metadata into thermal reference values:

```python
from eismaps.utils.width2velocity import width2velocity

t_max, v_therm = width2velocity('Fe', 12)
```

## Typical End-to-End Script

The core package can be driven with a short script that chains search, download, fitting, map creation, and full-disk assembly:

```python
from datetime import datetime

from eismaps.data.search import main as search_eis
from eismaps.data.download import eismaps as download_files
from eismaps import fit, maps
from eismaps.full_disk import make_helioprojective_map
from eismaps.utils.find import main as find_files

file_urls = search_eis(
    datetime(2013, 1, 13, 7, 48, 0),
    datetime(2013, 1, 13, 7, 50, 0),
)

downloaded = download_files(file_urls, local_top='data', datatree=False)
data_files = [path for path in downloaded if path.endswith('.data.h5')]

fit.batch(data_files, lines_to_fit=['fe_12_195_119'], output_dir='fits')
fit_files = find_files('.fit.h5', 'fits')

maps.batch(fit_files, measurement=['int', 'vel', 'ntv'], output_dir='maps')
int_files = find_files('.int.fits', 'maps')

fd_map, overlap_map = make_helioprojective_map(int_files, overlap='mean')
fd_map.save('fd_intensity.fits', overwrite=True)
```