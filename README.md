# EISMaps

EISMaps is a Python package for making raster and full-disk maps from Hinode/EIS observations. It works on top of [EISPAC](https://github.com/USNavalResearchLaboratory/eispac) and provides functions for:

- fitting EIS spectral lines in batches
- radiometric calibration
- SunPy maps of intensity, Doppler velocity and non-thermal velocity
- full-disk maps in helioprojective or Carrington coordinates

Contact: James McKevitt (jm2@mssl.ucl.ac.uk)

Licence: CC BY-NC-SA 4.0. See [LICENSE](LICENSE).

## Citation And Acknowledgement

If you use EISMaps in a publication, please cite:

- McKevitt, J., et al. (2026). Coronal non-thermal and Doppler plasma flows driven by photospheric flux in 28 active regions. Publications of the Astronomical Society of Japan. https://doi.org/10.1093/pasj/psag024

Please also acknowledge the software in your acknowledgements section. Recommended text:

> This work made use of version X.X of the EISMaps Python package (DOI).

The DOI of each version is on the [Zenodo release page](https://doi.org/10.5281/zenodo.17640972).

## Installation

```bash
python -m pip install git+https://github.com/jamesmckevitt/eismaps.git
```

## Tutorial

The notebook [full_disk_tutorial.ipynb](full_disk_tutorial.ipynb) goes through making full-disk mosaics, from downloading the data to plotting the maps.

## Public API

The package exports these functions:

```python
from eismaps import (
    apply_calibration,
    fit,
    list_fit_lines,
    make_carrington_map,
    make_helioprojective_map,
    make_maps,
)
```

### Fitting

`list_fit_lines()` lists the lines covered by the EISPAC templates that match one or more rasters:

```python
from eismaps import list_fit_lines

lines = list_fit_lines(['eis_20130113_074850.data.h5'])
print(lines)
```

`fit()` runs EISPAC fits on one or more rasters:

```python
from eismaps import fit

fit_results = fit(
    ['eis_20130113_074850.data.h5'],
    lines_to_fit=['fe_12_195_119'],
    ncpu='max',
    save=False,
)
```

Main options:

- `lines_to_fit='all'` fits every line that has a template
- `ncpu='max'` uses all available CPU cores
- `filter_chi2=<float>` sets fitted values to NaN where chi-squared is above this value, before saving
- `output_dir='...'` writes `.fit.h5` files, and the fit results are still returned in memory
- `skip_done=True` (the default) loads existing `.fit.h5` files from `output_dir` instead of fitting again
- `lock_to_window=True` keeps one fit result per spectral window

### Raster Maps

`make_maps()` turns fit results into SunPy maps:

```python
from eismaps import make_maps

map_products = make_maps(
    fit_results,
    measurement=['int', 'vel', 'ntv'],
    ncpu='max',
    save=False,
)
```

Behaviour:

- one measurement gives a SunPy map, or a `MapSequence` for several rasters
- several measurements give a dictionary keyed by measurement name
- the measurements are `int`, `vel`, `wid`, `ntv`, `chi2`, and the fit uncertainties `int_err`, `vel_err` and `wid_err`
- pixels without a usable fit are NaN in every measurement: spectra EISPAC could not fit, fits whose line amplitude is zero or has no error, and fits with no degrees of freedom (infinite chi-squared)
- `vel` and `vel_err` are also NaN where the fitted centroid is not a measurement, and `wid`, `wid_err` and `ntv` where the width is not. A parameter is not a measurement if it ended the fit on one of its template limits, or if MPFIT gave it a zero error. MPFIT does this for parameters on a limit, for parameters fixed or tied in the template, and when it cannot compute the errors.
- `vel` is EISPAC's Doppler velocity minus the median of each raster column, with the median taken over measured centroids only
- `int_err` is EISPAC's intensity error, which has no width term where the width error is zero (for example a width on a limit)

Useful options:

- `clip=True` sets outliers to NaN before saving
- `output_dir='...'` writes the maps as FITS files
- `vel_los_correct=True` applies a line-of-sight correction to velocity maps, assuming the velocity is radial

### Calibration

`apply_calibration()` converts intensity maps from EISPAC's pre-flight calibration to one of several published calibrations. It takes a map, a list of maps or a dict of maps:

```python
from eismaps import apply_calibration

calibrated = apply_calibration(
    map_products['int'],
    method='del_zanna_2025',
    ncpu='max',
)
```

Supported `method=` values are `ground`, `ground_cal`, `preflight`, `warren_2014`, `del_zanna_2013`, and `del_zanna_2025`. The first three keep EISPAC's pre-flight values (a factor of 1). `int_err` maps can be calibrated in the same way. The factor applied is stored in each calibrated map's `calfact` keyword, and `eismaps.calibration.calibration_factor(date, wavelength, method)` returns it directly.

### Full-Disk Assembly

`make_helioprojective_map()` combines raster maps into a full-disk helioprojective map:

```python
from eismaps import make_helioprojective_map

fd_map, overlap_map = make_helioprojective_map(
    calibrated,
    overlap='mean',
    apply_rotation=True,
    preserve_limb=True,
    remove_off_disk='after',
    algorithm='interpolation',
    ncpu='max',
)
```

Main options:

- `overlap='mean'` averages overlapping rasters
- `overlap='max'` keeps the value of largest magnitude in overlaps
- `apply_rotation=True` differentially rotates the rasters to the time of the reference map before combining them
- `preserve_limb=True` keeps off-limb signal during reprojection
- `algorithm='exact'`, `'interpolation'`, or `'adaptive'` selects the reprojection method
- `remove_off_disk='before'` or `'after'` sets off-disk pixels to NaN in each raster before combining, or in the combined map

To carry uncertainties and other per-raster quantities through the mosaic, pass them with the values:

```python
fd_int, overlap_map, extras = make_helioprojective_map(
    calibrated_intensity,
    overlap='max',
    algorithm='exact',
    uncertainty=calibrated_intensity_errors,
    companions={'chi2': map_products['chi2']},
)
fd_int_err = extras['uncertainty']
fd_chi2 = extras['chi2']
```

- `uncertainty` and `companions` need `algorithm='exact'`, and the values are then the same as without them
- with `overlap='max'`, each pixel's uncertainty and companions come from the raster that supplied its value; with `overlap='mean'` the uncertainty is that of the mean
- the uncertainty of a resampled pixel accounts for the EIS pixels it averages, assuming their errors are independent
- by default a NaN raster pixel makes every output pixel it overlaps NaN, as in `reproject_exact`. `skip_nan=True` leaves NaN pixels out, so each output pixel averages the valid pixels it overlaps (needs `algorithm='exact'`).

`make_carrington_map()` makes a Carrington map instead of a helioprojective one.

## Typical API Flow

The functions chain together:

```python
from eismaps import apply_calibration, fit, make_helioprojective_map, make_maps

fit_results = fit(data_files, lines_to_fit=['fe_12_195_119'])
map_products = make_maps(fit_results, measurement=['int', 'vel', 'ntv'])
calibrated_intensity = apply_calibration(map_products['int'], method='del_zanna_2025')
fd_map, overlap_map = make_helioprojective_map(calibrated_intensity, overlap='mean')
```

## Developer note: Calibration Assets

The calibration code reads files in `eismaps/calibration_data/`, and the non-thermal velocity code reads `eismaps/eis_width2velocity.dat`. They are copies of files in a local SolarSoft (SSW) tree, under `hinode/eis/response` and `hinode/eis/idl/atest/hwarren`.

### What is in the package

| File | Where it comes from in SSW | Used by |
|------|----------------------------|---------|
| `fit_eis_ea_YYYY-MM-DD.sav` | `hinode/eis/idl/atest/hwarren/calibration/new/fit_eis_ea_*.sav` (latest) | `del_zanna_2025` time-interpolated effective area |
| `EIS_EffArea_A.004` | `hinode/eis/response/EIS_EffArea_A.004` (exact) | `del_zanna_2013` (long-wave pinned ground area) |
| `EIS_EffArea_B.004` | `hinode/eis/response/EIS_EffArea_B.004` (exact) | `del_zanna_2013` (short-wave pinned ground area) |
| `EIS_EffArea_A.NNN`, `EIS_EffArea_B.NNN` | `hinode/eis/response/EIS_EffArea_*.*` (latest) | `warren_2014` and `del_zanna_2025` (pre-flight area in the calibration factor); `del_zanna_2013` if the `.004` files are missing |
| `eis_width2velocity.dat` (package root) | `hinode/eis/idl/atest/hwarren/eis_width2velocity.dat` | Non-thermal velocity helper (`eismaps.utils.width2velocity`) |
| `sources.json` | written by the sync helper | Provenance manifest of the copied files |

The Warren 2014 NRL coefficients and the Del Zanna 2013 spline points are not files: they are constants in `eismaps/calibration.py` (`WARREN_2014_*` and `GDZ_2013_*`). SSW's `eis_ltds.pro` also holds the Del Zanna 2013 points in its source, while `eis_ea_nrl.pro` reads the Warren 2014 coefficients from `nrl_ea_coeff_v1.3.genx`. Both sets are stable: the Warren 2014 NRL set is at v1.3 (Feb 2016) and the GDZ 2013 set has not been revised.

### Refreshing the assets from SolarSoft

To update the files from a local SSW tree (`$SSW`), run:

```bash
eismaps-sync-calibration --ssw-root "$SSW"
```

Or call the helper from Python:

```python
from eismaps.calibration import sync_solarsoft_calibration_data

manifest = sync_solarsoft_calibration_data(ssw_root='/path/to/ssw')
print(manifest['copied'])
```

This:

1. Copies the latest `fit_eis_ea_*.sav` into `calibration_data/`.
2. Copies the latest `EIS_EffArea_A.*` and `EIS_EffArea_B.*` text tables into `calibration_data/`.
3. Copies `EIS_EffArea_A.004` and `EIS_EffArea_B.004` by name, as the Del Zanna 2013 model is defined against these versions.
4. Copies `eis_width2velocity.dat` into the package root, next to `__init__.py`.
5. Writes a manifest, `calibration_data/sources.json`, with the source and destination paths, the SSW root and the UTC time of the sync.

The Warren 2014 coefficients are not copied, as they are constants in the source (see above).

## Full-Disk Scan Dataset

The table below gives the time coverage of each full-disk scan in the dataset of "Full-Disk Spectroscopy of the Solar Corona Across a Solar Cycle with Hinode/EIS" (McKevitt et al., 2026). The start and end times are the filename timestamps of the first and last EIS raster file in each scan directory, in UTC.

| Disk | Start (UTC) | End (UTC) |
|------|-------------|-----------|
| 20130116 | 2013-01-16 09:37:20 | 2013-01-18 06:00:44 |
| 20130225 | 2013-02-25 08:11:49 | 2013-02-28 10:46:20 |
| 20150401 | 2015-04-01 09:14:49 | 2015-04-03 00:44:13 |
| 20151018 | 2015-10-18 10:27:19 | 2015-10-20 00:26:12 |
| 20171021 | 2017-10-21 10:49:49 | 2017-10-23 02:24:12 |
| 20180825 | 2018-08-25 12:25:41 | 2018-08-27 05:23:13 |
| 20181028 | 2018-10-28 10:32:49 | 2018-10-30 00:30:41 |
| 20190413 | 2019-04-13 17:37:41 | 2019-04-15 10:14:42 |
| 20190505 | 2019-05-05 12:16:13 | 2019-05-07 05:17:42 |
| 20190912 | 2019-09-12 12:02:20 | 2019-09-14 02:50:43 |
| 20200118 | 2020-01-18 11:07:20 | 2020-01-20 01:05:12 |
| 20200422 | 2020-04-22 13:12:40 | 2020-04-24 06:11:12 |
| 20200906 | 2020-09-06 14:56:43 | 2020-09-08 06:02:41 |
| 20210418 | 2021-04-18 00:37:43 | 2021-04-19 17:52:12 |
| 20211016 | 2021-10-16 12:17:20 | 2021-10-18 02:47:12 |
| 20220507 | 2022-05-07 12:31:50 | 2022-05-09 05:31:12 |
| 20220925 | (no data) | (no data) |
| 20230429 | 2023-04-29 10:59:20 | 2023-05-01 03:57:42 |
| 20230905 | (no data) | (no data) |
| 20240310 | 2024-03-10 10:16:10 | 2024-03-11 21:35:41 |
| 20240320 | 2024-03-20 05:04:49 | 2024-03-22 01:18:43 |