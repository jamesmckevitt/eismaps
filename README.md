# EISMaps

EISMaps is a Python package for building science-ready raster and full-disk products from Hinode/EIS observations. It is designed to sit on top of [EISPAC](https://github.com/USNavalResearchLaboratory/eispac) and provide a compact, function-oriented workflow for:

- batch fitting EIS spectral lines
- applying radiometric calibration
- making intensity, Doppler velocity, and non-thermal velocity SunPy maps
- assembling full-disk helioprojective or Carrington products

Contact: James McKevitt (jm2@mssl.ucl.ac.uk)

Licence: CC BY-NC-SA 4.0. See [LICENSE](LICENSE).

## Citation And Acknowledgement

If you use EISMaps in a publication, please cite:

- McKevitt, J., et al. (2026). Coronal non-thermal and Doppler plasma flows driven by photospheric flux in 28 active regions. Publications of the Astronomical Society of Japan. https://doi.org/10.1093/pasj/psag024

You should also acknowledge the software in your acknowledgements section. Recommended text:

> This work made use of version X.X of the EISMaps Python package (DOI).

The DOI for each version can be found on the [Zenodo release page](https://doi.org/10.5281/zenodo.17640972).

## Installation

```bash
python -m pip install git+https://github.com/jamesmckevitt/eismaps.git
```

## Tutorial

A full notebook tutorial on how to assemble full-disk mosaics can be found in [full_disk_tutorial.ipynb](full_disk_tutorial.ipynb).

## Public API

The top-level package exports a small function-oriented interface:

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

Use `list_fit_lines()` to inspect which template lines are available in one or more rasters:

```python
from eismaps import list_fit_lines

lines = list_fit_lines(['eis_20130113_074850.data.h5'])
print(lines)
```

Use `fit()` to run EISPAC fits on one or more rasters:

```python
from eismaps import fit

fit_results = fit(
    ['eis_20130113_074850.data.h5'],
    lines_to_fit=['fe_12_195_119'],
    ncpu='max',
    save=False,
)
```

Key options:

- `lines_to_fit='all'` fits every available template line
- `ncpu='max'` uses all available CPU cores
- `filter_chi2=<float>` masks poor fits before saving
- `output_dir='...'` writes `.fit.h5` products while still returning in-memory fit objects
- `lock_to_window=True` keeps one fit result per spectral window

### Raster Maps

Use `make_maps()` to convert fit results into SunPy map products:

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

- a single requested measurement returns a SunPy map or `MapSequence`
- multiple requested measurements return a dictionary keyed by measurement name
- supported measurements are `int`, `vel`, `wid`, `ntv`, and `chi2`

Useful options:

- `clip=True` applies outlier clipping before saving to catch bad data
- `output_dir='...'` writes FITS map products
- `vel_los_correct=True` applies a line-of-sight correction to velocity maps (assuming the velocity is radial)

### Calibration

`apply_calibration()` converts gives an intensity map (or a list / dict of maps) in calibrated radiance, using one of several published methods:

```python
from eismaps import apply_calibration

calibrated = apply_calibration(
    map_products['int'],
    method='del_zanna_2025',
    ncpu='max',
)
```

Supported `method=` values are `ground`, `ground_cal`, `preflight`, `warren_2014`, `del_zanna_2013`, and `del_zanna_2025`.

### Full-Disk Assembly

Use `make_helioprojective_map()` to combine raster maps into a full-disk helioprojective product:

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

Important options:

- `overlap='mean'` averages overlapping rasters
- `overlap='max'` keeps the largest absolute value in overlaps
- `apply_rotation=True` applies differential rotation before reprojection
- `preserve_limb=True` keeps off-limb signal during reprojection
- `algorithm='exact'`, `'interpolation'`, or `'adaptive'` selects the reprojection method
- `remove_off_disk='before'` or `'after'` controls whether off-disk pixels are masked before or after reprojection

Use `make_carrington_map()` when you want a Carrington projection instead of a helioprojective full-disk map.

## Typical API Flow

The public functions are intended to chain:

```python
from eismaps import apply_calibration, fit, make_helioprojective_map, make_maps

fit_results = fit(data_files, lines_to_fit=['fe_12_195_119'])
map_products = make_maps(fit_results, measurement=['int', 'vel', 'ntv'])
calibrated_intensity = apply_calibration(map_products['int'], method='del_zanna_2025')
fd_map, overlap_map = make_helioprojective_map(calibrated_intensity, overlap='mean')
```

## Developer note: Calibration Assets

The calibration helpers expect reference assets in `eismaps/calibration_data/` (plus `eismaps/eis_width2velocity.dat` at the package root for the non-thermal velocity helper). These files are mirrored from a local SolarSoft (SSW) checkout and the originals live in the `hinode/eis/response` and `hinode/eis/idl/atest/hwarren` trees there.

### What is in the package

| File | Where it comes from in SSW | Used by |
|------|----------------------------|---------|
| `fit_eis_ea_YYYY-MM-DD.sav` | `hinode/eis/idl/atest/hwarren/calibration/new/fit_eis_ea_*.sav` (latest) | `del_zanna_2025` time-interpolated effective area |
| `EIS_EffArea_A.004` | `hinode/eis/response/EIS_EffArea_A.004` (exact) | `del_zanna_2013` (long-wave pinned ground area) |
| `EIS_EffArea_B.004` | `hinode/eis/response/EIS_EffArea_B.004` (exact) | `del_zanna_2013` (short-wave pinned ground area) |
| `EIS_EffArea_A.NNN`, `EIS_EffArea_B.NNN` | `hinode/eis/response/EIS_EffArea_*.*` (latest) | `ground_cal` and Del Zanna long-term decay model |
| `eis_width2velocity.dat` (package root) | `hinode/eis/idl/atest/hwarren/eis_width2velocity.dat` | Non-thermal velocity helper (`eismaps.utils.width2velocity`) |
| `sources.json` | written by the sync helper | Provenance manifest of the copied files |

The Warren 2014 NRL coefficients and the Del Zanna 2013 spline points are not stored as files. Both are inlined directly in `eismaps/calibration.py` (as `WARREN_2014_*` and `GDZ_2013_*` constants), exactly the way the SSW IDL routines `eis_ea_nrl.pro` and `eis_ltds.pro` embed them in source. Their published values are stable - the Warren 2014 NRL set is at v1.3 (Feb 2016) and the GDZ 2013 set has not been revised.

### Refreshing the assets from SolarSoft

If you keep SSW locally (`$SSW`) and want to update the bundled calibration files, run the sync entry point:

```bash
eismaps-sync-calibration --ssw-root "$SSW"
```

Or call the helper from Python:

```python
from eismaps.calibration import sync_solarsoft_calibration_data

manifest = sync_solarsoft_calibration_data(ssw_root='/path/to/ssw')
print(manifest['copied'])
```

This will:

1. Copy the latest `fit_eis_ea_*.sav` into `calibration_data/`.
2. Copy the latest `EIS_EffArea_A.*` and `EIS_EffArea_B.*` text tables into `calibration_data/`.
3. Pin and copy `EIS_EffArea_A.004` and `EIS_EffArea_B.004` exactly (these are the ground areas the Del Zanna 2013 model is defined against).
4. Copy `eis_width2velocity.dat` into the package root next to `__init__.py`.
5. Write a manifest at `calibration_data/sources.json` recording source paths, destination paths, the SSW root used, and the UTC sync timestamp.

The Warren 2014 `.sav` cache is intentionally left alone by the sync (see note above).

## Full-Disk Scan Dataset

The table below lists the time coverage of each full-disk scan in the dataset used in Full-Disk Spectroscopy of the Solar Corona Across a Solar Cycle with Hinode/EIS (McKevitt et al., 2026). Start and end times are taken from the filename timestamps of the first and last EIS raster file in each scan directory, and are in UTC.

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