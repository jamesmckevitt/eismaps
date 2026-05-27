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

Behavior:

- a single requested measurement returns a SunPy map or `MapSequence`
- multiple requested measurements return a dictionary keyed by measurement name
- supported measurements are `int`, `vel`, `wid`, `ntv`, and `chi2`

Useful options:

- `clip=True` applies outlier clipping before saving to catch bad data
- `output_dir='...'` writes FITS map products
- `vel_los_correct=True` applies a line-of-sight correction to velocity maps (assuming the velocity is radial)

### Calibration

`apply_calibration()` converts an intensity map (or a list / dict of maps) from raw EIS units into calibrated radiance, using one of several published methods. Each method gives the effective area `Aeff(wavelength, date)` and the helper divides the map by the appropriate `Aeff` ratio:

```python
from eismaps import apply_calibration

calibrated = apply_calibration(
    map_products['int'],
    method='del_zanna_2025',
    ncpu='max',
)
```

Supported user-facing calibration methods:

| `method=` | What it does | Backing files |
|-----------|--------------|---------------|
| `ground_cal` | Pre-flight ground calibration only; no time decay applied. Useful as a sanity baseline. | `EIS_EffArea_A.*`, `EIS_EffArea_B.*` |
| `warren_2014` | Warren et al. (2014) spline-knot model. Time-dependent: at each knot wavelength `Aeff = A0 * exp(-t / TAU)` so the calibration ratio evolves with observation date. | inlined constants (`WARREN_2014_*` in `calibration.py`) |
| `del_zanna_2013` | Del Zanna (2013) recalibration. Pins ground areas to `EIS_EffArea_A.004` / `EIS_EffArea_B.004` and applies the Del Zanna degradation correction. | `EIS_EffArea_A.004`, `EIS_EffArea_B.004` |
| `del_zanna_2025` | Latest time-interpolated calibration based on hot/cold line-pair tracking. Best for science use. | `fit_eis_ea_YYYY-MM-DD.sav` |

`del_zanna_2023` is kept as an alias for `del_zanna_2025` for backwards compatibility.

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

The public functions are intended to chain naturally:

```python
from eismaps import apply_calibration, fit, make_helioprojective_map, make_maps

fit_results = fit(data_files, lines_to_fit=['fe_12_195_119'])
map_products = make_maps(fit_results, measurement=['int', 'vel', 'ntv'])
calibrated_intensity = apply_calibration(map_products['int'], method='del_zanna_2025')
fd_map, overlap_map = make_helioprojective_map(calibrated_intensity, overlap='mean')
```

For data discovery and download, see the notebook tutorial, which shows the current SunPy Fido plus EISPAC-client workflow.

## Calibration Assets

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

The Warren 2014 NRL coefficients and the Del Zanna 2013 spline points are not stored as files. Both are inlined directly in `eismaps/calibration.py` (as `WARREN_2014_*` and `GDZ_2013_*` constants), exactly the way the SSW IDL routines `eis_ea_nrl.pro` and `eis_ltds.pro` embed them in source. Their published values are stable - the Warren 2014 NRL set is at v1.3 (Feb 2016) and the GDZ 2013 set has not been revised. If a new coefficient set is ever released, update those constants in `calibration.py`.

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

### Numerical sanity check

For Fe XII 195.119 A, the bundled ground/preflight effective area gives `eis_ea(195.119) = 0.302 cm^2`, matching the value quoted in Del Zanna (2013). Time-dependent areas at 2013-01-16 are `~0.298 cm^2` (Warren 2014) and `~0.329 cm^2` (Del Zanna 2025), consistent with the published degradation curves.