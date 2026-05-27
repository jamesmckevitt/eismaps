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

Use `apply_calibration()` to calibrate a map, a list of maps, or a dictionary of map collections:

```python
from eismaps import apply_calibration

calibrated = apply_calibration(
    map_products['int'],
    method='del_zanna_2025',
    ncpu='max',
)
```

Supported user-facing calibration methods:

- `ground_cal`
- `del_zanna_2013`
- `warren_2014`
- `del_zanna_2025`

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

The calibration helpers expect reference assets in `eismaps/calibration_data/`. To sync them from a local SolarSoft checkout before a release or asset update, run:

```bash
eismaps-sync-calibration --ssw-root "$SSW"
```

You can also call the same helper from Python:

```python
from eismaps.calibration import sync_solarsoft_calibration_data

manifest = sync_solarsoft_calibration_data(ssw_root='/path/to/ssw')
print(manifest['copied'])
```