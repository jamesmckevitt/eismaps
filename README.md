# EISMaps

EISMaps is a Python package for processing data from the EIS (Extreme-ultraviolet Imaging Spectrometer) instrument on the Hinode spacecraft. It assembles Hinode/EIS slit full-disk maps with various options. It also provides useful wrapping functionality to the fantastic [eispac](https://github.com/USNavalResearchLaboratory/eispac) code for batch processing of large EIS datasets.

Contact: James McKevitt (jm2@mssl.ucl.ac.uk).

Licence: CC BY-NC-SA 4.0 (see [LICENSE](LICENSE) for details).

Cite: McKevitt, J. (2025). Coronal non-thermal and Doppler plasma flows driven by photospheric flux in 28 active regions. In prep.

## Installation

```bash
python -m pip install git+https://github.com/jamesmckevitt/eismaps.git
```

## Quick Start: Minimal Full Disk Pipeline

This guide walks you through the complete pipeline for creating full-disk maps from Hinode/EIS data, following the processing steps used in McKevitt (2025).

### Step 1: Spectral Line Fitting

First, fit spectral lines in EIS data using the `fit.batch()` function:

```python
import eismaps.utils.find as find
from eismaps.proc import fit

# Batch fit Fe XII 195.119 Å line (wrapping eispac fitting)
fit.batch(
    data_files,                          # List of .data.h5 files to process
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
from eismaps.proc import maps

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
from eismaps.proc import full_disk
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
    apply_los_correction=False          # Apply line-of-sight viewing angle correction
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
    apply_los_correction=True           # Apply line-of-sight viewing angle correction (if not applied earlier)
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