"""Finding and downloading EIS level 1 HDF5 files.

Files are taken from the MSSL mirror if a complete copy is there, and from NRL
otherwise. Some files on the mirror are truncated, so each download is opened
before it is accepted.
"""

import os
import re
import urllib.error
import urllib.request

import h5py

SERVERS = ('https://vsolar.mssl.ucl.ac.uk/eispac/hdf5', 'https://eis.nrl.navy.mil/level1/hdf5')
TIMEOUT = 120  # seconds


class RasterUnavailable(Exception):
    """No complete HDF5 files for a raster on any server."""


def _day_path(timestamp):
    return f'{timestamp[:4]}/{timestamp[4:6]}/{timestamp[6:8]}'


def list_rasters(date):
    """Return the timestamps of the rasters of a day on any server.

    Parameters
    ----------
    date : str
        Day as 'YYYYMMDD' or 'YYYY-MM-DD'.

    Returns
    -------
    list[str]
        Sorted timestamps such as '20130116_093720'.

    Raises
    ------
    ConnectionError
        If no server answers, so that an outage is not taken for an empty day.
    """
    day = date.replace('-', '')
    found = set()
    answered = False
    for server in SERVERS:
        try:
            with urllib.request.urlopen(f'{server}/{_day_path(day)}/', timeout=TIMEOUT) as response:
                listing = response.read().decode('utf-8', 'replace')
        except urllib.error.HTTPError as error:
            # A day with no observations has no directory.
            answered = answered or error.code == 404
            continue
        except (urllib.error.URLError, OSError):
            continue
        answered = True
        found.update(re.findall(r'eis_(\d{8}_\d{6})\.data\.h5', listing))
    if not answered:
        raise ConnectionError(f'No server answered for {date}')
    return sorted(found)


def _download(url, path):
    """Download a file, replacing `path` only when complete."""
    partial = path + '.part'
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response, open(partial, 'wb') as output:
        while True:
            block = response.read(1 << 22)
            if not block:
                break
            output.write(block)
    os.replace(partial, path)


def _readable(path):
    """Return True if an HDF5 file opens, which fails if it is truncated."""
    try:
        with h5py.File(path, 'r'):
            return True
    except OSError:
        return False


def fetch_raster(timestamp, directory):
    """Download the data and header files of a raster.

    Parameters
    ----------
    timestamp : str
        Raster start time as 'YYYYMMDD_HHMMSS'.
    directory : str
        Directory for the files.

    Returns
    -------
    str
        Path of the data file.

    Raises
    ------
    RasterUnavailable
        If no server has complete files.
    """
    os.makedirs(directory, exist_ok=True)
    paths = {kind: os.path.join(directory, f'eis_{timestamp}.{kind}.h5') for kind in ('data', 'head')}
    if all(os.path.exists(p) and _readable(p) for p in paths.values()):
        return paths['data']
    for server in SERVERS:
        try:
            for kind, path in paths.items():
                _download(f'{server}/{_day_path(timestamp)}/eis_{timestamp}.{kind}.h5', path)
        except (urllib.error.URLError, OSError):
            continue
        if all(_readable(p) for p in paths.values()):
            return paths['data']
    for path in paths.values():
        if os.path.exists(path):
            os.remove(path)
    raise RasterUnavailable(f'No complete HDF5 files for {timestamp}')
