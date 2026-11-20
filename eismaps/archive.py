"""Finding and downloading EIS level 1 HDF5 files and level 0 FITS files.

Level 1 files are taken from the MSSL mirror if a complete copy is there, and
from NRL otherwise. Some files on the mirror are truncated, so each download is
opened before it is accepted. Level 0 files come from the archive at RAL.
"""

import http.client
import os
import re
import ssl
import urllib.error
import urllib.request
import zlib

import certifi
import h5py
from astropy.io import fits

SERVERS = ('https://vsolar.mssl.ucl.ac.uk/eispac/hdf5', 'https://eis.nrl.navy.mil/level1/hdf5')
LEVEL0_SERVER = 'http://solar.ads.rl.ac.uk/MSSL-data/eis/level0'
TIMEOUT = 120  # seconds

# Largest part of a level 0 file read to find its primary header, in bytes.
MAX_HEADER_BYTES = 1 << 20

# certifi's trusted roots are added to the system's, as some systems lack the
# root that signs NRL's certificate.
CONTEXT = ssl.create_default_context()
CONTEXT.load_verify_locations(cafile=certifi.where())


class RasterUnavailable(Exception):
    """No complete file for a raster on any server."""


def _day_path(timestamp):
    return f'{timestamp[:4]}/{timestamp[4:6]}/{timestamp[6:8]}'


def list_rasters(date):
    """Return the timestamps of the rasters of a day on any server.

    Rasters with only a level 0 file are included.

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
        If a server does not answer, as the day could then be incomplete.
    """
    day = date.replace('-', '')
    found = set()
    patterns = [(server, r'eis_(\d{8}_\d{6})\.data\.h5') for server in SERVERS]
    patterns.append((LEVEL0_SERVER, r'eis_l0_(\d{8}_\d{6})\.fits(?:\.gz)?"'))
    for server, pattern in patterns:
        try:
            with urllib.request.urlopen(f'{server}/{_day_path(day)}/', timeout=TIMEOUT, context=CONTEXT) as response:
                listing = response.read().decode('utf-8', 'replace')
        except urllib.error.HTTPError as error:
            # A day with no observations has no directory.
            if error.code == 404:
                continue
            raise ConnectionError(f'{server} gave status {error.code} for {date}') from error
        except (urllib.error.URLError, http.client.HTTPException, OSError) as error:
            raise ConnectionError(f'{server} did not answer for {date}') from error
        found.update(re.findall(pattern, listing))
    return sorted(found)


def _download(url, path):
    """Download a file, replacing `path` only when complete."""
    partial = path + '.part'
    with urllib.request.urlopen(url, timeout=TIMEOUT, context=CONTEXT) as response, open(partial, 'wb') as output:
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
        If every server answers but none has complete files.
    ConnectionError
        If no server has complete files and one did not answer, so that the
        raster can be tried again later.
    """
    os.makedirs(directory, exist_ok=True)
    paths = {kind: os.path.join(directory, f'eis_{timestamp}.{kind}.h5') for kind in ('data', 'head')}
    if all(os.path.exists(p) and _readable(p) for p in paths.values()):
        return paths['data']
    failures = []
    for server in SERVERS:
        try:
            for kind, path in paths.items():
                _download(f'{server}/{_day_path(timestamp)}/eis_{timestamp}.{kind}.h5', path)
        except urllib.error.HTTPError as error:
            if error.code != 404:
                failures.append(f'{server} gave status {error.code}')
            continue
        except (urllib.error.URLError, http.client.HTTPException, OSError):
            failures.append(f'{server} did not answer')
            continue
        if all(_readable(p) for p in paths.values()):
            return paths['data']
    for path in paths.values():
        if os.path.exists(path):
            os.remove(path)
    if failures:
        raise ConnectionError(f'No complete HDF5 files for {timestamp}, as ' + ' and '.join(failures))
    raise RasterUnavailable(f'No complete HDF5 files for {timestamp}')


def _level0_url(timestamp):
    return f'{LEVEL0_SERVER}/{_day_path(timestamp)}/eis_l0_{timestamp}.fits.gz'


def level0_header(timestamp):
    """Return the primary header of a raster's level 0 file without downloading all of it.

    Raises
    ------
    RasterUnavailable
        If the file cannot be read.
    """
    end = b'END'.ljust(80)
    inflate = zlib.decompressobj(wbits=31)
    text = b''
    read = 0
    try:
        with urllib.request.urlopen(_level0_url(timestamp), timeout=TIMEOUT, context=CONTEXT) as response:
            while read < MAX_HEADER_BYTES:
                block = response.read(1 << 16)
                if not block:
                    break
                read += len(block)
                text += inflate.decompress(block)
                cards = [text[i:i + 80] for i in range(0, len(text) - 79, 80)]
                if end in cards:
                    return fits.Header.fromstring(text[:80 * (cards.index(end) + 1)].decode('ascii', 'replace'))
    except (urllib.error.URLError, http.client.HTTPException, OSError, zlib.error, ValueError) as error:
        raise RasterUnavailable(f'No level 0 file for {timestamp}') from error
    raise RasterUnavailable(f'No level 0 header found for {timestamp}')


def _readable_level0(path):
    """Return True if a level 0 file opens and its data can be read."""
    try:
        with fits.open(path) as hdul:
            hdul[1].data.field(0)
        return True
    except (OSError, ValueError, IndexError, EOFError, TypeError):
        return False


def fetch_level0(timestamp, directory):
    """Download the level 0 file of a raster.

    Parameters
    ----------
    timestamp : str
        Raster start time as 'YYYYMMDD_HHMMSS'.
    directory : str
        Directory for the file.

    Returns
    -------
    str
        Path of the level 0 file.

    Raises
    ------
    RasterUnavailable
        If there is no complete file.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f'eis_l0_{timestamp}.fits.gz')
    try:
        _download(_level0_url(timestamp), path)
    except (urllib.error.URLError, http.client.HTTPException, OSError) as error:
        raise RasterUnavailable(f'No level 0 file for {timestamp}') from error
    if not _readable_level0(path):
        os.remove(path)
        raise RasterUnavailable(f'Incomplete level 0 file for {timestamp}')
    return path
