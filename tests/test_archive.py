"""Tests of raster listing and download."""

import io
import urllib.error

import gzip

import h5py
import numpy as np
import pytest
from astropy.io import fits

from eismaps import archive


def listing_of(names):
    def urlopen(url, timeout, context=None):
        return io.BytesIO(''.join(f'<a href="{n}">{n}</a>' for n in names).encode())
    return urlopen


def http_error(code):
    def urlopen(url, timeout, context=None):
        raise urllib.error.HTTPError(url, code, 'error', None, None)
    return urlopen


def test_list_rasters(monkeypatch):
    names = ['eis_20130116_093720.data.h5', 'eis_20130116_093720.head.h5', 'eis_20130116_104840.data.h5']
    monkeypatch.setattr('urllib.request.urlopen', listing_of(names))
    assert archive.list_rasters('2013-01-16') == ['20130116_093720', '20130116_104840']


def test_day_with_no_directory_is_empty(monkeypatch):
    monkeypatch.setattr('urllib.request.urlopen', http_error(404))
    assert archive.list_rasters('2010-06-15') == []


def test_outage_is_not_an_empty_day(monkeypatch):
    monkeypatch.setattr('urllib.request.urlopen', http_error(503))
    with pytest.raises(ConnectionError):
        archive.list_rasters('2013-01-16')


def test_one_server_down_is_an_outage(monkeypatch):
    # A 404 from one server says nothing about a day the other could not list.
    def urlopen(url, timeout, context=None):
        if 'mssl' in url:
            raise urllib.error.HTTPError(url, 404, 'error', None, None)
        raise urllib.error.URLError('timed out')

    monkeypatch.setattr('urllib.request.urlopen', urlopen)
    with pytest.raises(ConnectionError):
        archive.list_rasters('2013-01-16')


def test_truncated_file_is_refused_and_the_next_server_used(tmp_path, monkeypatch):
    good = tmp_path / 'good.h5'
    with h5py.File(good, 'w') as f:
        f['x'] = [1, 2, 3]
    whole = good.read_bytes()

    def urlopen(url, timeout, context=None):
        return io.BytesIO(whole[:200] if 'mssl' in url else whole)

    monkeypatch.setattr('urllib.request.urlopen', urlopen)
    path = archive.fetch_raster('20251002_200728', str(tmp_path / 'raster'))
    with h5py.File(path, 'r') as f:
        assert f['x'].shape == (3,)


def test_no_complete_copy_raises(tmp_path, monkeypatch):
    monkeypatch.setattr('urllib.request.urlopen', lambda url, timeout, context=None: io.BytesIO(b'not hdf5'))
    with pytest.raises(archive.RasterUnavailable):
        archive.fetch_raster('20251002_200728', str(tmp_path))


def test_requests_use_the_certifi_roots(tmp_path, monkeypatch):
    contexts = []

    def urlopen(url, timeout, context=None):
        contexts.append(context)
        return io.BytesIO(b'')

    monkeypatch.setattr('urllib.request.urlopen', urlopen)
    archive.list_rasters('2013-01-16')
    assert contexts and all(c is archive.CONTEXT for c in contexts)


def test_rasters_with_only_a_level0_file_are_listed(monkeypatch):
    def urlopen(url, timeout, context=None):
        if 'level0' in url:
            return io.BytesIO(b'<a href="eis_l0_20130116_093720.fits.gz">x</a><a href="eis_l0_20130116_120000.fits.gz">y</a>')
        return io.BytesIO(b'<a href="eis_20130116_093720.data.h5">z</a>')

    monkeypatch.setattr('urllib.request.urlopen', urlopen)
    assert archive.list_rasters('2013-01-16') == ['20130116_093720', '20130116_120000']


def level0_bytes(tmp_path):
    primary = fits.PrimaryHDU()
    primary.header['SLIT_ID'] = '40"'
    primary.header['NWIN'] = 1
    table = fits.BinTableHDU.from_columns([fits.Column(name='Fe XII 195.120', format='4I', array=np.ones((3, 4), dtype=np.int16))])
    path = tmp_path / 'eis_l0_20130116_093720.fits.gz'
    fits.HDUList([primary, table]).writeto(path)
    return path.read_bytes()


def test_level0_header_is_read_from_the_start_of_the_file(tmp_path, monkeypatch):
    whole = level0_bytes(tmp_path)
    monkeypatch.setattr('urllib.request.urlopen', lambda url, timeout, context=None: io.BytesIO(whole))
    assert archive.level0_header('20130116_093720')['SLIT_ID'] == '40"'


def test_truncated_level0_file_is_refused(tmp_path, monkeypatch):
    whole = level0_bytes(tmp_path)
    monkeypatch.setattr('urllib.request.urlopen', lambda url, timeout, context=None: io.BytesIO(whole[:len(whole) // 2]))
    with pytest.raises(archive.RasterUnavailable):
        archive.fetch_level0('20130116_093720', str(tmp_path / 'download'))
    monkeypatch.setattr('urllib.request.urlopen', lambda url, timeout, context=None: io.BytesIO(whole))
    path = archive.fetch_level0('20130116_093720', str(tmp_path / 'download'))
    assert path.endswith('eis_l0_20130116_093720.fits.gz')
