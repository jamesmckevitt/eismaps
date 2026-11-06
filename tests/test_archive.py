"""Tests of raster listing and download."""

import io
import urllib.error

import h5py
import pytest

from eismaps import archive


def listing_of(names):
    def urlopen(url, timeout):
        return io.BytesIO(''.join(f'<a href="{n}">{n}</a>' for n in names).encode())
    return urlopen


def http_error(code):
    def urlopen(url, timeout):
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


def test_truncated_file_is_refused_and_the_next_server_used(tmp_path, monkeypatch):
    good = tmp_path / 'good.h5'
    with h5py.File(good, 'w') as f:
        f['x'] = [1, 2, 3]
    whole = good.read_bytes()

    def urlopen(url, timeout):
        return io.BytesIO(whole[:200] if 'mssl' in url else whole)

    monkeypatch.setattr('urllib.request.urlopen', urlopen)
    path = archive.fetch_raster('20251002_200728', str(tmp_path / 'raster'))
    with h5py.File(path, 'r') as f:
        assert f['x'].shape == (3,)


def test_no_complete_copy_raises(tmp_path, monkeypatch):
    monkeypatch.setattr('urllib.request.urlopen', lambda url, timeout: io.BytesIO(b'not hdf5'))
    with pytest.raises(archive.RasterUnavailable):
        archive.fetch_raster('20251002_200728', str(tmp_path))
