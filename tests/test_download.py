from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from recsys.config import Config
from recsys.data import download


def _make_archive(path: Path, entries: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as bundle:
        for name, content in entries.items():
            bundle.writestr(name, content)
    return path


def test_verify_md5_accepts_matching_and_rejects_other(tmp_path: Path):
    target = tmp_path / "f.bin"
    target.write_bytes(b"movielens")
    download.verify_md5(target, download.md5_of(target))
    with pytest.raises(download.ChecksumError):
        download.verify_md5(target, "0" * 32)


def test_safe_extract_refuses_path_traversal(tmp_path: Path):
    archive = _make_archive(tmp_path / "evil.zip", {"../outside.txt": "x"})
    with pytest.raises(download.UnsafeArchiveError):
        download.safe_extract(archive, tmp_path / "dest")
    assert not (tmp_path / "outside.txt").exists()


def test_ensure_dataset_downloads_verifies_and_extracts(tmp_path: Path, config: Config):
    source = _make_archive(tmp_path / "source.zip", {"ml-test/ratings.csv": "userId\n1\n"})
    config.dataset.url = source.as_uri()
    config.dataset.md5 = download.md5_of(source)

    folder = download.ensure_dataset(config)

    assert (folder / "ratings.csv").read_text() == "userId\n1\n"


def test_ensure_dataset_is_idempotent(config: Config, raw_dir: Path, monkeypatch):
    def _fail(*_args, **_kwargs):
        raise AssertionError("aucun telechargement attendu")

    monkeypatch.setattr(download, "download_file", _fail)
    assert download.ensure_dataset(config) == raw_dir


def test_ensure_dataset_rejects_corrupted_archive(tmp_path: Path, config: Config):
    source = _make_archive(tmp_path / "source.zip", {"ml-test/ratings.csv": "x"})
    config.dataset.url = source.as_uri()
    with pytest.raises(download.ChecksumError):
        download.ensure_dataset(config)
    assert not config.raw_dataset_dir.exists()
