"""Telechargement verifie et extraction sure du dataset MovieLens."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

from recsys.config import Config
from recsys.utils.logging import get_logger

logger = get_logger(__name__)

_CHUNK = 1024 * 1024


class ChecksumError(RuntimeError):
    """L'archive telechargee ne correspond pas a la somme attendue."""


class UnsafeArchiveError(RuntimeError):
    """L'archive contient un chemin qui sortirait du dossier de destination."""


def md5_of(path: Path) -> str:
    """Somme MD5 d'un fichier, lue par blocs pour ne pas le charger en memoire."""
    digest = hashlib.md5()  # noqa: S324 - controle d'integrite, pas de securite
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_md5(path: Path, expected: str) -> None:
    actual = md5_of(path)
    if actual != expected.lower():
        raise ChecksumError(f"MD5 inattendu pour {path.name} : {actual} (attendu {expected})")


def download_file(url: str, dest: Path) -> Path:
    """Telecharge `url` vers `dest` via un fichier temporaire (ecriture atomique)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    logger.info("Telechargement de %s", url)
    with urllib.request.urlopen(url) as response, partial.open("wb") as out:  # noqa: S310
        shutil.copyfileobj(response, out, length=_CHUNK)
    partial.replace(dest)
    return dest


def safe_extract(archive: Path, dest_dir: Path) -> list[Path]:
    """Extrait un zip en refusant toute entree qui sortirait de `dest_dir`."""
    dest_root = dest_dir.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            target = (dest_root / member).resolve()
            if not target.is_relative_to(dest_root):
                raise UnsafeArchiveError(f"Entree refusee dans {archive.name} : {member}")
        bundle.extractall(dest_root)
        return [dest_root / name for name in bundle.namelist()]


def ensure_dataset(config: Config, force: bool = False) -> Path:
    """Garantit la presence du dataset extrait et retourne son dossier.

    Idempotent : ne retelecharge pas si les CSV sont deja la, sauf `force`.
    """
    target_dir = config.raw_dataset_dir
    if target_dir.is_dir() and any(target_dir.glob("*.csv")) and not force:
        logger.info("Dataset deja present : %s", target_dir)
        return target_dir

    archive = config.paths.raw_dir / f"{config.dataset.name}.zip"
    if force or not archive.is_file():
        download_file(config.dataset.url, archive)
    verify_md5(archive, config.dataset.md5)
    logger.info("Somme MD5 verifiee, extraction dans %s", config.paths.raw_dir)
    safe_extract(archive, config.paths.raw_dir)
    if not target_dir.is_dir():
        raise FileNotFoundError(f"Dossier attendu absent apres extraction : {target_dir}")
    return target_dir
