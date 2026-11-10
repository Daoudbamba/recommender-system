from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from recsys.config import PROJECT_ROOT, Config, load_config


def test_default_config_is_valid_and_paths_are_absolute():
    config = load_config()
    assert config.paths.raw_dir.is_absolute()
    assert config.paths.raw_dir == PROJECT_ROOT / "data" / "raw"
    assert config.raw_dataset_dir.name == config.dataset.name


def test_missing_config_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "absent.yaml")


def test_invalid_md5_is_rejected(config: Config):
    payload = config.model_dump()
    payload["dataset"]["md5"] = "pas-un-md5"
    with pytest.raises(ValidationError):
        Config.model_validate(payload)


def test_inverted_rating_bounds_are_rejected(config: Config):
    payload = config.model_dump()
    payload["preprocessing"].update(rating_min=5.0, rating_max=0.5)
    with pytest.raises(ValidationError):
        Config.model_validate(payload)
