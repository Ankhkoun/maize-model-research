import pytest

from src.data.manifest import normalize_split


def test_source_val_alias_is_normalized_to_validation() -> None:
    assert normalize_split("val") == "validation"
    assert normalize_split(" validation ") == "validation"


def test_unknown_split_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid split"):
        normalize_split("development")
