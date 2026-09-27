from pathlib import Path

import pytest

from marketforge.data import RESERVE_BYTES, assert_budget, sha256_file


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "x"
    path.write_bytes(b"marketforge")
    assert sha256_file(path) == "91d63ebe565f767f5bb1e52b829398eb0107dfd91bb7ba185391b9c54fda907e"


def test_reserve_is_forty_gib() -> None:
    assert RESERVE_BYTES == 40 * 1024**3


def test_impossible_budget_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="storage reserve violation"):
        assert_budget(10**18)
