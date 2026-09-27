from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from .probe import atomic_json
from .state import ROOT

GIB = 1024**3
RESERVE_BYTES = 40 * GIB


@dataclass(frozen=True)
class PublicFile:
    source: str
    dataset: str
    schema: str
    filename: str
    url: str
    expected_bytes: int | None
    expected_sha256: str | None
    license_class: str
    official_page: str


CME_MBO_SAMPLE = PublicFile(
    source="databento",
    dataset="GLBX.MDP3",
    schema="mbo",
    filename="glbx-mdp3-futures-mbo.csv",
    url="https://hist.databento.com/v0/dataset/sample/download/glbx.mdp3/mbo?sample_type=futures",
    expected_bytes=None,
    expected_sha256=None,
    license_class="DATABENTO_PUBLIC_SAMPLE_TERMS",
    official_page="https://databento.com/tick-data",
)

CME_MBP10_SAMPLE = PublicFile(
    source="databento",
    dataset="GLBX.MDP3",
    schema="mbp-10",
    filename="glbx-mdp3-futures-mbp10.csv",
    url="https://hist.databento.com/v0/dataset/sample/download/glbx.mdp3/mbp-10?sample_type=futures",
    expected_bytes=None,
    expected_sha256=None,
    license_class="DATABENTO_PUBLIC_SAMPLE_TERMS",
    official_page="https://databento.com/tick-data",
)


def data_root() -> Path:
    return Path(os.environ.get("MARKETFORGE_DATA_ROOT", ROOT / "data")).resolve()


def assert_budget(expected_additional: int) -> dict[str, int]:
    usage = shutil.disk_usage(data_root().anchor)
    projected = usage.free - expected_additional
    if projected < RESERVE_BYTES:
        raise RuntimeError(
            f"storage reserve violation: free={usage.free}, additional={expected_additional}, "
            f"projected={projected}, reserve={RESERVE_BYTES}"
        )
    return {"free_before": usage.free, "projected_free": projected, "reserve": RESERVE_BYTES}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def acquire(spec: PublicFile) -> Path:
    root = data_root()
    target = root / "raw" / spec.source / spec.filename
    part = target.with_suffix(target.suffix + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        digest = sha256_file(target)
        if spec.expected_sha256 and digest != spec.expected_sha256:
            raise RuntimeError("existing file checksum mismatch")
        return target
    estimate = spec.expected_bytes or 600_000_000
    budget = assert_budget(estimate)
    offset = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(spec.url, headers={"User-Agent": "MarketForge-QTR/0.1"})
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    with urllib.request.urlopen(request, timeout=120) as response:
        if offset and response.status != 206:
            offset = 0
        mode = "ab" if offset else "wb"
        with part.open(mode) as output:
            shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
    os.replace(part, target)
    actual_size = target.stat().st_size
    digest = sha256_file(target)
    if spec.expected_bytes and actual_size != spec.expected_bytes:
        raise RuntimeError(f"size mismatch: {actual_size} != {spec.expected_bytes}")
    if spec.expected_sha256 and digest != spec.expected_sha256:
        raise RuntimeError("download checksum mismatch")
    manifest = {
        **asdict(spec),
        **budget,
        "resolved_path": str(target),
        "actual_bytes": actual_size,
        "sha256": digest,
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "parser_version": "pending",
        "row_count": None,
        "timestamp_bounds": None,
        "quality_status": "DOWNLOADED_UNAUDITED",
    }
    manifest_path = root / "manifests" / f"{spec.source}_{spec.dataset}_{spec.schema}.json"
    atomic_json(manifest_path, manifest)
    return target


def acquisition_summary(path: Path) -> str:
    return json.dumps(
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}, indent=2
    )
