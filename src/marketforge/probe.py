from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from .state import ROOT


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def environment_probe() -> Path:
    usage = shutil.disk_usage(ROOT)
    nvidia = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "repo": str(ROOT),
        "data_root": str(Path(os.environ.get("MARKETFORGE_DATA_ROOT", ROOT / "data")).resolve()),
        "disk_total_bytes": usage.total,
        "disk_free_bytes": usage.free,
        "reserve_bytes": 40 * 1024**3,
        "nvidia_smi_exit": nvidia.returncode,
        "nvidia_gpus": [line for line in nvidia.stdout.splitlines() if line.strip()],
    }
    target = ROOT / "artifacts" / "system" / "environment_runtime.json"
    atomic_json(target, payload)
    return target


def cuda_probe() -> Path:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA is mandatory but unavailable; CPU fallback is forbidden")
    device = torch.device("cuda:0")
    torch.manual_seed(1701)
    x = torch.randn(256, 128, device=device, requires_grad=True)
    layer = torch.nn.Linear(128, 64, device=device)
    loss = layer(x).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    props = torch.cuda.get_device_properties(device)
    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": True,
        "device_count": torch.cuda.device_count(),
        "device_name": props.name,
        "total_vram_bytes": props.total_memory,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "forward_loss": float(loss.detach().cpu()),
        "backward_gradient_finite": bool(torch.isfinite(x.grad).all().item()),
        "tensor_device": str(x.device),
        "model_device": str(next(layer.parameters()).device),
        "max_allocated_bytes": torch.cuda.max_memory_allocated(device),
    }
    if not payload["backward_gradient_finite"]:
        raise RuntimeError("CUDA backward pass produced non-finite gradients")
    target = ROOT / "artifacts" / "system" / "cuda_runtime.json"
    atomic_json(target, payload)
    return target
