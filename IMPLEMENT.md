# MarketForge-QTR v2 operational runbook

PowerShell 7 and Python 3.12 are canonical:

```powershell
.\.venv\Scripts\python.exe -m marketforge.cli <stage>
.\scripts\mf.ps1 <stage>
```

The completed v2 result is immutable and idempotent:

```powershell
.\scripts\mf.ps1 v2-final-evaluation  # returns existing artifact; does not recompute
.\scripts\mf.ps1 v2-report            # regenerates reports from completed evidence
.\scripts\mf.ps1 status
```

Quality audit:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check src tests scripts
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m pytest -q
```

Never run the v1 `full`, `freeze-final`, or `final-evaluation` commands to reinterpret v1. Never overwrite either final evaluation. V2 raw/canonical/replay files, `OPENED_ONCE.json`, the original freeze, amendments 001–003, and structural-failure evidence are permanent.

Resource rules remain: one GPU-heavy lease at a time; deep models assert CUDA; Windows multiprocessing is spawn-safe; no Bash/WSL/Make/Snakemake requirement; maintain 40 GiB free; use only official lawful data paths.
