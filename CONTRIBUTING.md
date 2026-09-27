# Contributing

Thank you for helping improve MarketForge-QTR.

## Development setup

Use native Windows and PowerShell 7 with Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,research,ml]"
```

Before opening a pull request, run:

```powershell
.\.venv\Scripts\python.exe -m ruff format --check src tests scripts
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m pytest -q
```

## Research contributions

- Make every experiment deterministic or record why it cannot be.
- Keep split, scaler, calibration, and lockbox boundaries explicit.
- Support scientific claims with generated artifacts and exact commands.
- Preserve negative, null, and resource-limited findings.
- Label simulated execution as `SIMULATED_L3`; never call it observed execution.
- Do not tune on test or final-lockbox results.
- New methods suggested after seeing final data must be identified as future-protocol candidates.

## Data and credentials

Never include raw proprietary, licensed, or provider-restricted market data in a pull request.
Do not commit API keys, tokens, `.env` files, credentials, private keys, model-provider secrets,
or private model weights. Use synthetic/minimal fixtures for tests and document independent
data-acquisition steps against official provider terms.

## Pull requests

Keep changes focused. Explain the scientific or engineering rationale, tests performed,
affected artifacts, and whether any result or protocol claim changes. Changes to authoritative
research claims require an explicitly versioned new protocol; they must not rewrite v2 history.
