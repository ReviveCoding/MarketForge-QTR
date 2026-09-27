param(
    [Parameter(Position = 0, Mandatory = $true)]
    [ValidateSet('bootstrap','probe','data','audit','canonicalize','study','features','splits','smoke','baselines','pretrain','posttrain','calibrate','simulator-validation','backtest','experiments','analyze','report','full-dev','freeze-final','final-evaluation','full','status','resume','v2-audit','v2-freeze-final','v2-final-evaluation','v2-report','v2-completion-audit','v2-posthoc')]
    [string]$Stage
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Missing .venv. Run: py -3.12 -m venv .venv' }
& $python -m marketforge.cli $Stage
exit $LASTEXITCODE
