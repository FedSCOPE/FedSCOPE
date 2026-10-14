$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Test-Path ".venv\Scripts\python.exe")) { py -3.11 -m venv .venv }
$PY = ".\.venv\Scripts\python.exe"
& $PY -m pip install --upgrade pip setuptools wheel
& $PY -m pip install -r requirements-lock.txt
& $PY -m pip install -e . --no-deps
& $PY -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "TESTS FAILED" }
& $PY -m fedscope.data.download --data-dir data/raw
if ($LASTEXITCODE -ne 0) { throw "DATA VERIFY/DOWNLOAD FAILED" }
Remove-Item results_final -Recurse -Force -ErrorAction SilentlyContinue
& $PY -m fedscope.pipeline --config configs/paper.yaml
if ($LASTEXITCODE -ne 0) { throw "PRIMARY RUN FAILED" }
& $PY -m fedscope.audit_run results_final
