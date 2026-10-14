$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$PY = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $PY)) { throw "Missing .venv. Run RUN_COMPLETE.ps1 first." }
$required = @("results_final", "paper_suite_final", "publication_audits")
$missing = $required | Where-Object { -not (Test-Path $_) }
if ($missing.Count -gt 0) { throw "Missing output directories: $($missing -join ', ')" }
& $PY .\analysis\verify_results.py --project-root .
if ($LASTEXITCODE -ne 0) { throw "REPRODUCTION CHECK FAILED" }
