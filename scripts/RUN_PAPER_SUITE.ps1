$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$PY = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $PY)) { throw "Run RUN_PRIMARY.ps1 once first, or use RUN_COMPLETE.ps1." }
Remove-Item paper_suite_final -Recurse -Force -ErrorAction SilentlyContinue
& $PY -m fedscope.suite --config configs/paper.yaml --output-root paper_suite_final --skip-primary --include-focal-rotation
if ($LASTEXITCODE -ne 0) { throw "PAPER SUITE FAILED" }
Compress-Archive -Path paper_suite_final -DestinationPath FedSCOPE_paper_suite_results.zip -Force
Write-Host "Created FedSCOPE_paper_suite_results.zip"
