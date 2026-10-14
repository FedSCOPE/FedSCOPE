$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$PY = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $PY)) { throw "Run RUN_PRIMARY.ps1 or RUN_COMPLETE.ps1 first." }
if (-not (Test-Path ".\results_final")) { throw "results_final not found." }
if (-not (Test-Path ".\paper_suite_final")) { throw "paper_suite_final not found. Run the paper suite first for full publication audits." }

Remove-Item publication_audits -Recurse -Force -ErrorAction SilentlyContinue
& $PY -m fedscope.publication_audit --heavy-root . --output publication_audits
if ($LASTEXITCODE -ne 0) { throw "PUBLICATION AUDIT FAILED" }

if (Test-Path ".\FedSCOPE_publication_audits.zip") { Remove-Item ".\FedSCOPE_publication_audits.zip" -Force }
Compress-Archive -Path ".\publication_audits\*" -DestinationPath ".\FedSCOPE_publication_audits.zip" -Force
Write-Host "Created FedSCOPE_publication_audits.zip"
