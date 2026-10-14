$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$PY = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $PY)) { throw "Run RUN_COMPLETE.ps1 first." }
if (-not (Test-Path ".\publication_audits")) { throw "publication_audits not found. Run RUN_PUBLICATION_AUDITS.ps1 first." }

Remove-Item publication_figures -Recurse -Force -ErrorAction SilentlyContinue
& $PY .\analysis\make_publication_figures.py --project-root . --audit-dir publication_audits --output publication_figures
if ($LASTEXITCODE -ne 0) { throw "PUBLICATION FIGURE GENERATION FAILED" }

if (Test-Path ".\FedSCOPE_publication_figures.zip") { Remove-Item ".\FedSCOPE_publication_figures.zip" -Force }
Compress-Archive -Path ".\publication_figures\*" -DestinationPath ".\FedSCOPE_publication_figures.zip" -Force
Write-Host "Created FedSCOPE_publication_figures.zip"
