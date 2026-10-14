$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Test-Path ".venv\Scripts\python.exe")) { py -3.11 -m venv .venv }
$PY = ".\.venv\Scripts\python.exe"

Write-Host "[FedSCOPE] Installing locked environment..."
& $PY -m pip install --upgrade pip setuptools wheel
& $PY -m pip install -r requirements-lock.txt
& $PY -m pip install -e . --no-deps

Write-Host "[FedSCOPE] Running tests..."
& $PY -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "TESTS FAILED" }

Write-Host "[FedSCOPE] Downloading/verifying locked OPSD data..."
& $PY -m fedscope.data.download --data-dir data/raw
if ($LASTEXITCODE -ne 0) { throw "DATA VERIFY/DOWNLOAD FAILED" }

Remove-Item results_final -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item paper_suite_final -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item publication_audits -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item publication_figures -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "[FedSCOPE] Running primary expanding experiment..."
& $PY -m fedscope.pipeline --config configs/paper.yaml
if ($LASTEXITCODE -ne 0) { throw "PRIMARY RUN FAILED" }

Write-Host "[FedSCOPE] Running hard scientific audit..."
& $PY -m fedscope.audit_run results_final
if ($LASTEXITCODE -ne 0) { throw "PRIMARY AUDIT FAILED" }

Write-Host "[FedSCOPE] Running prespecified paper suite..."
& $PY -m fedscope.suite --config configs/paper.yaml --output-root paper_suite_final --skip-primary --include-focal-rotation
if ($LASTEXITCODE -ne 0) { throw "PAPER SUITE FAILED" }

Write-Host "[FedSCOPE] Running lightweight publication audits (HSIC, MI, headroom, federation diagnostics)..."
& $PY -m fedscope.publication_audit --heavy-root . --output publication_audits
if ($LASTEXITCODE -ne 0) { throw "PUBLICATION AUDIT FAILED" }

Write-Host "[FedSCOPE] Generating publication figures..."
& $PY .\analysis\make_publication_figures.py --project-root . --audit-dir publication_audits --output publication_figures
if ($LASTEXITCODE -ne 0) { throw "PUBLICATION FIGURE GENERATION FAILED" }

Write-Host "[FedSCOPE] Verifying key reproduction metrics..."
& $PY .\analysis\verify_results.py --project-root .
if ($LASTEXITCODE -ne 0) { throw "REPRODUCTION CHECK FAILED" }

if (Test-Path ".\FedSCOPE_final_results.zip") { Remove-Item ".\FedSCOPE_final_results.zip" -Force }
Compress-Archive -Path results_final,paper_suite_final,publication_audits,publication_figures -DestinationPath FedSCOPE_final_results.zip -Force

Write-Host ""
Write-Host "FedSCOPE publication release complete."
Write-Host "Primary:            results_final"
Write-Host "Paper suite:         paper_suite_final"
Write-Host "Publication audits:  publication_audits"
Write-Host "Publication figures: publication_figures"
Write-Host "Results bundle:      FedSCOPE_final_results.zip"
