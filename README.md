# FedSCOPE

**Causal service-preserving strategic opacity for federated peer-to-peer emergency energy sharing in networked microgrids.**

FedSCOPE is the codebase used for the experiments in the accompanying study. It includes the forecasting, allocation, signaling, evaluation, and analysis pipeline, together with the scripts and tests needed to run the experiments from start to finish.

The main experiment can be reproduced with a single command, while each stage can also be run separately.

## Repository structure

```text
FedSCOPE/
├── .github/          GitHub Actions workflow
├── analysis/         Analysis and figure-generation code
├── configs/          Experiment configuration
├── scripts/          Run scripts
├── src/              Main Python package
├── tests/            Automated tests
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
└── requirements-lock.txt
```

## Requirements

- Windows 10 or 11
- Python 3.11 (64-bit)
- PowerShell
- Internet access for the initial data download

Python dependencies are pinned in `requirements-lock.txt`.

## Quick start

Clone the repository:

```powershell
git clone https://github.com/FedSCOPE/FedSCOPE.git
cd FedSCOPE
```

Run the full pipeline:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\RUN_COMPLETE.ps1
```

This command prepares the Python environment, installs the required packages, runs the tests, prepares the data, executes the experiments, runs the analysis, and generates the figures.

The full run can take some time because the forecasting models are refitted over expanding chronological training windows.

## Run scripts

The main scripts are available in `scripts/`.

### Full pipeline

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\RUN_COMPLETE.ps1
```

Runs the complete workflow.

### Primary experiment

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\RUN_PRIMARY.ps1
```

Runs the main FedSCOPE experiment.

### Experiment suite

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\RUN_PAPER_SUITE.ps1
```

Runs the additional experiment settings used in the study.

### Analysis

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\RUN_PUBLICATION_AUDITS.ps1
```

Runs the post-processing and evaluation steps on completed experiment results.

### Figures

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\RUN_PUBLICATION_FIGURES.ps1
```

Generates the figures from the experiment outputs.

## Configuration

Experiment settings are stored in `configs/`.

The main configuration is:

```text
configs/paper.yaml
```

This file contains the settings used by the main experiment and can be edited without changing the core source code.

## Source code

The main implementation is in:

```text
src/fedscope/
```

The package contains the code for:

- data preparation;
- forecasting;
- federated estimation;
- scarcity and allocation;
- signaling;
- dependence and performance evaluation;
- experiment execution.

The PowerShell scripts in `scripts/` provide the main entry points for running these components.

## Analysis

The `analysis/` folder contains the code used to process experiment outputs, compute summary metrics, run additional evaluations, and generate figures.

## Data

The experiments use the CoSSMic dataset from Open Power System Data (OPSD).

The required data are downloaded and prepared by the pipeline during setup, so no manual data download is needed before the first run.

## Tests

After the environment has been created, run:

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
```

The tests cover the main parts of the pipeline, including data processing, forecasting, allocation, federated estimation, signaling, evaluation, and experiment checks.

## Continuous integration

The GitHub Actions workflow is stored in `.github/` and runs the automated tests for repository updates.

## Citation

If you use this code in academic work, please cite the FedSCOPE study and the OPSD CoSSMic data source.

## License

MIT License. See `LICENSE`.
