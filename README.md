# FedSCOPE

This repository contains the implementation of the paper:

**“Causal Service-Preserving Strategic Opacity for Federated Peer-to-Peer Emergency Energy Sharing in Networked Microgrids.”**

It includes the code used to run the main experiments, sensitivity analyses, evaluation procedures, and figures reported in the study.

## Repository structure

```text
FedSCOPE/
├── .github/          GitHub Actions
├── analysis/         Analysis and figure generation
├── configs/          Experiment configuration
├── scripts/          Experiment run scripts
├── src/              FedSCOPE implementation
├── tests/            Tests
├── LICENSE
├── pyproject.toml
├── README.md
└── requirements-lock.txt
```

## Requirements

- Python 3.11
- Windows 10/11
- PowerShell

Dependencies are listed in `requirements-lock.txt`.

## Run the full experiment

Clone the repository:

```powershell
git clone https://github.com/FedSCOPE/FedSCOPE.git
cd FedSCOPE
```

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\RUN_COMPLETE.ps1
```

This runs the full experimental pipeline, including data preparation, the main experiment, sensitivity analyses, evaluation, and figure generation.

## Run individual stages

Main experiment:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\RUN_PRIMARY.ps1
```

Sensitivity experiments:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\RUN_PAPER_SUITE.ps1
```

Evaluation:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\RUN_PUBLICATION_AUDITS.ps1
```

Figures:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\RUN_PUBLICATION_FIGURES.ps1
```

## Configuration

The main experiment configuration is:

```text
configs/paper.yaml
```

## Data

The experiments use the **CoSSMic dataset** from Open Power System Data (OPSD).

The required data are downloaded and prepared automatically by the pipeline.

## Tests

Run the test suite with:

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
```

## Citation

If you use this repository, please cite the FedSCOPE paper and the OPSD CoSSMic dataset.

## License

MIT License.
