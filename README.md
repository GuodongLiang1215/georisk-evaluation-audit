# GeoRisk evaluation audit

Companion code and frozen-data analyses for **Evaluation unit sensitivity in selective Earth observation segmentation**. This package reproduces the study analyses and five main figures. It includes analysis code, per-tile scores and confusion statistics, calibration settings, model implementation, and study manifests.

**Repository:** [georisk-evaluation-audit](https://github.com/GuodongLiang1215/georisk-evaluation-audit) · **Snapshot:** 28 September 2026.

Author-original software and its associated documentation are licensed under the [MIT License](LICENSE). Dataset-derived materials and third-party components retain their separate terms; see [LICENSING](docs/LICENSING.md).

## Run the analyses

The default workflow uses CPU only and needs no imagery, checkpoints, GPU, network connection after dependency installation, or author-machine directories. Start from this package directory with Python 3.14 (tested on Windows x64):

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe run_all.py --output outputs
```

On Linux/macOS the environment's Python executable is `.venv/bin/python`; those platforms have not been tested. `requirements.txt` gives broader dependency ranges for other compatible environments; the lock file records the tested environment. Repeated executions require a new, empty output directory, for example `--output outputs_second_run`. The script never deletes an existing output directory.

The single command runs thirteen stages and produces:

- Recomputed binary and multiclass endpoint audits, calibration diagnostics, operating-point effects, and threshold workload.
- Exact binary SDC comparisons, target-matched fits, and temperature sensitivity from all 18 primary frozen runs.
- Fixed batch review-budget comparisons for all 27 runs (workload and original tile loss), with pooled risk and hazard-content outcomes for the 21 runs with frozen audit confusion counts; see [protocol](docs/FIXED_BUDGET_PROTOCOL.md).
- Five main figures in PDF, SVG, PNG and 600-dpi RGB TIFF, their plotted values, and a merged PDF.
- CSV exports of the six main and twenty-one supplementary table snapshots. Exporting these reviewed cells is distinct from recomputing each statistic; the [output map](docs/OUTPUT_MAP.md) specifies coverage.
- `validation.json`, comparing 86 recomputed CSV files with frozen reference results, and `reproduction_report.json`, recording every stage, dependency versions and input immutability.

The local test took about 30 seconds; timing varies with hardware. The final local test and its exact scope are recorded in [VALIDATION](docs/VALIDATION.md).

## Reproduction levels

| Level | Included | Verification scope |
|---|---|---|
| Frozen-data calculations | All inputs and one-command runner | Clean environment plus relocated-package test; numerical outputs compared with references |
| Frozen-model inference | Model code, primary comparator/temperature replay scripts, configurations, manifests, external material inventory | Source compilation and command-line import checks; raw imagery and trained checkpoints must be supplied separately |
| Training | Original study training implementation and configuration history | No training rerun as part of package verification; exact retraining is not claimed |

See [MODEL_REPLAY](docs/MODEL_REPLAY.md) for the second level. Do not interpret a successful CPU run as new independent validation of the models.

## Files and scientific conventions

`data/` contains immutable analysis inputs. `expected/` contains comparison targets only and is read solely by the validator. `analysis/` implements the CPU pipeline. `model_code/`, `model_replay/` and `model_materials/` support optional inference. Outputs are always written separately.

[METHODS](docs/METHODS.md) explains endpoint, coverage, fitting and uncertainty conventions. [DATA](docs/DATA.md) documents the frozen sources and export checks. [Figure captions](docs/Figure_captions.md) describe the figures. Follow the manuscript and current analysis scripts when interpreting results; historical configuration labels are retained for provenance and do not constitute preregistration.
