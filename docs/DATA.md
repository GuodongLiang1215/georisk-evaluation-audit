# Frozen data and provenance

`data/provenance.json` records logical source names, SHA-256 hashes, file sizes and transformations. Forty-five score tables were exported from the author Parquet files with exact numeric and row-order checks. All CPU calculations read portable CSVs; PyArrow is not required for that workflow.

| Input | Contents and role |
|---|---|
| `data/source/r5_endpoint_audit/` | Binary calibration/test tile statistics and confusion counts; endpoint and content-retention analyses |
| `data/source/r6_multiclass_audit/` | E1 multiclass tile statistics; class-set sensitivity |
| `data/source/consolidated/georisk_v2_metrics.csv` | Frozen aggregate results across 27 runs; block comparisons |
| `data/source/legacy_r4/threshold_coverage_runs.csv` | Original routing summary used as a cross-check, not refitted thresholds |
| `data/runs/` | Test scores and calibration JSON for 27 runs; calibration scores for the 18 primary runs |
| `data/replay/comparator_replay/` | Eighteen primary probability-replay CSVs with SDC sufficient statistics and confusion counts; JSON identifies runs |
| `data/replay/temperature_replay/` | Eighteen paired calibrated/T=1 replay CSVs with probability losses and component scores |
| `data/table_layout_data.json` | Reviewed table-cell snapshots, captions and notes |

Tile identifiers, event identifiers, split, block and seed retain the original ordering. Pixel counts and confusion entries are counts; loss, risk and coverage are unitless. Scores generally increase with risk; the exact directional controls are labeled explicitly. Missing or undefined statistics retain their NaN convention.

`expected/` stores frozen reference outputs for regression comparison. Two earlier R5 summary variants (`coverage_gain_decomposition.csv` and `external_comparator_summary.csv`) are retained for provenance but excluded from validation; their current endpoint/block counterparts are validated. No expected result is an analytic input.

The optional `model_materials/` directory contains six configurations, six manifests and two normalization-statistics files. Manifests refer to external imagery using relative paths or tortilla byte offsets. An updated download is not automatically equivalent to the frozen container: check its identity before using stored offsets.

Raw imagery, pixel masks, logits, model checkpoints and third-party DOFA implementation files are not bundled. The source datasets and model resources are listed in [LICENSING](LICENSING.md) and [MODEL_REPLAY](MODEL_REPLAY.md).
