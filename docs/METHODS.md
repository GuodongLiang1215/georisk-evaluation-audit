# Calculation conventions

The main scientific question concerns how evaluation functionals and coverage denominators affect interpretation of selective geospatial segmentation. The package preserves the study's existing sample splits, fitted scores, controls and endpoint audits. It adds no independent data or new segmentation training.

## Observations and aggregation

P1 and P2 contain U-Net and DOFA–UPerNet results on Sen1Floods11: three rotating event partitions and seeds 13, 37 and 71, giving nine correlated runs per model. The primary replay tables contain 2,640 calibration and 2,640 test records across these 18 runs. Repeated observations across models/seeds are not independent samples. External blocks E1–E3 contain three seeds each; these are target-calibrated evaluations, not universal zero-shot transfer claims.

Tile-mean loss and loss evaluated after pooling confusion counts are different functionals. Binary endpoints include present-class mIoU risk, foreground IoU risk and foreground Dice risk. The multiclass audit preserves present-class and fixed-class conventions. Consult `analyze_r5_endpoint_validity.py` and `analyze_r6_e1_endpoint_audit.py` for empty-class handling. The E1 self-overlap comparator is a custom multiclass extension; the exact binary SDC replay applies to P1/P2.

Lower scores are retained first, with stable sorting for ties. The nominal coverage grid is the original NumPy grid from 0.10 to 1.00 in increments of 0.05. Retained tile counts use `ceil(coverage * n)` with that grid, including its original floating-point representation. Realized count coverage can exceed the nominal fraction in small events. AURC is integrated on the retained grid; do not silently change grid construction or endpoint conventions.

Tile-count, valid-pixel and reference-hazard coverage have distinct denominators. Reference-hazard coverage is an evaluation quantity using reference labels, not a deployable selection score. Pooled tile coverage and event-macro coverage are reported separately. Threshold routing uses calibration thresholds and can depart from the nominal 80% coverage under shift.

## Comparators and calibration

Stored self-overlap and exact binary SDC remain separate scores. Replay sufficient statistics include the exact-SDC numerator/denominator, empty-prediction information, calibrated probability summaries and confusion counts. The analysis preserves the distinction between masks from stored float16 logits and the original unquantized confusion audit. These precision cases are explicitly compared rather than combined silently.

Target-matched fits use calibration observations only, event-balanced empirical CDFs, nonnegative coordinate-descent ridge coefficients (penalty 0.25, 100 sweeps), and L1-normalized weights. Alternative loss targets are specified in `compare_primary_targets.py`. Calibration LOEO and ridge sensitivity are post hoc diagnostics; they do not establish independent model selection.

Temperature sensitivity compares the frozen calibrated temperature with T=1, using three composite conditions: original mapping/weights, updated empirical CDF with fixed weights, and calibration-only refitting. Representation shift remains unchanged. Pixel disagreements, probability losses, rankings, retained sets, thresholds and workload are tracked separately.

## Numerical checks and interpretation

The validator requires matching file structure, row order and categorical values. Integer and Boolean columns must match exactly; floating values use absolute tolerance 1e-11 plus relative tolerance 1e-9. The temperature fit has a stricter absolute bound of 1e-12 and also requires exact calibrated ranking and retained-set equality. Actual errors are saved, not rounded away.

Frozen Parquet-to-CSV exports were checked for exact numeric equality and unchanged record order. CSV parsing uses round-trip float64 conversion. Primitive replay-score checks and calibrated AURC cross-checks remain exact. Figure source CSVs are compared with the original refined-figure values; scientific values are preserved even when portable fonts change layout.

Means, standard deviations, ranges and IQR shading are descriptive across correlated runs. They are not independent-event confidence intervals. Figure 1 is an explicitly constructed schematic with fixed counts, not an additional empirical result.

## Multiclass seed-level audit and interval correction

`audit_multiclass_scope.py` reconstructs the two class conventions from the same frozen confusion counts and retained sets. It exports all six seed–ordering pairs and 60 event–seed–ordering results. The partial integral includes the nominal 0.50 endpoint using absolute tolerance 1e-12; the mean fraction over 0.10–0.50 is 83.3%. A previous strict comparison inadvertently integrated only through 0.45 (78.6%). Full AURCs, the 32.7% relative displacement, risk curves and original selections are unchanged. These descriptive results cover one U-Net architecture, three seeds and the same 2,000 test tiles from 10 events.
