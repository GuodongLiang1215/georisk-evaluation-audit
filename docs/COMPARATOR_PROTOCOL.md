# Comparator fidelity and target alignment protocol

Written before the full primary replay and target-alignment results on 28 September 2026. This is a post-hoc revision analysis for GEOMAT-D-26-00230, not a new independent validation experiment.

## Fixed scope

Use all 18 completed primary runs: P1 U-Net and P2 DOFA–UPerNet, F1/F2/F3, seeds 13/37/71. Replay their original calibration and test splits from frozen checkpoints. Original files remain read-only. Expect 2,640 calibration records and 2,640 test records across both models. Do not train segmenters or alter folds, seeds, masks, temperature, original rankings or test data.

## Fidelity check

Reproduce inference and float16 logit storage from the original evaluation script. Recompute the legacy binary overlap and entropy using the saved temperature, and join by block, partition, seed, split and sample ID. Compare with original Parquet values, not rounded manuscript tables. Verify the re-inferred hard-mask confusion counts against the saved tile audit matrices.

SDC follows Borges et al. (2026), equation 22, DOI 10.1007/s10994-026-07096-w. On input-valid pixels define A=2 sum(p*yhat), B=sum(p)+sum(yhat). The exact published confidence is A/B for B>0 and zero for B=0. Compare with stored confidence (A+1e-7)/(B+1e-7). Compute both an unsmoothed version with legacy reduction precision and a float64 reduction version to distinguish smoothing from floating-point summation. Report score differences, changed orderings and ties, retained sets, AURC, risk and hazard coverage at 0.80. A small score difference alone is not evidence of identical orderings.

Keep the stable original sample order when breaking ties. Exact SDC assigns equal risk to all prediction-empty tiles; quantify the resulting tied-score boundary rather than breaking ties using reference labels.

## Target alignment

First reproduce the saved v2 score from the calibration features, ECDF transformation and original solver. Only after this gate passes, fit the same four-feature model under three alternative targets: tile macro-IoU loss, tile foreground-Dice loss and tile foreground-IoU loss. For the new target-aligned fits use exact binary SDC as the overlap feature, keeping entropy, representation distance and spatial inconsistency fixed. All arms use the same calibration tiles, equal-event weights, nonnegative coordinate-descent solver, ridge 0.25, 100 sweeps and L1 weight normalization. No tuning on test outcomes.

For Dice and foreground IoU, empty foreground on both reference and prediction has coefficient zero and risk one, consistent with the published Dice convention and the existing foreground-endpoint audit. State this convention explicitly; do not choose it after seeing results.

Evaluate every scoring arm under equal-tile mean macro-IoU, foreground Dice and foreground IoU losses, as well as event-pooled macro-IoU, foreground Dice and foreground IoU risk. Matching a per-tile Dice fitting target to pooled-event Dice is not sufficient by itself: the equal-tile Dice endpoint is the directly aligned comparison. Keep the original event-macro averaging and coverage grid (0.10–1.00 by 0.05) for comparability. Use stored nominal grid floats to avoid changing ceil(c*n) at integer boundaries.

Primary outputs are run-level paired AURC differences against exact SDC, paired risk differences at nominal 0.80, and differences in reference-hazard coverage. Report all runs and block summaries. The predefined 0.70–0.90 partial interval is supplementary. These correlated runs support descriptive comparisons; do not manufacture independent-sample significance claims.

## Interpretation and deliverables

### Precision alignment recorded before target-fit results

Replay validation found that the original risk-fitting table uses hard predictions from float16-stored logits, while the later R5 endpoint audit re-inferred hard predictions before that storage conversion. The difference occurs in P2. To keep the new target comparison matched, all three new fitting targets and their primary test endpoints use the stored-logit prediction convention. Every scoring arm is additionally evaluated under the original R5 hard-mask convention as a precision sensitivity check. Frozen R5 AURCs must be reproduced under that convention. Neither the affected pixels nor runs are removed. This clarification precedes the target-fit results; it does not change features, targets, calibration settings or coverage.

The AURC reproduction check further identified a CSV-serialization tie between two prediction-empty tiles in P2/F2/seed37 (Ghana_264787 and Ghana_8090). The original R5 output is reproduced with its own CSV scores and raw masks. New comparator analyses retain the original Parquet precision for every frozen score, with the CSV-versus-Parquet AURC difference reported separately; no precision choice is made based on comparative performance.

Use the results to distinguish a comparator implementation concern from a target-alignment concern. Assess whether the fixed-ordering evaluation-sensitivity conclusion persists without implying that the fitted score has the same label budget as the plug-in estimator or is universally superior. Keep the paper's main contribution in view and put implementation and per-run details in the supplement.

Deliver a key comparison table, reproducible source tables and scripts, and response paragraphs for R2.3, R3.M3 and R4.2. Do not mark the custom multiclass comparator as the published binary SDC or infer external-dataset validation from these primary runs.
