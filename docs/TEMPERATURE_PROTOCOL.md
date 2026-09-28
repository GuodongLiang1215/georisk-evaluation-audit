# Temperature ablation for R2.6

Fixed before full ablation results on 28 September 2026. This revision analysis uses the 18 frozen primary binary runs (P1/P2, F1/F2/F3, seeds 13/37/71), their original calibration/test splits and saved scalar temperatures. It neither retrains segmenters nor tunes a new temperature. Compare T=1 against each run's saved calibrated T using the same float16-stored logits. Replay both settings together; first reproduce the calibrated scores and the earlier comparator replay.

## Components and probability diagnostics

Recompute upper-tail normalized entropy, upper-tail MSP risk, spatial inconsistency, stored epsilon-smoothed binary overlap and unsmoothed SDC (float64 sums). Representation distances are unchanged because they do not use temperature-scaled probabilities. Preserve original input-valid support and top-tail fraction 0.10. Check hard-prediction equality directly; do not infer numerical equality solely from theoretical invariance of argmax to a positive scalar temperature.

For each component report T=1 minus calibrated raw-score changes, test Spearman correlation, within-event ordering and retained-set changes, six existing endpoints (tile-mean and event-pooled mIoU, foreground Dice and foreground IoU), nominal 0.80 risk, and 0.70–0.90 partial AURC. Compute event-pixel-mean NLL and binary foreground Brier error on reference-valid pixels as descriptive probability diagnostics. Probability calibration improvement and selective ranking improvement are separate observations; no temperature is chosen from test results.

## Composite decomposition

Use the original v2 four features, event weights, mIoU fitting target, nonnegative ridge 0.25, 100 sweeps and unit-L1 normalization. Keep the original epsilon-overlap feature here so the comparison isolates temperature rather than also changing the comparator implementation.

1. **Calibrated original v2:** reproduce the original feature ECDFs, fitted weights and scores at saved T.
2. **T=1 with frozen mapping:** apply T=1 features to the saved-T calibration ECDFs and original coefficients. This isolates changing probability inputs without downstream adjustment.
3. **T=1 with recalibrated ECDF and fixed coefficients:** recompute ECDF references on T=1 calibration features, keeping original coefficients. This distinguishes feature-scale adjustment from coefficient fitting.
4. **T=1 with refitting:** recompute calibration ECDFs and fit the same model to the same calibration mIoU outcomes at T=1. This is the end-to-end ablation with temperature scaling omitted; only calibration labels enter fitting.

Evaluate each fixed score vector with the saved-T stored-logit hard predictions to isolate changes in selection. If finite-precision hard predictions differ, also evaluate T=1 masks for the T=1 arms and report their sensitivity; retain all affected samples. Replay the original calibrated component scores and v2 exactly before interpreting differences.

## Routing and aggregation

Distinguish nominal within-event fixed-count coverage from transferred calibration thresholds. At nominal 0.80, report risk and tile/area/reference-hazard retention using the existing grid and stable original sample order. For routed workload, retain scores less than or equal to their event-balanced calibration 80th percentile, using the original quantile code and tie behavior. The frozen-mapping arm keeps the original v2 threshold; recalibrated-ECDF and refit arms obtain thresholds from their T=1 calibration scores. Primitive-score thresholds are recalculated on their corresponding calibration scores. A separate stale-threshold primitive diagnostic is not part of this analysis.

Report both event-macro realized coverage and pooled tile counts/review fractions. Summaries give the equal mean of the nine correlated event-macro run values per block, with complete paired run tables; do not treat records or seeds as independent geographic replications. Probability diagnostics first pool valid-pixel loss within each event and then average events equally. No new p values or independence-based confidence intervals.

## Deliverables and scope

Produce a concise component-sensitivity table and composite-ablation table with reproducible source data, integrate the principal finding in Sections 2.4 and the results, add full definitions and routing diagnostics to the supplement, and replace the pending R2.6 response with completed evidence. Preserve previous comparator revisions. This tests the primary binary experiments; it does not establish temperature robustness of the custom multiclass score or external datasets.

For continuity with the completed comparator revision, also summarize calibrated v2 minus calibrated exact SDC and fully refitted T=1 v2 minus T=1 exact SDC under both tile-mean and pooled mIoU. This compares each complete pipeline with its probability-matched baseline, using the already specified arms and endpoints.
