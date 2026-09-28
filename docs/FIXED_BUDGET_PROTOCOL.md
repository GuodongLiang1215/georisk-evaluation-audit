# Fixed batch review budget comparison

Protocol fixed before the numerical comparison on 28 September 2026. This is a revision-selected descriptive analysis of frozen GeoRisk v2 scores, not a new development or deployment validation.

## Question and scope

Compare the saved calibration threshold with a fixed review-count rule on every one of the 27 saved test runs (18 primary binary, 3 Kuro Siwo multiclass, and 6 Burn Scars binary). Each run's complete test set is one available batch. Do not allocate 20% separately to each event, pool different runs, retrain models, refit scores, or use test labels to route samples.

## Routing rules

- Threshold: automatically retain each tile whose frozen v2 score is at or below its saved calibration threshold.
- Fixed budget: for N tiles, review B = floor(N/5), and automatically retain N-B. Sort by ascending frozen score, breaking exact ties by original saved input position; retain the first N-B positions. Thus later original positions are reviewed first within a boundary tie. This is a complete-batch rule, not an online arrival policy. The integer review count never exceeds 20% of N and leaves less than one tile of rounding slack.
- Keep every run, regardless of the direction of any change. Do not tune the budget or tie rule after examining outcomes.

## Evaluation

Routing reads only scores, input order, N and (for the threshold policy) the saved calibration threshold. Join reference confusion counts only for retrospective evaluation. Report actual review count/fraction, auto-retained tile-mean 1-mIoU, batch-pooled 1-mIoU, valid-pixel retention, and reference-hazard content retained automatically. Report the unchanged full-batch risks as context.

Batch-pooled mIoU uses a fixed class set defined by nonzero prediction-or-reference union in the full batch. A full-batch class absent from the retained subset receives zero IoU. This avoids attributing a change in the scoring class set to the routing policy. It is a batch-level endpoint, distinct from the event-macro curves at nominal 0.80 elsewhere in the paper. Tile-mean loss uses the original present-class convention. For binary blocks the hazard is class 1 (water for P1/P2, burn scar for E2/E3); for Kuro Siwo it is flood class 2, with permanent-water class 1 reported separately. Content retention is reference pixels in auto-retained tiles divided by reference pixels in the entire batch, not segmentation recall or final-map loss. Zero reference totals yield an undefined ratio, not an invented zero or one.

Record per-event counts to show how the batch budget is distributed; E2/E3 have no event IDs and use only a dataset bucket. Summarize paired differences as fixed minus threshold, with all per-run values and descriptive mean/range/counts within each block. No cross-dataset mean, independent-replicate inference, confidence interval, or human-review benefit is claimed.

## Verification and reporting

Verify unique sample joins, unchanged original score ordering, valid-pixel/confusion totals, tile-loss reconstruction, threshold counts against the earlier 27-run audit, exact budget counts, monotone risk ordering, nested retained sets, and unchanged input hashes. Include a tied-score and integer-rounding check. Report both workload control and any error/content tradeoff. The fixed-count rule controls the review count by construction; risk and content outcomes remain empirical. Prospective error/content constraints would require domain-appropriate calibration and subsequent monitoring, and are not guaranteed by a fixed count.

## Input-availability clarification before policy outcomes

Input validation found that frozen per-tile confusion counts cover P1/P2/E1 (21 runs), while E2/E3 retain only scores and original tile losses. Workload and original tile-mean loss therefore cover all 27 runs; pooled risk, valid-pixel and hazard retention cover the 21 runs with existing counts. E2/E3 content fields remain unavailable rather than inferred from tile loss. No extra model inference is introduced.

The earlier audit counts were recovered by deterministic checkpoint re-inference using raw hard logits, whereas the stored score pipeline used float16-exported logits. Recomputed tile loss can consequently differ from stored tile loss (observed input-validation maxima: P1 1.1e-16, P2 0.0002332, E1 0.0013728). The primary tile-mean endpoint uses the original saved loss for all 27 runs. Recomputed audit tile-mean loss is an additional provenance/sensitivity field for the 21 covered runs; pooled risk uses those same frozen audit counts for both policies. This clarification precedes the first successful policy comparison and does not change any routing rule or omit any run.
