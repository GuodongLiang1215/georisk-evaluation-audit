# Manuscript output map

All paths below are relative to the chosen output directory unless marked as package inputs. Table CSVs under `tables/` are exports of the reviewed manuscript cells. They are not a claim that every cell has been independently recalculated. The analysis sources identify what is recomputed and what is a frozen descriptive record.

| Figure | Recomputed plotted data |
|---|---|
| 1 | `figures/plotted_data/figure_1_illustrative_geometry.csv` (explicit schematic) |
| 2 | `figure_2_calibration_deltas.csv`, `figure_2_test_deltas.csv`, `figure_2_curve_mean_iqr.csv` in `figures/plotted_data/` |
| 3 | `figures/plotted_data/figure_3_event_seed_mean_correlations.csv`; R5 coupling and endpoint audit outputs |
| 4 | `figures/plotted_data/figure_4_event_mean_range.csv`, `figure_4_mean_curves.csv` |
| 5 | `figures/plotted_data/figure_5_block_paired_deltas.csv`, `source/r6_multiclass_audit/e1_class_set_coverage_summary.csv` |

The numerical names are also enumerated in `validation.json`. Each figure has a matching PDF, SVG, PNG and TIFF under `figures/`.

| Table | Calculation / source coverage |
|---|---|
| 1 | Study-design record; package configurations/manifests, not a computed statistic |
| 2, S1 | Event composition and roles; package manifests plus reviewed table snapshot; metadata not regenerated from raw imagery |
| 3, S3 | `source/r5_endpoint_audit/calibration_loeo_endpoint_results.csv` and `calibration_loeo_endpoint_summary.csv` |
| 4, S6 | `source/r5_endpoint_audit/hazard_area_retention_by_run_event.csv` and `hazard_area_retention_summary.csv` |
| 5 | Frozen consolidated block metrics summarized into `figures/plotted_data/figure_5_block_paired_deltas.csv`; primary alternative endpoints recalculated in R5 |
| 6 | Reporting checklist; prose snapshot |
| S2 | Frozen per-run `data/runs/**/calibration/calibration.json` in the package; primary coefficient refits are additionally verified in comparator/temperature analysis |
| S4 | `source/r5_endpoint_audit/endpoint_paired_effects.csv` and `endpoint_aurc_summary.csv` |
| S5 | `source/r5_endpoint_audit/score_correlations_by_run_event.csv` and `score_correlation_summary.csv` |
| S7 | Original threshold summary cross-checked against scores/thresholds in `evidence/threshold_routing_audit.csv` and `threshold_denominator_audit.csv`; the existing table retains its original aggregation |
| S8 | Frozen external-event composition/AURC table snapshot; geographic metadata and every displayed event-level cell are not rebuilt by the default pipeline |
| S9 | `source/r6_multiclass_audit/e1_endpoint_aurc.csv` and `e1_endpoint_paired_effects.csv` |
| S10 | Frozen consolidated metric comparisons; default figure/block outputs cover the same-comparator effect; strongest-comparator table snapshot is not independently rebuilt |
| S11 | Frozen 18-tile NGA case snapshot; geographic bounds and individual case presentation are not recreated from rasters |
| S12 | `evidence/operating_point_summary.csv` and its run/event-level files |
| S13 | `evidence/external_seed_effects.csv`, `external_descriptive_summary.csv` |
| S14 | `evidence/threshold_denominator_audit.csv`, `threshold_event_counts.csv`; block summary derived from these values |
| S15 | `evidence/comparator_analysis/fidelity_summary.csv`, `ordering_changes.csv` |
| S16 | `evidence/comparator_analysis/key_comparison_table.csv` |
| S17 | `evidence/comparator_analysis/paired_summary.csv`, `retention_paired_summary.csv` |
| S18 | `evidence/temperature_analysis/component_sensitivity_table.csv` |
| S20 | `evidence/multiclass_scope/class_set_shift_by_seed.csv`; direct reconstruction from frozen confusion counts |
| S19 | `evidence/temperature_analysis/composite_ablation_table.csv`, `composite_routing_paired.csv` |
| S21 | `evidence/fixed_budget/summary_by_block.csv`, `paired_by_run.csv` and `policy_by_run.csv`; all 27 runs contribute workload and original tile loss, while 21 runs contribute pooled risk and content |

The package regenerates five main figures; it does not recreate the seven embedded supplementary images or the final Word pagination. The manuscript and supplementary document are separate submission artifacts.

The class-set partial integral uses the inclusive nominal interval 0.10–0.50, with tolerance for the stored 0.50 floating-point grid value. Its share is 83.3%; 78.6% described 0.10–0.45 in the earlier draft. Full curves, AURCs and selections are unchanged.

The binary coverage decomposition uses the same inclusive interval check. Pooled-mIoU gain shares are 62.1%/67.8% for P1/P2 and foreground-Dice shares are 34.0%/56.7%; the earlier 55.9%/60.3% and 27.8%/47.2% covered 0.10–0.45. Only the two partial-integral summary columns change. Plotted curves, full AURCs, selection counts and the 0.80 results are unchanged. See [consistency corrections](CONSISTENCY_CORRECTIONS.md).
