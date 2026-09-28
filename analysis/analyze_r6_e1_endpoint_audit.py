from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


from package_paths import OUTPUT_ROOT as ROOT, SOURCE_ROOT, RUNS_ROOT, read_tile_scores
INPUT = SOURCE_ROOT / "r6_multiclass_audit" / "e1_tile_audit_scores.csv"
OUTPUT = SOURCE_ROOT / "r6_multiclass_audit"
GRID = np.arange(0.10, 1.001, 0.05)
SCORES = ("georisk_v2", "self_dice_risk_multiclass", "oracle_tile_loss")
ENDPOINTS = (
    "present_class_miou_risk",
    "fixed_full_event_class_miou_risk",
    "permanent_water_iou_risk",
    "flood_iou_risk",
)


def confusion_columns(frame):
    return sorted(
        [column for column in frame.columns if column.startswith("confusion_")],
        key=lambda value: tuple(int(part) for part in value.split("_")[1:]),
    )


def confusion_from_rows(frame, columns):
    return frame[columns].to_numpy(dtype=np.int64).sum(axis=0).reshape(3, 3)


def endpoint_risks(confusion, full_event_classes):
    confusion = np.asarray(confusion, dtype=np.float64)
    true_positive = np.diag(confusion)
    union = confusion.sum(axis=0) + confusion.sum(axis=1) - true_positive
    present = union > 0
    iou = np.divide(true_positive, union, out=np.zeros_like(true_positive), where=present)
    return {
        "present_class_miou_risk": 1.0 - float(iou[present].mean()),
        "fixed_full_event_class_miou_risk": 1.0 - float(iou[full_event_classes].mean()),
        "permanent_water_iou_risk": 1.0 - float(iou[1]),
        "flood_iou_risk": 1.0 - float(iou[2]),
    }, present


def main():
    frame = pd.read_csv(INPUT, dtype={"sample_id": str, "event_id": str})
    columns = confusion_columns(frame)
    curve_rows = []
    aurc_rows = []
    omission_rows = []
    for seed, run in frame.groupby("seed", sort=True):
        event_ids = sorted(run["event_id"].unique())
        for score in SCORES:
            event_curves = {
                event_id: {endpoint: [] for endpoint in ENDPOINTS}
                for event_id in event_ids
            }
            for event_id in event_ids:
                event = run[run["event_id"] == event_id]
                full_confusion = confusion_from_rows(event, columns)
                diagonal = np.diag(full_confusion)
                full_union = full_confusion.sum(axis=0) + full_confusion.sum(axis=1) - diagonal
                full_classes = full_union > 0
                ordered = event.sort_values(score, kind="mergesort").reset_index(drop=True)
                for coverage in GRID:
                    retained_count = max(1, int(np.ceil(coverage * len(ordered))))
                    confusion = confusion_from_rows(ordered.iloc[:retained_count], columns)
                    risks, present = endpoint_risks(confusion, full_classes)
                    omission_rows.append(
                        {
                            "seed": int(seed),
                            "score": score,
                            "event_id": event_id,
                            "coverage": float(coverage),
                            "permanent_water_omitted": int(full_classes[1] and not present[1]),
                            "flood_omitted": int(full_classes[2] and not present[2]),
                        }
                    )
                    for endpoint, risk in risks.items():
                        event_curves[event_id][endpoint].append(risk)
                        curve_rows.append(
                            {
                                "seed": int(seed),
                                "score": score,
                                "endpoint": endpoint,
                                "event_id": event_id,
                                "coverage": float(coverage),
                                "risk": risk,
                            }
                        )
            for endpoint in ENDPOINTS:
                event_matrix = np.asarray(
                    [event_curves[event_id][endpoint] for event_id in event_ids],
                    dtype=float,
                )
                macro_curve = event_matrix.mean(axis=0)
                for coverage, risk in zip(GRID, macro_curve):
                    curve_rows.append(
                        {
                            "seed": int(seed),
                            "score": score,
                            "endpoint": endpoint,
                            "event_id": "__event_macro__",
                            "coverage": float(coverage),
                            "risk": float(risk),
                        }
                    )
                aurc_rows.append(
                    {
                        "seed": int(seed),
                        "score": score,
                        "endpoint": endpoint,
                        "event_macro_aurc": float(np.trapezoid(macro_curve, GRID)),
                    }
                )

    curves = pd.DataFrame(curve_rows)
    aurc = pd.DataFrame(aurc_rows)
    omissions = pd.DataFrame(omission_rows)
    effects = []
    for endpoint, group in aurc.groupby("endpoint", sort=True):
        pivot = group.pivot(index="seed", columns="score", values="event_macro_aurc")
        for comparator in ("self_dice_risk_multiclass", "oracle_tile_loss"):
            delta = pivot["georisk_v2"] - pivot[comparator]
            effects.append(
                {
                    "endpoint": endpoint,
                    "comparison": f"georisk_v2_minus_{comparator}",
                    "mean_delta": float(delta.mean()),
                    "sd_delta": float(delta.std(ddof=1)),
                    "minimum_delta": float(delta.min()),
                    "maximum_delta": float(delta.max()),
                    "seeds_favoring_georisk": int((delta < 0).sum()),
                    "seeds": int(delta.size),
                }
            )
    omission_summary = (
        omissions.groupby(["score"], as_index=False)[
            ["permanent_water_omitted", "flood_omitted"]
        ]
        .agg(["sum", "mean"])
        .reset_index()
    )
    deployable_scores = ("georisk_v2", "self_dice_risk_multiclass")
    macro = curves[
        curves["event_id"].eq("__event_macro__")
        & curves["score"].isin(deployable_scores)
        & curves["endpoint"].isin(
            ("present_class_miou_risk", "fixed_full_event_class_miou_risk")
        )
    ]
    macro = macro.pivot_table(
        index=["seed", "score", "coverage"],
        columns="endpoint",
        values="risk",
    ).reset_index()
    macro["class_set_shift"] = (
        macro["fixed_full_event_class_miou_risk"]
        - macro["present_class_miou_risk"]
    )
    score_shift = macro.pivot_table(
        index="coverage",
        columns="score",
        values="class_set_shift",
        aggfunc="mean",
    ).reset_index()
    score_shift = score_shift.rename(
        columns={
            "georisk_v2": "georisk_v2_class_set_shift",
            "self_dice_risk_multiclass": "self_overlap_class_set_shift",
        }
    )
    omission_curve = (
        omissions[omissions["score"].isin(deployable_scores)]
        .groupby("coverage", as_index=False)[
            ["permanent_water_omitted", "flood_omitted"]
        ]
        .mean()
        .rename(
            columns={
                "permanent_water_omitted": "permanent_water_omission_rate",
                "flood_omitted": "flood_omission_rate",
            }
        )
    )
    decomposition = score_shift.merge(omission_curve, on="coverage", validate="one_to_one")
    decomposition["mean_class_set_shift"] = decomposition[
        ["georisk_v2_class_set_shift", "self_overlap_class_set_shift"]
    ].mean(axis=1)
    decomposition = decomposition[
        [
            "coverage",
            "georisk_v2_class_set_shift",
            "self_overlap_class_set_shift",
            "mean_class_set_shift",
            "permanent_water_omission_rate",
            "flood_omission_rate",
        ]
    ]
    mean_curve = decomposition.set_index("coverage")["mean_class_set_shift"]
    low_curve = mean_curve[mean_curve.index <= 0.50 + 1e-12]
    summary_rows = []
    for score, group in macro.groupby("score", sort=True):
        curve = group.groupby("coverage")["class_set_shift"].mean()
        summary_rows.append(
            {
                "score_scope": score,
                "integrated_class_set_shift": float(np.trapezoid(curve, curve.index)),
                "fraction_of_shift_at_or_below_0_50": "",
                "class_set_shift_at_0_80": float(
                    curve.loc[np.isclose(curve.index.to_numpy(dtype=float), 0.80)].iloc[0]
                ),
            }
        )
    integrated_mean_shift = float(np.trapezoid(mean_curve, mean_curve.index))
    summary_rows.append(
        {
            "score_scope": "mean_georisk_v2_and_self_overlap",
            "integrated_class_set_shift": integrated_mean_shift,
            "fraction_of_shift_at_or_below_0_50": float(
                np.trapezoid(low_curve, low_curve.index) / integrated_mean_shift
            ),
            "class_set_shift_at_0_80": float(
                mean_curve.loc[
                    np.isclose(mean_curve.index.to_numpy(dtype=float), 0.80)
                ].iloc[0]
            ),
        }
    )
    curves.to_csv(OUTPUT / "e1_endpoint_curves.csv", index=False)
    aurc.to_csv(OUTPUT / "e1_endpoint_aurc.csv", index=False)
    pd.DataFrame(effects).to_csv(OUTPUT / "e1_endpoint_paired_effects.csv", index=False)
    omissions.to_csv(OUTPUT / "e1_class_omission_records.csv", index=False)
    omission_summary.to_csv(OUTPUT / "e1_class_omission_summary.csv", index=False)
    decomposition.to_csv(
        OUTPUT / "e1_class_set_coverage_decomposition.csv", index=False
    )
    pd.DataFrame(summary_rows).to_csv(
        OUTPUT / "e1_class_set_coverage_summary.csv", index=False
    )
    print(pd.DataFrame(effects).to_string(index=False))
    print(omission_summary.to_string(index=False))


if __name__ == "__main__":
    main()
