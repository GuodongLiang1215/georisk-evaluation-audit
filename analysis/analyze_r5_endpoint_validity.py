from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from package_paths import OUTPUT_ROOT as ROOT, SOURCE_ROOT, RUNS_ROOT, read_tile_scores
RUNS = RUNS_ROOT
OUTPUT = SOURCE_ROOT / "r5_endpoint_audit"
GRID = np.arange(0.10, 1.001, 0.05)
PRIMARY_SCORES = (
    "georisk_v2",
    "soft_dice_risk_binary",
    "predicted_foreground_fraction",
    "predicted_foreground_scarcity",
    "reference_foreground_fraction",
    "oracle_tile_loss",
)
SCORE_LABELS = {
    "georisk_v2": "GeoRisk v2",
    "soft_dice_risk_binary": "Soft Dice-style self-overlap",
    "predicted_foreground_fraction": "Predicted foreground prevalence",
    "predicted_foreground_scarcity": "Predicted foreground scarcity",
    "reference_foreground_fraction": "Reference foreground prevalence (label-informed)",
    "oracle_tile_loss": "Label-informed tile-loss ordering",
}
ENDPOINTS = (
    "present_class_miou_risk",
    "fixed_full_event_class_miou_risk",
    "foreground_iou_risk",
    "foreground_dice_risk",
)


def event_balanced_ecdf(value, calibration_values, calibration_events, event_balanced):
    calibration_values = np.asarray(calibration_values, dtype=np.float64)
    calibration_events = np.asarray(calibration_events, dtype=str)
    if not event_balanced:
        return float(np.mean(calibration_values <= value))
    event_values = []
    for event_id in np.unique(calibration_events):
        event_values.append(
            np.mean(calibration_values[calibration_events == event_id] <= value)
        )
    return float(np.mean(event_values))


def fit_event_balanced_nonnegative_ridge(
    calibration_rows,
    components,
    event_balanced,
    ridge,
    iterations,
):
    calibration_events = np.asarray(
        [row["event_id"] for row in calibration_rows], dtype=str
    )
    calibration_values = {
        component: np.asarray(
            [row[component] for row in calibration_rows], dtype=np.float64
        )
        for component in components
    }
    design = np.column_stack(
        [
            [
                event_balanced_ecdf(
                    value,
                    calibration_values[component],
                    calibration_events,
                    event_balanced,
                )
                for value in calibration_values[component]
            ]
            for component in components
        ]
    )
    target = np.asarray(
        [row["tile_loss"] for row in calibration_rows], dtype=np.float64
    )
    sample_weights = np.ones(len(calibration_rows), dtype=np.float64)
    if event_balanced:
        for event_id in np.unique(calibration_events):
            event_mask = calibration_events == event_id
            sample_weights[event_mask] = 1.0 / event_mask.sum()
        sample_weights /= sample_weights.mean()
    weighted_design = design * np.sqrt(sample_weights)[:, None]
    weighted_target = target * np.sqrt(sample_weights)
    weights = np.full(len(components), 1.0 / len(components), dtype=np.float64)
    for _ in range(iterations):
        for index in range(len(components)):
            residual = weighted_target - weighted_design @ weights
            residual += weighted_design[:, index] * weights[index]
            numerator = weighted_design[:, index] @ residual
            denominator = weighted_design[:, index] @ weighted_design[:, index] + ridge
            weights[index] = max(0.0, numerator / denominator)
    return weights / weights.sum(), calibration_values, calibration_events


def confusion_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        [column for column in frame.columns if column.startswith("confusion_")],
        key=lambda value: tuple(int(part) for part in value.split("_")[1:]),
    )


def confusion_from_rows(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    classes = int(math.sqrt(len(columns)))
    return frame[columns].to_numpy(dtype=np.int64).sum(axis=0).reshape(classes, classes)


def endpoint_risks(confusion: np.ndarray, full_event_classes: np.ndarray) -> dict[str, float]:
    confusion = np.asarray(confusion, dtype=np.float64)
    true_positive = np.diag(confusion)
    false_positive = confusion.sum(axis=0) - true_positive
    false_negative = confusion.sum(axis=1) - true_positive
    union = true_positive + false_positive + false_negative
    present = union > 0
    iou = np.divide(true_positive, union, out=np.zeros_like(true_positive), where=present)
    present_miou = float(iou[present].mean())
    fixed_miou = float(iou[full_event_classes].mean())
    foreground_union = float(union[1])
    foreground_iou = float(true_positive[1] / foreground_union) if foreground_union > 0 else 0.0
    foreground_dice_denominator = float(
        2 * true_positive[1] + false_positive[1] + false_negative[1]
    )
    foreground_dice = (
        float(2 * true_positive[1] / foreground_dice_denominator)
        if foreground_dice_denominator > 0
        else 0.0
    )
    return {
        "present_class_miou_risk": 1.0 - present_miou,
        "fixed_full_event_class_miou_risk": 1.0 - fixed_miou,
        "foreground_iou_risk": 1.0 - foreground_iou,
        "foreground_dice_risk": 1.0 - foreground_dice,
    }


def run_key_values(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "block": str(frame["block"].iloc[0]),
        "partition": str(frame["partition"].iloc[0]),
        "seed": int(frame["seed"].iloc[0]),
    }


def build_curves(
    frame: pd.DataFrame,
    scores: tuple[str, ...],
    *,
    subset_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = confusion_columns(frame)
    curve_rows: list[dict[str, object]] = []
    aurc_rows: list[dict[str, object]] = []
    for (_, _, _), run in frame.groupby(["block", "partition", "seed"], sort=True):
        run_keys = run_key_values(run)
        event_ids = sorted(run["event_id"].astype(str).unique())
        for score in scores:
            macro_by_endpoint = {endpoint: [] for endpoint in ENDPOINTS}
            event_curves: dict[str, dict[str, list[float]]] = {}
            for event_id in event_ids:
                event = run[run["event_id"].astype(str) == event_id]
                full_confusion = confusion_from_rows(event, columns)
                true_positive = np.diag(full_confusion)
                full_union = (
                    true_positive
                    + full_confusion.sum(axis=0)
                    - true_positive
                    + full_confusion.sum(axis=1)
                    - true_positive
                )
                full_event_classes = full_union > 0
                ordered = event.sort_values(score, kind="mergesort").reset_index(drop=True)
                event_curves[event_id] = {endpoint: [] for endpoint in ENDPOINTS}
                for coverage in GRID:
                    retained_count = max(1, int(np.ceil(coverage * len(ordered))))
                    retained = ordered.iloc[:retained_count]
                    confusion = confusion_from_rows(retained, columns)
                    risks = endpoint_risks(confusion, full_event_classes)
                    for endpoint, risk in risks.items():
                        event_curves[event_id][endpoint].append(risk)
                        curve_rows.append(
                            {
                                **run_keys,
                                "subset": subset_name,
                                "score": score,
                                "score_label": SCORE_LABELS.get(score, score),
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
                macro_by_endpoint[endpoint] = macro_curve
                for coverage, risk in zip(GRID, macro_curve):
                    curve_rows.append(
                        {
                            **run_keys,
                            "subset": subset_name,
                            "score": score,
                            "score_label": SCORE_LABELS.get(score, score),
                            "endpoint": endpoint,
                            "event_id": "__event_macro__",
                            "coverage": float(coverage),
                            "risk": float(risk),
                        }
                    )
                aurc_rows.append(
                    {
                        **run_keys,
                        "subset": subset_name,
                        "score": score,
                        "score_label": SCORE_LABELS.get(score, score),
                        "endpoint": endpoint,
                        "event_macro_aurc": float(np.trapezoid(macro_curve, GRID)),
                    }
                )
    return pd.DataFrame(curve_rows), pd.DataFrame(aurc_rows)


def summarize_aurc(aurc: pd.DataFrame) -> pd.DataFrame:
    return (
        aurc.groupby(["block", "subset", "endpoint", "score", "score_label"], as_index=False)
        ["event_macro_aurc"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


def paired_effects(aurc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (block, subset, endpoint), group in aurc.groupby(
        ["block", "subset", "endpoint"], sort=True
    ):
        pivot = group.pivot(
            index=["partition", "seed"],
            columns="score",
            values="event_macro_aurc",
        )
        for comparator in (
            "soft_dice_risk_binary",
            "predicted_foreground_fraction",
            "predicted_foreground_scarcity",
            "reference_foreground_fraction",
            "oracle_tile_loss",
        ):
            if comparator not in pivot.columns:
                continue
            delta = pivot["georisk_v2"] - pivot[comparator]
            rows.append(
                {
                    "block": block,
                    "subset": subset,
                    "endpoint": endpoint,
                    "comparison": f"georisk_v2_minus_{comparator}",
                    "mean_delta": float(delta.mean()),
                    "sd_delta": float(delta.std(ddof=1)),
                    "minimum_delta": float(delta.min()),
                    "maximum_delta": float(delta.max()),
                    "runs_favoring_georisk": int((delta < 0).sum()),
                    "runs": int(delta.size),
                }
            )
    return pd.DataFrame(rows)


def coverage_decomposition(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    selected = curves[
        (curves["subset"] == "all_tiles")
        & (curves["event_id"] == "__event_macro__")
        & (curves["score"].isin(["georisk_v2", "soft_dice_risk_binary"]))
    ]
    for (block, endpoint), group in selected.groupby(["block", "endpoint"], sort=True):
        mean = (
            group.groupby(["score", "coverage"], as_index=False)["risk"]
            .mean()
            .pivot(index="coverage", columns="score", values="risk")
            .reset_index()
            .sort_values("coverage")
        )
        mean["delta_v2_minus_self_overlap"] = (
            mean["georisk_v2"] - mean["soft_dice_risk_binary"]
        )
        gain = -mean["delta_v2_minus_self_overlap"].to_numpy(dtype=float)
        coverage = mean["coverage"].to_numpy(dtype=float)
        total = float(np.trapezoid(gain, coverage))
        # Include the nominal 0.50 grid point (stored as 0.5000000000000001).
        low = coverage <= 0.5 + 1e-12
        low_gain = float(np.trapezoid(gain[low], coverage[low]))
        for record in mean.to_dict("records"):
            rows.append(
                {
                    "block": block,
                    "endpoint": endpoint,
                    **record,
                    "total_integrated_gain": total,
                    "gain_at_coverage_le_0_5": low_gain,
                    "fraction_gain_at_coverage_le_0_5": (
                        low_gain / total if total != 0 else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def score_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    outcomes = (
        "tile_loss",
        "reference_foreground_fraction",
        "predicted_foreground_fraction",
    )
    scores = (
        "georisk_v2",
        "soft_dice_risk_binary",
        "predicted_foreground_fraction",
        "predicted_foreground_scarcity",
    )
    for (block, partition, seed, event_id), event in frame.groupby(
        ["block", "partition", "seed", "event_id"], sort=True
    ):
        for score in scores:
            for outcome in outcomes:
                rho, p_value = spearmanr(event[score], event[outcome])
                rows.append(
                    {
                        "block": block,
                        "partition": partition,
                        "seed": int(seed),
                        "event_id": event_id,
                        "tiles": int(len(event)),
                        "score": score,
                        "outcome": outcome,
                        "spearman_rho": float(rho),
                        "two_sided_p_uncorrected": float(p_value),
                    }
                )
    return pd.DataFrame(rows)


def correlation_summary(correlations: pd.DataFrame) -> pd.DataFrame:
    return (
        correlations.groupby(["block", "score", "outcome"], as_index=False)["spearman_rho"]
        .agg(["mean", "std", "median", "min", "max", "count"])
        .reset_index()
    )


def random_retention_null(
    reference_foreground: np.ndarray,
    retained_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    n = len(reference_foreground)
    combinations = math.comb(n, retained_count)
    total = reference_foreground.sum()
    if combinations <= 200_000:
        values = [
            reference_foreground[list(indices)].sum() / total
            for indices in itertools.combinations(range(n), retained_count)
        ]
        return np.asarray(values, dtype=float), "exact"
    draws = 20_000
    random_order = rng.random((draws, n))
    retained = np.argpartition(random_order, retained_count - 1, axis=1)[:, :retained_count]
    values = reference_foreground[retained].sum(axis=1) / total
    return values, "monte_carlo_20000"


def hazard_retention(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(20260829)
    rows = []
    for (block, partition, seed, event_id), event in frame.groupby(
        ["block", "partition", "seed", "event_id"], sort=True
    ):
        reference_foreground = event["reference_foreground_pixels"].to_numpy(dtype=np.int64)
        total_foreground = int(reference_foreground.sum())
        retained_count = max(1, int(np.ceil(0.8 * len(event))))
        null, null_method = random_retention_null(reference_foreground, retained_count, rng)
        for score in (
            "georisk_v2",
            "soft_dice_risk_binary",
            "predicted_foreground_fraction",
            "predicted_foreground_scarcity",
        ):
            ordered = event.sort_values(score, kind="mergesort")
            retained = ordered.iloc[:retained_count]
            foreground_fraction = float(
                retained["reference_foreground_pixels"].sum() / total_foreground
            )
            valid_fraction = float(
                retained["valid_pixels"].sum() / event["valid_pixels"].sum()
            )
            if null_method == "exact":
                p_value = float(np.mean(null <= foreground_fraction + 1e-15))
            else:
                p_value = float(
                    (np.count_nonzero(null <= foreground_fraction) + 1) / (len(null) + 1)
                )
            rows.append(
                {
                    "block": block,
                    "partition": partition,
                    "seed": int(seed),
                    "event_id": event_id,
                    "score": score,
                    "tiles": int(len(event)),
                    "retained_tiles": retained_count,
                    "tile_count_coverage": retained_count / len(event),
                    "valid_pixel_coverage": valid_fraction,
                    "reference_foreground_coverage": foreground_fraction,
                    "foreground_minus_count_coverage": (
                        foreground_fraction - retained_count / len(event)
                    ),
                    "foreground_to_count_coverage_ratio": (
                        foreground_fraction / (retained_count / len(event))
                    ),
                    "null_method": null_method,
                    "one_sided_p_below_random": p_value,
                    "null_mean": float(null.mean()),
                }
            )
    return pd.DataFrame(rows)


def hazard_summary(retention: pd.DataFrame) -> pd.DataFrame:
    return (
        retention.groupby(["block", "score"], as_index=False)
        .agg(
            run_events=("event_id", "size"),
            mean_tile_count_coverage=("tile_count_coverage", "mean"),
            mean_valid_pixel_coverage=("valid_pixel_coverage", "mean"),
            mean_reference_foreground_coverage=("reference_foreground_coverage", "mean"),
            median_reference_foreground_coverage=("reference_foreground_coverage", "median"),
            mean_foreground_minus_count=("foreground_minus_count_coverage", "mean"),
            run_events_below_random_p05=(
                "one_sided_p_below_random",
                lambda values: int((values < 0.05).sum()),
            ),
        )
    )


def hazard_coverage_curves(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scores = (
        "georisk_v2",
        "soft_dice_risk_binary",
        "predicted_foreground_fraction",
        "predicted_foreground_scarcity",
    )
    for (block, partition, seed, event_id), event in frame.groupby(
        ["block", "partition", "seed", "event_id"], sort=True
    ):
        total_reference = float(event["reference_foreground_pixels"].sum())
        total_prediction = float(event["predicted_foreground_pixels"].sum())
        total_valid = float(event["valid_pixels"].sum())
        for score in scores:
            ordered = event.sort_values(score, kind="mergesort").reset_index(drop=True)
            for coverage in GRID:
                retained_count = max(1, int(np.ceil(coverage * len(ordered))))
                retained = ordered.iloc[:retained_count]
                rows.append(
                    {
                        "block": block,
                        "partition": partition,
                        "seed": int(seed),
                        "event_id": event_id,
                        "score": score,
                        "coverage": float(coverage),
                        "realized_tile_count_coverage": retained_count / len(ordered),
                        "valid_pixel_coverage": float(
                            retained["valid_pixels"].sum() / total_valid
                        ),
                        "reference_foreground_coverage": float(
                            retained["reference_foreground_pixels"].sum()
                            / total_reference
                        ),
                        "predicted_foreground_coverage": float(
                            retained["predicted_foreground_pixels"].sum()
                            / total_prediction
                        ),
                    }
                )
    return pd.DataFrame(rows)


def event_content_coupling(
    curves: pd.DataFrame,
    hazard_curves: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = curves[
        (curves["subset"] == "all_tiles")
        & (curves["event_id"] != "__event_macro__")
        & (curves["endpoint"] == "present_class_miou_risk")
        & (curves["score"].isin(["georisk_v2", "soft_dice_risk_binary"]))
    ]
    event_aurc = (
        selected.groupby(
            ["block", "partition", "seed", "event_id", "score"], as_index=False
        )
        .apply(
            lambda group: pd.Series(
                {
                    "event_aurc": float(
                        np.trapezoid(
                            group.sort_values("coverage")["risk"],
                            group.sort_values("coverage")["coverage"],
                        )
                    )
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    aurc_pivot = event_aurc.pivot(
        index=["block", "partition", "seed", "event_id"],
        columns="score",
        values="event_aurc",
    ).reset_index()
    aurc_pivot["aurc_delta_v2_minus_self_overlap"] = (
        aurc_pivot["georisk_v2"] - aurc_pivot["soft_dice_risk_binary"]
    )

    content = hazard_curves[
        hazard_curves["score"].isin(["georisk_v2", "soft_dice_risk_binary"])
    ].copy()
    at_eighty = content[np.isclose(content["coverage"], 0.8)].pivot(
        index=["block", "partition", "seed", "event_id"],
        columns="score",
        values="reference_foreground_coverage",
    ).reset_index()
    at_eighty["foreground_coverage_delta_v2_minus_self_overlap_at_0_8"] = (
        at_eighty["georisk_v2"] - at_eighty["soft_dice_risk_binary"]
    )
    area = (
        content.groupby(
            ["block", "partition", "seed", "event_id", "score"], as_index=False
        )
        .apply(
            lambda group: pd.Series(
                {
                    "area_under_foreground_coverage": float(
                        np.trapezoid(
                            group.sort_values("coverage")["reference_foreground_coverage"],
                            group.sort_values("coverage")["coverage"],
                        )
                    )
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    area_pivot = area.pivot(
        index=["block", "partition", "seed", "event_id"],
        columns="score",
        values="area_under_foreground_coverage",
    ).reset_index()
    area_pivot["foreground_coverage_area_delta_v2_minus_self_overlap"] = (
        area_pivot["georisk_v2"] - area_pivot["soft_dice_risk_binary"]
    )
    keys = ["block", "partition", "seed", "event_id"]
    run_event = (
        aurc_pivot[keys + ["aurc_delta_v2_minus_self_overlap"]]
        .merge(
            at_eighty[
                keys
                + ["foreground_coverage_delta_v2_minus_self_overlap_at_0_8"]
            ],
            on=keys,
            validate="one_to_one",
        )
        .merge(
            area_pivot[
                keys + ["foreground_coverage_area_delta_v2_minus_self_overlap"]
            ],
            on=keys,
            validate="one_to_one",
        )
    )
    by_event = (
        run_event.groupby(["block", "event_id"], as_index=False)[
            [
                "aurc_delta_v2_minus_self_overlap",
                "foreground_coverage_delta_v2_minus_self_overlap_at_0_8",
                "foreground_coverage_area_delta_v2_minus_self_overlap",
            ]
        ]
        .mean()
    )
    summary_rows = []
    for block, group in by_event.groupby("block", sort=True):
        for content_delta in (
            "foreground_coverage_delta_v2_minus_self_overlap_at_0_8",
            "foreground_coverage_area_delta_v2_minus_self_overlap",
        ):
            rho, p_value = spearmanr(
                group["aurc_delta_v2_minus_self_overlap"], group[content_delta]
            )
            summary_rows.append(
                {
                    "block": block,
                    "content_delta": content_delta,
                    "events": int(len(group)),
                    "spearman_rho": float(rho),
                    "two_sided_p_uncorrected": float(p_value),
                    "mean_aurc_delta_v2_minus_self_overlap": float(
                        group["aurc_delta_v2_minus_self_overlap"].mean()
                    ),
                    "mean_content_delta_v2_minus_self_overlap": float(
                        group[content_delta].mean()
                    ),
                }
            )
    return run_event, by_event, pd.DataFrame(summary_rows)


def empirical_cdf(reference, query):
    ordered = np.sort(np.asarray(reference, dtype=np.float64))
    return np.searchsorted(
        ordered, np.asarray(query, dtype=np.float64), side="right"
    ) / ordered.size


def fit_loeo_ridge(train_ranks: np.ndarray, train: pd.DataFrame) -> np.ndarray:
    event_sizes = train.groupby("event_id")["event_id"].transform("size").to_numpy()
    sample_weights = 1.0 / event_sizes
    sample_weights = sample_weights / sample_weights.mean()
    weighted_design = train_ranks * np.sqrt(sample_weights)[:, None]
    weighted_target = train["tile_loss"].to_numpy() * np.sqrt(sample_weights)
    coefficients = np.full(4, 0.25, dtype=np.float64)
    for _ in range(100):
        for index in range(4):
            residual = weighted_target - weighted_design @ coefficients
            residual += weighted_design[:, index] * coefficients[index]
            numerator = weighted_design[:, index] @ residual
            denominator = weighted_design[:, index] @ weighted_design[:, index] + 0.25
            coefficients[index] = max(0.0, numerator / denominator)
    return coefficients / coefficients.sum()


def calibration_loeo_endpoint_comparison(
    calibration: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    components = [
        "soft_dice_risk_binary",
        "uncertainty",
        "representation_shift",
        "spatial_inconsistency",
    ]
    columns = confusion_columns(calibration)
    rows = []
    for (block, partition, seed), run in calibration.groupby(
        ["block", "partition", "seed"], sort=True
    ):
        for held_event in sorted(run["event_id"].astype(str).unique()):
            train = run[run["event_id"].astype(str) != held_event].reset_index(drop=True)
            held = run[run["event_id"].astype(str) == held_event].reset_index(drop=True)
            train_ranks = np.column_stack(
                [empirical_cdf(train[component], train[component]) for component in components]
            )
            held_ranks = np.column_stack(
                [empirical_cdf(train[component], held[component]) for component in components]
            )
            score_values = {
                "soft_dice": held["soft_dice_risk_binary"].to_numpy(dtype=float),
                "nonnegative_ridge": held_ranks @ fit_loeo_ridge(train_ranks, train),
            }
            full_confusion = confusion_from_rows(held, columns)
            diagonal = np.diag(full_confusion)
            full_union = (
                diagonal
                + full_confusion.sum(axis=0)
                - diagonal
                + full_confusion.sum(axis=1)
                - diagonal
            )
            full_classes = full_union > 0
            for method, values in score_values.items():
                ordered = held.assign(_score=values).sort_values(
                    "_score", kind="mergesort"
                )
                endpoint_values = {"tile_mean_present_class_miou_risk": []}
                endpoint_values.update({endpoint: [] for endpoint in ENDPOINTS})
                for coverage in GRID:
                    retained_count = max(1, int(np.ceil(coverage * len(ordered))))
                    retained = ordered.iloc[:retained_count]
                    endpoint_values["tile_mean_present_class_miou_risk"].append(
                        float(retained["tile_loss"].mean())
                    )
                    risks = endpoint_risks(
                        confusion_from_rows(retained, columns), full_classes
                    )
                    for endpoint, value in risks.items():
                        endpoint_values[endpoint].append(value)
                for endpoint, values_at_coverage in endpoint_values.items():
                    rows.append(
                        {
                            "block": block,
                            "partition": partition,
                            "seed": int(seed),
                            "held_event": held_event,
                            "tiles": int(len(held)),
                            "method": method,
                            "endpoint": endpoint,
                            "aurc": float(np.trapezoid(values_at_coverage, GRID)),
                        }
                    )
    results = pd.DataFrame(rows)
    summary = (
        results.groupby(["block", "method", "endpoint"], as_index=False)
        .agg(mean_aurc=("aurc", "mean"), sd_aurc=("aurc", "std"), held_events=("held_event", "size"))
    )
    pivot = results.pivot(
        index=["block", "partition", "seed", "held_event", "endpoint"],
        columns="method",
        values="aurc",
    ).reset_index()
    pivot["ridge_minus_self_overlap"] = (
        pivot["nonnegative_ridge"] - pivot["soft_dice"]
    )
    effects = (
        pivot.groupby(["block", "endpoint"], as_index=False)
        .agg(
            mean_delta=("ridge_minus_self_overlap", "mean"),
            sd_delta=("ridge_minus_self_overlap", "std"),
            held_events=("held_event", "size"),
            held_events_favoring_ridge=(
                "ridge_minus_self_overlap", lambda values: int((values < 0).sum())
            ),
        )
    )
    return results, summary, effects


def ridge_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    components = [
        "soft_dice_risk_binary",
        "uncertainty",
        "representation_shift",
        "spatial_inconsistency",
    ]
    columns = confusion_columns(frame)
    for (block, partition, seed), run in frame.groupby(
        ["block", "partition", "seed"], sort=True
    ):
        run_dir = RUNS / block / str(partition).lower() / f"seed_{int(seed)}" / "risk_v2"
        calibration = read_tile_scores(
            run_dir / "predictions" / "calibration_tile_scores.parquet"
        )
        calibration_rows = calibration.to_dict("records")
        calibration_events = np.asarray(calibration["event_id"], dtype=str)
        for ridge in (0.0, 0.25, 1.0):
            for sweeps in (25, 100, 400):
                weights, calibration_values, calibration_event_ids = (
                    fit_event_balanced_nonnegative_ridge(
                        calibration_rows,
                        components,
                        True,
                        ridge,
                        sweeps,
                    )
                )
                scored = run.copy()
                values = []
                for record in scored.to_dict("records"):
                    calibrated = np.asarray(
                        [
                            event_balanced_ecdf(
                                record[component],
                                calibration_values[component],
                                calibration_event_ids,
                                True,
                            )
                            for component in components
                        ],
                        dtype=float,
                    )
                    values.append(float(calibrated @ weights))
                scored["sensitivity_score"] = values
                macro_present = []
                macro_fixed = []
                for coverage in GRID:
                    event_present = []
                    event_fixed = []
                    for _, event in scored.groupby("event_id", sort=True):
                        full_confusion = confusion_from_rows(event, columns)
                        true_positive = np.diag(full_confusion)
                        full_union = (
                            true_positive
                            + full_confusion.sum(axis=0)
                            - true_positive
                            + full_confusion.sum(axis=1)
                            - true_positive
                        )
                        ordered = event.sort_values("sensitivity_score", kind="mergesort")
                        retained_count = max(1, int(np.ceil(coverage * len(ordered))))
                        confusion = confusion_from_rows(ordered.iloc[:retained_count], columns)
                        risks = endpoint_risks(confusion, full_union > 0)
                        event_present.append(risks["present_class_miou_risk"])
                        event_fixed.append(risks["fixed_full_event_class_miou_risk"])
                    macro_present.append(np.mean(event_present))
                    macro_fixed.append(np.mean(event_fixed))
                rows.append(
                    {
                        "block": block,
                        "partition": partition,
                        "seed": int(seed),
                        "ridge": ridge,
                        "sweeps": sweeps,
                        "weight_self_overlap": float(weights[0]),
                        "weight_uncertainty": float(weights[1]),
                        "weight_representation_shift": float(weights[2]),
                        "weight_spatial_inconsistency": float(weights[3]),
                        "present_class_miou_aurc": float(np.trapezoid(macro_present, GRID)),
                        "fixed_full_event_class_miou_aurc": float(
                            np.trapezoid(macro_fixed, GRID)
                        ),
                    }
                )
        print(f"Sensitivity completed: {block} {partition} seed={int(seed)}", flush=True)
    return pd.DataFrame(rows)


def ridge_sensitivity_summary(sensitivity: pd.DataFrame) -> pd.DataFrame:
    return (
        sensitivity.groupby(["block", "ridge", "sweeps"], as_index=False)
        .agg(
            present_class_miou_aurc_mean=("present_class_miou_aurc", "mean"),
            present_class_miou_aurc_sd=("present_class_miou_aurc", "std"),
            fixed_full_event_class_miou_aurc_mean=(
                "fixed_full_event_class_miou_aurc",
                "mean",
            ),
            fixed_full_event_class_miou_aurc_sd=(
                "fixed_full_event_class_miou_aurc",
                "std",
            ),
        )
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(OUTPUT / "tile_audit_scores.csv")
    frame["predicted_foreground_scarcity"] = (
        1.0 - frame["predicted_foreground_fraction"]
    )
    empty_both = (
        (frame["reference_foreground_pixels"] == 0)
        & (frame["predicted_foreground_pixels"] == 0)
    )
    full_curves, full_aurc = build_curves(
        frame,
        PRIMARY_SCORES,
        subset_name="all_tiles",
    )
    print("All-tile endpoint curves completed", flush=True)
    nonempty_curves, nonempty_aurc = build_curves(
        frame[~empty_both].copy(),
        ("georisk_v2", "soft_dice_risk_binary"),
        subset_name="exclude_reference_and_prediction_empty",
    )
    print("Empty-tile sensitivity curves completed", flush=True)
    curves = pd.concat([full_curves, nonempty_curves], ignore_index=True)
    aurc = pd.concat([full_aurc, nonempty_aurc], ignore_index=True)
    summary = summarize_aurc(aurc)
    print("AURC summary completed", flush=True)
    effects = paired_effects(aurc)
    print("Paired endpoint effects completed", flush=True)
    decomposition = coverage_decomposition(curves)
    print("Coverage decomposition completed", flush=True)
    correlations = score_correlations(frame)
    correlations_summary = correlation_summary(correlations)
    print("Run-event correlations completed", flush=True)
    retention = hazard_retention(frame)
    retention_summary = hazard_summary(retention)
    print("Hazard-area retention nulls completed", flush=True)
    hazard_curves = hazard_coverage_curves(frame)
    coupling_run_event, coupling_event, coupling_summary = event_content_coupling(
        curves, hazard_curves
    )
    print("Risk-content coupling analysis completed", flush=True)
    calibration = pd.read_csv(OUTPUT / "calibration_tile_audit_scores.csv")
    loeo_results, loeo_summary, loeo_effects = calibration_loeo_endpoint_comparison(
        calibration
    )
    print("Calibration LOEO endpoint comparison completed", flush=True)
    sensitivity = ridge_sensitivity(frame)
    sensitivity_summary = ridge_sensitivity_summary(sensitivity)

    curves.to_csv(OUTPUT / "endpoint_risk_coverage_by_run_event.csv", index=False)
    aurc.to_csv(OUTPUT / "endpoint_aurc_by_run.csv", index=False)
    summary.to_csv(OUTPUT / "endpoint_aurc_summary.csv", index=False)
    effects.to_csv(OUTPUT / "endpoint_paired_effects.csv", index=False)
    decomposition.to_csv(OUTPUT / "coverage_gain_decomposition_endpoints.csv", index=False)
    correlations.to_csv(OUTPUT / "score_correlations_by_run_event.csv", index=False)
    correlations_summary.to_csv(OUTPUT / "score_correlation_summary.csv", index=False)
    retention.to_csv(OUTPUT / "hazard_area_retention_by_run_event.csv", index=False)
    retention_summary.to_csv(OUTPUT / "hazard_area_retention_summary.csv", index=False)
    hazard_curves.to_csv(OUTPUT / "hazard_coverage_curves.csv", index=False)
    coupling_run_event.to_csv(
        OUTPUT / "risk_content_coupling_by_run_event.csv", index=False
    )
    coupling_event.to_csv(OUTPUT / "risk_content_coupling_by_event.csv", index=False)
    coupling_summary.to_csv(OUTPUT / "risk_content_coupling_summary.csv", index=False)
    loeo_results.to_csv(OUTPUT / "calibration_loeo_endpoint_results.csv", index=False)
    loeo_summary.to_csv(OUTPUT / "calibration_loeo_endpoint_summary.csv", index=False)
    loeo_effects.to_csv(OUTPUT / "calibration_loeo_endpoint_effects.csv", index=False)
    sensitivity.to_csv(OUTPUT / "ridge_sweep_sensitivity_by_run.csv", index=False)
    sensitivity_summary.to_csv(OUTPUT / "ridge_sweep_sensitivity_summary.csv", index=False)

    headline_effects = effects[
        (effects["subset"] == "all_tiles")
        & (effects["comparison"] == "georisk_v2_minus_soft_dice_risk_binary")
    ].to_dict("records")
    report = {
        "analysis_status": "post_hoc_exploratory_endpoint_validity_audit",
        "tile_rows": int(len(frame)),
        "reference_and_prediction_empty_rows": int(empty_both.sum()),
        "reference_and_prediction_empty_fraction": float(empty_both.mean()),
        "headline_paired_effects": headline_effects,
        "correlation_summary": correlations_summary.to_dict("records"),
        "hazard_retention_summary": retention_summary.to_dict("records"),
        "risk_content_coupling_summary": coupling_summary.to_dict("records"),
        "calibration_loeo_endpoint_effects": loeo_effects.to_dict("records"),
        "ridge_sensitivity_range": (
            sensitivity_summary.groupby("block")
            .agg(
                present_aurc_min=("present_class_miou_aurc_mean", "min"),
                present_aurc_max=("present_class_miou_aurc_mean", "max"),
                fixed_aurc_min=("fixed_full_event_class_miou_aurc_mean", "min"),
                fixed_aurc_max=("fixed_full_event_class_miou_aurc_mean", "max"),
            )
            .reset_index()
            .to_dict("records")
        ),
    }
    (OUTPUT / "endpoint_validity_summary.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
