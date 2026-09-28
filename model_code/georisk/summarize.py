import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEPLOYABLE_SCORES = [
    "top_tail_msp",
    "uncertainty",
    "soft_dice_risk_binary",
]


def paired_bootstrap_interval(differences, repetitions=5000, seed=20260824):
    differences = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.choice(
        differences, size=(repetitions, differences.size), replace=True
    ).mean(axis=1)
    return np.quantile(samples, [0.025, 0.975])


def event_aurc_table(curves):
    event_curves = curves[curves["event_id"] != "__event_macro__"].copy()
    rows = []
    group_columns = ["block", "dataset", "model", "partition", "seed", "score", "event_id"]
    for keys, group in event_curves.groupby(group_columns):
        group = group.sort_values("coverage")
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "event_aurc": float(np.trapezoid(group["risk"], group["coverage"])),
            }
        )
    return pd.DataFrame(rows)


def primary_effects(event_aurc):
    rows = []
    for block in ["P1", "P2"]:
        block_table = event_aurc[event_aurc["block"] == block]
        averaged = (
            block_table.groupby(["model", "score", "event_id"], as_index=False)["event_aurc"]
            .mean()
        )
        available_baselines = [
            score for score in DEPLOYABLE_SCORES if score in set(averaged["score"])
        ]
        baseline_means = (
            averaged[averaged["score"].isin(available_baselines)]
            .groupby("score")["event_aurc"]
            .mean()
        )
        strongest_baseline = baseline_means.idxmin()
        georisk = averaged[averaged["score"] == "georisk_full"].set_index("event_id")
        baseline = averaged[averaged["score"] == strongest_baseline].set_index("event_id")
        paired_events = georisk.index.intersection(baseline.index)
        differences = (
            georisk.loc[paired_events, "event_aurc"]
            - baseline.loc[paired_events, "event_aurc"]
        ).to_numpy()
        lower, upper = paired_bootstrap_interval(differences)
        rows.append(
            {
                "block": block,
                "model": averaged["model"].iloc[0],
                "comparison": f"georisk_full - {strongest_baseline}",
                "strongest_baseline": strongest_baseline,
                "event_count": differences.size,
                "mean_paired_delta_aurc": differences.mean(),
                "ci95_lower": lower,
                "ci95_upper": upper,
                "hypothesis_supported": bool(differences.mean() < 0 and upper < 0),
            }
        )
    return pd.DataFrame(rows)


def summarize(root):
    root = Path(root)
    results_dir = root / "results"
    metrics = pd.read_csv(results_dir / "georisk_metrics.csv")
    curves = pd.read_csv(results_dir / "risk_coverage.csv")

    metric_columns = [
        "event_macro_aurc",
        "selective_risk_at_calibrated_80pct_threshold",
        "realized_coverage_at_calibrated_80pct_threshold",
        "worst_event_selective_risk",
        "score_loss_spearman",
        "full_coverage_miou",
        "full_coverage_macro_f1",
    ]
    group_columns = ["block", "dataset", "model", "partition", "score"]
    summary = metrics.groupby(group_columns)[metric_columns].agg(["mean", "std"])
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    summary.reset_index().to_csv(results_dir / "georisk_seed_summary.csv", index=False)

    event_aurc = event_aurc_table(curves)
    event_aurc.to_csv(results_dir / "event_aurc.csv", index=False)
    primary_effects(event_aurc).to_csv(results_dir / "primary_effects.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize completed GeoRisk evaluations")
    parser.add_argument("--root", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    summarize(arguments.root)
