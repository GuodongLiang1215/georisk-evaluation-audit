import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import load_config, resolve_root, run_directory
from .data import DATASET_INFO, GeoRiskDataset, build_datasets
from .models import build_model
from .risk import (
    add_adaptive_risk_score,
    add_composite_scores,
    aggregate_selective_metrics,
    event_balanced_quantile,
    fit_temperature,
    raw_scores,
    representation_distances,
    score_names,
)


def native_copy(dataset):
    return GeoRiskDataset(
        dataset.readers,
        dataset.dataset_name,
        dataset.sample_ids,
        dataset.partition,
        dataset.crop_size,
        mode="native",
    )


@torch.inference_mode()
def predict_embeddings(model, dataset, device, amp_enabled):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    embeddings = []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            _, embedding = model(image)
        embeddings.append(embedding[0].float().cpu().numpy())
    return np.stack(embeddings)


@torch.inference_mode()
def predict_records(model, dataset, device, amp_enabled):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    records = []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            logits, embedding = model(image)
        records.append(
            {
                "sample_id": batch["sample_id"][0],
                "event_id": batch["event_id"][0],
                "logits": logits[0].float().cpu().numpy().astype(np.float16),
                "embedding": embedding[0].float().cpu().numpy(),
                "target": batch["target"][0].numpy().astype(np.uint8),
                "input_valid_mask": batch["input_valid_mask"][0].numpy(),
                "evaluation_mask": batch["evaluation_mask"][0].numpy(),
            }
        )
    return records


def json_ready_component_calibration(values, events):
    return {
        component: {
            "values": [float(value) for value in component_values],
            "events": [str(event) for event in events],
        }
        for component, component_values in values.items()
    }


def rows_for_table(rows):
    return [
        {
            key: value
            for key, value in row.items()
            if key != "confusion"
        }
        for row in rows
    ]


def evaluate_one(config, block_name, partition, seed, risk_v2_config=None):
    root = resolve_root(config)
    block = config["blocks"][block_name]
    dataset_name = block["dataset"]
    model_name = block["model"]
    dataset_info = DATASET_INFO[dataset_name]
    model_config = config["models"][model_name]
    training_config = config["training"]
    output_dir = run_directory(root, block_name, partition, seed)
    evaluation_dir = (
        output_dir / risk_v2_config["output_subdirectory"]
        if risk_v2_config
        else output_dir
    )
    prediction_dir = evaluation_dir / "predictions"
    calibration_dir = evaluation_dir / "calibration"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    calibration_dir.mkdir(parents=True, exist_ok=True)

    datasets = build_datasets(
        root,
        dataset_name,
        partition,
        seed,
        training_config["internal_validation_fraction_per_train_event"],
        training_config["crop_size"][0],
    )
    device = torch.device("cuda")
    model = build_model(model_name, dataset_info, root, model_config).to(device)
    checkpoint = torch.load(
        output_dir / "checkpoints" / "best.pt",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    amp_enabled = bool(config["environment"]["mixed_precision"])

    reference_embeddings = predict_embeddings(
        model, native_copy(datasets["train"]), device, amp_enabled
    )
    calibration_records = predict_records(
        model, datasets["calibration"], device, amp_enabled
    )
    test_records = predict_records(model, datasets["test"], device, amp_enabled)

    event_balanced = bool(dataset_info["event_grouped"])
    temperature = fit_temperature(calibration_records, event_balanced)
    calibration_distances = representation_distances(
        reference_embeddings,
        [record["embedding"] for record in calibration_records],
        config["risk"]["representation_shift"]["nearest_neighbors"],
    )
    test_distances = representation_distances(
        reference_embeddings,
        [record["embedding"] for record in test_records],
        config["risk"]["representation_shift"]["nearest_neighbors"],
    )
    tail_fraction = config["risk"]["uncertainty"]["upper_tail_fraction"]
    calibration_rows = raw_scores(
        calibration_records, temperature, calibration_distances, tail_fraction
    )
    test_rows = raw_scores(test_records, temperature, test_distances, tail_fraction)
    component_values = add_composite_scores(
        calibration_rows, test_rows, event_balanced
    )
    adaptive_fit = None
    additional_scores = None
    if risk_v2_config:
        component_key = "binary" if dataset_info["classes"] == 2 else "multiclass"
        components = risk_v2_config["components"][component_key]
        weight_fit = risk_v2_config["weight_fit"]
        adaptive_fit = add_adaptive_risk_score(
            calibration_rows,
            test_rows,
            components,
            event_balanced,
            risk_v2_config["score_name"],
            weight_fit["ridge"],
            weight_fit["iterations"],
        )
        additional_scores = [risk_v2_config["score_name"]]

    coverage_config = config["selective_prediction"]["coverage_grid"]
    coverage_grid = np.arange(
        coverage_config["start"],
        coverage_config["stop"] + coverage_config["step"] / 2,
        coverage_config["step"],
    )
    calibration_events = [row["event_id"] for row in calibration_rows]
    metric_rows = []
    curve_rows = []
    thresholds = {}
    for score_name in score_names(dataset_info["classes"], additional_scores):
        threshold = event_balanced_quantile(
            [row[score_name] for row in calibration_rows],
            calibration_events,
            config["selective_prediction"]["threshold_quantile"],
            event_balanced,
        )
        thresholds[score_name] = threshold
        metrics = aggregate_selective_metrics(
            test_rows, score_name, threshold, coverage_grid, dataset_info["classes"]
        )
        metric_rows.append(
            {
                "block": block_name,
                "dataset": dataset_name,
                "model": model_name,
                "partition": partition,
                "seed": seed,
                "score": score_name,
                "threshold": threshold,
                **{key: value for key, value in metrics.items() if key != "curve_rows"},
            }
        )
        for row in metrics["curve_rows"]:
            curve_rows.append(
                {
                    "block": block_name,
                    "dataset": dataset_name,
                    "model": model_name,
                    "partition": partition,
                    "seed": seed,
                    **row,
                }
            )

    pd.DataFrame(rows_for_table(calibration_rows)).to_parquet(
        prediction_dir / "calibration_tile_scores.parquet", index=False
    )
    pd.DataFrame(rows_for_table(test_rows)).to_parquet(
        prediction_dir / "test_tile_scores.parquet", index=False
    )
    pd.DataFrame(metric_rows).to_csv(evaluation_dir / "score_metrics.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(evaluation_dir / "risk_coverage.csv", index=False)
    (calibration_dir / "calibration.json").write_text(
        json.dumps(
            {
                "temperature": temperature,
                "event_balanced": event_balanced,
                "target_coverage": config["selective_prediction"]["threshold_quantile"],
                "thresholds": thresholds,
                "component_calibration": json_ready_component_calibration(
                    component_values, calibration_events
                ),
                "adaptive_fit": adaptive_fit,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[{block_name} {partition} seed={seed}] evaluation completed; "
        f"temperature={temperature:.5f}"
    )


def consolidate_results(root, risk_v2_config=None):
    if risk_v2_config:
        subdirectory = risk_v2_config["output_subdirectory"]
        metric_files = sorted(
            (Path(root) / "runs").glob(f"*/*/seed_*/{subdirectory}/score_metrics.csv")
        )
        curve_files = sorted(
            (Path(root) / "runs").glob(f"*/*/seed_*/{subdirectory}/risk_coverage.csv")
        )
        metric_output = risk_v2_config["aggregate_metrics"]
        curve_output = risk_v2_config["aggregate_risk_coverage"]
    else:
        metric_files = sorted((Path(root) / "runs").glob("*/*/seed_*/score_metrics.csv"))
        curve_files = sorted((Path(root) / "runs").glob("*/*/seed_*/risk_coverage.csv"))
        metric_output = "georisk_metrics.csv"
        curve_output = "risk_coverage.csv"
    results_dir = Path(root) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.concat([pd.read_csv(path) for path in metric_files], ignore_index=True)
    curves = pd.concat([pd.read_csv(path) for path in curve_files], ignore_index=True)
    metrics.to_csv(results_dir / metric_output, index=False)
    curves.to_csv(results_dir / curve_output, index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GeoRisk experiment blocks")
    parser.add_argument("--config", default="configs/georisk_experiment_v1.yaml")
    parser.add_argument("--block", required=True, choices=["P1", "P2", "E1", "E2", "E3"])
    parser.add_argument("--partition")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--risk-config")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    risk_v2_config = load_config(args.risk_config) if args.risk_config else None
    block = config["blocks"][args.block]
    partitions = [args.partition] if args.partition else block["partitions"]
    seeds = [args.seed] if args.seed is not None else config["training"]["seeds"]
    for partition in partitions:
        for seed in seeds:
            evaluate_one(config, args.block, partition, seed, risk_v2_config)
    consolidate_results(resolve_root(config), risk_v2_config)


if __name__ == "__main__":
    main()
