from collections import defaultdict

import numpy as np
from scipy.ndimage import convolve
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.stats import rankdata

from .metrics import metrics_from_confusion


def softmax(logits, temperature=1.0):
    scaled = np.asarray(logits, dtype=np.float32) / float(temperature)
    scaled = scaled - scaled.max(axis=0, keepdims=True)
    exponent = np.exp(scaled)
    return exponent / exponent.sum(axis=0, keepdims=True)


def fit_temperature(records, event_balanced):
    def objective(log_temperature):
        temperature = float(np.exp(log_temperature))
        loss_sum = defaultdict(float)
        pixel_count = defaultdict(int)
        for record in records:
            logits = np.asarray(record["logits"], dtype=np.float32) / temperature
            target = record["target"]
            valid = record["evaluation_mask"]
            selected_logits = logits[:, valid].T
            selected_target = target[valid].astype(np.int64)
            losses = logsumexp(selected_logits, axis=1) - selected_logits[
                np.arange(selected_target.size), selected_target
            ]
            key = str(record["event_id"]) if event_balanced else "__all__"
            loss_sum[key] += float(losses.sum())
            pixel_count[key] += int(losses.size)
        return float(np.mean([loss_sum[key] / pixel_count[key] for key in loss_sum]))

    result = minimize_scalar(
        objective,
        bounds=(np.log(0.05), np.log(10.0)),
        method="bounded",
        options={"xatol": 1e-4},
    )
    return float(np.exp(result.x))


def upper_tail_mean(values, fraction=0.10):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    count = max(1, int(np.ceil(values.size * fraction)))
    boundary = values.size - count
    return float(np.partition(values, boundary)[boundary:].mean())


def uncertainty_score(probabilities, input_valid_mask, fraction=0.10):
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-7, 1.0))).sum(axis=0)
    entropy = entropy / np.log(probabilities.shape[0])
    return upper_tail_mean(entropy[input_valid_mask], fraction)


def msp_score(probabilities, input_valid_mask, fraction=0.10):
    risk = 1.0 - probabilities.max(axis=0)
    return upper_tail_mean(risk[input_valid_mask], fraction)


def spatial_inconsistency_score(probabilities, input_valid_mask, fraction=0.10):
    kernel = np.ones((3, 3), dtype=np.float32)
    valid = input_valid_mask.astype(np.float32)
    neighbor_count = convolve(valid, kernel, mode="constant", cval=0.0)
    local = np.stack(
        [
            convolve(channel * valid, kernel, mode="constant", cval=0.0)
            / np.maximum(neighbor_count, 1.0)
            for channel in probabilities
        ]
    )
    midpoint = 0.5 * (probabilities + local)
    first = (
        probabilities
        * (np.log(np.clip(probabilities, 1e-7, 1.0)) - np.log(np.clip(midpoint, 1e-7, 1.0)))
    ).sum(axis=0)
    second = (
        local
        * (np.log(np.clip(local, 1e-7, 1.0)) - np.log(np.clip(midpoint, 1e-7, 1.0)))
    ).sum(axis=0)
    divergence = 0.5 * (first + second) / np.log(2.0)
    return upper_tail_mean(divergence[input_valid_mask], fraction)


def soft_dice_risk(probabilities, input_valid_mask):
    foreground_probability = probabilities[1]
    prediction = probabilities.argmax(axis=0) == 1
    valid_prediction = prediction & input_valid_mask
    numerator = 2.0 * foreground_probability[valid_prediction].sum()
    denominator = valid_prediction.sum() + foreground_probability[input_valid_mask].sum()
    confidence = float((numerator + 1e-7) / (denominator + 1e-7))
    return 1.0 - confidence


def multiclass_self_dice_risk(probabilities, input_valid_mask):
    prediction = probabilities.argmax(axis=0)
    confidences = []
    for class_index in range(probabilities.shape[0]):
        predicted_class = (prediction == class_index) & input_valid_mask
        if not predicted_class.any():
            continue
        numerator = 2.0 * probabilities[class_index][predicted_class].sum()
        denominator = (
            predicted_class.sum()
            + probabilities[class_index][input_valid_mask].sum()
        )
        confidences.append(float((numerator + 1e-7) / (denominator + 1e-7)))
    return 1.0 - float(np.mean(confidences))


def normalized_embeddings(embeddings):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    return embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)


def representation_distances(reference_embeddings, query_embeddings, neighbors=5):
    reference = normalized_embeddings(reference_embeddings)
    query = normalized_embeddings(query_embeddings)
    output = []
    for embedding in query:
        distances = np.clip(
            1.0 - np.sum(reference * embedding[None, :], axis=1),
            0.0,
            2.0,
        )
        nearest = np.partition(distances, neighbors - 1)[:neighbors]
        output.append(np.median(nearest))
    return np.asarray(output, dtype=np.float32)


def spearman_rank_correlation(first, second):
    first_ranks = rankdata(np.asarray(first, dtype=np.float64), method="average")
    second_ranks = rankdata(np.asarray(second, dtype=np.float64), method="average")
    first_centered = first_ranks - first_ranks.mean()
    second_centered = second_ranks - second_ranks.mean()
    denominator = np.sqrt(
        np.sum(first_centered * first_centered)
        * np.sum(second_centered * second_centered)
    )
    if denominator == 0.0:
        return float("nan")
    return float(np.sum(first_centered * second_centered) / denominator)


def event_balanced_ecdf(value, calibration_values, calibration_events, event_balanced):
    calibration_values = np.asarray(calibration_values, dtype=np.float64)
    calibration_events = np.asarray(calibration_events, dtype=str)
    if not event_balanced:
        return float(np.mean(calibration_values <= value))
    event_values = []
    for event_id in np.unique(calibration_events):
        event_values.append(np.mean(calibration_values[calibration_events == event_id] <= value))
    return float(np.mean(event_values))


def event_balanced_quantile(values, events, quantile, event_balanced):
    values = np.asarray(values, dtype=np.float64)
    events = np.asarray(events, dtype=str)
    if not event_balanced:
        return float(np.quantile(values, quantile, method="higher"))
    weights = np.zeros(values.size, dtype=np.float64)
    unique_events = np.unique(events)
    for event_id in unique_events:
        mask = events == event_id
        weights[mask] = 1.0 / (unique_events.size * mask.sum())
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, quantile, side="left"))
    return float(values[order[min(index, values.size - 1)]])


def tile_confusion(record, probabilities):
    prediction = probabilities.argmax(axis=0)
    target = np.asarray(record["target"])
    valid = target != 255
    indices = target[valid].astype(np.int64) * probabilities.shape[0] + prediction[valid]
    return np.bincount(
        indices, minlength=probabilities.shape[0] ** 2
    ).reshape(probabilities.shape[0], probabilities.shape[0])


def raw_scores(records, temperature, shift_distances, tail_fraction):
    output = []
    for record, shift_distance in zip(records, shift_distances):
        probabilities = softmax(record["logits"], temperature)
        input_valid = record["input_valid_mask"]
        confusion = tile_confusion(record, probabilities)
        row = {
            "sample_id": str(record["sample_id"]),
            "event_id": str(record["event_id"]),
            "top_tail_msp": msp_score(probabilities, input_valid, tail_fraction),
            "uncertainty": uncertainty_score(probabilities, input_valid, tail_fraction),
            "representation_shift": float(shift_distance),
            "spatial_inconsistency": spatial_inconsistency_score(
                probabilities, input_valid, tail_fraction
            ),
            "tile_loss": 1.0 - metrics_from_confusion(confusion)["miou"],
            "confusion": confusion.tolist(),
        }
        if probabilities.shape[0] == 2:
            row["soft_dice_risk_binary"] = soft_dice_risk(probabilities, input_valid)
        else:
            row["self_dice_risk_multiclass"] = multiclass_self_dice_risk(
                probabilities, input_valid
            )
        output.append(row)
    return output


def add_composite_scores(calibration_rows, test_rows, event_balanced):
    components = ["uncertainty", "representation_shift", "spatial_inconsistency"]
    calibration_events = [row["event_id"] for row in calibration_rows]
    calibration_values = {
        component: [row[component] for row in calibration_rows]
        for component in components
    }
    for rows in (calibration_rows, test_rows):
        for row in rows:
            calibrated = {
                component: event_balanced_ecdf(
                    row[component],
                    calibration_values[component],
                    calibration_events,
                    event_balanced,
                )
                for component in components
            }
            row["uncertainty_plus_shift"] = 0.5 * (
                calibrated["uncertainty"] + calibrated["representation_shift"]
            )
            row["uncertainty_plus_spatial"] = 0.5 * (
                calibrated["uncertainty"] + calibrated["spatial_inconsistency"]
            )
            row["shift_plus_spatial"] = 0.5 * (
                calibrated["representation_shift"] + calibrated["spatial_inconsistency"]
            )
            row["georisk_full"] = sum(calibrated.values()) / 3.0
            row["oracle_tile_loss"] = row["tile_loss"]
    return calibration_values


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
            denominator = (
                weighted_design[:, index] @ weighted_design[:, index] + ridge
            )
            weights[index] = max(0.0, numerator / denominator)
    return weights / weights.sum(), calibration_values, calibration_events


def add_adaptive_risk_score(
    calibration_rows,
    test_rows,
    components,
    event_balanced,
    score_name,
    ridge,
    iterations,
):
    weights, calibration_values, calibration_events = (
        fit_event_balanced_nonnegative_ridge(
            calibration_rows,
            components,
            event_balanced,
            ridge,
            iterations,
        )
    )
    for rows in (calibration_rows, test_rows):
        for row in rows:
            calibrated = np.asarray(
                [
                    event_balanced_ecdf(
                        row[component],
                        calibration_values[component],
                        calibration_events,
                        event_balanced,
                    )
                    for component in components
                ],
                dtype=np.float64,
            )
            row[score_name] = float(calibrated @ weights)
    return {
        "score_name": score_name,
        "components": components,
        "weights": [float(weight) for weight in weights],
        "ridge": float(ridge),
        "iterations": int(iterations),
    }


def score_names(num_classes, additional_scores=None):
    names = [
        "top_tail_msp",
        "uncertainty",
        "representation_shift",
        "spatial_inconsistency",
        "uncertainty_plus_shift",
        "uncertainty_plus_spatial",
        "shift_plus_spatial",
        "georisk_full",
        "oracle_tile_loss",
    ]
    if num_classes == 2:
        names.insert(2, "soft_dice_risk_binary")
    else:
        names.insert(2, "self_dice_risk_multiclass")
    if additional_scores:
        names[-1:-1] = list(additional_scores)
    return names


def aggregate_selective_metrics(rows, score_name, threshold, coverage_grid, num_classes):
    groups = defaultdict(list)
    for row in rows:
        groups[row["event_id"]].append(row)

    curve_rows = []
    macro_curve = []
    for coverage in coverage_grid:
        event_risks = []
        for event_id, event_rows in groups.items():
            ordered = sorted(event_rows, key=lambda row: row[score_name])
            retained_count = max(1, int(np.ceil(coverage * len(ordered))))
            retained = ordered[:retained_count]
            confusion = np.sum(
                [np.asarray(row["confusion"], dtype=np.int64) for row in retained], axis=0
            )
            risk = 1.0 - metrics_from_confusion(confusion)["miou"]
            event_risks.append(risk)
            curve_rows.append(
                {
                    "score": score_name,
                    "event_id": event_id,
                    "coverage": coverage,
                    "risk": risk,
                }
            )
        macro_risk = float(np.mean(event_risks))
        macro_curve.append(macro_risk)
        curve_rows.append(
            {
                "score": score_name,
                "event_id": "__event_macro__",
                "coverage": coverage,
                "risk": macro_risk,
            }
        )

    fixed_event_risk = []
    fixed_event_coverage = []
    full_event_f1 = []
    for event_rows in groups.values():
        full_confusion = np.sum(
            [np.asarray(row["confusion"], dtype=np.int64) for row in event_rows], axis=0
        )
        full_event_f1.append(metrics_from_confusion(full_confusion)["macro_f1"])
        retained = [row for row in event_rows if row[score_name] <= threshold]
        fixed_event_coverage.append(len(retained) / len(event_rows))
        if retained:
            confusion = np.sum(
                [np.asarray(row["confusion"], dtype=np.int64) for row in retained], axis=0
            )
            fixed_event_risk.append(1.0 - metrics_from_confusion(confusion)["miou"])
        else:
            fixed_event_risk.append(1.0)
    losses = [row["tile_loss"] for row in rows]
    scores = [row[score_name] for row in rows]
    correlation = spearman_rank_correlation(scores, losses)
    return {
        "event_macro_aurc": float(np.trapezoid(macro_curve, coverage_grid)),
        "selective_risk_at_calibrated_80pct_threshold": float(np.mean(fixed_event_risk)),
        "realized_coverage_at_calibrated_80pct_threshold": float(
            np.mean(fixed_event_coverage)
        ),
        "worst_event_selective_risk": float(np.max(fixed_event_risk)),
        "score_loss_spearman": float(correlation),
        "full_coverage_risk": float(macro_curve[-1]),
        "full_coverage_miou": float(1.0 - macro_curve[-1]),
        "full_coverage_macro_f1": float(np.mean(full_event_f1)),
        "curve_rows": curve_rows,
    }
