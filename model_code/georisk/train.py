import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import load_config, resolve_root, run_directory
from .data import DATASET_INFO, build_datasets
from .losses import segmentation_loss
from .metrics import EventMetricAccumulator
from .models import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(dataset, batch_size, shuffle, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        generator=generator,
    )


@torch.inference_mode()
def evaluate_internal(model, loader, device, num_classes, amp_enabled):
    model.eval()
    accumulator = EventMetricAccumulator(num_classes)
    losses = []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        if not torch.any(target != 255):
            continue
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            logits, _ = model(image)
            loss = segmentation_loss(logits, target)
        losses.append(float(loss.item()))
        accumulator.update(logits, target, batch["event_id"])
    metrics = accumulator.compute()
    metrics["loss"] = float(np.mean(losses))
    return metrics


def train_epoch(model, loader, optimizer, scaler, device, accumulation_steps, amp_enabled):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for batch_index, batch in enumerate(loader):
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        if not torch.any(target != 255):
            continue
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            logits, _ = model(image)
            loss = segmentation_loss(logits, target)
            scaled_loss = loss / accumulation_steps
        scaler.scale(scaled_loss).backward()
        is_update = (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(loader)
        if is_update:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.item()))
    return float(np.mean(losses))


def write_split_manifest(path, datasets):
    payload = {
        split_name: dataset.sample_ids
        for split_name, dataset in datasets.items()
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_one(config, block_name, partition, seed):
    root = resolve_root(config)
    block = config["blocks"][block_name]
    dataset_name = block["dataset"]
    model_name = block["model"]
    dataset_info = DATASET_INFO[dataset_name]
    training_config = config["training"]
    model_config = config["models"][model_name]
    output_dir = run_directory(root, block_name, partition, seed)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    datasets = build_datasets(
        root,
        dataset_name,
        partition,
        seed,
        training_config["internal_validation_fraction_per_train_event"],
        training_config["crop_size"][0],
    )
    write_split_manifest(output_dir / "split_samples.json", datasets)

    train_loader = make_loader(
        datasets["train"], model_config["batch_size"], True, seed
    )
    validation_loader = make_loader(
        datasets["internal_validation"], model_config["batch_size"], False, seed
    )

    device = torch.device("cuda")
    model = build_model(model_name, dataset_info, root, model_config).to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=model_config["learning_rate"],
        weight_decay=model_config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=model_config["epochs"]
    )
    amp_enabled = bool(config["environment"]["mixed_precision"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    accumulation_steps = model_config["gradient_accumulation_steps"]

    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "epoch",
                "learning_rate",
                "train_loss",
                "validation_loss",
                "validation_event_macro_miou",
                "validation_event_macro_f1",
            ],
        )
        writer.writeheader()

    best_metric = -1.0
    for epoch in range(1, model_config["epochs"] + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            accumulation_steps,
            amp_enabled,
        )
        validation = evaluate_internal(
            model,
            validation_loader,
            device,
            dataset_info["classes"],
            amp_enabled,
        )
        learning_rate = optimizer.param_groups[0]["lr"]
        scheduler.step()

        row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_loss,
            "validation_loss": validation["loss"],
            "validation_event_macro_miou": validation["event_macro_miou"],
            "validation_event_macro_f1": validation["event_macro_f1"],
        }
        with metrics_path.open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=row.keys()).writerow(row)

        if validation["event_macro_miou"] > best_metric:
            best_metric = validation["event_macro_miou"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "validation_event_macro_miou": best_metric,
                    "block": block_name,
                    "partition": partition,
                    "seed": seed,
                    "dataset": dataset_name,
                    "model_name": model_name,
                },
                checkpoint_dir / "best.pt",
            )

        print(
            f"[{block_name} {partition} seed={seed}] "
            f"epoch={epoch}/{model_config['epochs']} "
            f"train_loss={train_loss:.5f} "
            f"val_mIoU={validation['event_macro_miou']:.5f}"
        )

    (output_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "best_validation_event_macro_miou": best_metric,
                "block": block_name,
                "partition": partition,
                "seed": seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train frozen GeoRisk experiment blocks")
    parser.add_argument("--config", default="configs/georisk_experiment_v1.yaml")
    parser.add_argument("--block", required=True, choices=["P1", "P2", "E1", "E2", "E3"])
    parser.add_argument("--partition")
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    block = config["blocks"][args.block]
    partitions = [args.partition] if args.partition else block["partitions"]
    seeds = [args.seed] if args.seed is not None else config["training"]["seeds"]
    for partition in partitions:
        for seed in seeds:
            run_one(config, args.block, partition, seed)


if __name__ == "__main__":
    main()
