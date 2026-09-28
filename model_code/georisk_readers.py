import csv
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.io import MemoryFile


IGNORE_INDEX = 255


def _csv_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _read_raster(path):
    with rasterio.open(path) as dataset:
        return dataset.read()


def _clip_standardize(image, clip_min, clip_max, mean, std):
    clip_min = np.asarray(clip_min, dtype=np.float32)[:, None, None]
    clip_max = np.asarray(clip_max, dtype=np.float32)[:, None, None]
    mean = np.asarray(mean, dtype=np.float32)[:, None, None]
    std = np.asarray(std, dtype=np.float32)[:, None, None]
    image = image.astype(np.float32, copy=False)
    image = np.where(np.isfinite(image), image, clip_min)
    image = np.clip(image, clip_min, clip_max)
    image = (image - clip_min) / (clip_max - clip_min)
    return (image - mean) / std


class GeoRiskReaders:
    def __init__(self, root):
        self.root = Path(root)
        self.sen1_root = self.root / "sen1floods11"
        self.containers = {
            "burn_scars": self.root / "geobench2" / "burn_scars" / "geobench_burn_scars.tortilla",
            "kuro_siwo": self.root / "geobench2" / "kuro_siwo" / "geobench_kuro_siwo.tortilla",
        }

        sen1_rows = _csv_rows(self.root / "manifests" / "sen1floods11_handlabeled_samples.csv")
        self.sen1_samples = {row["sample_id"]: row for row in sen1_rows}
        self.fold_rows = _csv_rows(self.root / "manifests" / "sen1floods11_event_folds_v2.csv")

        asset_rows = _csv_rows(self.root / "manifests" / "geobench2_tortilla_assets.csv")
        self.assets = {
            (row["dataset"], row["sample_id"], row["asset_id"]): row
            for row in asset_rows
        }

        burn_rows = _csv_rows(self.root / "manifests" / "geobench2_burn_scars_samples.csv")
        kuro_rows = _csv_rows(self.root / "manifests" / "geobench2_kuro_siwo_samples.csv")
        self.burn_samples = {row["sample_id"]: row for row in burn_rows}
        self.kuro_samples = {row["sample_id"]: row for row in kuro_rows}

        self.sen1_stats = json.loads(
            (self.root / "configs" / "sen1floods11_fold_normalization_v1.json").read_text()
        )
        self.burn_stats = json.loads(
            (self.root / "geobench2" / "burn_scars" / "burn_scars_stats_clip_rescale.json").read_text()
        )
        self.kuro_stats = json.loads(
            (self.root / "geobench2" / "kuro_siwo" / "kuro_siwo_stats_clip_rescale.json").read_text()
        )

    def _read_asset(self, dataset, sample_id, asset_id):
        row = self.assets[(dataset, str(sample_id), asset_id)]
        with self.containers[dataset].open("rb") as stream:
            stream.seek(int(row["byte_offset"]))
            payload = stream.read(int(row["byte_length"]))
        with MemoryFile(payload) as memory_file:
            with memory_file.open() as raster:
                return raster.read()

    def sen1_ids(self, fold, role):
        events = {
            row["event_code"]
            for row in self.fold_rows
            if row["fold"] == fold and row["role"] == role
        }
        return sorted(
            sample_id
            for sample_id, row in self.sen1_samples.items()
            if row["event_code"] in events
        )

    def burn_ids(self, split):
        return sorted(
            sample_id
            for sample_id, row in self.burn_samples.items()
            if row["official_split"] == split
        )

    def kuro_ids(self, split):
        return sorted(
            sample_id
            for sample_id, row in self.kuro_samples.items()
            if row["official_split"] == split
        )

    def load_sen1(self, sample_id, fold, normalize=True):
        row = self.sen1_samples[str(sample_id)]
        image = _read_raster(self.sen1_root / row["image_relpath"]).astype(np.float32)
        raw_target = _read_raster(self.sen1_root / row["label_relpath"])[0]
        input_valid = np.all(np.isfinite(image), axis=0)
        evaluation_mask = ((raw_target == 0) | (raw_target == 1)) & input_valid
        target = np.full(raw_target.shape, IGNORE_INDEX, dtype=np.uint8)
        target[evaluation_mask] = raw_target[evaluation_mask].astype(np.uint8)

        if normalize:
            stats = self.sen1_stats["folds"][fold]
            image = _clip_standardize(
                image,
                stats["clip_min_p02"],
                stats["clip_max_p98"],
                stats["normalized_mean"],
                stats["normalized_std"],
            )

        return {
            "image": image,
            "target": target,
            "input_valid_mask": input_valid,
            "evaluation_mask": evaluation_mask,
            "channels": ["VV", "VH"],
            "sample_id": str(sample_id),
            "event_id": row["event_code"],
            "split_role": next(
                item["role"]
                for item in self.fold_rows
                if item["fold"] == fold and item["event_code"] == row["event_code"]
            ),
        }

    def load_burn_scars(self, sample_id, normalize=True):
        sample_id = str(sample_id)
        image = self._read_asset("burn_scars", sample_id, "image").astype(np.float32)
        raw_target = self._read_asset("burn_scars", sample_id, "label")[0]
        input_valid = np.all(np.isfinite(image), axis=0)
        evaluation_mask = ((raw_target == 0) | (raw_target == 1)) & input_valid
        target = np.full(raw_target.shape, IGNORE_INDEX, dtype=np.uint8)
        target[evaluation_mask] = raw_target[evaluation_mask].astype(np.uint8)

        stats = self.burn_stats["input_stats"]["image"]
        if normalize:
            image = _clip_standardize(
                image,
                [stats["clip_min_used"]] * 6,
                [stats["clip_max_used"]] * 6,
                stats["mean"],
                stats["std"],
            )

        return {
            "image": image,
            "target": target,
            "input_valid_mask": input_valid,
            "evaluation_mask": evaluation_mask,
            "channels": stats["band_names"],
            "sample_id": sample_id,
            "official_split": self.burn_samples[sample_id]["official_split"],
        }

    def load_kuro_siwo(self, sample_id, normalize=True):
        sample_id = str(sample_id)
        asset_names = ["pre_event_1", "pre_event_2", "post_event", "dem"]
        arrays = [self._read_asset("kuro_siwo", sample_id, name).astype(np.float32) for name in asset_names]
        raw_target = self._read_asset("kuro_siwo", sample_id, "mask")[0]
        raw_validity = self._read_asset("kuro_siwo", sample_id, "invalid_data")[0]
        finite = np.all(np.isfinite(np.concatenate(arrays, axis=0)), axis=0)
        input_valid = (raw_validity == 1) & finite
        evaluation_mask = input_valid & (raw_target != 3)
        target = np.full(raw_target.shape, IGNORE_INDEX, dtype=np.uint8)
        for raw_value in (0, 1, 2):
            target[evaluation_mask & (raw_target == raw_value)] = raw_value

        if normalize:
            stat_names = ["image_pre_1", "image_pre_2", "image_post", "image_dem"]
            normalized = []
            for array, stat_name in zip(arrays, stat_names):
                stats = self.kuro_stats["input_stats"][stat_name]
                normalized.append(
                    _clip_standardize(
                        array,
                        [stats["clip_min_used"]] * array.shape[0],
                        [stats["clip_max_used"]] * array.shape[0],
                        stats["norm_mean"],
                        stats["norm_std"],
                    )
                )
            arrays = normalized

        image = np.concatenate(arrays, axis=0)
        metadata = self.kuro_samples[sample_id]
        return {
            "image": image,
            "target": target,
            "input_valid_mask": input_valid,
            "evaluation_mask": evaluation_mask,
            "channels": [
                "pre1_VV", "pre1_VH", "pre2_VV", "pre2_VH",
                "post_VV", "post_VH", "DEM",
            ],
            "sample_id": sample_id,
            "event_id": metadata["actid"],
            "aoi_id": metadata["aoiid"],
            "official_split": metadata["official_split"],
        }
