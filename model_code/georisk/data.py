import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from georisk_readers import GeoRiskReaders  # noqa: E402


DATASET_INFO = {
    "sen1floods11": {
        "channels": 2,
        "classes": 2,
        "class_names": ["no_water", "water"],
        "event_grouped": True,
        "dofa_band_descriptors": [5.405, 5.405],
    },
    "kuro_siwo": {
        "channels": 7,
        "classes": 3,
        "class_names": ["no_water", "permanent_water", "flood"],
        "event_grouped": True,
        "dofa_band_descriptors": None,
    },
    "burn_scars": {
        "channels": 6,
        "classes": 2,
        "class_names": ["background", "burn_scar"],
        "event_grouped": False,
        "dofa_band_descriptors": [0.49, 0.56, 0.665, 0.865, 1.61, 2.19],
    },
}


def event_lookup(readers, dataset_name):
    if dataset_name == "sen1floods11":
        return {sample_id: row["event_code"] for sample_id, row in readers.sen1_samples.items()}
    if dataset_name == "kuro_siwo":
        return {sample_id: row["actid"] for sample_id, row in readers.kuro_samples.items()}
    return {sample_id: "__dataset__" for sample_id in readers.burn_samples}


def split_internal_validation(sample_ids, events, fraction, seed, event_grouped):
    groups = defaultdict(list)
    for sample_id in sorted(sample_ids):
        key = events[sample_id] if event_grouped else "__dataset__"
        groups[key].append(sample_id)

    rng = np.random.default_rng(seed)
    train_ids = []
    validation_ids = []
    for ids in groups.values():
        shuffled = np.asarray(ids, dtype=object)
        rng.shuffle(shuffled)
        validation_count = int(round(len(shuffled) * fraction))
        validation_ids.extend(shuffled[:validation_count].tolist())
        train_ids.extend(shuffled[validation_count:].tolist())
    return sorted(train_ids), sorted(validation_ids)


def build_id_sets(readers, dataset_name, partition, seed, validation_fraction):
    if dataset_name == "sen1floods11":
        train_all = readers.sen1_ids(partition, "train")
        calibration = readers.sen1_ids(partition, "calibration")
        test = readers.sen1_ids(partition, "test")
    elif dataset_name == "kuro_siwo":
        train_all = readers.kuro_ids("train")
        calibration = readers.kuro_ids("validation")
        test = readers.kuro_ids("test")
    else:
        train_all = readers.burn_ids("train")
        calibration = readers.burn_ids("validation")
        test = readers.burn_ids("test")

    events = event_lookup(readers, dataset_name)
    train, internal_validation = split_internal_validation(
        train_all,
        events,
        validation_fraction,
        seed,
        DATASET_INFO[dataset_name]["event_grouped"],
    )
    return {
        "train": train,
        "internal_validation": internal_validation,
        "calibration": calibration,
        "test": test,
    }


class GeoRiskDataset(Dataset):
    def __init__(
        self,
        readers,
        dataset_name,
        sample_ids,
        partition,
        crop_size=224,
        mode="native",
    ):
        self.readers = readers
        self.dataset_name = dataset_name
        self.sample_ids = list(sample_ids)
        self.partition = partition
        self.crop_size = int(crop_size)
        self.mode = mode

    def __len__(self):
        return len(self.sample_ids)

    def _load(self, sample_id):
        if self.dataset_name == "sen1floods11":
            return self.readers.load_sen1(sample_id, self.partition)
        if self.dataset_name == "kuro_siwo":
            return self.readers.load_kuro_siwo(sample_id)
        return self.readers.load_burn_scars(sample_id)

    def _crop(self, arrays):
        height, width = arrays[0].shape[-2:]
        size = self.crop_size
        if self.mode == "train":
            top = int(torch.randint(0, height - size + 1, (1,)).item())
            left = int(torch.randint(0, width - size + 1, (1,)).item())
        else:
            top = (height - size) // 2
            left = (width - size) // 2
        return [array[..., top : top + size, left : left + size] for array in arrays]

    def _augment(self, arrays):
        rotation = int(torch.randint(0, 4, (1,)).item())
        arrays = [np.rot90(array, rotation, axes=(-2, -1)) for array in arrays]
        if bool(torch.randint(0, 2, (1,)).item()):
            arrays = [np.flip(array, axis=-1) for array in arrays]
        if bool(torch.randint(0, 2, (1,)).item()):
            arrays = [np.flip(array, axis=-2) for array in arrays]
        return [np.ascontiguousarray(array) for array in arrays]

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]
        sample = self._load(sample_id)
        arrays = [
            sample["image"],
            sample["target"],
            sample["input_valid_mask"],
            sample["evaluation_mask"],
        ]
        if self.mode in {"train", "center_crop"}:
            arrays = self._crop(arrays)
        if self.mode == "train":
            arrays = self._augment(arrays)

        image, target, input_valid_mask, evaluation_mask = arrays
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)).float(),
            "target": torch.from_numpy(np.ascontiguousarray(target)).long(),
            "input_valid_mask": torch.from_numpy(np.ascontiguousarray(input_valid_mask)).bool(),
            "evaluation_mask": torch.from_numpy(np.ascontiguousarray(evaluation_mask)).bool(),
            "sample_id": sample["sample_id"],
            "event_id": sample.get("event_id", "__dataset__"),
        }


def build_datasets(root, dataset_name, partition, seed, validation_fraction, crop_size):
    readers = GeoRiskReaders(root)
    id_sets = build_id_sets(readers, dataset_name, partition, seed, validation_fraction)
    return {
        "train": GeoRiskDataset(
            readers, dataset_name, id_sets["train"], partition, crop_size, mode="train"
        ),
        "internal_validation": GeoRiskDataset(
            readers,
            dataset_name,
            id_sets["internal_validation"],
            partition,
            crop_size,
            mode="center_crop",
        ),
        "calibration": GeoRiskDataset(
            readers, dataset_name, id_sets["calibration"], partition, crop_size, mode="native"
        ),
        "test": GeoRiskDataset(
            readers, dataset_name, id_sets["test"], partition, crop_size, mode="native"
        ),
    }
