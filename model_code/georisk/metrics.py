from collections import defaultdict

import numpy as np
import torch


def confusion_matrix(prediction, target, num_classes, ignore_index=255):
    valid = target != ignore_index
    indices = target[valid] * num_classes + prediction[valid]
    counts = torch.bincount(indices, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes).cpu().numpy().astype(np.int64)


def metrics_from_confusion(confusion):
    confusion = np.asarray(confusion, dtype=np.float64)
    true_positive = np.diag(confusion)
    false_positive = confusion.sum(axis=0) - true_positive
    false_negative = confusion.sum(axis=1) - true_positive
    union = true_positive + false_positive + false_negative
    present_iou = union > 0
    iou = np.divide(true_positive, union, out=np.zeros_like(true_positive), where=present_iou)
    f1_denominator = 2 * true_positive + false_positive + false_negative
    present_f1 = f1_denominator > 0
    f1 = np.divide(
        2 * true_positive,
        f1_denominator,
        out=np.zeros_like(true_positive),
        where=present_f1,
    )
    return {
        "miou": float(iou[present_iou].mean()),
        "macro_f1": float(f1[present_f1].mean()),
    }


class EventMetricAccumulator:
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusions = defaultdict(
            lambda: np.zeros((num_classes, num_classes), dtype=np.int64)
        )

    def update(self, logits, target, event_ids):
        prediction = logits.argmax(dim=1)
        for index, event_id in enumerate(event_ids):
            self.confusions[str(event_id)] += confusion_matrix(
                prediction[index], target[index], self.num_classes, self.ignore_index
            )

    def compute(self):
        per_event = {
            event_id: metrics_from_confusion(confusion)
            for event_id, confusion in self.confusions.items()
        }
        return {
            "event_macro_miou": float(np.mean([value["miou"] for value in per_event.values()])),
            "event_macro_f1": float(np.mean([value["macro_f1"] for value in per_event.values()])),
            "per_event": per_event,
        }
