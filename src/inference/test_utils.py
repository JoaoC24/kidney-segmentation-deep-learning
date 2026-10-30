from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F


CSV_FIELDS = [
    "input_file",
    "label_file",
    "model",
    "checkpoint",
    "dice",
    "precision",
    "recall",
    "hausdorff_distance",
    "iou",
    "segmented_pixels",
    "ground_truth_pixels",
    "inference_time_seconds",
    "inference_file",
]

SUMMARY_CSV_FIELDS = [
    "model",
    "checkpoint",
    "num_volumes",
    "overall_dice",
    "overall_precision",
    "overall_recall",
    "overall_iou",
    "overall_hausdorff_distance",
    "mean_dice",
    "mean_precision",
    "mean_recall",
    "mean_hausdorff_distance",
    "mean_iou",
    "mean_volume_inference_time_seconds",
    "finite_hausdorff_distance_count",
    "infinite_hausdorff_distance_count",
    "total_segmented_pixels",
    "total_ground_truth_pixels",
]


class VolumeSegmentationTester(L.LightningModule):
    def __init__(
        self,
        model: L.LightningModule,
        model_name: str,
        checkpoint_path: str | Path,
        output_csv_path: str | Path,
        summary_csv_path: str | Path | None = None,
        slice_batch_size: int = 32,
        save_inference_volumes: bool = False,
        inference_output_dir: str | Path | None = None,
    ):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.checkpoint_path = Path(checkpoint_path)
        self.output_csv_path = Path(output_csv_path)
        self.summary_csv_path = Path(summary_csv_path) if summary_csv_path is not None else None
        self.slice_batch_size = slice_batch_size
        self.save_inference_volumes = save_inference_volumes
        self.inference_output_dir = Path(inference_output_dir) if inference_output_dir is not None else None
        self.rows: list[dict[str, object]] = []
        self.total_tp = 0.0
        self.total_fp = 0.0
        self.total_fn = 0.0

    def on_test_start(self) -> None:
        self.output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        if self.summary_csv_path is not None:
            self.summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
        if self.save_inference_volumes:
            if self.inference_output_dir is None:
                raise ValueError("inference_output_dir must be set when save_inference_volumes=True.")
            self.inference_output_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_csv_header(self.output_csv_path, CSV_FIELDS)
        if self.summary_csv_path is not None:
            self._ensure_csv_header(self.summary_csv_path, SUMMARY_CSV_FIELDS)

    def test_step(self, batch, batch_idx):
        volumes, labels, image_paths, label_paths = batch
        if volumes.shape[0] != 1:
            raise ValueError("Volume testing expects batch_size=1 so each CSV row maps to one input volume.")

        volume = volumes.squeeze(0)
        label = labels.squeeze(0).long()
        pred, inference_time = self._timed_predict_volume(volume)
        inference_path = self._save_inference_volume(pred, image_paths[0])

        metrics = volume_metrics(pred > 0, label > 0)
        metrics["inference_time_seconds"] = inference_time
        counts = confusion_counts(pred > 0, label > 0)
        self.total_tp += counts["tp"]
        self.total_fp += counts["fp"]
        self.total_fn += counts["fn"]
        row = {
            "input_file": image_paths[0],
            "label_file": label_paths[0],
            "model": self.model_name,
            "checkpoint": self.checkpoint_path.name,
            "inference_file": str(inference_path) if inference_path is not None else "",
            **metrics,
        }
        self.rows.append(row)
        self._append_row(row)

        log_metrics = {
            "test_dice": metrics["dice"],
            "test_precision": metrics["precision"],
            "test_recall": metrics["recall"],
            "test_iou": metrics["iou"],
            "test_segmented_pixels": metrics["segmented_pixels"],
            "test_ground_truth_pixels": metrics["ground_truth_pixels"],
            "test_volume_inference_time_seconds": metrics["inference_time_seconds"],
        }
        if math.isfinite(metrics["hausdorff_distance"]):
            log_metrics["test_hausdorff_distance"] = metrics["hausdorff_distance"]

        self.log_dict(log_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=False, batch_size=1)
        return log_metrics

    def on_test_epoch_end(self) -> None:
        if self.summary_csv_path is None:
            return
        self._append_summary_row(self._summary_row())

    def _predict_volume(self, volume: torch.Tensor) -> torch.Tensor:
        predictions = []
        for start in range(0, volume.shape[0], self.slice_batch_size):
            slices = volume[start : start + self.slice_batch_size].to(self.device)
            logits = self._slice_logits(slices)
            predictions.append(torch.argmax(logits, dim=1).detach().cpu())
        return torch.cat(predictions, dim=0)

    def _timed_predict_volume(self, volume: torch.Tensor) -> tuple[torch.Tensor, float]:
        _sync_cuda_if_needed(self.device)
        start_time = time.perf_counter()
        pred = self._predict_volume(volume)
        _sync_cuda_if_needed(self.device)
        return pred, time.perf_counter() - start_time

    def _slice_logits(self, slices: torch.Tensor) -> torch.Tensor:


        if hasattr(self.model, "inference_prune"):
            return self.model._inference_logits(slices)

        outputs = self.model(slices)
        if hasattr(self.model, "_inference_logits"):
            return self.model._inference_logits(outputs)
        if isinstance(outputs, list):
            return outputs[-1]
        return outputs

    def _append_row(self, row: dict[str, object]) -> None:
        with self.output_csv_path.open("a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writerow(row)

    def _append_summary_row(self, row: dict[str, object]) -> None:
        if self.summary_csv_path is None:
            return
        with self.summary_csv_path.open("a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_CSV_FIELDS)
            writer.writerow(row)

    def _summary_row(self) -> dict[str, object]:
        finite_hausdorff = [
            float(row["hausdorff_distance"])
            for row in self.rows
            if math.isfinite(float(row["hausdorff_distance"]))
        ]
        return {
            "model": self.model_name,
            "checkpoint": self.checkpoint_path.name,
            "num_volumes": len(self.rows),
            "overall_dice": _safe_divide(2 * self.total_tp, 2 * self.total_tp + self.total_fp + self.total_fn),
            "overall_precision": _safe_divide(self.total_tp, self.total_tp + self.total_fp),
            "overall_recall": _safe_divide(self.total_tp, self.total_tp + self.total_fn),
            "overall_iou": _safe_divide(self.total_tp, self.total_tp + self.total_fp + self.total_fn),
            "overall_hausdorff_distance": (
                sum(finite_hausdorff) / len(finite_hausdorff) if finite_hausdorff else math.inf
            ),
            "mean_dice": _mean_from_rows(self.rows, "dice"),
            "mean_precision": _mean_from_rows(self.rows, "precision"),
            "mean_recall": _mean_from_rows(self.rows, "recall"),
            "mean_hausdorff_distance": sum(finite_hausdorff) / len(finite_hausdorff) if finite_hausdorff else math.inf,
            "mean_iou": _mean_from_rows(self.rows, "iou"),
            "mean_volume_inference_time_seconds": _mean_from_rows(self.rows, "inference_time_seconds"),
            "finite_hausdorff_distance_count": len(finite_hausdorff),
            "infinite_hausdorff_distance_count": len(self.rows) - len(finite_hausdorff),
            "total_segmented_pixels": sum(float(row["segmented_pixels"]) for row in self.rows),
            "total_ground_truth_pixels": sum(float(row["ground_truth_pixels"]) for row in self.rows),
        }

    @staticmethod
    def _ensure_csv_header(path: Path, fieldnames: list[str]) -> None:
        if path.exists():
            return
        with path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

    def _save_inference_volume(self, pred: torch.Tensor, image_path: str) -> Path | None:
        if not self.save_inference_volumes:
            return None
        if self.inference_output_dir is None:
            raise ValueError("inference_output_dir must be set when save_inference_volumes=True.")

        input_path = Path(image_path)
        output_name = f"{input_path.stem}_inference.npy"
        output_path = self.inference_output_dir / output_name
        np.save(output_path, pred.detach().cpu().numpy().astype(np.uint8))
        return output_path

    def configure_optimizers(self):
        return None


def volume_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = pred.detach().cpu().bool()
    target = target.detach().cpu().bool()

    counts = confusion_counts(pred, target)
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]

    return {
        "dice": _safe_divide(2 * tp, 2 * tp + fp + fn),
        "precision": _safe_divide(tp, tp + fp),
        "recall": _safe_divide(tp, tp + fn),
        "hausdorff_distance": hausdorff_distance_3d(pred, target),
        "iou": _safe_divide(tp, tp + fp + fn),
        "segmented_pixels": float(pred.sum().item()),
        "ground_truth_pixels": float(target.sum().item()),
    }


def confusion_counts(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = pred.detach().cpu().bool()
    target = target.detach().cpu().bool()
    return {
        "tp": float(torch.logical_and(pred, target).sum().item()),
        "fp": float(torch.logical_and(pred, ~target).sum().item()),
        "fn": float(torch.logical_and(~pred, target).sum().item()),
    }


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 1.0
    return float(numerator / denominator)


def hausdorff_distance_3d(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_points = mask_boundary_points_3d(pred)
    target_points = mask_boundary_points_3d(target)

    if pred_points.numel() == 0 and target_points.numel() == 0:
        return 0.0
    if pred_points.numel() == 0 or target_points.numel() == 0:
        return math.inf

    pred_points = pred_points.float()
    target_points = target_points.float()
    pred_to_target = directed_hausdorff(pred_points, target_points)
    target_to_pred = directed_hausdorff(target_points, pred_points)
    return float(max(pred_to_target, target_to_pred))


def mask_boundary_points_3d(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    if mask.sum() == 0:
        return torch.empty((0, 3), dtype=torch.long)

    padded = F.pad(mask.unsqueeze(0).unsqueeze(0).float(), (1, 1, 1, 1, 1, 1))
    eroded = F.max_pool3d(1.0 - padded, kernel_size=3, stride=1, padding=0) == 0
    boundary = mask & ~eroded.squeeze(0).squeeze(0)
    return boundary.nonzero(as_tuple=False)


def directed_hausdorff(source_points: torch.Tensor, target_points: torch.Tensor, chunk_size: int = 1024) -> float:
    max_min_distance = 0.0
    for start in range(0, source_points.shape[0], chunk_size):
        source_chunk = source_points[start : start + chunk_size]
        distances = torch.cdist(source_chunk, target_points)
        chunk_max = distances.min(dim=1).values.max().item()
        max_min_distance = max(max_min_distance, chunk_max)
    return max_min_distance


def _mean_from_rows(rows: list[dict[str, object]], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(float(row[key]) for row in rows) / len(rows))


def _sync_cuda_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
