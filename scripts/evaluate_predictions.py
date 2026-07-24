#!/usr/bin/env python3
"""Evaluate baseline and direct-injection predictions against versioned box annotations."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from _common import ensure_output_directory, load_json, require_file, require_output_file, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="data/annotations/annotations.json")
    parser.add_argument("--baseline-predictions", default="outputs/baseline/predictions.jsonl")
    parser.add_argument("--direct-predictions", default="outputs/s0_direct/predictions.jsonl")
    parser.add_argument("--baseline-metadata", default="outputs/baseline/metadata.json")
    parser.add_argument("--direct-metadata", default="outputs/s0_direct/metadata.json")
    parser.add_argument("--output-dir", default="outputs/comparisons")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser.parse_args()


def box_iou(left: list[float], right: list[float]) -> float:
    if len(left) != 4 or len(right) != 4:
        raise ValueError("Every box must contain four xyxy values")
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number} is not a JSON object")
        records.append(value)
    return records


def validate_annotations(value: dict[str, Any]) -> list[dict[str, Any]]:
    if value.get("schema_version") != 1 or not isinstance(value.get("images"), list):
        raise ValueError("Annotations must use schema_version 1 and contain an images list")
    images = value["images"]
    if not images:
        raise ValueError("Annotations contain no images")
    ids = [image.get("image_id") for image in images]
    if any(not isinstance(item, str) or not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("Annotation image_id values must be unique non-empty strings")
    for image in images:
        if not isinstance(image.get("objects"), list):
            raise ValueError(f"Annotation {image['image_id']} lacks an objects list")
        for obj in image["objects"]:
            if not isinstance(obj.get("id"), str) or not isinstance(obj.get("box_xyxy"), list):
                raise ValueError(f"Malformed object in annotation {image['image_id']}")
            if not obj.get("positive_prompt_ids"):
                raise ValueError(f"Object {obj['id']} has no positive_prompt_ids")
            obj.setdefault("evaluable_hard_negative_prompt_ids", [])
            obj.setdefault("mission_relevant", True)
    return images


def average_precision(rows: list[tuple[float, bool]], ground_truth_count: int) -> float:
    if ground_truth_count == 0:
        raise ValueError("AP is undefined without ground truth")
    rows = sorted(rows, reverse=True)
    tp, fp, recalls, precisions = 0, 0, [], []
    for _, matched in rows:
        tp += int(matched)
        fp += int(not matched)
        recalls.append(tp / ground_truth_count)
        precisions.append(tp / (tp + fp))
    recalls = [0.0, *recalls, 1.0]
    precisions = [1.0, *precisions, 0.0]
    for index in range(len(precisions) - 2, -1, -1):
        precisions[index] = max(precisions[index], precisions[index + 1])
    return sum(
        (recalls[index] - recalls[index - 1]) * precisions[index]
        for index in range(1, len(recalls))
        if recalls[index] != recalls[index - 1]
    )


def evaluate(records: list[dict[str, Any]], images: list[dict[str, Any]], threshold: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    objects_by_image = {image["image_id"]: image["objects"] for image in images}
    image_ids = set(objects_by_image)
    if {record.get("image_id") for record in records} - image_ids:
        raise ValueError("Predictions contain image IDs absent from annotations")
    candidates: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        candidates[(str(record["image_id"]), int(record["prompt_id"]))].append(record)
    evaluated: list[dict[str, Any]] = []
    gt_by_prompt: dict[int, int] = defaultdict(int)
    mission_gt, mission_tp = 0, 0
    false_negatives: dict[int, int] = defaultdict(int)
    ap_rows: dict[int, list[tuple[float, bool]]] = defaultdict(list)

    all_prompt_ids = {int(record["prompt_id"]) for record in records}
    for image in images:
        for obj in image["objects"]:
            all_prompt_ids.update(int(value) for value in obj["positive_prompt_ids"])
            all_prompt_ids.update(int(value) for value in obj["evaluable_hard_negative_prompt_ids"])
    for image in images:
        image_id = image["image_id"]
        for prompt_id in sorted(all_prompt_ids):
            gt_objects = [obj for obj in image["objects"] if prompt_id in obj["positive_prompt_ids"]]
            gt_by_prompt[prompt_id] += len(gt_objects)
            unmatched = {obj["id"]: obj for obj in gt_objects}
            prompt_predictions = sorted(candidates.get((image_id, prompt_id), []), key=lambda item: item["confidence"], reverse=True)
            for record in prompt_predictions:
                best_id, best_iou = None, 0.0
                for object_id, obj in unmatched.items():
                    overlap = box_iou(record["box_xyxy"], obj["box_xyxy"])
                    if overlap > best_iou:
                        best_id, best_iou = object_id, overlap
                matched = best_id is not None and best_iou >= threshold
                updated = dict(record, matched_ground_truth=best_id if matched else None, iou=best_iou)
                evaluated.append(updated)
                ap_rows[prompt_id].append((float(record["confidence"]), matched))
                if matched:
                    unmatched.pop(best_id)
            false_negatives[prompt_id] += len(unmatched)

    true_positives = sum(bool(record["matched_ground_truth"]) for record in evaluated)
    false_positives = len(evaluated) - true_positives
    ground_truth_total = sum(gt_by_prompt.values())
    ap_by_prompt = {
        str(prompt_id): average_precision(ap_rows[prompt_id], count)
        for prompt_id, count in gt_by_prompt.items()
        if count > 0
    }
    margins: list[dict[str, Any]] = []
    for image in images:
        for obj in image["objects"]:
            def best_score(prompt_ids: list[int]) -> float:
                scores = [
                    float(record["confidence"])
                    for prompt_id in prompt_ids
                    for record in candidates.get((image["image_id"], int(prompt_id)), [])
                    if box_iou(record["box_xyxy"], obj["box_xyxy"]) >= threshold
                ]
                return max(scores, default=0.0)
            correct = best_score(obj["positive_prompt_ids"])
            negative = best_score(obj["evaluable_hard_negative_prompt_ids"])
            margins.append({
                "image_id": image["image_id"],
                "object_id": obj["id"],
                "correct_confidence": correct,
                "hard_negative_confidence": negative,
                "margin": correct - negative,
                "mission_relevant": bool(obj.get("mission_relevant", True)),
                "hard_negative_evaluable": bool(obj["evaluable_hard_negative_prompt_ids"]),
            })
    mission_gt = sum(item["mission_relevant"] for item in margins)
    mission_tp = sum(item["mission_relevant"] and item["correct_confidence"] > 0 for item in margins)
    metrics = {
        "box_precision": true_positives / len(evaluated) if evaluated else 0.0,
        "box_recall": true_positives / ground_truth_total if ground_truth_total else 0.0,
        "map50": statistics.mean(ap_by_prompt.values()) if ap_by_prompt else 0.0,
        "ap50_by_prompt": ap_by_prompt,
        "false_positives_per_image": false_positives / len(images),
        "false_negatives_per_prompt": {str(key): value for key, value in sorted(false_negatives.items())},
        "mission_object_recall": mission_tp / mission_gt if mission_gt else 0.0,
        "missed_object_rate": 1.0 - (mission_tp / mission_gt) if mission_gt else 0.0,
        "hard_negative_margins": margins,
        "median_hard_negative_margin": (
            statistics.median(item["margin"] for item in margins if item["hard_negative_evaluable"])
            if any(item["hard_negative_evaluable"] for item in margins)
            else None
        ),
        "ground_truth_prompt_instances": ground_truth_total,
        "prediction_count": len(evaluated),
    }
    return metrics, evaluated


def decision(baseline: dict[str, Any], direct: dict[str, Any]) -> dict[str, Any]:
    baseline_map = float(baseline["map50"])
    relative_map_drop = (baseline_map - float(direct["map50"])) / baseline_map if baseline_map > 0 else None
    base_margin = baseline["median_hard_negative_margin"]
    direct_margin = direct["median_hard_negative_margin"]
    checks = {
        "recall_drop_le_5pp": float(baseline["box_recall"]) - float(direct["box_recall"]) <= 0.05,
        "relative_map50_drop_le_10pct": relative_map_drop is not None and relative_map_drop <= 0.10,
        "false_positive_increase_le_0_5_per_image": float(direct["false_positives_per_image"]) - float(baseline["false_positives_per_image"]) <= 0.5,
        "median_margin_positive": direct_margin is not None and direct_margin > 0,
        "median_margin_at_least_80pct_baseline": base_margin is not None and direct_margin is not None and direct_margin >= 0.8 * base_margin,
    }
    passed = all(checks.values())
    return {
        "provisionally_close": passed,
        "checks": checks,
        "relative_map50_drop": relative_map_drop,
        "recommendation": (
            "Run a larger labeled evaluation; this smoke dataset cannot establish general compatibility."
            if passed
            else "Evaluate a linear adapter, then RepRTA/YOLOE-head alignment fine-tuning if adapter alignment is insufficient."
        ),
    }


def main() -> None:
    args = parse_args()
    annotations = validate_annotations(load_json(require_file(args.annotations, "annotations"), "annotations"))
    baseline_path = require_output_file(args.baseline_predictions, "baseline predictions")
    direct_path = require_output_file(args.direct_predictions, "direct predictions")
    baseline_meta = load_json(require_output_file(args.baseline_metadata, "baseline metadata"), "baseline metadata")
    direct_meta = load_json(require_output_file(args.direct_metadata, "direct metadata"), "direct metadata")
    if baseline_meta.get("run_signature_sha256") != direct_meta.get("run_signature_sha256"):
        raise ValueError("Baseline and direct metadata run signatures differ")
    baseline_metrics, baseline_records = evaluate(load_jsonl(baseline_path), annotations, args.iou_threshold)
    direct_metrics, direct_records = evaluate(load_jsonl(direct_path), annotations, args.iou_threshold)
    output_dir = ensure_output_directory(args.output_dir)
    write_jsonl(output_dir / "baseline_evaluated_predictions.jsonl", baseline_records)
    write_jsonl(output_dir / "s0_direct_evaluated_predictions.jsonl", direct_records)
    result = {
        "schema_version": 1,
        "iou_threshold": args.iou_threshold,
        "baseline": baseline_metrics,
        "s0_direct": direct_metrics,
        "decision": decision(baseline_metrics, direct_metrics),
    }
    write_json(output_dir / "detection_comparison.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
