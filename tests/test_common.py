from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from _common import (  # noqa: E402
    EXPERIMENT_ROOT,
    canonical_sha256,
    ensure_output_directory,
    image_manifest,
    load_prompts,
    tensor_summary,
    validate_embeddings,
)
from _yoloe_runner import run_signature_material  # noqa: E402
from evaluate_predictions import average_precision, box_iou, evaluate  # noqa: E402


def test_output_path_confined_to_experiment() -> None:
    with pytest.raises(ValueError, match="Output must remain"):
        ensure_output_directory("../../../outside")


def test_prompt_loading_rejects_duplicates(tmp_path: Path) -> None:
    config = tmp_path / "prompts.yaml"
    config.write_text(yaml.safe_dump({"chosen": ["car", "car"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicates"):
        load_prompts(config, "chosen")


def test_embedding_validation_and_summary() -> None:
    tensor = torch.nn.functional.normalize(torch.randn(1, 6, 512), dim=-1)
    validate_embeddings(tensor, prompt_count=6, expected_ndim=3, require_normalized=True)
    summary = tensor_summary(tensor)
    assert summary["finite"] is True
    assert summary["shape"] == [1, 6, 512]


def test_embedding_validation_rejects_nonfinite() -> None:
    tensor = torch.ones(2, 4)
    tensor[0, 0] = torch.nan
    with pytest.raises(FloatingPointError, match="NaN or Inf"):
        validate_embeddings(tensor, prompt_count=2, expected_ndim=2, require_normalized=False)


def test_image_manifest_is_ordered_and_content_addressed(tmp_path: Path) -> None:
    first = tmp_path / "a.jpg"
    second = tmp_path / "b.jpg"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    manifest = image_manifest([first, second])
    assert [item["image_id"] for item in manifest] == ["a", "b"]
    assert all(len(item["sha256"]) == 64 for item in manifest)


def test_image_manifest_rejects_duplicate_stems(tmp_path: Path) -> None:
    one = tmp_path / "one"
    two = tmp_path / "two"
    one.mkdir()
    two.mkdir()
    first = one / "same.jpg"
    second = two / "same.png"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    with pytest.raises(ValueError, match="stems must be unique"):
        image_manifest([first, second])


def test_run_signature_changes_with_inference_settings() -> None:
    base = run_signature_material(
        prompts=["car"],
        manifest=[{"image_id": "x", "path": "/x", "size_bytes": 1, "sha256": "0" * 64}],
        checkpoint_sha256="1" * 64,
        image_size=640,
        confidence=0.25,
        iou=0.7,
        max_detections=300,
    )
    changed = dict(base, image_size=320)
    assert canonical_sha256(base) != canonical_sha256(changed)
    assert canonical_sha256(base) == canonical_sha256(json.loads(json.dumps(base)))


def test_iou_and_average_precision() -> None:
    assert box_iou([0, 0, 10, 10], [5, 5, 15, 15]) == pytest.approx(25 / 175)
    assert average_precision([(0.9, True), (0.8, False)], 1) == pytest.approx(1.0)


def test_evaluate_matches_once_and_counts_false_positive() -> None:
    images = [{
        "image_id": "sample",
        "objects": [{
            "id": "car-1",
            "box_xyxy": [0, 0, 10, 10],
            "positive_prompt_ids": [0],
            "evaluable_hard_negative_prompt_ids": [1],
            "mission_relevant": True,
        }],
    }]
    records = [
        {"image_id": "sample", "prompt_id": 0, "box_xyxy": [0, 0, 10, 10], "confidence": 0.9},
        {"image_id": "sample", "prompt_id": 0, "box_xyxy": [0, 0, 10, 10], "confidence": 0.8},
        {"image_id": "sample", "prompt_id": 1, "box_xyxy": [0, 0, 10, 10], "confidence": 0.2},
    ]
    metrics, evaluated = evaluate(records, images, 0.5)
    assert sum(bool(item["matched_ground_truth"]) for item in evaluated) == 1
    assert metrics["box_recall"] == pytest.approx(1.0)
    assert metrics["median_hard_negative_margin"] == pytest.approx(0.7)
