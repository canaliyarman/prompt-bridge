#!/usr/bin/env python3
"""Shared YOLOE loading, validation, prediction, and result serialization."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from _common import (
    EXPERIMENT_ROOT,
    add_source_checkout,
    canonical_sha256,
    collect_images,
    configure_offline_runtime,
    ensure_output_directory,
    git_commit,
    image_manifest,
    require_directory,
    require_file,
    sha256_file,
    synchronize,
    tensor_summary,
    validate_device,
    write_json,
    write_jsonl,
)


def load_yoloe(checkpoint: Path, repository_arg: str | None, device_name: str) -> tuple[Any, torch.device, Path | None]:
    """Load an explicitly provisioned YOLOE checkpoint without network fallback."""
    configure_offline_runtime()
    repository = add_source_checkout(repository_arg, "YOLOE source checkout") if repository_arg else None
    ultralytics = importlib.import_module("ultralytics")
    model = ultralytics.YOLOE(str(checkpoint))
    device = validate_device(device_name)
    model.to(device)
    model.model.eval()
    return model, device, repository


def get_head(model: Any) -> Any:
    internal = getattr(model, "model", None)
    layers = getattr(internal, "model", None)
    if layers is None or len(layers) == 0:
        raise TypeError("Loaded checkpoint does not expose a YOLOE layer stack")
    head = layers[-1]
    if "YOLOE" not in type(head).__name__:
        raise TypeError(f"Expected a YOLOE detection head, got {type(head).__name__}")
    return head


def validate_dynamic_head(model: Any) -> Any:
    head = get_head(model)
    if not hasattr(head, "reprta") or not callable(getattr(head, "get_tpe", None)):
        raise RuntimeError("YOLOE RepRTA text-adaptation path is absent")
    if bool(getattr(head, "is_fused", False)):
        raise RuntimeError("YOLOE head is already fused; dynamic class embeddings are unavailable")
    if hasattr(head, "lrpc"):
        raise RuntimeError("Prompt-free YOLOE checkpoint does not accept dynamic class embeddings")
    if not hasattr(head, "embed"):
        raise RuntimeError("YOLOE head does not publish its expected text-embedding dimension")
    return head


def class_names(model: Any) -> list[str]:
    names = getattr(model.model, "names", None)
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    if isinstance(names, list):
        return [str(name) for name in names]
    raise TypeError(f"Unexpected YOLOE class-name container: {type(names).__name__}")


def install_classes(model: Any, prompts: list[str], embeddings: torch.Tensor) -> None:
    """Install embeddings and verify exact prompt order and tensor installation."""
    model.set_classes(prompts, embeddings)
    installed_names = class_names(model)
    if installed_names != prompts:
        raise RuntimeError(f"set_classes() changed prompt order or count: {installed_names!r}")
    installed = getattr(model.model, "pe", None)
    if not isinstance(installed, torch.Tensor):
        raise RuntimeError("set_classes() did not install a text embedding tensor")
    if installed is not embeddings:
        raise RuntimeError("set_classes() did not retain the supplied embedding tensor identity")
    if installed.shape != embeddings.shape or not torch.equal(installed, embeddings):
        raise RuntimeError("Installed YOLOE embeddings differ from the supplied tensor")


def load_offline_blt_encoder(
    *, checkpoint: Path, clip_repository_arg: str, device: torch.device, model: Any
) -> tuple[Any, Path]:
    """Attach the official Ultralytics B(LT) wrapper using an explicit local TorchScript file."""
    clip_repository = require_directory(clip_repository_arg, "Ultralytics CLIP tokenizer checkout")
    add_source_checkout(clip_repository, "Ultralytics CLIP tokenizer checkout")
    text_model_module = importlib.import_module("ultralytics.nn.text_model")
    encoder = text_model_module.MobileCLIPTS(device=device, weight=str(checkpoint))
    encoder.eval()
    model.model.clip_model = encoder
    return encoder, clip_repository


def get_official_embeddings(model: Any, encoder: Any, head: Any, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate raw B(LT) and official post-RepRTA embeddings through the supported YOLOE path."""
    with torch.inference_mode():
        official = model.model.get_text_pe(prompts, cache_clip_model=True)
        tokens = encoder.tokenize(prompts)
        raw = encoder.encode_text(tokens).detach().reshape(1, len(prompts), -1)
        independently_adapted = head.get_tpe(raw)
    if independently_adapted is None or not torch.allclose(official, independently_adapted, atol=1e-6, rtol=1e-6):
        raise RuntimeError("YOLOE get_text_pe output differs from the explicit B(LT)-through-RepRTA path")
    return raw, official


def predict_and_save(
    *,
    model: Any,
    device: torch.device,
    images_source: str,
    prompts: list[str],
    output_dir_arg: str,
    image_size: int,
    confidence: float,
    iou: float,
    warmup_runs: int,
    repetitions: int,
    max_detections: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run identical class-aware inference and save records plus diagnostics."""
    if warmup_runs < 0 or repetitions < 1:
        raise ValueError("warmup-runs must be non-negative and repetitions must be at least one")
    images = collect_images(images_source)
    manifest = image_manifest(images)
    output_dir = ensure_output_directory(output_dir_arg)
    rendered_dir = output_dir / "rendered"
    rendered_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    image_summaries: list[dict[str, Any]] = []

    prediction_args = {
        "imgsz": image_size,
        "conf": confidence,
        "iou": iou,
        "device": str(device),
        "verbose": False,
        "save": False,
        "agnostic_nms": False,
        "max_det": max_detections,
    }
    for image_path in images:
        for _ in range(warmup_runs):
            with torch.inference_mode():
                warmup_result = model.predict(source=str(image_path), **prediction_args)
            if len(warmup_result) != 1:
                raise RuntimeError(f"Expected one warm-up result for {image_path}")

        latencies_ms: list[float] = []
        results = None
        for _ in range(repetitions):
            synchronize(device)
            start = time.perf_counter()
            with torch.inference_mode():
                results = model.predict(source=str(image_path), **prediction_args)
            synchronize(device)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
        assert results is not None
        median_ms = statistics.median(latencies_ms)
        if len(results) != 1:
            raise RuntimeError(f"Expected one result for {image_path}, received {len(results)}")
        result = results[0]
        boxes = result.boxes
        if boxes is None:
            raise RuntimeError(f"YOLOE returned no boxes container for {image_path}")
        xyxy = boxes.xyxy.detach().cpu().float()
        confidence_values = boxes.conf.detach().cpu().float()
        class_values = boxes.cls.detach().cpu().to(torch.int64)
        if not torch.isfinite(xyxy).all() or not torch.isfinite(confidence_values).all():
            raise FloatingPointError(f"Non-finite prediction output for {image_path}")
        for index in range(len(xyxy)):
            prompt_id = int(class_values[index].item())
            if not 0 <= prompt_id < len(prompts):
                raise IndexError(f"Prediction class {prompt_id} is outside the {len(prompts)} requested prompts")
            records.append(
                {
                    "image_id": image_path.stem,
                    "prompt_id": prompt_id,
                    "prompt_text": prompts[prompt_id],
                    "box_xyxy": [float(value) for value in xyxy[index].tolist()],
                    "confidence": float(confidence_values[index].item()),
                    "matched_ground_truth": None,
                    "iou": None,
                    "inference_time_ms": median_ms,
                }
            )
        rendered_path = rendered_dir / f"{image_path.stem}.jpg"
        plotted = result.plot()
        cv2 = importlib.import_module("cv2")
        if not cv2.imwrite(str(rendered_path), plotted):
            raise OSError(f"Failed to write rendered prediction: {rendered_path}")
        image_summaries.append(
            {
                "image_id": image_path.stem,
                "detections": len(xyxy),
                "warmup_runs": warmup_runs,
                "measured_repetitions": repetitions,
                "latencies_ms": latencies_ms,
                "median_inference_time_ms": median_ms,
                "rendered_path": str(rendered_path.relative_to(EXPERIMENT_ROOT)),
            }
        )

    write_jsonl(output_dir / "predictions.jsonl", records)
    write_json(output_dir / "images.json", image_summaries)
    write_json(output_dir / "image_manifest.json", manifest)
    return records, image_summaries, manifest


def run_signature_material(
    *,
    prompts: list[str],
    manifest: list[dict[str, Any]],
    checkpoint_sha256: str,
    image_size: int,
    confidence: float,
    iou: float,
    max_detections: int,
) -> dict[str, Any]:
    return {
        "prompts": prompts,
        "image_manifest": manifest,
        "yoloe_checkpoint_sha256": checkpoint_sha256,
        "image_size": image_size,
        "confidence_threshold": confidence,
        "iou_threshold": iou,
        "agnostic_nms": False,
        "max_detections": max_detections,
    }


def environment_metadata(
    checkpoint: Path,
    repository: Path | None,
    model: Any,
    head: Any,
    *,
    blt_checkpoint: Path | None = None,
    clip_repository: Path | None = None,
) -> dict[str, Any]:
    try:
        ultralytics_version = importlib.metadata.version("ultralytics")
    except importlib.metadata.PackageNotFoundError:
        ultralytics_version = "editable-unversioned"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    text_model = getattr(model.model, "text_model", None)
    return {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_model": gpu,
        "gpu_compute_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "ultralytics_version": ultralytics_version,
        "yoloe_checkpoint": str(checkpoint),
        "yoloe_checkpoint_sha256": sha256_file(checkpoint),
        "yoloe_repository": str(repository) if repository else None,
        "yoloe_repository_commit": git_commit(repository) if repository else "installed-package",
        "blt_checkpoint": str(blt_checkpoint) if blt_checkpoint else None,
        "blt_checkpoint_sha256": sha256_file(blt_checkpoint) if blt_checkpoint else None,
        "clip_repository": str(clip_repository) if clip_repository else None,
        "clip_repository_commit": git_commit(clip_repository) if clip_repository else None,
        "checkpoint_type": type(model.model).__name__,
        "detection_head_class": type(head).__name__,
        "detection_scales": len(getattr(head, "stride", [])),
        "head_embedding_dimension": int(getattr(head, "embed")),
        "text_model_checkpoint_metadata": text_model,
        "resolved_official_text_model": "mobileclip:blt",
        "reprta_present": hasattr(head, "reprta"),
        "fused": bool(getattr(head, "is_fused", False)),
        "dynamic_class_embeddings": not hasattr(head, "lrpc") and not bool(getattr(head, "is_fused", False)),
    }


def print_metadata(metadata: dict[str, Any]) -> None:
    print(json.dumps(metadata, indent=2, sort_keys=True))


__all__ = [
    "canonical_sha256",
    "class_names",
    "environment_metadata",
    "get_head",
    "get_official_embeddings",
    "install_classes",
    "load_offline_blt_encoder",
    "load_yoloe",
    "predict_and_save",
    "print_metadata",
    "run_signature_material",
    "tensor_summary",
    "validate_dynamic_head",
    "write_json",
]
