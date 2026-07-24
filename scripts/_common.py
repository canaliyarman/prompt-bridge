#!/usr/bin/env python3
"""Shared, experiment-local utilities."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = EXPERIMENT_ROOT.parents[1]
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def resolve_experiment_path(value: str | Path) -> Path:
    """Resolve a user path relative to the experiment root."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (EXPERIMENT_ROOT / path).resolve()


def require_file(value: str | Path, label: str) -> Path:
    """Resolve a required file and fail rather than downloading it."""
    path = resolve_experiment_path(value)
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}. Provision it explicitly; downloads are disabled.")
    return path


def require_directory(value: str | Path, label: str) -> Path:
    """Resolve a required directory."""
    path = resolve_experiment_path(value)
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def ensure_output_directory(value: str | Path) -> Path:
    """Create an output directory within this experiment."""
    path = resolve_experiment_path(value)
    try:
        path.relative_to(EXPERIMENT_ROOT)
    except ValueError as exc:
        raise ValueError(f"Output must remain under {EXPERIMENT_ROOT}: {path}") from exc
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_output_file(value: str | Path, label: str) -> Path:
    """Resolve a required file and require it to live below the experiment outputs."""
    path = require_file(value, label)
    outputs_root = EXPERIMENT_ROOT / "outputs"
    try:
        path.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError(f"{label} must remain under {outputs_root}: {path}") from exc
    return path


def load_prompts(config_path: str | Path, key: str) -> list[str]:
    """Load and validate a named prompt list."""
    path = require_file(config_path, "prompt configuration")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    prompts = data.get(key) if isinstance(data, dict) else None
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"{path} must contain a non-empty '{key}' list")
    if not all(isinstance(item, str) and item.strip() for item in prompts):
        raise ValueError(f"Every entry in '{key}' must be a non-empty string")
    if len(prompts) != len(set(prompts)):
        raise ValueError(f"Prompt list '{key}' contains duplicates")
    return prompts


def set_deterministic(seed: int) -> None:
    """Seed supported random number generators and request deterministic kernels."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def validate_device(device: str) -> torch.device:
    """Validate an explicitly requested CPU or CUDA device."""
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    return resolved


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a file SHA-256 digest without loading the whole checkpoint."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repository: Path) -> str:
    """Return a source commit or an explicit non-Git marker."""
    if not (repository / ".git").exists():
        return "not-a-git-checkout"
    completed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def add_source_checkout(repository: str | Path | None, label: str) -> Path | None:
    """Prepend an optional editable source checkout to sys.path."""
    if repository is None:
        return None
    path = require_directory(repository, label)
    sys.path.insert(0, str(path))
    return path


def configure_offline_runtime() -> None:
    """Disable known implicit network paths used by model tooling."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    """Return serializable tensor shape, dtype, finite-state, and norm statistics."""
    if tensor.numel() == 0:
        raise ValueError("Embedding tensor is empty")
    finite = bool(torch.isfinite(tensor).all().item())
    norms = tensor.detach().float().norm(dim=-1)
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "finite": finite,
        "norm_min": float(norms.min().item()),
        "norm_max": float(norms.max().item()),
        "norm_mean": float(norms.mean().item()),
        "norm_std": float(norms.std(unbiased=False).item()),
    }


def validate_embeddings(
    tensor: torch.Tensor,
    *,
    prompt_count: int,
    expected_ndim: int,
    require_normalized: bool,
    atol: float = 1e-5,
) -> None:
    """Fail on malformed, non-finite, or unexpectedly unnormalized embeddings."""
    if tensor.ndim != expected_ndim:
        raise ValueError(f"Expected {expected_ndim} embedding dimensions, got shape {tuple(tensor.shape)}")
    prompt_axis = 0 if expected_ndim == 2 else 1
    if tensor.shape[prompt_axis] != prompt_count:
        raise ValueError(
            f"Embedding prompt axis has {tensor.shape[prompt_axis]} entries; expected {prompt_count}"
        )
    if not torch.isfinite(tensor).all():
        raise FloatingPointError("Embeddings contain NaN or Inf values")
    if require_normalized:
        norms = tensor.float().norm(dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=atol, rtol=atol):
            delta = float((norms - 1).abs().max().item())
            raise ValueError(f"Embeddings are not L2-normalized; maximum norm error is {delta:.3e}")


def collect_images(source: str | Path) -> list[Path]:
    """Collect deterministically ordered images from a file or directory."""
    path = resolve_experiment_path(source)
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Image source not found: {path}")
    images = sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"No supported images found under: {path}")
    return images


def image_manifest(images: Iterable[Path]) -> list[dict[str, Any]]:
    """Build an ordered, content-addressed image manifest."""
    manifest: list[dict[str, Any]] = []
    for image in images:
        resolved = image.resolve()
        manifest.append({
            "image_id": resolved.stem,
            "path": str(resolved),
            "size_bytes": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
        })
    image_ids = [item["image_id"] for item in manifest]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Image stems must be unique because they are used as image_id values")
    return manifest


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-serializable value using a stable encoding."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    """Load a JSON object with a clear type error."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain a JSON object: {path}")
    return value


def load_tensor(path: Path) -> torch.Tensor:
    """Load a tensor-only artifact without permitting arbitrary pickle objects."""
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected a tensor in {path}, got {type(value).__name__}")
    return value


def write_json(path: Path, value: Any) -> None:
    """Write deterministic, human-readable JSON."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Write one JSON object per line."""
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")


def synchronize(device: torch.device) -> None:
    """Synchronize CUDA for accurate wall-clock latency."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
