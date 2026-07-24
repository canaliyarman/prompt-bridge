#!/usr/bin/env python3
"""Generate and validate normalized MobileCLIP-S0 text embeddings."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import time
from typing import Any

import numpy as np
import torch

from _common import (
    EXPERIMENT_ROOT,
    add_source_checkout,
    configure_offline_runtime,
    ensure_output_directory,
    git_commit,
    load_prompts,
    require_file,
    set_deterministic,
    sha256_file,
    synchronize,
    tensor_summary,
    validate_device,
    validate_embeddings,
    write_json,
)

PINNED_S0_SHA256 = "809b408eff74f8058843e86a1f92967097d42ba782450e85b8f4867b7f0ca0b7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/mobileclip_s0.pt")
    parser.add_argument("--expected-checkpoint-sha256", default=PINNED_S0_SHA256)
    parser.add_argument("--mobileclip-repo", default="third_party/ml-mobileclip")
    parser.add_argument("--model-arch", default="mobileclip_s0")
    parser.add_argument("--encoder-name", default="MobileCLIP-S0")
    parser.add_argument("--prompts-config", default="configs/prompts.yaml")
    parser.add_argument("--prompt-set", default="visdrone_prompts")
    parser.add_argument("--output-dir", default="outputs/embeddings")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--norm-atol", type=float, default=1e-5)
    parser.add_argument("--determinism-atol", type=float, default=1e-7)
    return parser.parse_args()


def encode_text(model: Any, tokenizer: Any, prompts: list[str], device: torch.device) -> tuple[torch.Tensor, float]:
    tokens = tokenizer(prompts)
    if not isinstance(tokens, torch.Tensor):
        raise TypeError(f"Tokenizer returned {type(tokens).__name__}, expected torch.Tensor")
    tokens = tokens.to(device)
    synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        embeddings = model.encode_text(tokens).float()
        norms = embeddings.norm(dim=-1, keepdim=True)
        if not torch.isfinite(embeddings).all() or not torch.isfinite(norms).all():
            raise FloatingPointError("Raw MobileCLIP-S0 embeddings contain NaN or Inf")
        if bool((norms <= torch.finfo(torch.float32).eps).any()):
            raise ValueError("MobileCLIP-S0 produced a zero-length text embedding")
        embeddings = embeddings / norms
    synchronize(device)
    return embeddings, (time.perf_counter() - start) * 1000.0


def main() -> None:
    args = parse_args()
    configure_offline_runtime()
    set_deterministic(args.seed)
    device = validate_device(args.device)
    checkpoint = require_file(args.checkpoint, "MobileCLIP-S0 checkpoint")
    checkpoint_hash = sha256_file(checkpoint)
    if args.expected_checkpoint_sha256 and checkpoint_hash != args.expected_checkpoint_sha256:
        raise ValueError(
            f"MobileCLIP-S0 checkpoint SHA-256 mismatch: expected {args.expected_checkpoint_sha256}, got {checkpoint_hash}"
        )
    repository = add_source_checkout(args.mobileclip_repo, "MobileCLIP source checkout")
    prompts_config = require_file(args.prompts_config, "prompt configuration")
    prompts = load_prompts(prompts_config, args.prompt_set)
    output_dir = ensure_output_directory(args.output_dir)

    mobileclip = importlib.import_module("mobileclip")
    model, _, _ = mobileclip.create_model_and_transforms(args.model_arch, pretrained=str(checkpoint))
    tokenizer = mobileclip.get_tokenizer(args.model_arch)
    model = model.to(device)
    model.eval()

    embeddings_first, latency_first_ms = encode_text(model, tokenizer, prompts, device)
    embeddings_second, latency_second_ms = encode_text(model, tokenizer, prompts, device)
    for embeddings in (embeddings_first, embeddings_second):
        validate_embeddings(
            embeddings,
            prompt_count=len(prompts),
            expected_ndim=2,
            require_normalized=True,
            atol=args.norm_atol,
        )
    max_delta = float((embeddings_first - embeddings_second).abs().max().item())
    if not torch.allclose(embeddings_first, embeddings_second, atol=args.determinism_atol, rtol=0.0):
        raise RuntimeError(f"Repeated text encoding is not deterministic; maximum delta is {max_delta:.3e}")

    embeddings_cpu = embeddings_first.detach().to(device="cpu", dtype=torch.float32).contiguous()
    tensor_path = output_dir / "mobileclip_s0_embeddings.pt"
    numpy_path = output_dir / "mobileclip_s0_embeddings.npy"
    metadata_path = output_dir / "mobileclip_s0_metadata.json"
    torch.save(embeddings_cpu, tensor_path)
    np.save(numpy_path, embeddings_cpu.numpy(), allow_pickle=False)

    try:
        mobileclip_version = importlib.metadata.version("mobileclip")
    except importlib.metadata.PackageNotFoundError:
        mobileclip_version = "editable-unversioned"
    metadata = {
        "schema_version": 1,
        "encoder": args.encoder_name,
        "model_architecture": args.model_arch,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "repository": str(repository),
        "repository_commit": git_commit(repository),
        "mobileclip_package_version": mobileclip_version,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "embedding_dimension": int(embeddings_cpu.shape[-1]),
        "embedding_shape": list(embeddings_cpu.shape),
        "dtype": "float32",
        "normalized": True,
        "prompts": prompts,
        "prompts_config_sha256": sha256_file(prompts_config),
        "prompt_set": args.prompt_set,
        "device": str(device),
        "seed": args.seed,
        "encoding_latency_ms": {
            "first": latency_first_ms,
            "repeat": latency_second_ms,
            "per_prompt_first": latency_first_ms / len(prompts),
        },
        "deterministic_repeat": True,
        "repeat_max_abs_delta": max_delta,
        "tensor_summary": tensor_summary(embeddings_cpu),
        "payload_bytes": embeddings_cpu.numel() * embeddings_cpu.element_size(),
        "outputs": {
            "pytorch": str(tensor_path.relative_to(EXPERIMENT_ROOT)),
            "pytorch_sha256": sha256_file(tensor_path),
            "numpy": str(numpy_path.relative_to(EXPERIMENT_ROOT)),
            "numpy_sha256": sha256_file(numpy_path),
        },
    }
    write_json(metadata_path, metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
