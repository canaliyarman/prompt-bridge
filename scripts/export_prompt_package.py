#!/usr/bin/env python3
"""Export a float32 prompt package only after the detection-quality decision passes."""

from __future__ import annotations

import argparse
import json

import numpy as np

from _common import ensure_output_directory, load_json, load_tensor, require_output_file, sha256_file, validate_embeddings, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", default="outputs/embeddings/mobileclip_s0_embeddings.pt")
    parser.add_argument("--embedding-metadata", default="outputs/embeddings/mobileclip_s0_metadata.json")
    parser.add_argument("--quality-decision", default="outputs/comparisons/detection_comparison.json")
    parser.add_argument("--output-dir", default="outputs/prompt_packages")
    parser.add_argument("--adapter", default=None, help="Optional trained adapter checkpoint; omitted for direct embeddings")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embedding_path = require_output_file(args.embeddings, "prompt embeddings")
    metadata_path = require_output_file(args.embedding_metadata, "embedding metadata")
    decision_path = require_output_file(args.quality_decision, "quality decision")
    metadata = load_json(metadata_path, "embedding metadata")
    comparison = load_json(decision_path, "quality decision")
    if comparison.get("decision", {}).get("provisionally_close") is not True:
        raise RuntimeError("Prompt export is blocked until the detection-quality decision is provisionally close")
    embeddings = load_tensor(embedding_path).detach().cpu().float().contiguous()
    prompts = metadata.get("prompts")
    if not isinstance(prompts, list):
        raise ValueError("Embedding metadata lacks an ordered prompts list")
    validate_embeddings(embeddings, prompt_count=len(prompts), expected_ndim=2, require_normalized=True)
    recorded_hash = metadata.get("outputs", {}).get("pytorch_sha256")
    if not recorded_hash or sha256_file(embedding_path) != recorded_hash:
        raise ValueError("Embedding tensor hash differs from metadata")
    adapter_path = require_output_file(args.adapter, "adapter checkpoint") if args.adapter else None
    output_dir = ensure_output_directory(args.output_dir)
    binary_path = output_dir / "mobileclip_s0_prompt_embeddings.npy"
    np.save(binary_path, embeddings.numpy(), allow_pickle=False)
    package = {
        "schema_version": 1,
        "encoder": "MobileCLIP-S0",
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "adapter": str(adapter_path) if adapter_path else None,
        "adapter_sha256": sha256_file(adapter_path) if adapter_path else None,
        "embedding_dimension": int(embeddings.shape[-1]),
        "dtype": "float32",
        "normalized": True,
        "prompts": [
            {"id": index, "text": prompt, "vector": embeddings[index].tolist()}
            for index, prompt in enumerate(prompts)
        ],
    }
    sidecar = {
        key: value for key, value in package.items() if key != "prompts"
    }
    sidecar.update(
        {
            "prompts": [{"id": index, "text": prompt} for index, prompt in enumerate(prompts)],
            "binary_format": "npy",
            "binary_file": binary_path.name,
            "binary_sha256": sha256_file(binary_path),
            "shape": list(embeddings.shape),
            "payload_bytes": embeddings.numel() * embeddings.element_size(),
            "quality_decision_sha256": sha256_file(decision_path),
        }
    )
    write_json(output_dir / "mobileclip_s0_prompt_package.json", package)
    write_json(output_dir / "mobileclip_s0_prompt_embeddings.metadata.json", sidecar)
    print(json.dumps(sidecar, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
