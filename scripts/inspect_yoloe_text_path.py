#!/usr/bin/env python3
"""Inspect and validate YOLOE's dynamic text-embedding path without fusing the model."""

from __future__ import annotations

import argparse

from _common import ensure_output_directory, load_json, load_prompts, load_tensor, require_file, require_output_file, set_deterministic, tensor_summary, validate_embeddings
from _yoloe_runner import environment_metadata, get_official_embeddings, install_classes, load_offline_blt_encoder, load_yoloe, print_metadata, validate_dynamic_head, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/yoloe-v8s-seg.pt")
    parser.add_argument("--blt-checkpoint", default="artifacts/checkpoints/mobileclip_blt.ts")
    parser.add_argument("--yoloe-repo", default="third_party/yoloe")
    parser.add_argument("--clip-repo", default="third_party/clip")
    parser.add_argument("--s0-embeddings", default="outputs/embeddings/mobileclip_s0_embeddings.pt")
    parser.add_argument("--s0-metadata", default="outputs/embeddings/mobileclip_s0_metadata.json")
    parser.add_argument("--prompts-config", default="configs/prompts.yaml")
    parser.add_argument("--prompt-set", default="visdrone_prompts")
    parser.add_argument("--output-dir", default="outputs/comparisons")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_deterministic(args.seed)
    checkpoint = require_file(args.checkpoint, "YOLOE-v8s checkpoint")
    blt_checkpoint = require_file(args.blt_checkpoint, "MobileCLIP-B(LT) TorchScript checkpoint")
    s0_path = require_output_file(args.s0_embeddings, "MobileCLIP-S0 embeddings")
    s0_metadata_path = require_output_file(args.s0_metadata, "MobileCLIP-S0 metadata")
    prompts = load_prompts(args.prompts_config, args.prompt_set)
    s0_metadata = load_json(s0_metadata_path, "MobileCLIP-S0 metadata")
    if s0_metadata.get("prompts") != prompts:
        raise ValueError("S0 metadata prompts do not exactly match the requested ordered prompts")
    s0 = load_tensor(s0_path).float()
    validate_embeddings(s0, prompt_count=len(prompts), expected_ndim=2, require_normalized=True)

    model, device, repository = load_yoloe(checkpoint, args.yoloe_repo, args.device)
    head = validate_dynamic_head(model)
    encoder, clip_repository = load_offline_blt_encoder(
        checkpoint=blt_checkpoint,
        clip_repository_arg=args.clip_repo,
        device=device,
        model=model,
    )
    raw_official, adapted_official = get_official_embeddings(model, encoder, head, prompts)
    validate_embeddings(raw_official, prompt_count=len(prompts), expected_ndim=3, require_normalized=True)
    validate_embeddings(adapted_official, prompt_count=len(prompts), expected_ndim=3, require_normalized=True)
    expected_dimension = int(head.embed)
    if raw_official.shape[-1] != expected_dimension:
        raise ValueError(f"Official B(LT) dimension {raw_official.shape[-1]} differs from RepRTA input {expected_dimension}")
    if s0.shape[-1] != expected_dimension:
        raise ValueError(f"S0 dimension {s0.shape[-1]} differs from RepRTA input {expected_dimension}")
    install_classes(model, prompts, adapted_official)

    metadata = environment_metadata(
        checkpoint,
        repository,
        model,
        head,
        blt_checkpoint=blt_checkpoint,
        clip_repository=clip_repository,
    )
    metadata.update(
        {
            "schema_version": 1,
            "inspection_status": "passed",
            "device": str(device),
            "requested_prompt_count": len(prompts),
            "installed_prompt_count": len(model.model.names),
            "official_raw_embedding": tensor_summary(raw_official),
            "official_post_reprta_embedding": tensor_summary(adapted_official),
            "mobileclip_s0_embedding": tensor_summary(s0),
            "dimensional_compatibility": True,
            "api_compatibility": "not_tested_by_inspection",
            "semantic_compatibility": "not_tested_by_inspection",
            "prompt_fusion_called": False,
            "checkpoint_modified": False,
        }
    )
    output_dir = ensure_output_directory(args.output_dir)
    write_json(output_dir / "yoloe_text_path_inspection.json", metadata)
    print_metadata(metadata)


if __name__ == "__main__":
    main()
