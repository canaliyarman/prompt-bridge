#!/usr/bin/env python3
"""Run the official offline B(LT) YOLOE text-embedding baseline."""

from __future__ import annotations

import argparse

import torch

from _common import ensure_output_directory, load_prompts, require_file, set_deterministic, sha256_file, tensor_summary, validate_embeddings
from _yoloe_runner import (
    canonical_sha256,
    environment_metadata,
    get_official_embeddings,
    install_classes,
    load_offline_blt_encoder,
    load_yoloe,
    predict_and_save,
    print_metadata,
    run_signature_material,
    validate_dynamic_head,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/yoloe-v8s-seg.pt")
    parser.add_argument("--blt-checkpoint", default="artifacts/checkpoints/mobileclip_blt.ts")
    parser.add_argument("--yoloe-repo", default="third_party/yoloe")
    parser.add_argument("--clip-repo", default="third_party/clip")
    parser.add_argument("--prompts-config", default="configs/prompts.yaml")
    parser.add_argument("--prompt-set", default="visdrone_prompts")
    parser.add_argument("--images", default="data/images")
    parser.add_argument("--output-dir", default="outputs/baseline")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_deterministic(args.seed)
    checkpoint = require_file(args.checkpoint, "YOLOE-v8s checkpoint")
    blt_checkpoint = require_file(args.blt_checkpoint, "MobileCLIP-B(LT) TorchScript checkpoint")
    prompts = load_prompts(args.prompts_config, args.prompt_set)
    output_dir = ensure_output_directory(args.output_dir)
    model, device, repository = load_yoloe(checkpoint, args.yoloe_repo, args.device)
    head = validate_dynamic_head(model)
    encoder, clip_repository = load_offline_blt_encoder(
        checkpoint=blt_checkpoint,
        clip_repository_arg=args.clip_repo,
        device=device,
        model=model,
    )
    raw_embeddings, official_embeddings = get_official_embeddings(model, encoder, head, prompts)
    validate_embeddings(raw_embeddings, prompt_count=len(prompts), expected_ndim=3, require_normalized=True)
    validate_embeddings(official_embeddings, prompt_count=len(prompts), expected_ndim=3, require_normalized=True)
    if int(raw_embeddings.shape[-1]) != int(head.embed):
        raise ValueError(f"B(LT) dimension {raw_embeddings.shape[-1]} differs from RepRTA input {head.embed}")
    install_classes(model, prompts, official_embeddings)

    raw_path = output_dir / "official_blt_raw_embeddings.pt"
    official_path = output_dir / "official_yoloe_embeddings.pt"
    torch.save(raw_embeddings.detach().cpu().float(), raw_path)
    torch.save(official_embeddings.detach().cpu().float(), official_path)
    records, images, manifest = predict_and_save(
        model=model,
        device=device,
        images_source=args.images,
        prompts=prompts,
        output_dir_arg=args.output_dir,
        image_size=args.image_size,
        confidence=args.confidence,
        iou=args.iou,
        warmup_runs=args.warmup_runs,
        repetitions=args.repetitions,
        max_detections=args.max_detections,
    )
    checkpoint_hash = sha256_file(checkpoint)
    signature_material = run_signature_material(
        prompts=prompts,
        manifest=manifest,
        checkpoint_sha256=checkpoint_hash,
        image_size=args.image_size,
        confidence=args.confidence,
        iou=args.iou,
        max_detections=args.max_detections,
    )
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
            "mode": "official_yoloe_baseline",
            "gate_status": "passed",
            "device": str(device),
            "seed": args.seed,
            "prompts": prompts,
            "raw_blt_embedding": tensor_summary(raw_embeddings),
            "official_reprta_embedding": tensor_summary(official_embeddings),
            "raw_blt_embedding_sha256": sha256_file(raw_path),
            "official_embedding_sha256": sha256_file(official_path),
            "image_size": args.image_size,
            "confidence_threshold": args.confidence,
            "iou_threshold": args.iou,
            "agnostic_nms": False,
            "max_detections": args.max_detections,
            "warmup_runs": args.warmup_runs,
            "measured_repetitions": args.repetitions,
            "image_count": len(images),
            "detection_count": len(records),
            "image_manifest": manifest,
            "run_signature_material": signature_material,
            "run_signature_sha256": canonical_sha256(signature_material),
        }
    )
    write_json(output_dir / "metadata.json", metadata)
    print_metadata(metadata)


if __name__ == "__main__":
    main()
