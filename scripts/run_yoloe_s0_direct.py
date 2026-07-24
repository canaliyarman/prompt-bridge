#!/usr/bin/env python3
"""Inject normalized MobileCLIP-S0 embeddings through YOLOE's existing RepRTA path."""

from __future__ import annotations

import argparse

import torch

from _common import (
    canonical_sha256,
    collect_images,
    ensure_output_directory,
    image_manifest,
    load_json,
    load_prompts,
    load_tensor,
    require_file,
    require_output_file,
    set_deterministic,
    sha256_file,
    tensor_summary,
    validate_embeddings,
)
from _yoloe_runner import (
    environment_metadata,
    install_classes,
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
    parser.add_argument("--yoloe-repo", default="third_party/yoloe")
    parser.add_argument("--embeddings", default="outputs/embeddings/mobileclip_s0_embeddings.pt")
    parser.add_argument("--embedding-metadata", default="outputs/embeddings/mobileclip_s0_metadata.json")
    parser.add_argument("--baseline-metadata", default="outputs/baseline/metadata.json")
    parser.add_argument("--prompts-config", default="configs/prompts.yaml")
    parser.add_argument("--prompt-set", default="visdrone_prompts")
    parser.add_argument("--images", default="data/images")
    parser.add_argument("--output-dir", default="outputs/s0_direct")
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
    embedding_path = require_output_file(args.embeddings, "MobileCLIP-S0 embeddings")
    embedding_metadata_path = require_output_file(args.embedding_metadata, "MobileCLIP-S0 metadata")
    baseline_metadata_path = require_output_file(args.baseline_metadata, "passed baseline metadata")
    prompts = load_prompts(args.prompts_config, args.prompt_set)
    saved_metadata = load_json(embedding_metadata_path, "MobileCLIP-S0 metadata")
    baseline_metadata = load_json(baseline_metadata_path, "baseline metadata")
    if baseline_metadata.get("mode") != "official_yoloe_baseline" or baseline_metadata.get("gate_status") != "passed":
        raise ValueError("Direct injection requires a completed, passed official YOLOE baseline artifact")
    if saved_metadata.get("prompts") != prompts:
        raise ValueError("Saved embedding prompts do not exactly match the requested ordered prompt set")
    if saved_metadata.get("encoder") != "MobileCLIP-S0":
        raise ValueError(f"Expected MobileCLIP-S0 metadata, got {saved_metadata.get('encoder')!r}")
    expected_tensor_hash = saved_metadata.get("outputs", {}).get("pytorch_sha256")
    if not expected_tensor_hash or sha256_file(embedding_path) != expected_tensor_hash:
        raise ValueError("MobileCLIP-S0 tensor SHA-256 does not match its metadata")

    raw_embeddings = load_tensor(embedding_path).float()
    validate_embeddings(raw_embeddings, prompt_count=len(prompts), expected_ndim=2, require_normalized=True)
    if int(saved_metadata.get("embedding_dimension", -1)) != int(raw_embeddings.shape[-1]):
        raise ValueError("Embedding metadata dimension does not match the saved tensor")

    current_manifest = image_manifest(collect_images(args.images))
    checkpoint_hash = sha256_file(checkpoint)
    signature_material = run_signature_material(
        prompts=prompts,
        manifest=current_manifest,
        checkpoint_sha256=checkpoint_hash,
        image_size=args.image_size,
        confidence=args.confidence,
        iou=args.iou,
        max_detections=args.max_detections,
    )
    signature = canonical_sha256(signature_material)
    if baseline_metadata.get("run_signature_sha256") != signature:
        raise ValueError("Baseline/direct configuration mismatch: prompts, images, checkpoint, or inference settings changed")
    if baseline_metadata.get("run_signature_material") != signature_material:
        raise ValueError("Baseline/direct signature collision guard failed: canonical settings differ")

    output_dir = ensure_output_directory(args.output_dir)
    model, device, repository = load_yoloe(checkpoint, args.yoloe_repo, args.device)
    head = validate_dynamic_head(model)
    expected_dimension = int(head.embed)
    if int(raw_embeddings.shape[-1]) != expected_dimension:
        raise ValueError(
            f"Dimensional incompatibility: MobileCLIP-S0 emits {raw_embeddings.shape[-1]} values, "
            f"while YOLOE RepRTA expects {expected_dimension}"
        )

    raw_batched = raw_embeddings.unsqueeze(0).to(device)
    with torch.inference_mode():
        adapted_embeddings = head.get_tpe(raw_batched)
    if adapted_embeddings is None:
        raise RuntimeError("YOLOE RepRTA unexpectedly returned no embeddings")
    validate_embeddings(adapted_embeddings, prompt_count=len(prompts), expected_ndim=3, require_normalized=True)
    install_classes(model, prompts, adapted_embeddings)

    raw_copy_path = output_dir / "mobileclip_s0_raw_embeddings.pt"
    adapted_path = output_dir / "mobileclip_s0_after_reprta.pt"
    torch.save(raw_embeddings.detach().cpu().float(), raw_copy_path)
    torch.save(adapted_embeddings.detach().cpu().float(), adapted_path)
    records, images, resulting_manifest = predict_and_save(
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
    if resulting_manifest != current_manifest:
        raise RuntimeError("Image manifest changed during direct-injection execution")
    metadata = environment_metadata(checkpoint, repository, model, head)
    metadata.update(
        {
            "schema_version": 1,
            "mode": "mobileclip_s0_direct_injection",
            "gate_status": "execution_passed_quality_pending",
            "interpretation": "API, dimensional, and numerical execution only; semantic compatibility requires metrics.",
            "baseline_metadata": str(baseline_metadata_path),
            "baseline_metadata_sha256": sha256_file(baseline_metadata_path),
            "baseline_run_signature_sha256": signature,
            "device": str(device),
            "seed": args.seed,
            "prompts": prompts,
            "mobileclip_metadata": saved_metadata,
            "raw_embedding": tensor_summary(raw_embeddings),
            "reprta_embedding": tensor_summary(adapted_embeddings),
            "raw_embedding_sha256": sha256_file(raw_copy_path),
            "reprta_embedding_sha256": sha256_file(adapted_path),
            "reprta_applied": True,
            "image_size": args.image_size,
            "confidence_threshold": args.confidence,
            "iou_threshold": args.iou,
            "agnostic_nms": False,
            "max_detections": args.max_detections,
            "warmup_runs": args.warmup_runs,
            "measured_repetitions": args.repetitions,
            "image_count": len(images),
            "detection_count": len(records),
            "image_manifest": resulting_manifest,
            "run_signature_material": signature_material,
            "run_signature_sha256": signature,
        }
    )
    write_json(output_dir / "metadata.json", metadata)
    print_metadata(metadata)


if __name__ == "__main__":
    main()
