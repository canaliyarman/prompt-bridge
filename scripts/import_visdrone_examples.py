#!/usr/bin/env python3
"""Copy selected VisDrone pairs and emit schema-v1 native-class annotations."""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

from _common import ensure_output_directory, load_prompts, require_directory, set_deterministic, sha256_file, write_json

VISDRONE_CATEGORIES = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning tricycle",
    9: "bus",
    10: "motorcycle",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="VisDrone split containing images/ and annotations/")
    parser.add_argument("--image-ids", nargs="+", required=True, help="Image stems without extensions")
    parser.add_argument("--prompts-config", default="configs/prompts.yaml")
    parser.add_argument("--prompt-set", default="visdrone_prompts")
    parser.add_argument("--images-output", default="data/images")
    parser.add_argument("--annotations-output", default="data/annotations")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def copy_verified(source: Path, destination: Path) -> None:
    """Copy a file or verify that the existing destination is byte-identical."""
    if destination.exists():
        if sha256_file(source) != sha256_file(destination):
            raise FileExistsError(f"Destination exists with different content: {destination}")
        return
    shutil.copy2(source, destination)


def parse_objects(
    annotation_path: Path,
    image_id: str,
    prompt_ids: dict[int, int],
) -> tuple[list[dict[str, object]], Counter[int]]:
    """Convert valid VisDrone xywh rows to experiment xyxy objects for all ten native classes."""
    objects: list[dict[str, object]] = []
    counts: Counter[int] = Counter()
    all_prompt_ids = list(prompt_ids.values())
    for line_number, line in enumerate(annotation_path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split(",")
        if len(fields) != 8:
            raise ValueError(f"{annotation_path}:{line_number}: expected 8 comma-separated fields")
        x, y, width, height, score, category, _, _ = (int(value) for value in fields)
        if score == 0 or category not in VISDRONE_CATEGORIES:
            continue
        if width <= 0 or height <= 0:
            raise ValueError(f"{annotation_path}:{line_number}: non-positive box dimensions")
        prompt_id = prompt_ids[category]
        counts[category] += 1
        label = VISDRONE_CATEGORIES[category].replace(" ", "-")
        objects.append(
            {
                "id": f"{image_id}-{label}-{counts[category]:04d}",
                "box_xyxy": [x, y, x + width, y + height],
                "positive_prompt_ids": [prompt_id],
                "evaluable_hard_negative_prompt_ids": [value for value in all_prompt_ids if value != prompt_id],
                "mission_relevant": True,
            }
        )
    return objects, counts


def main() -> None:
    args = parse_args()
    set_deterministic(args.seed)
    source = require_directory(args.source, "VisDrone source split")
    source_images = source / "images"
    source_annotations = source / "annotations"
    if not source_images.is_dir() or not source_annotations.is_dir():
        raise FileNotFoundError(f"{source} must contain images/ and annotations/")
    prompts = load_prompts(args.prompts_config, args.prompt_set)
    missing = [name for name in VISDRONE_CATEGORIES.values() if name not in prompts]
    if missing:
        raise ValueError(f"Prompt set {args.prompt_set!r} lacks VisDrone categories: {missing}")
    prompt_ids = {category: prompts.index(name) for category, name in VISDRONE_CATEGORIES.items()}
    if len(set(prompt_ids.values())) != len(VISDRONE_CATEGORIES):
        raise ValueError("VisDrone categories do not map to unique prompt IDs")
    images_output = ensure_output_directory(args.images_output)
    annotations_output = ensure_output_directory(args.annotations_output)
    raw_output = annotations_output / "visdrone_raw"
    raw_output.mkdir(parents=True, exist_ok=True)

    image_entries: list[dict[str, object]] = []
    provenance: list[dict[str, object]] = []
    seen: set[str] = set()
    total_objects = 0
    for image_id in args.image_ids:
        if image_id in seen:
            raise ValueError(f"Duplicate requested image ID: {image_id}")
        seen.add(image_id)
        source_image = source_images / f"{image_id}.jpg"
        source_annotation = source_annotations / f"{image_id}.txt"
        if not source_image.is_file() or not source_annotation.is_file():
            raise FileNotFoundError(f"Missing VisDrone pair for {image_id}")
        destination_image = images_output / source_image.name
        destination_annotation = raw_output / source_annotation.name
        copy_verified(source_image, destination_image)
        copy_verified(source_annotation, destination_annotation)
        objects, counts = parse_objects(source_annotation, image_id, prompt_ids)
        total_objects += len(objects)
        image_entries.append(
            {
                "image_id": image_id,
                "file_name": source_image.name,
                "scenario_tags": ["visdrone", "aerial", "supplemental_native_class_smoke"],
                "objects": objects,
            }
        )
        provenance.append(
            {
                "image_id": image_id,
                "source_image": str(source_image),
                "source_annotation": str(source_annotation),
                "image_sha256": sha256_file(destination_image),
                "raw_annotation_sha256": sha256_file(destination_annotation),
                "class_counts": {VISDRONE_CATEGORIES[key]: counts[key] for key in sorted(counts)},
                "object_count": len(objects),
            }
        )

    write_json(annotations_output / "annotations.json", {"schema_version": 1, "images": image_entries})
    write_json(
        annotations_output / "visdrone_import_metadata.json",
        {
            "schema_version": 1,
            "dataset": "VisDrone2019-DET-test-dev",
            "source": str(source),
            "source_format": "x,y,width,height,score,category,truncation,occlusion",
            "prompt_set": args.prompt_set,
            "prompts": prompts,
            "category_mapping": {
                str(category): {"prompt_id": prompt_ids[category], "prompt_text": name}
                for category, name in VISDRONE_CATEGORIES.items()
            },
            "attribute_labels_available": False,
            "manufacturer_labels_available": False,
            "hard_negative_definition": "other VisDrone native categories",
            "images": provenance,
        },
    )
    print(f"Imported {len(image_entries)} VisDrone pairs with {total_objects} native-class objects")


if __name__ == "__main__":
    main()
