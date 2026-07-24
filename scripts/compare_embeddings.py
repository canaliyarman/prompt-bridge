#!/usr/bin/env python3
"""Compare relational structure in official B(LT) and MobileCLIP-S0 text spaces."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from _common import ensure_output_directory, load_prompts, load_tensor, require_output_file, tensor_summary, validate_embeddings, write_json

RELATIONSHIPS = [
    ("red Fiat car", "Fiat car"),
    ("red Fiat car", "red car"),
    ("red Fiat car", "white Fiat car"),
    ("red Fiat car", "red Renault car"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", default="outputs/baseline/official_blt_raw_embeddings.pt")
    parser.add_argument("--s0", default="outputs/embeddings/mobileclip_s0_embeddings.pt")
    parser.add_argument("--prompts-config", default="configs/prompts.yaml")
    parser.add_argument("--prompt-set", default="visdrone_prompts")
    parser.add_argument("--output-dir", default="outputs/comparisons")
    return parser.parse_args()


def as_matrix(path: str, prompts: list[str], label: str) -> torch.Tensor:
    tensor = load_tensor(require_output_file(path, label)).float()
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    validate_embeddings(tensor, prompt_count=len(prompts), expected_ndim=2, require_normalized=True)
    return tensor


def relational_summary(matrix: torch.Tensor, prompts: list[str]) -> dict[str, object]:
    cosine = matrix @ matrix.T
    rankings: dict[str, list[dict[str, object]]] = {}
    for row, prompt in enumerate(prompts):
        order = torch.argsort(cosine[row], descending=True).tolist()
        rankings[prompt] = [
            {"prompt": prompts[index], "cosine": float(cosine[row, index])}
            for index in order
            if index != row
        ]
    relationships = []
    for left, right in RELATIONSHIPS:
        if left in prompts and right in prompts:
            relationships.append(
                {"left": left, "right": right, "cosine": float(cosine[prompts.index(left), prompts.index(right)])}
            )
    return {"cosine_matrix": cosine.tolist(), "rankings": rankings, "selected_relationships": relationships}


def coordinates(matrix: torch.Tensor) -> np.ndarray:
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    dimensions = min(2, vh.shape[0])
    projected = centered @ vh[:dimensions].T
    if dimensions == 1:
        projected = torch.cat([projected, torch.zeros_like(projected)], dim=1)
    return projected.cpu().numpy()


def save_plots(official: torch.Tensor, s0: torch.Tensor, prompts: list[str], output_dir: object) -> None:
    import matplotlib.pyplot as plt

    output = output_dir
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for axis, matrix, title in zip(axes, (official, s0), ("Official B(LT)", "MobileCLIP-S0")):
        image = axis.imshow((matrix @ matrix.T).cpu().numpy(), vmin=-1, vmax=1, cmap="coolwarm")
        axis.set_xticks(range(len(prompts)), prompts, rotation=60, ha="right")
        axis.set_yticks(range(len(prompts)), prompts)
        axis.set_title(title)
    fig.colorbar(image, ax=axes, shrink=0.8)
    fig.savefig(output / "cosine_matrices.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for axis, matrix, title in zip(axes, (official, s0), ("Official B(LT) PCA", "MobileCLIP-S0 PCA")):
        xy = coordinates(matrix)
        axis.scatter(xy[:, 0], xy[:, 1])
        for index, prompt in enumerate(prompts):
            axis.annotate(prompt, xy[index])
        axis.set_title(title)
    fig.savefig(output / "prompt_clusters.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args.prompts_config, args.prompt_set)
    official = as_matrix(args.official, prompts, "official B(LT) embeddings")
    s0 = as_matrix(args.s0, prompts, "MobileCLIP-S0 embeddings")
    if official.shape != s0.shape:
        raise ValueError(f"Embedding shapes differ: official {tuple(official.shape)}, S0 {tuple(s0.shape)}")
    output_dir = ensure_output_directory(args.output_dir)
    result = {
        "schema_version": 1,
        "interpretation": "Relational comparisons only; element-wise cross-space differences are not semantic errors.",
        "prompts": prompts,
        "official": {"tensor": tensor_summary(official), **relational_summary(official, prompts)},
        "mobileclip_s0": {"tensor": tensor_summary(s0), **relational_summary(s0, prompts)},
        "primary_conclusion_source": "detection metrics, not embedding visualization",
    }
    np.save(output_dir / "official_cosine_matrix.npy", (official @ official.T).cpu().numpy(), allow_pickle=False)
    np.save(output_dir / "s0_cosine_matrix.npy", (s0 @ s0.T).cpu().numpy(), allow_pickle=False)
    save_plots(official, s0, prompts, output_dir)
    write_json(output_dir / "embedding_comparison.json", result)


if __name__ == "__main__":
    main()
