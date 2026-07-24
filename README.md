# Prompt Bridge

Prompt Bridge is a workstation evaluation harness for connecting alternative text encoders to open-vocabulary detectors. Its first experiment asks one concrete question:

> Can an unmodified YOLOE-v8s detector meaningfully consume normalized MobileCLIP-S0 text embeddings, or does it require explicit embedding-space alignment?

The answer from the current five-image VisDrone smoke test is: **the tensors are accepted, but direct S0 injection is not semantically compatible**. YOLOE's official MobileCLIP-B(LT) path produced detections at two confidence thresholds; direct MobileCLIP-S0 injection produced none. The next experiment is a learned linear alignment adapter.

[![tests](https://github.com/canaliyarman/prompt-bridge/actions/workflows/tests.yml/badge.svg)](https://github.com/canaliyarman/prompt-bridge/actions/workflows/tests.yml)

## What this repository establishes

Prompt Bridge deliberately separates claims that are easy to conflate:

| Gate | Meaning | Current result |
| --- | --- | --- |
| Numerical validity | Embeddings are finite, float32, normalized, and deterministic | Pass |
| Dimensional compatibility | S0 output matches YOLOE RepRTA's discovered input width | Pass: 512 |
| API compatibility | YOLOE executes through its supported dynamic-class/RepRTA path | Pass |
| Semantic compatibility | Injected embeddings retain useful detection behavior | Fail on current smoke set |
| Deployment suitability | Quality is sufficient to package/transmit embeddings | Blocked |

Equal tensor dimensions are not evidence that two encoders use the same semantic coordinate system.

## System flow

```text
Mode A: prompt -> MobileCLIP-B(LT) -> YOLOE RepRTA -> YOLOE visual matching -> boxes
Mode B: prompt -> MobileCLIP-S0    -> YOLOE RepRTA -> YOLOE visual matching -> boxes
                                  same images, prompts, thresholds, and NMS
                                                    |
                                                    v
                                          quantitative comparison
```

YOLOE's official aligned B(LT) path is always run first. Mode B is rejected unless its prompt order, image hashes, checkpoint hash, resolution, confidence, NMS configuration, and detection cap match the signed baseline metadata.

## Current result

The active smoke set uses five VisDrone images, 460 native-category annotations, and these prompts:

`pedestrian`, `people`, `bicycle`, `car`, `van`, `truck`, `tricycle`, `awning tricycle`, `bus`, and `motorcycle`.

| Confidence | Official B(LT) predictions | Direct S0 predictions | Baseline recall | Direct recall |
| ---: | ---: | ---: | ---: | ---: |
| 0.25 | 34 | 0 | 0.0565 | 0 |
| 0.05 | 134 | 0 | 0.1130 | 0 |

This is a small, difficult aerial smoke test—not a general YOLOE benchmark. VisDrone has no manufacturer or color labels, so it supports no Fiat, Renault, or vehicle-color claim. See [the full workstation report](reports/mobileclip_s0_yoloe_v8s_report.md).

## Repository layout

```text
.
├── artifacts/checkpoints/       # local weights; ignored by Git
├── configs/
│   ├── experiment.yaml
│   ├── prompts.yaml
│   └── visdrone_examples.yaml
├── data/
│   ├── images/                  # local/licensed images; ignored
│   └── annotations/             # generated annotations; schema is tracked
├── docs/
│   ├── architecture.md
│   ├── evaluation.md
│   └── setup.md
├── outputs/                     # generated embeddings/predictions; ignored
├── reports/                     # reviewed result reports
├── scripts/
├── tests/
└── third_party/                 # pinned editable upstream checkouts; ignored
```

## Quick start

Python 3.10 is the tested version. Model weights and upstream checkouts are intentionally never downloaded by the scripts.

```bash
python3.10 -m venv .venv
source .venv/bin/activate

# Choose the correct PyTorch build for your hardware first.
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Provision the pinned source checkouts and checkpoints by following [docs/setup.md](docs/setup.md). Every missing checkpoint fails explicitly.

Run the unit tests:

```bash
python -m pytest -q
```

## Reproduce the VisDrone smoke test

Import the selected local VisDrone pairs:

```bash
python scripts/import_visdrone_examples.py \
  --source /path/to/VisDrone2019-DET-test-dev \
  --image-ids 0000078_06777_d_0000020 0000078_02363_d_0000007 \
    9999952_00000_d_0000009 0000278_02951_d_0000008 0000063_00500_d_0000001
```

Then execute the baseline-first pipeline:

```bash
python scripts/encode_mobileclip_s0.py --device cuda:0 --prompt-set visdrone_prompts
python scripts/inspect_yoloe_text_path.py --device cuda:0 --prompt-set visdrone_prompts
python scripts/run_yoloe_baseline.py --device cuda:0 --prompt-set visdrone_prompts
python scripts/run_yoloe_s0_direct.py --device cuda:0 --prompt-set visdrone_prompts
python scripts/compare_embeddings.py --prompt-set visdrone_prompts
python scripts/evaluate_predictions.py
```

CPU is the safe default; GPU execution must be explicitly requested. On the tested GTX 1060 (`sm_61`), the workstation's CUDA 12.8 PyTorch build was incompatible, while PyTorch 2.7.1 with CUDA 11.8 executed correctly.

Detailed commands, offline guarantees, output contracts, metrics, and quality gates are in [docs/evaluation.md](docs/evaluation.md).

## What happens next

The current evidence calls for Mode C:

1. Generate paired S0 and official B(LT) embeddings for a phrase corpus.
2. Train a normalized `Linear(512, 512)` mapping after confirming both dimensions.
3. Repeat the exact signed evaluation.
4. If alignment remains insufficient, freeze S0 and fine-tune YOLOE RepRTA plus its visual/contrastive heads.

Prompt export, quantization, FPGA compilation, and ZCU104 integration remain blocked until float32 workstation quality passes.

## Safety and scope

This repository does not:

- download model weights implicitly;
- modify upstream MobileCLIP or Ultralytics source;
- quantize or compile YOLOE;
- execute MobileCLIP on the ZCU104;
- modify Vitis AI or board runtime code;
- treat a successful forward pass as proof of detection quality.
