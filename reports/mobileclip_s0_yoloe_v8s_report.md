# MobileCLIP-S0 with YOLOE-v8s Workstation Report

## Result status

**Observed result:** Unmodified YOLOE-v8s accepts normalized MobileCLIP-S0 embeddings through its existing RepRTA path, so API, dimensional, and numerical compatibility pass. Semantic compatibility fails on the five-image native VisDrone smoke set: direct S0 produced zero detections at both confidence 0.25 and 0.05, while the official MobileCLIP-B(LT) baseline produced 34 and 134 detections respectively.

**Decision:** Direct MobileCLIP-S0 injection is not acceptable on the current evidence. The next authorized experiment should be a text-only linear adapter trained from S0 embeddings to official B(LT) embeddings. If that is insufficient, align RepRTA and the YOLOE visual/contrastive heads. No prompt package was exported, and no Vitis AI, quantization, or ZCU104 files were changed.

**Scope limit:** This result is an aerial native-class smoke test, not a general YOLOE benchmark. VisDrone does not label vehicle color or manufacturer, so Fiat/Renault/color conclusions remain outside this dataset.

## 1. Environment and source versions

Observed environment:

- Python 3.10.19
- PyTorch 2.7.1+cu118
- torchvision 0.22.1
- CUDA runtime reported by PyTorch: 11.8
- GPU: NVIDIA GeForce GTX 1060 with Max-Q Design, compute capability 6.1
- Ultralytics 8.4.14
- MobileCLIP package 0.1.0
- Apple MobileCLIP commit: `aecfb5453d022e9deff12f81a150ea8f35194baa`
- Ultralytics commit: `fc3d6e29133f97db7c0cb59b6a9cf21cb9d9be74`
- Ultralytics CLIP commit: `c4b6ea0932a2c0f39a0fa528af5ec4982ff15cab`

A real CUDA tensor operation and all measured experiment runs completed on the GTX 1060 with `--device cuda:0`. CPU remains the CLI default for portability. The workstation's base Python 3.13 PyTorch 2.10/CUDA 12.8 build supports `sm_70` and newer, so it cannot execute on this `sm_61` GPU; the dedicated CUDA 11.8 environment resolves that issue.

**Compatibility exception:** The Apple checkout currently declares PyTorch >=2.8 and torchvision >=0.23, while the tested CUDA 11.8 environment uses PyTorch 2.7.1 and torchvision 0.22.1 for GTX 1060 support. The exercised text-only S0 path passes, but this declared-version mismatch should remain recorded in any frozen environment.

## 2. Checkpoint hashes

| Checkpoint | SHA-256 |
| --- | --- |
| `mobileclip_s0.pt` | `809b408eff74f8058843e86a1f92967097d42ba782450e85b8f4867b7f0ca0b7` |
| `mobileclip_blt.ts` | `a67804d1b0f07b8b9a20c1761ec0847f34660f5fa338ec70e8f3fce68ed95e54` |
| `yoloe-v8s-seg.pt` | `ac2b90ed23011495a3e86d89caeb3432a15129cac8d849ba121293c8fc1e0536` |

The S0 cache and workspace are on different filesystems, so a hard link was impossible. The checkpoint was copied byte-for-byte and verified by SHA-256. The YOLOE checkpoint resides under `workspace/models/source/`. Missing weights cause failure; none of the experiment scripts implicitly downloads them.

## 3. Prompt set

The VisDrone run used these ten native categories in exact ID order:

| ID | Prompt | Annotated instances |
| ---: | --- | ---: |
| 0 | pedestrian | 257 |
| 1 | people | 37 |
| 2 | bicycle | 28 |
| 3 | car | 96 |
| 4 | van | 11 |
| 5 | truck | 3 |
| 6 | tricycle | 1 |
| 7 | awning tricycle | 3 |
| 8 | bus | 13 |
| 9 | motorcycle | 11 |

No Fiat, Renault, color, or negative-language prompts were used for the VisDrone result. The original manufacturer/color prompt set remains configured for a future appropriately labeled dataset.

## 4. Test dataset description

Five paired images and raw annotations were imported from `/mnt/xilinx/prompt-inference/Vitis-AI-3.0/VisDrone2019-DET-test-dev`:

- `0000078_06777_d_0000020`
- `0000078_02363_d_0000007`
- `9999952_00000_d_0000009`
- `0000278_02951_d_0000008`
- `0000063_00500_d_0000001`

The converted schema contains 460 evaluable objects. Images are in `data/images/`; converted annotations, import provenance, and copied source TXT files are in `data/annotations/`. The examples span sparse and dense aerial scenes, including very small objects.

**Assumption:** The local VisDrone copy is an appropriate source for workstation testing. This report does not make a licensing determination.

## 5. Official YOLOE baseline results

Both runs used the same five images, 640-pixel inference size, IoU/NMS threshold 0.7, class-aware NMS, one warm-up, and five measured repetitions per image.

| Confidence | Predictions | Precision | Recall | mAP50 | FP/image | Mission recall | Missed-object rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | 34 | 0.7647 | 0.0565 | 0.0342 | 1.6 | 0.0565 | 0.9435 |
| 0.05 | 134 | 0.3881 | 0.1130 | 0.1090 | 16.4 | 0.1152 | 0.8848 |

The low recall is itself an observed limitation of this 640-pixel smoke configuration on small aerial objects. Lowering confidence improves recall and mAP50 while substantially increasing false positives. Therefore, the baseline should not be interpreted as a strong VisDrone detector benchmark.

The signed run signatures were:

- confidence 0.25: `7b70b72548a15ce955da018a6c3b1a0ff35f69df4c5b14b3bb525d1dd4820066`
- confidence 0.05: `2eb2f9b731e0c2c20f8503992e34bf6a30e4af8d31f75a8c3f7ca5c8f00fe9cc`

## 6. MobileCLIP-S0 embedding validation

Observed for the ten native prompts on `cuda:0`:

- shape: `[10, 512]`
- dtype: float32
- finite values: yes
- L2 norm range: 0.9999999404 to 1.0
- deterministic repeat maximum absolute delta: 0.0
- first encode latency: 276.506 ms
- repeated encode latency: 21.172 ms
- float32 payload: 20,480 bytes, or 2,048 bytes per prompt

The first timing includes initialization effects. Only `encode_text` ran; the MobileCLIP image encoder was not invoked.

## 7. Direct-injection results

The direct path loaded the validated `[10, 512]` S0 tensor without silent reshaping, added the required batch axis, passed `[1, 10, 512]` through the checkpoint's existing `head.get_tpe()` RepRTA path, and installed it with `set_classes()`. The head was unfused, dynamic, finite, and prompt-count preserving.

This establishes:

- API compatibility: pass
- dimensional compatibility: pass
- numerical validity: pass
- semantic compatibility: fail on the current smoke set

Direct S0 produced zero predictions at confidence 0.25 and zero at confidence 0.05. Its precision, recall, mAP50, mission recall, and hard-negative margins were therefore all zero, with a missed-object rate of 1.0.

## 8. Detection comparison

At confidence 0.25, direct S0 lost all 34 baseline predictions and recall fell by 5.65 percentage points. At confidence 0.05, direct S0 lost all 134 baseline predictions and recall fell by 11.30 percentage points. Relative mAP50 drop was 100% in both runs.

The provisional-close gates failed:

- recall drop <=5 percentage points: fail
- relative mAP50 drop <=10%: fail
- false-positive increase <=0.5/image: pass only because direct S0 emitted nothing
- median hard-negative margin positive: fail
- median hard-negative margin >=80% of baseline: numerically vacuous because both medians are zero

The 0.05 sensitivity result is important: direct S0 remaining empty while the matched baseline quadrupled its predictions argues against the failure being only a confidence-threshold calibration issue.

## 9. Hard-negative and embedding analysis

Both raw spaces contain finite normalized `[10, 512]` tensors. Within-space cosine matrices and rankings are saved under `outputs/visdrone/comparisons/`. They preserve some intuitive neighborhoods, but cross-space coordinates were not compared element-by-element and the visualization was not used as the quality decision.

Detection-level median hard-negative margin was 0.0 for the baseline and direct run because most of the 460 small objects were unmatched. This smoke set is too weak for a meaningful aggregate margin claim. The decisive observation is the complete absence of direct-S0 detections under two matched confidence settings.

## 10. Latency and payload measurements

Median across the five per-image median inference times:

| Confidence | Official B(LT) baseline | Direct S0 |
| ---: | ---: | ---: |
| 0.25 | 31.459 ms | 28.087 ms |
| 0.05 | 34.242 ms | 30.904 ms |

The slightly lower direct latency is not a benefit claim: it accompanies zero detections and may reflect postprocessing work avoided by the empty outputs. The prompt payload is 20,480 bytes for ten float32 embeddings. Deployment suitability is not approved because semantic quality failed.

## 11. Identified compatibility problems

**Source-code facts:**

- The loaded checkpoint is `YOLOESegModel` with a `YOLOESegment` head, three detection scales, RepRTA present, an unfused state, dynamic-class support, and a 512-dimensional text input.
- The checkpoint's `text_model` metadata is absent. Current Ultralytics resolves the official default to `mobileclip:blt`; this experiment makes that resolution explicit with the local B(LT) checkpoint.
- Official B(LT) and MobileCLIP-S0 embeddings have equal dimensions and valid unit norms.

**Observed problem:** Equal dimensions do not provide aligned semantic coordinates. Direct S0 passes the supported injection path but produces no detections where the aligned B(LT) baseline does.

**Hypothesis:** RepRTA and the YOLOE visual/text contrastive space were trained for B(LT), so S0 requires an explicit learned mapping or detector-side alignment.

## 12. Recommended next step

Proceed with Mode C as a separately gated experiment:

1. Build a phrase corpus appropriate to VisDrone and broader detection categories.
2. Generate paired S0 and official B(LT) embeddings.
3. Confirm both discovered dimensions, then train `Linear(512, 512)` with cosine loss and optional MSE.
4. Normalize adapter outputs and repeat the exact signed baseline/direct evaluation.
5. If adapter-only alignment remains insufficient, freeze S0 and fine-tune RepRTA plus the YOLOE visual/contrastive heads.

Use a larger, less class-imbalanced labeled VisDrone split for the next aerial evaluation. Separately source manufacturer/color annotations if Fiat/Renault/attribute behavior is still a requirement.

## 13. Alignment decision

**Explicit decision:** Adapter training is required before MobileCLIP-S0 embeddings should be considered usable by unmodified YOLOE-v8s. Start with the linear adapter. YOLOE RepRTA/head fine-tuning is conditionally recommended if the adapter does not recover the official baseline. Direct prompt-package export and any ZCU104/DPU integration remain blocked until a larger float32 evaluation passes.

The earlier VisDrone run that incorrectly paired these images with the manufacturer/color minimal prompt set was moved, not deleted, to `outputs/archive/minimal_prompt_visdrone/` for traceability.
