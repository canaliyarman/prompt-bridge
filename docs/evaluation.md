# Evaluation workflow

## Baseline-first sequence

Run commands from the repository root.

### 1. Encode MobileCLIP-S0 prompts

```bash
python scripts/encode_mobileclip_s0.py \
  --device cuda:0 \
  --prompt-set visdrone_prompts
```

Required outputs:

- `outputs/embeddings/mobileclip_s0_embeddings.pt`
- `outputs/embeddings/mobileclip_s0_embeddings.npy`
- `outputs/embeddings/mobileclip_s0_metadata.json`

The gate requires finite float32 `[N, D]` output, unit norms, deterministic repetition, matching checkpoint provenance, and an exact prompt list.

### 2. Inspect the YOLOE text path

```bash
python scripts/inspect_yoloe_text_path.py \
  --device cuda:0 \
  --prompt-set visdrone_prompts
```

Inspection verifies a three-scale unfused dynamic YOLOE segmentation head, RepRTA presence, expected embedding width, prompt count preservation, and both official/S0 tensor summaries. It never calls prompt fusion or modifies the checkpoint.

### 3. Run the official baseline

```bash
python scripts/run_yoloe_baseline.py \
  --device cuda:0 \
  --prompt-set visdrone_prompts
```

The baseline performs one warm-up and five measured repetitions per image. It writes predictions, rendered diagnostics, raw B(LT) and post-RepRTA tensors, an ordered content-addressed image manifest, and signed metadata.

### 4. Run direct S0 injection

```bash
python scripts/run_yoloe_s0_direct.py \
  --device cuda:0 \
  --prompt-set visdrone_prompts
```

Mode B only starts after validating the baseline metadata. Any mismatch in prompt order, image hashes, checkpoint, image size, confidence, IoU/NMS, or detection cap fails explicitly.

### 5. Compare embedding relationships

```bash
python scripts/compare_embeddings.py --prompt-set visdrone_prompts
```

The comparator reports norms, within-space cosine structure, pairwise rankings, and two-dimensional diagnostics. It does not interpret element-wise differences between unrelated embedding spaces.

### 6. Evaluate predictions

```bash
python scripts/evaluate_predictions.py
```

The evaluator writes matched prediction JSONL for both modes and a structured comparison/decision artifact.

## Prediction record contract

Each prediction record contains:

- `image_id`
- `prompt_id`
- `prompt_text`
- `box_xyxy`
- `confidence`
- `matched_ground_truth`
- `iou`
- `inference_time_ms`

Rendered images are diagnostics only; quantitative matching determines the result.

## Metrics and decision gates

Detections match at IoU >= 0.5. Reported metrics include box precision/recall, mAP50, false positives per image, false negatives per prompt, mission-object recall, missed-object rate, hard-negative confidence margins, embedding latency, inference latency, and float32 payload size.

Direct injection is only provisionally close when all conditions hold:

- recall drop <=5 percentage points;
- relative mAP50 drop <=10%;
- false positives rise <=0.5 per image;
- median hard-negative margin remains positive;
- median hard-negative margin is at least 80% of baseline.

Passing a five-image smoke test would only authorize a larger evaluation. It would not establish general compatibility.

## Current VisDrone result

| Confidence | Mode | Predictions | Precision | Recall | mAP50 | FP/image | Missed rate |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | B(LT) baseline | 34 | 0.7647 | 0.0565 | 0.0342 | 1.6 | 0.9435 |
| 0.25 | Direct S0 | 0 | 0 | 0 | 0 | 0 | 1.0 |
| 0.05 | B(LT) baseline | 134 | 0.3881 | 0.1130 | 0.1090 | 16.4 | 0.8848 |
| 0.05 | Direct S0 | 0 | 0 | 0 | 0 | 0 | 1.0 |

The confidence-0.05 sensitivity run shows that the direct result is not merely hidden just below 0.25. The baseline gains 100 detections while direct S0 remains empty.

## Export gate

`export_prompt_package.py` accepts only a provisionally-close decision artifact. The current failure therefore blocks float32 package export. INT8 is intentionally out of scope until float32 detection quality passes.

## Interpreting failure

- Direct runs but performs poorly: learn a linear adapter, then a small residual MLP if needed.
- Adapter remains poor: freeze S0 and fine-tune RepRTA plus YOLOE visual/contrastive heads.
- Broad categories work but attributes fail: add attribute-rich phrases, hard negatives, UAV imagery, and blur/compression augmentation.
- Manufacturer identity remains unreliable: evaluate high-resolution crops and consider a two-stage vehicle detector/classifier.
