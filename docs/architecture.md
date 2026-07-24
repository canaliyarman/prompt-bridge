# Architecture and compatibility model

## Goal

Prompt Bridge evaluates whether a text embedding produced by one encoder can replace the embedding expected by an open-vocabulary detector. The initial target is MobileCLIP-S0 feeding YOLOE-v8s, whose released dynamic-text checkpoint is aligned to MobileCLIP-B(LT).

The experiment is workstation-only. It produces evidence for a later edge boundary; it does not put either MobileCLIP tower into a DPU graph.

## Why matching dimensions are insufficient

Both S0 and B(LT) emit 512-dimensional normalized vectors. That proves the values can fit the same tensor interface, but not that dimension 27—or any direction in the space—has the same learned meaning.

YOLOE learns visual features against the B(LT)-aligned text space. Feeding S0 coordinates directly can therefore be syntactically valid and semantically meaningless. The gates are intentionally separate:

1. Numerical validity: finite float32 values and unit norms.
2. Dimensional compatibility: discovered widths agree.
3. API compatibility: the supported YOLOE path executes.
4. Semantic compatibility: quantitative detections remain close to baseline.
5. Deployment suitability: quality and payload meet downstream needs.

## Components

### MobileCLIP-S0 encoder

`encode_mobileclip_s0.py` loads an explicitly supplied checkpoint and matching tokenizer, switches to evaluation mode, uses inference mode, encodes text only, normalizes the output, repeats the encoding for determinism, and saves PT/NPY plus provenance metadata.

### Official YOLOE baseline

`run_yoloe_baseline.py` uses the explicit MobileCLIP-B(LT) TorchScript checkpoint. It follows the supported flow:

```python
text_embeddings = model.get_text_pe(prompts)
model.set_classes(prompts, text_embeddings)
```

The implementation verifies prompt order, installed tensor identity, class count, finite normalized tensors, and the unfused dynamic head.

### Direct S0 injection

`run_yoloe_s0_direct.py` validates the saved S0 tensor as `[N, D]`, discovers YOLOE's expected `D`, adds only the batch dimension to form `[1, N, D]`, and calls the existing `head.get_tpe()` RepRTA path before `set_classes()`.

It refuses fused heads, prompt-free checkpoints, missing RepRTA, non-finite tensors, prompt mismatches, dimension mismatches, or baseline setting mismatches.

### Evaluator

`evaluate_predictions.py` matches detections to annotations at IoU >= 0.5 and computes precision, recall, mAP50, false positives per image, false negatives per prompt, mission recall, missed-object rate, and correct-versus-hard-negative confidence margins.

## Experiment modes

- Mode A — official B(LT) baseline. Required first.
- Mode B — unmodified direct S0 injection. This tests compatibility, not training.
- Mode C — learned text-space adapter. Start with a linear mapping after discovering dimensions.
- Mode D — detector alignment fine-tuning. Freeze S0 initially; tune RepRTA and contrastive heads before deeper visual layers.

## Intended edge boundary

```text
Workstation/control plane                 ZCU104/data plane
-------------------------                 ------------------
prompt                                    camera frame
  -> S0 tokenizer                           -> YOLOE visual model
  -> S0 text encoder                        -> regional features
  -> optional trained adapter               -> embedding matching
  -> normalized float32 vector              -> boxes
  -> network transmission                   -> RoI compression
```

No MobileCLIP image or text operation belongs in the Vitis AI graph for this design.
