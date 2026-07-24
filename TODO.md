# TODO

This roadmap follows the observed result that MobileCLIP-S0 is numerically, dimensionally, and API compatible with YOLOE-v8s, but produced no detections on the current VisDrone smoke test. Complete tasks in order and preserve the baseline-first gates.

## P0 — Explain the score collapse

- [ ] Instrument YOLOE inference to save raw per-scale, pre-threshold class similarity/logit summaries for both official B(LT) and direct S0 modes.
- [ ] Record per-image and per-prompt minimum, maximum, mean, median, and percentile score distributions before confidence filtering and NMS.
- [ ] Add an explicitly named diagnostic run at confidence `0.001`; do not treat its detections as production-quality results.
- [ ] Verify that direct S0 scores are genuinely shifted/collapsed relative to B(LT), rather than being removed by decoding, class indexing, or NMS.
- [ ] Confirm exact prompt-to-class indices after `set_classes()` in both modes.
- [ ] Compare raw and post-RepRTA tensor norms, cosine structure, and score distributions without interpreting element-wise cross-space differences.
- [ ] Add regression tests for score-summary serialization, prompt indexing, and pre-NMS tensor shape validation.
- [ ] Document the diagnostic evidence in the workstation report.

Exit criterion: identify the stage at which B(LT) and S0 behavior diverges and rule out thresholding, decoding, prompt ordering, and NMS implementation errors.

## P1 — Linear embedding adapter

- [ ] Assemble a versioned phrase corpus covering VisDrone classes, broad detection vocabulary, attributes, hard negatives, and aerial phrasing.
- [ ] Generate paired official B(LT) and MobileCLIP-S0 embeddings with exact prompt/source/checkpoint provenance.
- [ ] Discover both embedding dimensions at runtime; use `Linear(512, 512)` only when both are confirmed as 512.
- [ ] Split phrases into train, validation, and held-out semantic groups to prevent near-duplicate prompt leakage.
- [ ] Train a linear adapter using cosine embedding loss and optional MSE.
- [ ] L2-normalize adapter outputs.
- [ ] Save adapter weights, configuration, corpus hashes, source commits, training curves, and SHA-256 provenance.
- [ ] Add a separately named `s0_linear_adapter` inference mode without changing Modes A or B.
- [ ] Repeat the exact signed VisDrone evaluation with identical images, preprocessing, thresholds, and class-aware NMS.
- [ ] Compare against both official B(LT) and unadapted S0.

Exit criterion: adapter evaluation satisfies all provisional-close gates:

- recall drop no greater than 5 percentage points;
- relative mAP50 drop no greater than 10%;
- false positives increase no greater than 0.5 per image;
- median hard-negative margin remains positive and at least 80% of baseline.

## P2 — Adapter refinements

Only start if the linear adapter improves direct S0 but does not pass the quality gates.

- [ ] Evaluate a small residual MLP adapter.
- [ ] Compare parameter count, latency, payload implications, and overfitting against the linear adapter.
- [ ] Expand hard negatives for visually adjacent categories such as car/van/truck/bus and bicycle/tricycle/motorcycle.
- [ ] Measure prompt-template sensitivity on held-out phrases.
- [ ] Keep MobileCLIP-S0 frozen.

Exit criterion: choose the smallest adapter that passes held-out text alignment and detection-quality gates.

## P3 — YOLOE alignment fine-tuning

Only start if text-only adapters remain insufficient.

- [ ] Freeze MobileCLIP-S0.
- [ ] Fine-tune RepRTA or the equivalent text-adaptation module first.
- [ ] Then fine-tune YOLOE visual embedding and contrastive classification heads.
- [ ] Consider later neck layers only after head alignment is measured.
- [ ] Consider later backbone stages only when earlier stages are insufficient.
- [ ] Use UAV imagery, small-object examples, hard negatives, blur, occlusion, and compression augmentation.
- [ ] Preserve an untouched official B(LT) baseline for every comparison.
- [ ] Record all trainable modules, datasets, seeds, checkpoints, and metrics.

Exit criterion: a held-out detection evaluation passes the quality gates without unacceptable mission-object recall loss.

## P4 — Larger evaluation

- [ ] Expand beyond the five-image smoke set to a versioned, class-balanced VisDrone evaluation.
- [ ] Report results by class, object size, occlusion, truncation, and scene density.
- [ ] Measure consecutive-frame stability for RoI use.
- [ ] Add RoI coverage and false-positive RoI area metrics.
- [ ] Source a separate licensed dataset if manufacturer or vehicle-color behavior remains a requirement.
- [ ] Evaluate manufacturer identity on high-resolution crops; consider a two-stage detector/classifier when image evidence is insufficient.

Exit criterion: the selected alignment approach passes a statistically meaningful labeled evaluation. A five-image pass alone is never sufficient.

## P5 — Prompt packaging and edge boundary

Blocked until float32 detection quality passes.

- [ ] Export normalized float32 prompt packages with JSON and binary tensor sidecars.
- [ ] Measure payload size and transmission latency.
- [ ] Specify the workstation-to-ZCU104 protocol and versioned metadata contract.
- [ ] Evaluate INT8 prompt quantization only after float32 quality is established.
- [ ] Keep MobileCLIP tokenization/text encoding on the workstation.
- [ ] Do not add MobileCLIP operations to the Vitis AI graph.
- [ ] Do not modify ZCU104 runtime or quantization code until the workstation report recommends a specific architecture.

## Always-required checks

- [ ] Use explicit local checkpoints; never allow implicit weight downloads.
- [ ] Preserve ordered prompts, image hashes, source commits, checkpoint hashes, seeds, and signed inference settings.
- [ ] Keep CPU as the portable default and require explicit GPU selection.
- [ ] Run unit tests and structured artifact validation before every result commit.
- [ ] Clearly label observations, source facts, assumptions, hypotheses, and recommendations.
