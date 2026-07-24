# Prompt Bridge contributor instructions

- Keep all evaluation work workstation-only until the float32 quality gate passes.
- Never download checkpoints implicitly; use `artifacts/checkpoints/`.
- Keep upstream MobileCLIP, Ultralytics, and CLIP checkouts unmodified under ignored `third_party/`.
- Keep datasets under `data/` and generated artifacts under `outputs/`; neither belongs in Git.
- Run and validate the official B(LT) baseline before direct S0 injection.
- Do not equate API or dimensional compatibility with semantic compatibility.
- Do not modify Vitis AI, quantization, FPGA, or ZCU104 runtime code from this repository.
- Preserve prompt order, signed run settings, checkpoint hashes, and image manifests across comparisons.
