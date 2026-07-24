# Setup and provisioning

## Tested environment

| Component | Tested value |
| --- | --- |
| OS-side Python | 3.10.19 |
| PyTorch | 2.7.1+cu118 |
| torchvision | 0.22.1+cu118 |
| CUDA runtime reported by PyTorch | 11.8 |
| GPU | NVIDIA GeForce GTX 1060 Max-Q, compute capability 6.1 |
| Ultralytics | 8.4.14 |
| MobileCLIP package | 0.1.0 |

The GTX 1060 requires a PyTorch build containing `sm_61` kernels. The workstation's PyTorch 2.10/CUDA 12.8 build only supported `sm_70` and newer, so this project uses a separate CUDA 11.8 environment. CPU remains the default when no device is selected.

Apple's current checkout declares PyTorch >=2.8, while the tested GTX-compatible environment uses 2.7.1. The exercised text-only path passes, but this is an intentional recorded dependency exception.

## Create the environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# CUDA 11.8 option used by the measured GTX 1060 run:
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
pip install -r requirements-dev.txt
```

For CPU-only use, install the corresponding PyTorch CPU wheels instead. Do not assume the newest CUDA build supports an older GPU.

## Pin the upstream repositories

The scripts add these local editable checkouts to `sys.path`; they do not modify them.

```bash
mkdir -p third_party

git clone https://github.com/apple/ml-mobileclip.git third_party/ml-mobileclip
git -C third_party/ml-mobileclip checkout aecfb5453d022e9deff12f81a150ea8f35194baa

git clone https://github.com/ultralytics/ultralytics.git third_party/yoloe
git -C third_party/yoloe checkout fc3d6e29133f97db7c0cb59b6a9cf21cb9d9be74

git clone https://github.com/ultralytics/CLIP.git third_party/clip
git -C third_party/clip checkout c4b6ea0932a2c0f39a0fa528af5ec4982ff15cab

pip install --no-deps -e third_party/ml-mobileclip
pip install --no-deps -e third_party/yoloe
pip install --no-deps -e third_party/clip
```

The directories are ignored by Git. Every output artifact records the resolved upstream commit.

## Provision checkpoints

Place the files under `artifacts/checkpoints/` with these exact names and verify their digests:

| File | SHA-256 |
| --- | --- |
| `mobileclip_s0.pt` | `809b408eff74f8058843e86a1f92967097d42ba782450e85b8f4867b7f0ca0b7` |
| `mobileclip_blt.ts` | `a67804d1b0f07b8b9a20c1761ec0847f34660f5fa338ec70e8f3fce68ed95e54` |
| `yoloe-v8s-seg.pt` | `ac2b90ed23011495a3e86d89caeb3432a15129cac8d849ba121293c8fc1e0536` |

```bash
sha256sum artifacts/checkpoints/mobileclip_s0.pt \
  artifacts/checkpoints/mobileclip_blt.ts \
  artifacts/checkpoints/yoloe-v8s-seg.pt
```

Checkpoints are ignored and must be provisioned explicitly. If a file is absent or has the wrong pinned S0 hash, the script fails; it does not fetch a replacement.

## Prepare VisDrone examples

VisDrone remains external to this repository. Import selected pairs from a local split:

```bash
python scripts/import_visdrone_examples.py \
  --source /path/to/VisDrone2019-DET-test-dev \
  --image-ids 0000078_06777_d_0000020 0000078_02363_d_0000007 \
    9999952_00000_d_0000009 0000278_02951_d_0000008 0000063_00500_d_0000001
```

The importer copies images to `data/images/`, preserves raw TXT annotations under `data/annotations/visdrone_raw/`, converts all ten native classes into the tracked schema, records hashes/provenance, and refuses conflicting existing files.

The selected IDs and class mapping are recorded in `configs/visdrone_examples.yaml`.

## Offline execution

The runtime sets:

- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`
- `YOLO_AUTOINSTALL=false`

Provision everything before disconnecting the network. A missing source checkout or checkpoint is an error.
