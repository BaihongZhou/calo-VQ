# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Implementation of **Calo-VQ** ([2405.06605](https://arxiv.org/abs/2405.06605)): a vector-quantized two-stage generative model for fast calorimeter-shower simulation. Adapted from [taming-transformers](https://github.com/CompVis/taming-transformers) (VQ-VAE + discriminator) and [minGPT](https://github.com/karpathy/minGPT) (autoregressive prior).

**This repo has been refactored from the paper's ATLAS cylindrical geometry to the DarkSHINE ECAL xyz geometry** ([2407.17800](https://arxiv.org/abs/2407.17800): a 21×21×11 staggered LYSO crystal calorimeter). The cylindrical-specific code (cylindrical convolution with azimuthal periodic padding, FFT azimuthal resampling, per-layer R normalization, ds1/ds2/ds3 dataset branches) has been **removed**; the model is now xyz-only. The original cylindrical Calo-VQ history is preserved in git (commits up to the `DarkSHINE` branch point); paper dataset-2 HDF5 files still live in `paper_data/` but the cylindrical code path that consumed them is gone.

## Environment

Conda env: `/Users/zhoubaihong/miniconda3/envs/caloVQ/bin/python` (Python 3.12). Key deps: **torch 2.9.0**, **pytorch-lightning 2.x** (2.6.5), torchvision 0.24.0, torchmetrics, omegaconf, einops, numpy 2.x, h5py, matplotlib, scipy, psutil. Platform is macOS with no CUDA — everything runs on CPU.

> Install gotcha: a local proxy (`127.0.0.1:7890`) makes large wheels (torch) flaky. Install from the Tsinghua mirror with the proxy off:
> `unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY`
> `pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple --retries 10 --timeout 300 ...`

The code was migrated from `pytorch-lightning==1.6.5` to PL 2.x. Notably `VQModel` uses **manual optimization** (PL2 removed `optimizer_idx`), and `main.py` builds the `Trainer` directly (no `add_argparse_args`/`from_argparse_args`/`--gpus`).

## Common commands

Training is **two-stage and sequential**: Step 1 trains the VQ-VAE; Step 2 trains the GPT prior over the frozen Step-1 codebook. The Step-2 config points at the Step-1 model via `model.params.vq_config` (`model_config` + `checkpoint`); pass the produced checkpoint on the CLI.

```bash
# --- DarkSHINE smoke test (1 epoch each, CPU) ---
PY=/Users/zhoubaihong/miniconda3/envs/caloVQ/bin/python

# Step 1 (VQ-VAE)
$PY main.py --base config/darkshine_step1.yaml -t True -l logs -n dss1

# Step 2 (GPT prior) — point vq_config.checkpoint at the Step-1 last.ckpt
CKPT=$(ls -t logs/*dss1*/checkpoints/last.ckpt | head -1)
$PY main.py --base config/darkshine_step2.yaml -t True -l logs -n dss2 \
  "model.params.vq_config.checkpoint=$CKPT"
```

`--gpus N` selects GPUs (omit / `0` → CPU). Trainer settings live in the `lightning.trainer:` config section (e.g. `max_epochs`, `limit_train_batches`). `nested.key=value` CLI args override config (dotlist). Learning rate is scaled by `ngpu * batch_size * accumulate_grad_batches` unless `--scale_lr False`.

The fixed geometry mask `DarkSHINE_data/geom_mask.npy` is required (see below); it is committed and loaded by the model/decoder/loss.

## DarkSHINE dataset

`DarkSHINE_data/export.h5` (10000 events, currently a single 4 GeV energy point). Three keys:

- `condition` (N, 1): incident particle energy [MeV] — the **stage-2 conditioning** (same role as ATLAS incident energy).
- `energy` (N, 43, 43, 11): per-cell **deposited** energy Eᵢ (the model input). Stored **channels-last**: `(x=43, y=43, depth=11)`.
- `label` (N, 43, 43, 11) int {0,1,2,4}: a **per-event hit flag** (`label>0` ⟺ `energy>0`). It is *not* a fixed crystal mask and is **not used directly** by the model.

### Geometry, staggering, and the 43×43 padding

The ECAL is 21×21 LYSO crystals per layer, 11 layers deep, with **adjacent layers staggered (offset by half a crystal)**. To stack center-aligned layers into one regular tensor, the grid is stored at **half-crystal resolution**: each physical crystal occupies a **2×2 cell block**, so a layer spans 42×42 cells, and `43 = 42 + 1` extra row/col holds the ±1 stagger between even/odd layers. Hence per layer the real region is a **contiguous 42×42 block offset by layer parity** (even layers `[1:43, 1:43]`, odd `[0:42, 0:42]`). Total real cells = 11·42·42 = 19404 (= 11·21·21·4 → 4851 crystals). The ~8% of the grid outside these blocks is **padding**.

### The fixed geometry mask

Because generation must place energy without knowing the truth hit pattern, the mask must be **identical for every sample**. `DarkSHINE_data/geom_mask.npy` (shape `(43,43,11)` bool) encodes the staggered real-crystal blocks above (built by filling the contiguous per-layer block; verified that zero deposited energy ever falls outside it). The model loads it channels-first as `(depth=11, x=43, y=43)` and uses it in **three places**:

1. **Normalization factor R**: `R = (ΣEⱼ over real cells) / E_incident` (`vqvae.py:preprocess`). Padding cells are excluded (they are 0 anyway).
2. **Decoder masked softmax** (`decoder.py:masked_voxel_softmax`): padding logits set to `-inf` so the per-shower distribution sums to 1 over **real cells only** — padding cannot steal probability mass.
3. **Reconstruction / EC / width loss** (`combined.py`): the recon difference is multiplied by the mask so padding produces no gradient and never enters a normalization denominator.

### Layout permute

Data is channels-last `(N, 43, 43, 11)`; PyTorch conv2d needs `(N, C, H, W)`. The dataloader (`data.py:CaloDarkSHINE`) permutes to **`(N, 11, 43, 43)`** = (batch, depth→channel, x→H, y→W). The encoder zero-pads the (x,y) plane 43→48 for clean stride-2 downsampling; the decoder crops 48→43 after the masked softmax.

## Architecture

Pipeline: encode → quantize → autoregress → decode, split across two `LightningModule`s.

- **`calo_ldm/models/vqvae.py` (`VQModel`)** — Step-1. `Encoder` → `VectorQuantizer` (codebook of `n_embed`) → `Decoder`, trained with `CombinedLoss` (masked recon + codebook/commitment + shower energy-centre/width + adversarial `Discriminator`). The decoder output is a **masked voxel-softmax** that sums to **R** per shower (energy ratio = E_dep/E_inc); the VQ-VAE models the *shape*, R carries the *scale*. Uses PL2 manual optimization for the two optimizers (AE + discriminator).
- **`calo_ldm/models/gpt.py` (`CondGPT`)** — Step-2. Loads and **freezes** a `VQModel`, encodes showers to a `sequence_shape=(6,6)` code grid, and trains a GPT to autoregressively predict codes conditioned on incident energy. With `predict_R: true`, R is quantized to `R_bits` and prepended to the token sequence. **DarkSHINE R ≈ 1e-3** (4 GeV in, ~3.9 GeV deposited), so set `R_max ≈ 0.0011` (not the paper's ~1.3) for usable resolution. `sample_fullchain(batch)` is the generation entry point.

### Geometry mapping (depth → channel, (x,y) → image)

Following the paper's "R→channel, (Z,A)→image" idea but for xyz:
- **layer depth (11) → channel dimension** (shower evolves with depth, no translation symmetry — analogous to the paper's radial axis).
- **(x, y) = 43×43 → 2D image plane** (translation-symmetric → 2D conv).

### Conv layer mini-DSL

Encoder/decoder/discriminator architectures are declared via `conv_spec` lists of colon-delimited strings, parsed by `parse_conv_spec` in `calo_ldm/util.py`. Layer ops (xyz, in `calo_ldm/layers/conv_xyz.py`):
- `pconv` — plain 2D conv over (x,y), **non-periodic** zero padding (replaces the removed cylindrical `cconv`).
- `pconvT` — plain 2D transposed conv (replaces `cconvT`).

Within a spec, fields are order/case-insensitive: `kAB`=kernel `(x,y)`, `sAB`=stride, `p`=zero-pad both x and y by `(k-1)//2`, `*N`/`/N`/`+N`/`-N`=scale channel count, `cN`=set channels to N. Example `'pconv : k4 : s2 : p : *2'` = 4×4 kernel, stride 2, padded (halves the plane), double channels. With the default 48×48 working size, `k4 s2 p` cleanly halves (48→24→12→6) and `pconvT k4 s2 p` doubles (6→12→24→48).

### Supporting modules

- `calo_ldm/data.py` — `CaloDarkSHINE` preloads `export.h5` into RAM and permutes to channels-first. `DataModuleFromConfig` uses **batched-index** sampling.
- `calo_ldm/losses/combined.py` (`CombinedLoss`) — Step-1 multi-term loss + GAN logic (xyz + mask).
- `calo_ldm/layers/vectorquantizer.py`, `layers/transformer.py` — the quantizer and GPT transformer blocks (geometry-agnostic, unchanged).
- `calo_ldm/util.py` — `instantiate_from_config` (`{target, params}` convention), `parse_conv_spec`, `load_geom_mask`.

### Config convention

Everything is instantiated from OmegaConf YAML via `instantiate_from_config`. `VQModel` runs in "overwrite mode": shared params (`cond_dim`, `log_scale_params`, `mask_path`) declared on `model.params` are pushed down into the encoder/decoder/loss sub-configs. `main.py` merges base configs left-to-right then applies `nested.key=value` CLI overrides.

## Differences from the original ATLAS-cylindrical Calo-VQ

| | Original (paper, cylindrical) | This repo (DarkSHINE, xyz) |
|---|---|---|
| Geometry | (R, Z, A), A periodic | (x, y, depth), no periodic axis |
| Conv | `CylinderConv` (cyclic A pad) + FFT A-resampling | `PlaneConv`/`PlaneConvTranspose`, zero pad |
| Channel axis | radial R | depth (11 layers) |
| Image plane | (Z, A) | (x, y) = 43×43 (padded 48) |
| Decoder softmax | over all voxels | **masked** over real crystals |
| Mask | none (full cylinder) | fixed staggered geometry mask, used in R / softmax / loss |
| Datasets | ds1/ds2/ds3 | DarkSHINE only |
| Lightning | 1.6.5 | 2.x (manual optimization) |

## Trained models

`models/` still holds the original cylindrical published checkpoints (git-lfs). They are **not loadable by the current xyz code** (architecture changed) — kept for reference only. `gen-tools.py` / `generation.sh` are for the original geometry and would need adapting to xyz (see `Agent_md/`).

There is no test suite, linter, or CI in this repo. Smoke configs: `config/darkshine_step1.yaml`, `config/darkshine_step2.yaml` (1-epoch CPU end-to-end). Original paper-ds2 smoke configs (`config/ds2_step1_smoke.yaml`, `ds2_step2_smoke.yaml`) target the now-removed cylindrical path and are kept only as historical reference.
