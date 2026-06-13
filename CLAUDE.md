# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Implementation of **Calo-VQ** ([2405.06605](https://arxiv.org/abs/2405.06605)): a vector-quantized two-stage generative model for fast calorimeter-shower simulation. Adapted from [taming-transformers](https://github.com/CompVis/taming-transformers) (VQ-VAE + discriminator) and [minGPT](https://github.com/karpathy/minGPT) (autoregressive prior).

**This repo has been refactored from the paper's ATLAS cylindrical geometry to the DarkSHINE ECAL xyz geometry** ([2407.17800](https://arxiv.org/abs/2407.17800): a 21×21×11 staggered LYSO crystal calorimeter). The cylindrical-specific code (cylindrical convolution with azimuthal periodic padding, FFT azimuthal resampling, per-layer R normalization, ds1/ds2/ds3 dataset branches) has been **removed**; the model is now xyz-only. The original cylindrical Calo-VQ history is preserved in git (commits up to the `DarkSHINE` branch point); the paper dataset-2 HDF5 files (`paper_data/`) and cylindrical checkpoints (`models/`) have been removed.

## Environment

Conda env: `/Users/zhoubaihong/miniconda3/envs/caloVQ/bin/python` (Python 3.12). Key deps: **torch 2.9.0**, **pytorch-lightning 2.x** (2.6.5), torchvision 0.24.0, torchmetrics, omegaconf, einops, numpy 2.x, h5py, matplotlib, scipy, psutil, **wandb** (validation-metric logging). Platform is macOS with no CUDA — everything runs on CPU.

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

- `condition` (N, 1): incident particle energy stored in **keV** (4e6 = 4 GeV) — the **stage-2 conditioning** (same role as ATLAS incident energy). `CaloDarkSHINE` converts it to **MeV** (`condition_scale=1e-3`) so it matches `energy`'s units; do not use the raw keV value.
- `energy` (N, 43, 43, 11): per-cell **deposited** energy Eᵢ (the model input). Stored **channels-last**: `(x=43, y=43, depth=11)`.
- `label` (N, 43, 43, 11) int {0,1,2,4}: a **per-event hit flag**. `label>0` ⟹ `energy>0` (every flagged cell has energy), but **not** the converse: 416 cells across 9 events have `energy>0` yet `label==0` (energies up to ~36 MeV — not sub-threshold noise), so `label>0` is a *strict subset* of `energy>0`, not an equivalence. It is *not* a fixed crystal mask and is **not used directly** by the model. Both `label` and `energy` are exactly 0 everywhere outside the geometry mask (verified over all 10000 events).

### Geometry, staggering, and the 43×43 padding

The ECAL is 21×21 LYSO crystals per layer, 11 layers deep, with **adjacent layers staggered (offset by half a crystal)**. To stack center-aligned layers into one regular tensor, the grid is stored at **half-crystal resolution**: each physical crystal occupies a **2×2 cell block**, so a layer spans 42×42 cells, and `43 = 42 + 1` extra row/col holds the ±1 stagger between even/odd layers. Per layer the nominal region is a **contiguous 42×42 block offset by layer parity** (even layers `[1:43, 1:43]`, odd `[0:42, 0:42]`). The ~8% of the grid outside these blocks is **padding**.

### The fixed geometry mask

Because generation must place energy without knowing the truth hit pattern, the mask must be **identical for every sample**. `DarkSHINE_data/geom_mask.npy` (shape `(43,43,11)` bool, **19404 real cells**) is the exact detector geometry as defined by the data producer's `ML_IO.cpp` (`FillECALEventEnergy`, `StaggeredUpsample=1`): every crystal `(idz∈1..11, idy,idx∈1..21)` fills a full **2×2 (py,px) block**, offset by layer parity (even 1-based layer → `[2·i-2, 2·i-1]`, odd → `[2·i-1, 2·i]`), so all 21×21 crystals tile a **contiguous 42×42 block per layer** = 11·42·42 = 19404 (4851 crystals × 4). Rebuild: see `DarkSHINE_data/build_geom_mask.py`. Note: do **not** derive the mask from `(label>0)`/`(energy>0)` unioned over events — that undercounts by ~700 deep-layer cells that are real crystals simply never hit in the finite sample (verified: every real cell tiles a crystal, and zero energy/label ever falls outside the mask). The model loads it channels-first as `(depth=11, x=43, y=43)` and uses it in **three places**:

1. **Normalization factor R**: `R = (ΣEⱼ over real cells) / E_incident` (`vqvae.py:preprocess`). Padding cells are excluded (they are 0 anyway).
2. **Decoder masked softmax** (`decoder.py:masked_voxel_softmax`): padding logits set to `-inf` so the per-shower distribution sums to 1 over **real cells only** — padding cannot steal probability mass.
3. **Reconstruction / EC / width loss** (`combined.py`): the recon difference is multiplied by the mask so padding produces no gradient and never enters a normalization denominator.

### Layout permute

Data is channels-last `(N, 43, 43, 11)`; PyTorch conv2d needs `(N, C, H, W)`. The dataloader (`data.py:CaloDarkSHINE`) permutes to **`(N, 11, 43, 43)`** = (batch, depth→channel, x→H, y→W). The encoder zero-pads the (x,y) plane 43→48 for clean stride-2 downsampling; the decoder crops 48→43 after the masked softmax.

## Architecture

Pipeline: encode → quantize → autoregress → decode, split across two `LightningModule`s.

- **`calo_ldm/models/vqvae.py` (`VQModel`)** — Step-1. `Encoder` → `VectorQuantizer` (codebook of `n_embed`) → `Decoder`, trained with `CombinedLoss` (masked recon + codebook/commitment + shower energy-centre/width + adversarial `Discriminator`). The decoder output is a **masked voxel-softmax** that sums to **R** per shower (energy ratio = E_dep/E_inc); the VQ-VAE models the *shape*, R carries the *scale*. Uses PL2 manual optimization for the two optimizers (AE + discriminator).
- **`calo_ldm/models/gpt.py` (`CondGPT`)** — Step-2. Loads and **freezes** a `VQModel`, encodes showers to a `sequence_shape=(6,6)` code grid, and trains a GPT to autoregressively predict codes conditioned on incident energy. With `predict_R: true`, R is quantized to `R_bits` and prepended to the token sequence. **DarkSHINE R ≈ 0.97** (4 GeV in, ~3.9 GeV deposited, units matched — see the unit fix below), so `R_max ≈ 1.1` (paper-like). NOTE: earlier configs used `R_max ≈ 0.0011` because `condition` was read in keV while `energy` is in MeV (a 1000× bug, now fixed in `CaloDarkSHINE`). `sample_fullchain(batch)` is the generation entry point.

### Geometry mapping (depth → channel, (x,y) → image)

Following the paper's "R→channel, (Z,A)→image" idea but for xyz:
- **layer depth (11) → channel dimension** (shower evolves with depth, no translation symmetry — analogous to the paper's radial axis).
- **(x, y) = 43×43 → 2D image plane** (translation-symmetric → 2D conv).

### Conv layer mini-DSL

Encoder/decoder/discriminator architectures are declared via `conv_spec` lists of colon-delimited strings, parsed by `parse_conv_spec` in `calo_ldm/util.py`. The layers build plain `nn.Conv2d`/`nn.ConvTranspose2d` over the (x,y) plane (the `pad_z` flag → `padding` via `conv_padding` in `util.py`; the old cylindrical `CylinderConv` / xyz `PlaneConv` wrappers have been removed). Layer-op keywords:
- `pconv` — plain 2D conv over (x,y), **non-periodic** zero padding (replaces the removed cylindrical `cconv`).
- `pconvT` — plain 2D transposed conv (replaces `cconvT`).

Within a spec, fields are order/case-insensitive: `kAB`=kernel `(x,y)`, `sAB`=stride, `p`=zero-pad both x and y by `(k-1)//2`, `*N`/`/N`/`+N`/`-N`=scale channel count, `cN`=set channels to N. Example `'pconv : k4 : s2 : p : *2'` = 4×4 kernel, stride 2, padded (halves the plane), double channels. With the default 48×48 working size, `k4 s2 p` cleanly halves (48→24→12→6) and `pconvT k4 s2 p` doubles (6→12→24→48).

### Supporting modules

- `calo_ldm/data.py` — `CaloDarkSHINE` preloads `export.h5` into RAM and permutes to channels-first. `DataModuleFromConfig` uses **batched-index** sampling.
- `calo_ldm/losses/combined.py` (`CombinedLoss`) — Step-1 multi-term loss + GAN logic (xyz + mask).
- `calo_ldm/layers/vectorquantizer.py`, `layers/transformer.py` — the quantizer and GPT transformer blocks (geometry-agnostic, unchanged).
- `calo_ldm/util.py` — `instantiate_from_config` (`{target, params}` convention), `parse_conv_spec`, `conv_padding`, `load_geom_mask`.
- `calo_ldm/geometry.py` — single source of truth for the crystal↔half-cell mapping (the `ML_IO.cpp` parity rule): `build_crystal_mask()` (used by `DarkSHINE_data/build_geom_mask.py`) and `downsample_to_crystals(E, mode)` (the 43×43→21×21 detector-resolution map, see below). `NZ,NY,NX,H,W` constants.
- `calo_ldm/metrics.py` — DarkSHINE validation physics metrics (`compute_observables`, `ShowerMetrics`). Replaces the removed ATLAS HLF/`ImageLogger` system. **All shower-shape observables are computed at real-detector (21×21×11 crystal) resolution** — `ShowerMetrics.update` downsamples the 43×43 grid first (computing shape on the half-cell grid is physically meaningless). Both models accumulate truth-vs-pred observables in their validation hooks (`VQModel`: reco, paired; `CondGPT`: generation, distribution-only) and emit per-shower distributions (E_tot, R, E_max fraction, longitudinal `<z>`/σ_z, lateral `<x>,<y>`/σ_x,σ_y, hit multiplicity, occupancy, cell spectrum) + profiles (per-layer energy & fraction, radial). Scalars go through `pl_module.log` every val epoch; `wandb.Histogram` + one combined overview figure every `metric_freq` epochs (`CondGPT` gates generation by `record_freq`). Knobs: `do_metric`, `metric_freq`, `metric_hit_threshold`, `downsample_mode` (and `record_freq` on `CondGPT`). The figure builder (`build_overview_figure`) and `scalar_summary`/`results` are reused by `eval-tools.py`.

### Config convention

Everything is instantiated from OmegaConf YAML via `instantiate_from_config`. `VQModel` runs in "overwrite mode": shared params (`cond_dim`, `log_scale_params`, `mask_path`) declared on `model.params` are pushed down into the encoder/decoder/loss sub-configs. `main.py` merges base configs left-to-right then applies `nested.key=value` CLI overrides.

## Differences from the original ATLAS-cylindrical Calo-VQ

| | Original (paper, cylindrical) | This repo (DarkSHINE, xyz) |
|---|---|---|
| Geometry | (R, Z, A), A periodic | (x, y, depth), no periodic axis |
| Conv | `CylinderConv` (cyclic A pad) + FFT A-resampling | plain `nn.Conv2d`/`nn.ConvTranspose2d`, zero pad |
| Channel axis | radial R | depth (11 layers) |
| Image plane | (Z, A) | (x, y) = 43×43 (padded 48) |
| Decoder softmax | over all voxels | **masked** over real crystals |
| Mask | none (full cylinder) | fixed staggered geometry mask, used in R / softmax / loss |
| Datasets | ds1/ds2/ds3 | DarkSHINE only |
| Lightning | 1.6.5 | 2.x (manual optimization) |

## Detector resolution & downsampling (training 43×43 vs detector 21×21)

Training **must** use the 11×43×43 half-cell grid — the staggered ECAL is artificially
aligned onto it (cell centres matched across layers), which is the core of the method.
But the model output is only physically meaningful **mapped back to the real detector
image** 11×21×21 = 4851 crystals: each crystal is a 2×2 cell block on the grid.

`calo_ldm/geometry.py:downsample_to_crystals(E, mode)` inverts the 2×2 upsample
(per-layer parity-offset 2×2 pooling, ONNX-friendly), with two modes:
- **`sum`** — crystal = Σ of its 4 cells (energy-conserving; recommended).
- **`min`** — crystal = 4 × min of its 4 cells (robustness check against model bias).

For **truth** the 4 cells are equal (each E/4) so `sum`==`4·min`==E; for VQ-VAE
reconstructions the 4 cells differ, so the two diverge. Two places use it:
1. **Validation metrics** always downsample to 21×21 before computing shower shape (Part above).
2. **`convert_to_detector_shape`** (config, default `False`): when `True`, `VQModel.postprocess`
   (hence `CondGPT.sample_fullchain` and gen-tools export) emits `(N,11,21,21)` instead of
   `(N,11,43,43)` — this is the ONNX / production output shape. Default `False` keeps 43×43 so
   reco/gen and truth share a shape and we downsample ourselves in evaluation. `downsample_mode`
   (config, default `sum`) selects the mode for both uses.

## Generation

`gen-tools.py` generates DarkSHINE showers from a trained stage-2 run and writes
an HDF5 with the **same schema as `export.h5`** (`condition (N,1)`, `energy (N,43,43,11)`
channels-last) so outputs are drop-in comparable with the real data:

```bash
$PY gen-tools.py --out darkshine_gen.h5 --model logs/<step2-run-dir> \
  --energy 4000.0 --nevts 10000 --batch-size 256
# detector resolution (N,21,21,11) for ONNX/production: --detector-shape [--mode sum|min]
# or draw E_inc from a file's `condition` column: --cond-file DarkSHINE_data/export.h5
```

It loads `configs/*.yaml` + `checkpoints/last.ckpt` from the run dir, builds the
conditioning batch (`vq_model.preprocess_cond` → `preprocess_cond`), and calls
`CondGPT.sample_fullchain`. `generation.sh` has ready-to-edit examples.

## Evaluation

`eval-tools.py` compares generated vs reference (truth) showers **at detector
resolution** (downsamples both to 21×21×11):

```bash
$PY eval-tools.py --gen darkshine_gen.h5 --ref DarkSHINE_data/export.h5 --out eval_out
# --mode sum|min, --nevts N, --hit-threshold MeV, --wandb
```

Outputs to `--out/`: `energy_truth.png` / `energy_gen.png` (12 deposited-energy
heatmaps each — 11 per-layer means + the layer-summed image), `shape_overview.png`
(the training shower-shape observables incl. E_tot and R=E_tot/E_inc, truth vs gen,
via `ShowerMetrics.build_overview_figure`), and `summary.json`. Accepts inputs at
either 43×43 or 21×21 (already-detector-shape files pass through).

## Logging

`main.py` defaults to **`WandbLogger` in offline mode** (no login; data under
`{logdir}/wandb`; `wandb sync` later, or set `lightning.logger` params to go online
with your entity/project). Physics metrics use native `wandb.Histogram`/`wandb.Image`
when wandb is active, else the overview figure falls back to `{log_dir}/metrics/`.

## Current DarkSHINE optimization note (2026-06-11)

A full 200-epoch baseline (`logs/2026-06-10T06-38-39_dss1` -> `logs/2026-06-10T11-41-38_dss2`) showed that the first DarkSHINE optimization pass did **not** learn a useful shower shape. The main failure was in Step 1, not only in the GPT prior:

- The old Step-1 config trained reconstruction in `R = E/E_inc` space with L2. Since DarkSHINE `R ~= 9.7e-4`, the pixel L2 loss was `O(1e-8)` and was dominated by codebook/GAN dynamics.
- The old Step-1 checkpoint used only 6 codes in a validation batch, with perplexity about 4.6. Its detector-level shape was over-broad: `E_max_frac` about 0.5x truth and width ratios about 1.6-1.8x truth, even though total energy/R were forced correct by construction.
- The old Step-2 free sampler also let the first R token use all 1024 codebook IDs, but with `R_bits=16` and 10 bits per code only 64 first-token values are legal. This could produce `R_pred > R_max`.

Conservative fix now in the configs/code:

- Step 1 trains shape with `reco_normalization: U`, `pixel_power: 1`, `disc_start: 10000`, and `disc_weight: 0.2`.
- Step 2 masks invalid first R-token logits during sampling and clamps decoded R integers into the representable range.
- `main.py` copies top-level `model.monitor` onto the instantiated model so existing YAML monitor fields drive `ModelCheckpoint`.
- `diagnostics/darkshine_diagnostics.py` records the failed baseline, gate metrics, and R-sampling range checks.

Use a staged gate before spending a full training budget: train optimized Step 1 for 30 epochs on two A800 GPUs, require at least 32 validation-batch unique codes and perplexity at least 16 plus materially improved detector-shape observables, then continue Step 1/Step 2 to 200 epochs only if the gate passes.

Gate result: `logs/2026-06-11T06-11-08_dss1_u_l1_gate` completed 30 epochs and failed the gate. `last.ckpt` reached `val/rec_loss=1.90604`, but only 2 validation-batch unique codes, perplexity 1.97, `E_max_frac` ratio 0.0024, `hits` ratio 23.7, and width ratios `z/x/y = 2.27/7.77/7.98`.

### 2026-06-11 pass 2 — ROOT-CAUSED and fixed (see `Agent_md/optimization_plan.md`, `diagnostics/ROOT_CAUSE_REPORT.md`)

The gate's "U + L1" pass failed for two **independent** reasons, both in Stage 1, neither of which the conservative pass addressed:

1. **Unit bug.** `condition` is keV (4e6 = 4 GeV), `energy` is MeV (~3900 = 3.9 GeV) — a 1000× mismatch. So `R ≈ 9.7e-4` was an *artifact*, not physics (true `R ≈ 0.97`). Through the fixed `LogScale(0,3000,7)` this starved the encoder (real-cell input mean 2e-5 vs the ~0.2–0.7 the convs need). Switching the loss to U-space did nothing because the encoder *input* scale was never fixed.
2. **VQ codebook/posterior collapse.** The plain gradient VQ cold-starts into collapse (1–2 live codes); DarkSHINE's single energy point makes it severe.

**Fixes applied:**
- Unit fix at the data boundary — `data.py::CaloDarkSHINE(condition_scale=1e-3)` (keV→MeV); `vqvae.preprocess_cond` offset −6.0→−3.6; `config/darkshine_step2.yaml` `R_max 0.0011→1.1`. `LogScale(0,3000,7)` unchanged (now correct).
- EMA codebook in `calo_ldm/layers/vectorquantizer.py` — EMA updates + data-dependent init + dead-code reinit (replaces the collapse-prone gradient codebook; training-only; `ema`/`ema_decay`/`reinit_threshold`/`reinit_every` knobs). Diagnostics-only dead-code-reinit on the plain gradient VQ was tried first and diverged to NaN — EMA is required.

**Validation.** Step 1 (`logs/2026-06-11T22-14-06_dss1_fix3`, best-val ckpt): **988 codes, perplexity 144, rec L1 0.289** (uniform=1.93); detector-shape ratios E_tot 1.000 / E_max_frac 0.992 / z_cog 0.978 / x_sigma 0.911 / hits 1.78. Step 2 (`logs/2026-06-12T14-24-34_dss2_fix3`, 50 ep) → generation @ 4 GeV (2.4 ms/shower): E_tot ratio 1.000, R 0.975 (physical), z_cog 0.989 / z_sigma 0.980, hits 1.81. From "never learned" to a working fast-sim; remaining blemish is hits/occupancy ~1.8× (sparsity is the next target). Full numbers in `Agent_md/optimization_plan.md`; reproducer `diagnostics/diagnose_stage1.py`.

**Two operational gotchas:** (1) always load data through `CaloDarkSHINE` (or apply `condition_scale`) when evaluating — reading `condition` raw feeds keV-scale inputs to a MeV-trained model and *fakes* an encoder collapse (this bit `diagnose_stage1.py` mid-investigation). (2) `main.py --max_epochs` is an argparse flag defaulting to 200 that **overrides** any `lightning.trainer.max_epochs` dotlist — use the flag (`--max_epochs 50`), not the dotlist. The GAN (`disc_start=10000`≈ep92, `disc_weight=0.2`) diverged to NaN at epoch 161 in the 200-epoch run; for >90-epoch runs raise `disc_start`/lower `disc_weight` or grad-clip.

**Hit/no-hit head (pass 3, occupancy fix).** The decoder has a second 1×1 head (`hit_head`, on *detached* features) trained with class-imbalance BCE on `E_cell>0.025 MeV`; at eval/generation a hard gate `sigmoid(hit_logits)>hit_gate_threshold` (default 0.4) masks the energy softmax and renormalises (`VQModel._apply_hit_gate`, preserves E_tot/R). It cuts generation `hits` 1.81→1.34 (thr 0.4) or →1.12 (thr 0.5) at a modest width cost (gating trims the outer halo, which also narrows the shower — a tunable tradeoff). Train it via `freeze_except_hit_head=True` (load a trained energy ckpt, freeze everything incl. the EMA codebook, train only the head, `disc_weight=0`) — a full fine-tune NaN's (fresh-Adam overshoot on converged weights). Artifacts: `logs/2026-06-12T23-09-55_dss1_hithead` (energy=fix3 + head; codebook identical so the GPT is reusable), `logs/2026-06-12T23-54-09_dss2_hit`. A cleaner occupancy/width decoupling needs a sparse output parameterisation (sparsemax / ReLU-normalised) — future work.


## Notes

The original cylindrical published checkpoints (`models/`) and paper-ds2 data
(`paper_data/`) have been removed (they were not loadable by the xyz code); the
cylindrical history remains in git. There is no test suite, linter, or CI in this
repo. Smoke configs: `config/darkshine_step1.yaml`, `config/darkshine_step2.yaml`
(1-epoch CPU end-to-end, exercising the full chain + metrics).
