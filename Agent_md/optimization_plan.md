# 2026-06-11 (pass 2) — root-caused the "model never learned" failure

Full evidence: `diagnostics/ROOT_CAUSE_REPORT.md`; reproducer: `diagnostics/diagnose_stage1.py`.

**Two independent root causes, both in Stage 1:**

1. **Unit mismatch (keV vs MeV).** `condition` is stored in keV (4e6 = 4 GeV); `energy`
   is in MeV (Σ ≈ 3900 = 3.9 GeV). The code computed `R = E/E_inc ≈ 9.7e-4` instead of
   the physical ~0.97. Through the fixed encoder front-end `LogScale(0,3000,7)` (tuned
   for the paper's R≈1.3) this mapped real-cell inputs to a mean of **2e-5** (paper
   regime ~0.2–0.7) → the encoder was **starved**. The earlier "U-space loss" patch did
   not help because it fixed the *loss* scale, not the *encoder-input* scale.
2. **VQ codebook / posterior collapse.** Even with healthy inputs the plain gradient VQ
   cold-starts into collapse (codebook init `±1/n_e` ≪ encoder outputs → 1 code nearest
   to everything → commitment loss drags the encoder onto it). DarkSHINE's single energy
   point (near-identical showers) makes this severe. Result: 1–2 live codes, decoder
   emits the over-broad mean shower.

**Fixes (minimal, traceable):**
- Unit fix at the data boundary: `data.py::CaloDarkSHINE(condition_scale=1e-3)` (keV→MeV);
  `vqvae.preprocess_cond` offset −6.0→−3.6 (recentre `log_E_inc`); `step2` `R_max 0.0011→1.1`.
  `LogScale(0,3000,7)` is left unchanged — now correct (real-cell mean 0.076 / p99 0.65).
- EMA codebook in `vectorquantizer.py`: EMA updates + data-dependent init + dead-code
  reinit (replaces the collapse-prone gradient codebook; no NaN). Knobs: `ema`,
  `ema_decay`, `reinit_threshold`, `reinit_every`.

**Validation (Step 1, 50 epochs, CPU, load_partial=1000/file, `--scale_lr True`):**
- Run: `logs/<ts>_dss1_fix3`. Diagnostic at **epoch 18** (corrected diagnostic, routed
  through the real dataloader):

  | metric | failed baseline | fix3 @ep18 |
  |---|---|---|
  | unique codes (/1024) | 2 | 796 |
  | perplexity | 1.97 | 74.6 |
  | rec L1 (uniform=1.93) | ~1.91 | 0.42 |
  | E_max_frac ratio | 0.0024 | 0.97 |
  | hits ratio | 23.6 | 0.79 |
  | z_cog / z_sigma ratio | 2.2 / 2.27 | 0.90 / 0.90 |
  | x_sigma / y_sigma ratio | 7.8 / 8.0 | 0.74 / 0.74 |

  Final Step-1 artifact = `logs/2026-06-11T22-14-06_dss1_fix3/checkpoints/last.ckpt`
  (= best-val `epoch=000160.ckpt`; the run's `max_epochs` was clobbered to 200 — see
  the `--max_epochs` gotcha below — and the GAN diverged at epoch 161, but the saved
  best-val model is healthy). At 1000 val events: **988 codes, perplexity 144, rec L1
  0.289**; ratios E_tot 1.000 / E_max_frac 0.992 / z_cog 0.978 / z_sigma 0.973 /
  x_sigma 0.911 / y_sigma 0.927 / hits 1.78.

**Step 2 (GPT prior) + generation** — `logs/2026-06-12T14-24-34_dss2_fix3` (50 ep,
`R_max=1.1`, R-loopback error 1.5e-5). `val/loss` 2.289 → 2.251 (uniform = log 1024 =
6.93). Generated 2000 showers @ 4 GeV (2.4 ms/shower) and compared to truth at 21×21
(`eval-tools.py`):

  | gen-vs-truth | failed baseline (old eval) | fix3 |
  |---|---|---|
  | E_tot ratio | 0.0094 | 1.000 |
  | R (physical) | garbage (units bug) | 0.975 vs 0.975, ratio 1.00 |
  | z_cog / z_sigma ratio | 2.20 / 1.89 | 0.989 / 0.980 |
  | hits ratio | 0.28 | 1.81 |

**Net:** from "model never learned" (E_max_frac 0.24% of truth, widths 7.8×, 2 live
codes, generation 1% of the energy) to a working fast-sim that reproduces total energy,
R, and the longitudinal/lateral shower shape. Remaining blemish: **hits/occupancy ~1.8×**
(energy spread into ~1.8× as many low-energy cells near the 0.1 MeV threshold) — next
target, e.g. a hit/no-hit head or sparsity-weighted reconstruction.

**Unit bookkeeping is now consistent end-to-end:** `condition` is keV in the h5;
`CaloDarkSHINE`, `gen-tools.py` (writes keV back), and `eval-tools.py` (`condition_scale`)
all convert to MeV, so R is physical (~0.97) everywhere.

## 2026-06-12 (pass 3) — hit/no-hit head for the occupancy (hits) blemish

The `hits` ratio ~1.8× (energy spread into too many low-energy cells) is structural:
the masked voxel-softmax can't emit exact zeros, so it leaves a soft outer halo.

**Added a hit/no-hit head** (`decoder.py` `hit_head`, a parallel 1×1 conv on
*detached* final features): trained with a masked, class-imbalance-weighted BCE on
`E_cell > 0.025 MeV` (`combined.py`, `hit_weight`); at eval/generation a hard gate
`sigmoid(hit_logits) > hit_gate_threshold` masks the energy softmax and renormalises
(`VQModel._apply_hit_gate`), preserving E_tot/R exactly. Generation needs no GPT change
(the gate decodes deterministically from the codes).

**Training notes / dead ends:**
- Full fine-tune of the converged fix3 model with the new head **NaN'd at step ~13**
  (fresh-Adam overshoot on converged weights). A fresh 50-epoch train was stable but
  under-occupied (hits 0.75 — occupancy is training-duration-dependent: early=peaked/few,
  late=broad/many). The robust path is `freeze_except_hit_head=True` (VQModel): load the
  validated fix3 energy model, freeze everything (incl. the EMA codebook), train ONLY the
  hit head (disc off). rec_loss stays 0.29 (energy untouched); no NaN.
- Stage-1 artifact: `logs/2026-06-12T23-09-55_dss1_hithead` (= fix3 energy + hit head;
  codebook identical to fix3, so the existing GPT is compatible). Stage 2 retrained:
  `logs/2026-06-12T23-54-09_dss2_hit`.

**The gate couples hits and width** (it trims the outer tail, which also narrows the
shower), so it is a tunable tradeoff via `hit_gate_threshold` (default 0.4). Generation
(2000 showers vs truth, ratios→1):

  | metric | no hit head | hit+gate (thr 0.4) | thr 0.5 |
  |---|---|---|---|
  | hits | 1.81 | **1.34** | 1.12 |
  | E_max_frac | 0.99 | 1.10 | 1.18 |
  | x/y_sigma | 0.92 / 0.96 | 0.82 / 0.83 | 0.78 / 0.78 |
  | z_cog / z_sigma | 0.99 / 0.98 | 0.96 / 0.93 | — |
  | E_tot / R | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 |

hits improves 1.81→1.34 (thr 0.4) or →1.12 (thr 0.5) at a modest width cost. A cleaner
fix (decouple occupancy from width) needs a genuinely sparse energy parameterisation
(e.g. sparsemax / ReLU-normalised output, or a learned per-cell threshold) — the next
redesign if tighter occupancy+width is required.

**Gotcha for the next agent:** evaluate models THROUGH `CaloDarkSHINE` (or apply
`condition_scale`), never by reading `condition` raw from the h5 — the raw value is keV
and will fake an encoder collapse on a correctly-trained model.

**Two operational gotchas found during this pass:**
- `main.py --max_epochs` is an argparse flag that **defaults to 200 and unconditionally
  overwrites** `lightning.trainer.max_epochs` (main.py:143). A dotlist
  `lightning.trainer.max_epochs=50` is silently clobbered — use the **flag** `--max_epochs 50`.
- **GAN late-divergence.** With `disc_start=10000` steps (~epoch 92 at load_partial=1000)
  and `disc_weight=0.2`, the `dss1_fix3` run was stable for ~70 more epochs then went to
  **NaN at epoch 161** (perplexity collapsed back to 1, rec_loss=nan). The best-val
  checkpoint (`last.ckpt` == `epoch=000160.ckpt`: 988 codes, rec L1 0.289, no NaN) was
  saved before the blow-up and is the Stage-1 artifact used for Stage 2. For runs that
  will exceed ~90 epochs, raise `disc_start` / lower `disc_weight` (or keep the GAN off —
  reconstruction is already excellent without it), and consider grad-clipping. At ≤50
  epochs the discriminator never activates, so it is a non-issue for the gate run.

---

# 2026-06-11 conservative DarkSHINE optimization pass

## Diagnosis

- Baseline runs exist at `logs/2026-06-10T06-38-39_dss1` and `logs/2026-06-10T11-41-38_dss2`, both trained for 200 epochs.
- Step 1 is the primary failure: old `reco_normalization: R` + L2 makes the reconstruction signal tiny (`R ~= 9.7e-4`, L2 around `1e-8`), so the model collapses to a broad template.
- Reproduced failed Step-1 numbers on `DarkSHINE_data/j14/export.h5`: 6 unique codes, perplexity 4.6, `E_max_frac` ratio 0.50, `z_sigma` ratio 1.64, `x_sigma` ratio 1.63, `y_sigma` ratio 1.80.
- Step 2 also had an R sampler bug: with `R_bits=16` and 10-bit code tokens, only 64 first-token values are legal; old sampling allowed all 1024.

## Patch status

- Step 1 config changed to `reco_normalization: U`, `pixel_power: 1`, `disc_start: 10000`, `disc_weight: 0.2`.
- Step 2 sampling now masks invalid first R-token logits and clamps decoded R integers. Old Step-2 checkpoint now samples `R` within `[0, R_max]` in `diagnostics/darkshine_diagnostics.py`.
- `main.py` now preserves existing top-level YAML `model.monitor` fields for `ModelCheckpoint`.
- Added `diagnostics/darkshine_diagnostics.py` for baseline/gate diagnostics and R-sampling assertions.

## Gate plan

- Run optimized Step 1 for 30 epochs on `--gpus 2`.
- Gate passes only if validation-batch unique code count is at least 32, perplexity is at least 16, and detector-shape observables move materially toward truth versus the failed baseline.
- If the gate passes, continue Step 1 to 200 epochs, train Step 2 for 200 epochs, generate samples, and run `eval-tools.py`. If it fails, stop before Step 2 and switch to a larger refactor such as sparse/hit-aware output modeling.

## 30-epoch gate result

- Run: `logs/2026-06-11T06-11-08_dss1_u_l1_gate`, trained with `--gpus 2 --max_epochs 30`.
- Best/last checkpoint: `logs/2026-06-11T06-11-08_dss1_u_l1_gate/checkpoints/last.ckpt` (`val/rec_loss` reached 1.90604 at epoch 29).
- Diagnostic output: `/tmp/calo-vq-gate-diagnostics.json`.
- Gate failed: validation-batch code usage collapsed further to 2 unique codes with perplexity 1.97, below the required 32 / 16.
- Detector-shape diagnostics also failed: `E_max_frac` ratio 0.0024, `hits` ratio 23.7, `z_sigma` ratio 2.27, `x_sigma` ratio 7.77, and `y_sigma` ratio 7.98. Total energy and R remain forced correct by the decoder normalization, so they are not evidence of shape learning.
- Decision: stop before Step 2/full training. The conservative `U` + L1 + GAN warmup pass is insufficient for DarkSHINE; the next optimization pass should be a larger Step-1 refactor focused on sparse/hit-aware modeling and codebook collapse, e.g. a hit/no-hit occupancy head or weighted sparse reconstruction, stronger codebook usage regularization/reinitialization, and possibly a less over-compressed latent grid.

---

# DarkSHINE Calo-VQ — migration notes & optimization plan

Status after the cylindrical→xyz refactor: both stages run end-to-end on CPU
(`config/darkshine_step1.yaml` then `config/darkshine_step2.yaml`, 1 epoch each).
This document records what was removed, what is still rough, and where to go next.

## 1. Cylindrical components removed

Deleted files:
- `calo_ldm/layers/conv.py`, `conv2.py` — `CylinderConv(2)` / `CylinderConvTranspose(2)` (azimuthal cyclic padding).
- `calo_ldm/layers/fft.py` — `FFTDownsample`/`FFTInterpolate(V2)` azimuthal resampling.
- `calo_ldm/layers/models.py`, `calo_ldm/models/clustering.py`, `dense.py`, `hybrid1d.py`, `decoderMH.py` — cylindrical/ds1/ds3-only modules (per-layer-norm decoder, clustering perceptual net scaffolding, flat ds1 dense net).

Replaced / rewritten:
- `calo_ldm/layers/conv_xyz.py` — new `PlaneConv` / `PlaneConvTranspose` (plain 2D conv, **zero** padding both x and y, no periodic axis).
- `encoder.py`, `decoder.py`, `discriminator.py` — xyz versions; pad 43→48 / crop 48→43; decoder uses a **masked voxel-softmax**.
- `vqvae.py` (`VQModel`) — slimmed to a single global-R xyz path (removed ds1/ds2/ds3 branches, `layer_seg`/`interleaveR`, `z_pad`/`z_padding_strategy`, cylindrical metric & plotting). PL2 manual optimization retained.
- `combined.py` (`CombinedLoss`) — masked recon + shower centre/width in (x,y) per depth-layer.
- `data.py` — added `CaloDarkSHINE` (channels-last → channels-first permute).
- `util.py` — added `load_geom_mask`; removed dead `parse_spec_str`/`cal_RZA`/`cal_ZA` helpers.

Also: `main.py` + `callbacks.py` migrated PL 1.6.5 → 2.x; `gpt.py` got a `darkshine`
branch and an R-loopback test scaled to `R_max`.

## 2. Known rough edges / further refactor

- **Geometry mask provenance.** `DarkSHINE_data/geom_mask.npy` is inferred empirically
  (contiguous 42×42 block per layer, parity offset). It matches the data perfectly
  (zero energy outside it; 4851 crystals), but it should be cross-checked against the
  exact arXiv:2407.17800 crystal map, especially the parity/stagger direction and any
  non-square edge effects. Provide a builder that reads the true geometry.
- **Single energy point.** `condition` is constant (4 GeV), so the conditioning is
  currently a constant. Re-validate `log_E_inc` normalization and `R_max`/`R_bits`
  once multi-energy data exists; the conditioning path is wired but untested across energies.
- **`gen-tools.py` / `generation.sh`** still target the cylindrical submission format
  (`{showers, incident_energies}`, ZA→flat reshape). Port to xyz: call
  `CondGPT.sample_fullchain`, reshape `(N,11,43,43)` back to channels-last `(N,43,43,11)`,
  and write whatever DarkSHINE evaluation expects.
- **Cluster-perceptual loss** (`losses/cluster_perceptual.py`) is unused (perceptual_weight=0)
  and references a cylindrical encoder checkpoint; either delete or retrain for xyz.
- **EMA / scheduler** paths are kept but untested on the xyz model.
- **Old artifacts**: `models/` checkpoints and the `config/ds2_*` + `config/ds1*/ds3*`
  configs are cylindrical and no longer loadable; prune once no longer needed for reference.

## 3. Physics evaluation metrics (the big TODO)

The paper evaluates with a **separation metric** over histograms of high-level features.
The original `plot.py` (`HighLevelFeatures2`, `phyPlotter`, `CaloPlotter`) is cylindrical
(polar η/φ binning via per-dataset XML) and is **not wired into the xyz model** (`do_metric=False`).
To adapt:

- **Shower centre / width**: change from η/φ (polar, derived from R,A) to **x/y** centroids
  and RMS widths. The loss already computes per-depth-layer EC_x/EC_y/width_x/width_y in
  `combined.py:_centre_width` — promote the same computation to a validation metric and
  histogram it (truth vs generated), masked to real cells.
- **Layer energies**: E per depth layer (sum over x,y) → 11-bin profile. Straightforward
  channels-first sum; histogram per layer.
- **Cell energy spectrum**: flatten real-cell energies (mask-selected) and histogram in log.
- **Total energy / R**: histogram `ΣE` and `R = ΣE/E_inc`; for a single energy point this is
  a tight peak — a good first sanity check that generation reproduces the scale.
- **Separation metric**: reimplement the triangular-distance/χ² separation between truth and
  generated histograms for each feature above. Write a small `darkshine_metrics.py` rather
  than reviving the cylindrical `plot.py`.

Wire these into `VQModel.val_accumulate` / a validation callback (currently a no-op) and
gate behind `do_metric` so smoke runs stay fast.

## 4. Performance & accuracy directions

- **Codebook utilization**: monitor `perplexity` / `cluster_usage` (already logged). If low,
  consider codebook re-init / EMA codebook / larger `sequence_shape` than (6,6).
- **Exploit sparsity**: per-event only ~4% of cells fire. A sparsity-aware reconstruction
  weighting (or a hit/no-hit head) could improve fidelity and speed.
- **Masked input**: the encoder currently sees zero-padding cells. Feeding the mask as an
  extra channel (or pre-masking) may help the encoder ignore padding.
- **R modelling**: with R≈1e-3 in a narrow band, verify `R_bits`/`R_max` give enough
  resolution; consider per-layer R if depth-energy spread is large (the layer-wise-R machinery
  was removed but the idea is sound for DarkSHINE).
- **Architecture**: 43→48 padding wastes a little compute; a conv stack designed natively for
  43 (with `output_padding`) would avoid the pad/crop. Tune depth/width/codebook for accuracy.
- **GAN stability**: keep the adaptive disc weight (load-bearing per the paper); tune
  `disc_start`, `disc_weight`, and consider spectral norm if training is unstable at scale.
- **Throughput**: real training needs GPU + full dataset (drop `load_partial`), more epochs,
  and `num_workers>0`; current smoke configs are CPU/tiny on purpose.

---

## 2026-06-14 — output-parameterisation experiment (sparsemax) — FAILED, reverted

**Hypothesis.** Generation's residual blemish is occupancy (`hits` over-count). The dense
masked voxel-softmax can never emit exact zeros, so it always leaves a non-zero halo →
too many hits and an irreducible U-space recon-L1 floor vs the genuinely-sparse truth.
Idea: replace softmax with a sparse output transform that can emit EXACT zeros
(sparsemax = Euclidean projection onto the simplex; relu_norm = relu+renormalise), still
summing to 1 over real cells so the whole downstream (R/disc/loss/gate/stage-2) is unchanged.
Implemented as a configurable `Decoder.output_activation` (default `voxel_softmax`), with
standalone configs `config/darkshine_step1_sparsemax.yaml` / `_relunorm.yaml` (hit_gate off,
hit_weight 0, so occupancy comes from the output transform alone).

**Result — sparsemax is clearly WORSE.** 200-epoch runs:
- baseline softmax + hit-head + gate: `logs/2026-06-13T10-17-36_dss1`
- sparsemax (no hit head): `logs/2026-06-13T13-35-53_dss1`

| metric (val/reco) | truth | baseline (softmax+gate) | sparsemax |
|---|---|---|---|
| `rec_loss` (U, L1) | — | **0.183** | 0.385 (2.1× worse) |
| `hits_ratio` | 1.0 | **0.94** | 0.53 (over-sparsified) |
| `L1_width_x` | — | **1.87** | 2.24 |
| `z_cog_ratio` | 1.0 | 0.947 | 0.944 |
| `z_sigma_ratio` | 1.0 | 0.920 | **1.012** |
| `E_tot_ratio` | 1.0 | 1.0 | 1.0 |
| perplexity / cluster_usage | — | 137 / 0.82 | 168 / 0.86 |

**Root cause (two compounding mistakes).**
1. **Wrong baseline in my head.** The blemish was the *un-gated* fix3 (`hits≈1.8×`). The
   current baseline already carries the hit-head + eval gate and reaches `hits≈0.94` —
   occupancy was already essentially solved. I optimised a problem that was no longer open.
2. **Wrong sparsity prior for the physics.** sparsemax is "peaky / winner-take-most": a
   single global threshold τ hard-cuts small entries to 0. But a shower's per-cell energy
   distribution is **long-tailed** (a few hot cells + a broad low-energy tail of many small
   hits). sparsemax truncates that tail (`hits` 206→108), then piles the mass back onto the
   survivors, distorting the shape → recon-L1 *doubles*. It over-corrected from "too many
   hits" all the way to "too few".

**Lesson (load-bearing).** **Occupancy is best modelled SEPARATELY from shape**, not folded
into the energy distribution. The existing decomposition — smooth softmax for shape + an
independent BCE hit-head + a hard gate for occupancy — is the right design; sparsemax
re-couples the two jobs into one transform under a sparsity prior that does not match the
long-tailed shower. `relu_norm` was not run (predicted to share the failure mode, softer);
not worth the compute given sparsemax's clear regression. The `output_activation` knob and
configs are kept in the tree (default `voxel_softmax`) but **sparse output is abandoned**;
step-1 of record stays the hit-head baseline (`logs/2026-06-13T10-17-36_dss1`).

**Reframe — where the real step-1 ceiling is.** The baseline is already strong (rec 0.183,
all ratios near 1, hits 0.94). The remaining shape residuals (width L1≈1.9, z_sigma 0.92)
are a **backbone/capacity** question, not an output-transform one — no output activation
changes how well the encoder/decoder capture shape. Next, low-risk step before any wholesale
ViT rewrite: add 1–2 self-attention blocks at the CNN **bottleneck (6×6)** (the
taming-transformers/VQGAN pattern) — cheap (36 tokens), keeps all CNN inductive bias, and
directly tests whether global mixing buys shape fidelity. If it helps, escalate toward ViT;
if not, the backbone is not the limit. Implemented as `bottleneck_attn` on Encoder/Decoder
(default 0 = off, i.e. identity), config `config/darkshine_step1_attn.yaml`.
