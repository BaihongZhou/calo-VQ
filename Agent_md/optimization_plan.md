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
