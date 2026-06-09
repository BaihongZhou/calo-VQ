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
