Calo-VQ — DarkSHINE (xyz) edition
=======

Implementation of **Calo-VQ** ([2405.06605](https://arxiv.org/abs/2405.06605)): a
vector-quantized two-stage generative model for fast calorimeter-shower
simulation. Stage 1 is a VQ-VAE (adapted from
[taming-transformers](https://github.com/CompVis/taming-transformers)); stage 2
is an autoregressive GPT prior over the frozen codebook (adapted from
[minGPT](https://github.com/karpathy/minGPT)).

This fork is **refactored from the paper's ATLAS cylindrical geometry to the
DarkSHINE ECAL xyz geometry** ([2407.17800](https://arxiv.org/abs/2407.17800)):
a 21×21×11 staggered LYSO crystal calorimeter, stored on a 43×43×11 half-crystal
grid. The cylindrical-specific code (azimuthal periodic convolution, FFT
resampling, ds1/ds2/ds3 branches) has been removed — the model is now xyz-only.
See `CLAUDE.md` for the full architecture/geometry write-up.

# Environment

Conda env (Python 3.12): **torch 2.9**, **pytorch-lightning 2.x**, torchmetrics,
omegaconf, einops, numpy 2.x, h5py, matplotlib, scipy, psutil. CPU-only on macOS
(no CUDA). See `environements.yaml`.

> The code was migrated from `pytorch-lightning==1.6.5` to PL 2.x — `VQModel`
> now uses manual optimization and `main.py` builds the `Trainer` directly
> (no `--gpus`/`add_argparse_args`).

# Data

`DarkSHINE_data/export.h5` (10000 events @ 4 GeV) with keys `condition`
(incident energy, the stage-2 conditioning), `energy` (the `(N,43,43,11)`
deposited-energy input) and `label` (per-event hit flag, unused by the model).
The fixed staggered geometry mask `DarkSHINE_data/geom_mask.npy` `(43,43,11)` is
committed and required — it is used in the R normalization, the decoder masked
softmax, and the reconstruction loss.

# Training

Two-stage and **sequential**: Step 1 trains the VQ-VAE; Step 2 trains the GPT
prior over the frozen Step-1 codebook. The Step-2 config points at the Step-1
checkpoint via `model.params.vq_config`, passed on the CLI.

```bash
PY=/Users/zhoubaihong/miniconda3/envs/caloVQ/bin/python

# Step 1 — VQ-VAE
$PY main.py --base config/darkshine_step1.yaml -t True -l logs -n dss1

# Step 2 — GPT prior (point vq_config.checkpoint at the Step-1 last.ckpt)
CKPT=$(ls -t logs/*dss1*/checkpoints/last.ckpt | head -1)
$PY main.py --base config/darkshine_step2.yaml -t True -l logs -n dss2 \
  "model.params.vq_config.checkpoint=$CKPT"
```

Notes:
- `config/darkshine_step1.yaml` / `darkshine_step2.yaml` are 1-epoch CPU smoke
  configs (end-to-end sanity check).
- Trainer settings (`max_epochs`, `limit_train_batches`, …) live in the
  `lightning.trainer:` section of the config.
- `nested.key=value` CLI args override config values (dotlist).
- `--gpus N` selects GPUs (omit / `0` → CPU). LR is scaled by
  `ngpu * batch_size * accumulate_grad_batches` unless `--scale_lr False`.

# Generation

`gen-tools.py` generates DarkSHINE showers from a trained Step-2 run and writes an
HDF5 with the **same schema as `export.h5`** (`condition (N,1)`, `energy (N,43,43,11)`),
so generated files are drop-in comparable with the real data:

```bash
$PY gen-tools.py --out darkshine_gen.h5 --model logs/<step2-run-dir> \
  --energy 4000.0 --nevts 10000 --batch-size 256
# real detector resolution (N,21,21,11) for ONNX/production: --detector-shape [--mode sum|min]
# multi-energy: draw E_inc from a file's `condition` column instead of --energy:
#   --cond-file DarkSHINE_data/export.h5
```

`sample_fullchain(batch)` on the Step-2 `CondGPT` is the underlying entry point.
See `generation.sh` for ready-to-edit examples.

## Detector resolution

Training uses the 11×43×43 half-cell grid (the staggered crystals are aligned
onto it). The physical detector image is **11×21×21 = 4851 crystals** — each
crystal is a 2×2 cell block. `calo_ldm/geometry.py:downsample_to_crystals` maps a
shower back to crystals with mode `sum` (Σ of the 4 cells, energy-conserving) or
`min` (4×min). The model emits detector shape when `convert_to_detector_shape:
true` (config, default false; the ONNX/production output); validation metrics and
`eval-tools.py` always evaluate at crystal resolution.

# Evaluation

`eval-tools.py` compares generated vs truth showers at detector resolution:

```bash
$PY eval-tools.py --gen darkshine_gen.h5 --ref DarkSHINE_data/export.h5 --out eval_out
# --mode sum|min, --nevts N, --hit-threshold MeV, --wandb
```

Produces 12 deposited-energy heatmaps each for truth & gen (11 per-layer means +
the layer-summed image), a shower-shape overview (E_tot, R=E_tot/E_inc,
longitudinal/lateral profiles & widths, hit multiplicity, cell spectrum, …), and
`summary.json`.

# Validation metrics & logging

Training logs DarkSHINE shower observables (E_tot, R, longitudinal/lateral
profiles & widths, hit multiplicity, cell spectrum, …) comparing
reconstruction/generation against truth — see `calo_ldm/metrics.py`. `main.py`
logs to **Weights & Biases** (offline by default; data under `logs/.../wandb`,
`wandb sync` to upload, or set `lightning.logger` to go online). Tune with the
`do_metric` / `metric_freq` / `metric_hit_threshold` model params (and
`record_freq` for Step-2 generation metrics).

# Thanks to

We would like to express our gratitude to the authors of the codes used in this
work: [taming-transformers](https://github.com/CompVis/taming-transformers) and
[minGPT](https://github.com/karpathy/minGPT).
</content>
</invoke>
