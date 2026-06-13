#!/usr/bin/env python
"""DarkSHINE evaluation: generated vs reference (truth) showers.

Everything is evaluated at REAL DETECTOR resolution (21x21x11 = 4851 crystals):
inputs on the 43x43 half-cell training grid are downsampled with
`downsample_to_crystals` (mode `sum`/`min`); inputs already at 21x21 are used
as-is. Outputs (PNG to --out, optionally also to a wandb run):

  * energy_truth.png / energy_gen.png -- 12 deposited-energy heatmaps each
    (per-layer mean over events for layers 0..10, plus the layer-summed image).
  * shape_overview.png -- the shower-shape observables tracked during training
    (E_tot, R=E_tot/E_inc, longitudinal/lateral profiles & widths, hit
    multiplicity, occupancy, cell spectrum, radial profile), truth vs gen.
  * a printed scalar summary (means + ratios).

Usage:
  eval-tools.py --gen darkshine_gen.h5 [--ref DarkSHINE_data/export.h5] --out eval_out
"""
import os
import sys
import json
import argparse

import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from calo_ldm.geometry import downsample_to_crystals, NZ, NY, NX
from calo_ldm.metrics import ShowerMetrics

torch.set_grad_enabled(False)


def load_showers(path, n=None, key_e="energy", key_c="condition", condition_scale=1e-3):
    # condition is stored in keV (4e6 = 4 GeV); convert to MeV so it matches the
    # per-cell deposited energy and R = E_tot/E_inc is physical (~0.97). Both the
    # reference export.h5 and gen-tools output use the keV schema, so this is applied
    # uniformly. (Mirrors CaloDarkSHINE's condition_scale.)
    with h5py.File(path, "r") as f:
        E = torch.from_numpy(f[key_e][:n]).float()       # (N, H, W, 11) channels-last
        C = torch.from_numpy(f[key_c][:n]).float() * condition_scale   # (N, 1) keV -> MeV
    E = E.permute(0, 3, 1, 2).contiguous()               # (N, 11, H, W)
    if C.dim() == 1:
        C = C.unsqueeze(-1)
    return E, C


def to_crystals(E, mode):
    return E if E.shape[-1] == NX else downsample_to_crystals(E, mode)


def energy_grid_figure(per_layer_mean, summed_mean, vmax, title):
    """12-panel grid: layers 0..10 (21x21 mean-energy heatmaps) + layer sum."""
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    axes = axes.ravel()
    for z in range(NZ):
        im = axes[z].imshow(per_layer_mean[z], origin="lower", cmap="viridis",
                            vmin=0, vmax=vmax[z])
        axes[z].set_title(f"layer {z}", fontsize=9)
        axes[z].tick_params(labelsize=6)
        fig.colorbar(im, ax=axes[z], fraction=0.046, pad=0.04)
    im = axes[11].imshow(summed_mean, origin="lower", cmap="viridis", vmin=0, vmax=vmax[11])
    axes[11].set_title("all layers (sum)", fontsize=9)
    axes[11].tick_params(labelsize=6)
    fig.colorbar(im, ax=axes[11], fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def mean_energy_images(E):
    # E: (N,11,21,21) -> (per-layer mean (11,21,21), summed-over-layers mean (21,21))
    per_layer = E.mean(0).cpu().numpy()                  # (11,21,21)
    summed = E.sum(1).mean(0).cpu().numpy()              # (21,21)
    return per_layer, summed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", required=True, help="generated showers HDF5")
    ap.add_argument("--ref", default="DarkSHINE_data/export.h5", help="reference (truth) HDF5")
    ap.add_argument("--out", default="eval_out", help="output directory for PNGs")
    ap.add_argument("--mode", default="sum", choices=["sum", "min"],
                    help="43x43 -> 21x21 crystal aggregation (only used for 43x43 inputs)")
    ap.add_argument("--nevts", type=int, default=None, help="cap events used from each file")
    ap.add_argument("--hit-threshold", type=float, default=0.1, help="[MeV] hit threshold")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--wandb", action="store_true", help="also log to a (offline) wandb run")
    ap.add_argument("--wandb-project", default="calo-vq-darkshine")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    gen_E, gen_C = load_showers(args.gen, args.nevts)
    ref_E, ref_C = load_showers(args.ref, args.nevts)
    n = min(gen_E.shape[0], ref_E.shape[0])
    if gen_E.shape[0] != ref_E.shape[0]:
        print(f"NOTE: gen has {gen_E.shape[0]} events, ref has {ref_E.shape[0]}; "
              f"using {n} from each.")
    gen_E, gen_C = gen_E[:n], gen_C[:n]
    ref_E, ref_C = ref_E[:n], ref_C[:n]

    gen_c = to_crystals(gen_E, args.mode)                # (n,11,21,21)
    ref_c = to_crystals(ref_E, args.mode)
    print(f"truth {tuple(ref_c.shape)}, gen {tuple(gen_c.shape)} @ crystal resolution")

    # --- shower-shape metrics (truth vs gen, unpaired distributions) ---
    metrics = ShowerMetrics(downsample_mode=args.mode, hit_threshold=args.hit_threshold)
    for i in range(0, n, args.batch_size):
        sl = slice(i, i + args.batch_size)
        metrics.update(ref_c[sl], gen_c[sl], ref_C[sl], pred_E_inc=gen_C[sl])
    t, p, tprof, pprof, tcell, pcell = metrics.results()
    summary = metrics.scalar_summary(t, p, tag="eval", paired=False)
    print("\n--- scalar summary (truth vs gen) ---")
    for k, v in summary.items():
        print(f"  {k}: {v:.4g}")
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    shape_fig = metrics.build_overview_figure("eval", None, t, p, tprof, pprof, tcell, pcell)
    shape_fig.savefig(os.path.join(args.out, "shape_overview.png"), dpi=100)

    # --- deposited-energy images (truth & gen), shared per-layer color scale ---
    ref_layers, ref_sum = mean_energy_images(ref_c)
    gen_layers, gen_sum = mean_energy_images(gen_c)
    vmax = [max(ref_layers[z].max(), gen_layers[z].max(), 1e-12) for z in range(NZ)]
    vmax.append(max(ref_sum.max(), gen_sum.max(), 1e-12))   # for the summed panel
    fig_t = energy_grid_figure(ref_layers, ref_sum, vmax, f"truth — deposited energy — N={n}")
    fig_g = energy_grid_figure(gen_layers, gen_sum, vmax, f"gen — deposited energy — N={n}")
    fig_t.savefig(os.path.join(args.out, "energy_truth.png"), dpi=100)
    fig_g.savefig(os.path.join(args.out, "energy_gen.png"), dpi=100)

    print(f"\nWrote: shape_overview.png, energy_truth.png, energy_gen.png, summary.json -> {args.out}/")

    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, name="eval", mode="offline",
                         dir=args.out)
        run.log({"eval/shape_overview": wandb.Image(shape_fig),
                 "eval/energy_truth": wandb.Image(fig_t),
                 "eval/energy_gen": wandb.Image(fig_g), **summary})
        run.finish()
        print("logged to wandb (offline)")

    for fig in (shape_fig, fig_t, fig_g):
        plt.close(fig)


if __name__ == "__main__":
    main()
