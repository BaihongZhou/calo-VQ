#!/usr/bin/env python
"""DarkSHINE ECAL offline evaluation: Calo-VQ generated vs Geant4 reference.

Adapted from the Fast Calorimeter Challenge 2022 evaluation
(`diagnostics/CaloChannelge/code/evaluate.py` + `evaluate_plotting_helper.py`)
to the DarkSHINE xyz geometry. Everything is evaluated at REAL DETECTOR
resolution (21x21x11 = 4851 crystals): inputs on the 43x43 half-cell training
grid are downsampled with `downsample_to_crystals`. At crystal resolution there
is no padding -- all crystals are real -- so no per-event `label` mask is needed.

Metrics produced (gen vs ref overlays, each with a separation-power chi2):
  1. E_tot           total deposited energy + the E_tot < 0.5*E_inc subset
                     (DarkSHINE low-deposition / partially-contained events).
  2. longitudinal    mean energy per z layer (0..10).
  3. centroid x, y   transverse energy centre  Sum(E*x)/Sum(E).
  4. width x, y      transverse RMS  sqrt(Sum(E*(x-xc)^2)/Sum(E)).
  5. (per-energy binning: not run -- single energy point; see --energy-bins.)
  6. pixel spectrum  pooled per-crystal E_i / E_tot (log-log).
  + extras           R=E_tot/E_inc, hit multiplicity, occupancy, average-shower
                     heatmaps, and the combined training-style overview figure.
  7. (classifier AUC/JSD: interface stub only, --classifier; not implemented.)

Usage:
  evaluation/evaluate.py --gen darkshine_gen.h5 [--ref DarkSHINE_data/export.h5] \
      --out evaluation_out [--mode sum] [--nevts N] [--hit-threshold 0.1]
"""
import os
import sys
import json
import argparse

# allow `import calo_ldm` / `import evaluation.*` when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")

from calo_ldm.geometry import NZ
from calo_ldm.metrics import ShowerMetrics

from evaluation.data_io import load_showers, to_crystals
from evaluation.observables import (
    low_deposit_mask, pixel_norm_energies, per_layer_observables)
from evaluation import plots

torch.set_grad_enabled(False)


def _log(msg):
    print(f"[eval] {msg}", flush=True)


def mean_energy_images(E):
    """E:(N,11,21,21) -> per-layer mean (11,21,21) + layer-summed mean (21,21)."""
    per_layer = E.mean(0).cpu().numpy()
    summed = E.sum(1).mean(0).cpu().numpy()
    return per_layer, summed


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", required=True, help="generated showers HDF5")
    ap.add_argument("--ref", default="DarkSHINE_data/j0/export.h5",
                    help="reference (Geant4 truth) HDF5 (data is sharded into j0..j15)")
    ap.add_argument("--out", default="evaluation_out", help="output directory")
    ap.add_argument("--mode", default="sum", choices=["sum", "min"],
                    help="43x43 -> 21x21 crystal aggregation (43x43 inputs only)")
    ap.add_argument("--nevts", type=int, default=None,
                    help="cap events used from each file")
    ap.add_argument("--hit-threshold", type=float, default=0.1,
                    help="[MeV] cell hit threshold (multiplicity/occupancy)")
    ap.add_argument("--lowdep-frac", type=float, default=0.5,
                    help="low-deposit selection: keep events with E_tot < frac*E_inc")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--energy-bins", default="none",
                    help="'none' (global only, default). Per-energy binning is a "
                         "no-op on the current single-energy dataset; reserved.")
    ap.add_argument("--classifier", action="store_true",
                    help="(stub) CaloChallenge cls-low/cls-high AUC/JSD -- not implemented yet")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ---------------------------------------------------------------- load
    _log(f"loading gen={args.gen}")
    gen_E, gen_C = load_showers(args.gen, args.nevts)
    _log(f"loading ref={args.ref}")
    ref_E, ref_C = load_showers(args.ref, args.nevts)
    n = min(gen_E.shape[0], ref_E.shape[0])
    if gen_E.shape[0] != ref_E.shape[0]:
        _log(f"gen has {gen_E.shape[0]} events, ref has {ref_E.shape[0]}; using {n} each")
    gen_E, gen_C = gen_E[:n], gen_C[:n]
    ref_E, ref_C = ref_E[:n], ref_C[:n]

    gen_c = to_crystals(gen_E, args.mode)               # (n,11,21,21)
    ref_c = to_crystals(ref_E, args.mode)
    _log(f"ref {tuple(ref_c.shape)}, gen {tuple(gen_c.shape)} @ crystal resolution; "
         f"E_inc ref={ref_C.mean():.1f} MeV, gen={gen_C.mean():.1f} MeV")

    if args.energy_bins != "none":
        _log(f"WARNING: --energy-bins={args.energy_bins} ignored; only global evaluation "
             "is implemented (single-energy dataset).")

    # --------------------- one accumulation pass over all observables ------
    _log("computing shower-shape observables (ShowerMetrics, crystal resolution)")
    metrics = ShowerMetrics(downsample_mode=args.mode, hit_threshold=args.hit_threshold)
    for i in range(0, n, args.batch_size):
        sl = slice(i, i + args.batch_size)
        metrics.update(ref_c[sl], gen_c[sl], ref_C[sl], pred_E_inc=gen_C[sl])
    t, p, tprof, pprof, tcell, pcell = metrics.results()

    summary = metrics.scalar_summary(t, p, tag="eval", paired=False)
    seps = {}

    # ----------------------------------------------------- required plots
    _log("metric 1/6: total deposited energy E_tot (+ low-deposit subset)")
    ref_low = low_deposit_mask(t["E_tot"], ref_C, args.lowdep_frac).numpy()
    gen_low = low_deposit_mask(p["E_tot"], gen_C, args.lowdep_frac).numpy()
    sep_etot, sep_lowdep = plots.etot_with_lowdep(
        t["E_tot"].numpy(), p["E_tot"].numpy(), ref_low, gen_low, args.lowdep_frac,
        os.path.join(args.out, "etot.png"))
    seps["E_tot"] = sep_etot
    seps[f"E_tot_lowdep(<{args.lowdep_frac:g}Einc)"] = sep_lowdep
    summary["eval/lowdep_frac_truth"] = float(ref_low.mean())
    summary["eval/lowdep_frac_pred"] = float(gen_low.mean())

    _log("metric 2/6: longitudinal profile (mean E per z layer)")
    seps["longitudinal_shape"] = plots.longitudinal_profile(
        tprof["E_layer"].numpy(), pprof["E_layer"].numpy(),
        os.path.join(args.out, "longitudinal_profile.png"))

    _log("metric 3/6: transverse centroids (x, y)")
    seps["x_centroid"] = plots.hist_compare(
        t["x_cog"].numpy(), p["x_cog"].numpy(), "x centroid [crystal]",
        os.path.join(args.out, "centroid_x.png"))
    seps["y_centroid"] = plots.hist_compare(
        t["y_cog"].numpy(), p["y_cog"].numpy(), "y centroid [crystal]",
        os.path.join(args.out, "centroid_y.png"))

    _log("metric 4/6: transverse widths / RMS (x, y)")
    seps["x_width"] = plots.hist_compare(
        t["x_sigma"].numpy(), p["x_sigma"].numpy(), "x RMS [crystal]",
        os.path.join(args.out, "width_x.png"))
    seps["y_width"] = plots.hist_compare(
        t["y_sigma"].numpy(), p["y_sigma"].numpy(), "y RMS [crystal]",
        os.path.join(args.out, "width_y.png"))

    # --------------------- per-layer versions of E_tot / centroid / width ----
    # In CaloChannelge these shape observables are per-layer (one histogram per
    # layer); we reproduce that for the 11 z layers (E per layer + transverse
    # centroid/width restricted to each layer).
    _log(f"per-layer metrics: E_tot / centroid x,y / width x,y over {NZ} z layers")
    ref_pl = per_layer_observables(ref_c)
    gen_pl = per_layer_observables(gen_c)
    pl_dir = os.path.join(args.out, "per_layer")
    os.makedirs(pl_dir, exist_ok=True)
    PL_SPECS = [
        ("E_layer", "E_tot",   r"$E_{\mathrm{tot}}$ in layer [MeV]", True),
        ("x_cog",   "xcenter", "x centroid in layer [crystal]",      False),
        ("y_cog",   "ycenter", "y centroid in layer [crystal]",      False),
        ("x_sigma", "xwidth",  "x RMS in layer [crystal]",           False),
        ("y_sigma", "ywidth",  "y RMS in layer [crystal]",           False),
    ]
    layer_seps = {}
    for key, fname, xlabel, logx in PL_SPECS:
        for z in range(NZ):
            r = ref_pl[key][:, z]
            g = gen_pl[key][:, z]
            sep = plots.hist_compare(
                r, g, xlabel, os.path.join(pl_dir, f"{fname}_layer{z:02d}.png"),
                title=f"{xlabel.split(' in layer')[0]} — layer {z}", logx=logx)
            layer_seps[f"{fname}/layer{z:02d}"] = sep
        _log(f"  {fname}: {NZ} layers done")

    _log("metric 6/6: pixel-level energy spectrum (E_i / E_tot, log-log)")
    ref_pix = pixel_norm_energies(ref_c)
    gen_pix = pixel_norm_energies(gen_c)
    seps["pixel_spectrum"] = plots.pixel_spectrum(
        ref_pix, gen_pix, os.path.join(args.out, "pixel_spectrum.png"))

    # ------------------------------------------------------------ extras
    _log("extras: R, hit multiplicity, occupancy")
    seps["R"] = plots.hist_compare(
        t["R"].numpy(), p["R"].numpy(), r"R = $E_{tot}/E_{inc}$",
        os.path.join(args.out, "R.png"))
    seps["hits"] = plots.hist_compare(
        t["hits"].numpy(), p["hits"].numpy(), "hit multiplicity",
        os.path.join(args.out, "hits.png"))
    seps["occupancy"] = plots.hist_compare(
        t["sparsity"].numpy(), p["sparsity"].numpy(), "occupancy (hits / crystals)",
        os.path.join(args.out, "occupancy.png"))

    _log("extras: average-shower heatmaps (12-panel summary + per-layer) + overview")
    ref_layers, ref_sum = mean_energy_images(ref_c)
    gen_layers, gen_sum = mean_energy_images(gen_c)
    vmax = [max(ref_layers[z].max(), gen_layers[z].max(), 1e-12) for z in range(NZ)]
    vmax.append(max(ref_sum.max(), gen_sum.max(), 1e-12))
    plots.average_shower_grid(ref_layers, ref_sum, vmax,
                              f"Geant4 ref - deposited energy - N={n}",
                              os.path.join(args.out, "avg_shower_ref.png"))
    plots.average_shower_grid(gen_layers, gen_sum, vmax,
                              f"Calo-VQ gen - deposited energy - N={n}",
                              os.path.join(args.out, "avg_shower_gen.png"))
    # each layer + the layer sum as its own image (e.g. to inspect only the front layers)
    sh_dir = os.path.join(args.out, "avg_shower_layers")
    os.makedirs(sh_dir, exist_ok=True)
    plots.average_shower_layers(ref_layers, ref_sum, vmax, "Geant4 ref", sh_dir, "ref")
    plots.average_shower_layers(gen_layers, gen_sum, vmax, "Calo-VQ gen", sh_dir, "gen")
    _log(f"  wrote per-layer heatmaps -> {sh_dir}/ (ref/gen_layer00..{NZ-1:02d}.png + _sum.png)")

    overview = metrics.build_overview_figure("eval", None, t, p, tprof, pprof, tcell, pcell)
    overview.savefig(os.path.join(args.out, "overview.png"), dpi=150)

    # ----------------------------------------------------- classifier stub
    if args.classifier:
        _log("WARNING: --classifier (cls-low/cls-high AUC/JSD) is not implemented; skipping.")

    # ------------------------------------------------------------- write
    def _clean(d):  # NaN (e.g. empty subset) -> null so summary.json stays valid JSON
        return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in d.items()}

    summary["eval/separation_power"] = _clean(seps)
    summary["eval/separation_power_per_layer"] = _clean(layer_seps)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, "separation.txt"), "w") as f:
        f.write("Separation power (triangular discrimination chi2, eq.15 of 2009.03796)\n")
        f.write("0 = identical gen/ref distributions; larger = more separable.\n\n")
        f.write("# global observables\n")
        for k, v in seps.items():
            f.write(f"{k:36s} {v:.5g}\n")
        f.write("\n# per-layer observables\n")
        for k, v in layer_seps.items():
            f.write(f"{k:36s} {v:.5g}\n")

    print("\n--- separation power (gen vs ref) ---")
    for k, v in seps.items():
        print(f"  {k:32s} {v:.5g}")
    print("\n--- scalar summary (means + ratios) ---")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4g}")

    _log(f"wrote PNGs + summary.json + separation.txt -> {args.out}/")


if __name__ == "__main__":
    main()
