"""Plotting layer for DarkSHINE offline evaluation.

Clean, self-consistent house style (not the CaloChallenge style): despined axes,
a light reference fill + a crisp generated line, and a gen/ref **ratio sub-panel**
under each 1D comparison so agreement is read off directly. Every 1D comparison
still returns its separation power (triangular-discrimination chi2) so the driver
can tabulate quantitative scores.

All figures are saved at >= 150 dpi. Energy is in MeV throughout; shower-shape
observables are at real detector (21x21x11 crystal) resolution.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from calo_ldm.geometry import NZ
from evaluation.observables import separation_power, make_bins

# --- house style -----------------------------------------------------------
_REF_LABEL = "Geant4 (ref)"
_GEN_LABEL = "Calo-VQ (gen)"
_REF = "#3B6FB6"      # muted blue  -> reference
_GEN = "#D1495B"      # muted red   -> generated
_DPI = 200

plt.rcParams.update({
    "figure.dpi": 110,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "legend.frameon": False,
    "legend.fontsize": 10,
})


def _hist_densities(ref, gen, bins):
    r, _ = np.histogram(np.asarray(ref).ravel(), bins=bins, density=True)
    g, _ = np.histogram(np.asarray(gen).ravel(), bins=bins, density=True)
    return r, g


def _draw_overlay(ax, ref_d, gen_d, bins):
    """Filled reference + line generated on a main axis (uses pre-binned densities)."""
    ax.stairs(ref_d, bins, fill=True, alpha=0.18, color=_REF, label=_REF_LABEL)
    ax.stairs(ref_d, bins, color=_REF, lw=1.6)
    ax.stairs(gen_d, bins, color=_GEN, lw=2.0, label=_GEN_LABEL)


def _sep_legend(ax, sep, loc="best"):
    """Legend with an extra invisible entry reporting the separation power."""
    ax.plot([], [], " ", label=f"sep = {sep:.3g}")
    ax.legend(loc=loc)


def _draw_ratio(ax, ref_d, gen_d, bins):
    """gen/ref ratio sub-panel with a unity reference line."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(ref_d > 0, gen_d / ref_d, np.nan)
    ax.axhline(1.0, color="0.4", lw=0.9, ls="--")
    ax.stairs(ratio, bins, color=_GEN, lw=1.8)
    finite = ratio[np.isfinite(ratio)]
    if finite.size:
        hi = min(np.nanpercentile(finite, 98), 3.0)
        lo = max(np.nanpercentile(finite, 2), 0.0)
        if hi <= lo:
            lo, hi = 0.0, 2.0
        ax.set_ylim(max(0.0, lo - 0.1), hi + 0.1)
    ax.set_ylabel("gen/ref", fontsize=9)
    ax.tick_params(labelsize=8)


def _fig_with_ratio(figsize=(6.2, 5.6)):
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.06, figure=fig)
    ax = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharex=ax)
    ax.tick_params(labelbottom=False)
    return fig, ax, axr


def hist_compare(ref, gen, xlabel, out_path, title=None, nbins=60,
                 logx=False, logy=False):
    """1D gen-vs-ref histogram with ratio panel. Saves PNG, returns sep power."""
    bins = make_bins(ref, gen, nbins=nbins, log=logx)
    ref_d, gen_d = _hist_densities(ref, gen, bins)
    sep = separation_power(ref_d, gen_d, bins)

    fig, ax, axr = _fig_with_ratio()
    _draw_overlay(ax, ref_d, gen_d, bins)
    _draw_ratio(axr, ref_d, gen_d, bins)
    if logx:
        ax.set_xscale("log"); axr.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_ylabel("normalized counts")
    ax.set_title(title or xlabel)
    _sep_legend(ax, sep)
    axr.set_xlabel(xlabel)
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return sep


def etot_with_lowdep(ref_Etot, gen_Etot, ref_lowmask, gen_lowmask, frac,
                     out_path, nbins=60):
    """E_tot distribution (left, with ratio) + the E_tot < frac*E_inc subset (right).

    The subset is the DarkSHINE low-deposition / partially-contained selection;
    the right panel's legend reports each sample's selected fraction.
    Returns (sep_full, sep_lowdep).
    """
    fig = plt.figure(figsize=(13, 5.6))
    gs = GridSpec(2, 2, height_ratios=[3, 1], hspace=0.06, wspace=0.22, figure=fig)
    ax0 = fig.add_subplot(gs[0, 0]); ax0r = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax1 = fig.add_subplot(gs[0, 1]); ax1r = fig.add_subplot(gs[1, 1], sharex=ax1)
    ax0.tick_params(labelbottom=False); ax1.tick_params(labelbottom=False)

    # full
    bins = make_bins(ref_Etot, gen_Etot, nbins=nbins)
    ref_d, gen_d = _hist_densities(ref_Etot, gen_Etot, bins)
    sep_full = separation_power(ref_d, gen_d, bins)
    _draw_overlay(ax0, ref_d, gen_d, bins)
    _draw_ratio(ax0r, ref_d, gen_d, bins)
    ax0.set_ylabel("normalized counts")
    ax0.set_title("total deposited energy")
    _sep_legend(ax0, sep_full)
    ax0r.set_xlabel(r"$E_{\mathrm{tot}}$ [MeV]")

    # low-deposit subset
    r_low = np.asarray(ref_Etot).ravel()[np.asarray(ref_lowmask).ravel()]
    g_low = np.asarray(gen_Etot).ravel()[np.asarray(gen_lowmask).ravel()]
    f_ref = float(np.mean(ref_lowmask)) if np.size(ref_lowmask) else 0.0
    f_gen = float(np.mean(gen_lowmask)) if np.size(gen_lowmask) else 0.0
    if r_low.size and g_low.size:
        bins_l = make_bins(r_low, g_low, nbins=max(20, nbins // 2))
        rl, gl = _hist_densities(r_low, g_low, bins_l)
        sep_low = separation_power(rl, gl, bins_l)
        ax1.stairs(rl, bins_l, fill=True, alpha=0.18, color=_REF,
                   label=f"{_REF_LABEL}: {100*f_ref:.2f}%")
        ax1.stairs(rl, bins_l, color=_REF, lw=1.6)
        ax1.stairs(gl, bins_l, color=_GEN, lw=2.0,
                   label=f"{_GEN_LABEL}: {100*f_gen:.2f}%")
        _draw_ratio(ax1r, rl, gl, bins_l)
        _sep_legend(ax1, sep_low)
    else:
        ax1.text(0.5, 0.5,
                 f"no low-deposit events\n(ref {100*f_ref:.2f}%, gen {100*f_gen:.2f}%)",
                 ha="center", va="center", transform=ax1.transAxes, fontsize=11)
        ax1r.axhline(1.0, color="0.4", lw=0.9, ls="--"); ax1r.set_ylabel("gen/ref", fontsize=9)
        sep_low = float("nan")
    ax1.set_ylabel("normalized counts")
    ax1.set_title(rf"low-deposit: $E_{{\mathrm{{tot}}}} < {frac:g}\,E_{{\mathrm{{inc}}}}$")
    ax1r.set_xlabel(r"$E_{\mathrm{tot}}$ [MeV]")

    fig.suptitle("DarkSHINE ECAL — total deposited energy", fontsize=13, y=0.99)
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return sep_full, sep_low


def longitudinal_profile(ref_E_layer_mean, gen_E_layer_mean, out_path):
    """Mean deposited energy per z layer (0..10) with ratio panel. Returns sep."""
    z = np.arange(NZ)
    r = np.asarray(ref_E_layer_mean).ravel()
    g = np.asarray(gen_E_layer_mean).ravel()
    rp = r / (r.sum() + 1e-16)
    gp = g / (g.sum() + 1e-16)
    sep = float(0.5 * ((rp - gp) ** 2 / (rp + gp + 1e-16)).sum())

    fig, ax, axr = _fig_with_ratio()
    ax.fill_between(z, r, color=_REF, alpha=0.16, step="mid")
    ax.plot(z, r, color=_REF, marker="o", ms=5, lw=1.8, label=_REF_LABEL)
    ax.plot(z, g, color=_GEN, marker="s", ms=5, lw=1.8, ls="--", label=_GEN_LABEL)
    ax.set_ylabel(r"mean $E$ per layer [MeV]")
    ax.set_title("longitudinal profile")
    ax.plot([], [], " ", label=f"shape sep = {sep:.3g}")
    ax.legend(loc="best")

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(r > 0, g / r, np.nan)
    axr.axhline(1.0, color="0.4", lw=0.9, ls="--")
    axr.plot(z, ratio, color=_GEN, marker="s", ms=4, lw=1.5)
    axr.set_ylabel("gen/ref", fontsize=9)
    axr.set_xlabel("z layer index")
    axr.set_xticks(z)
    axr.tick_params(labelsize=8)
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return sep


def pixel_spectrum(ref_vals, gen_vals, out_path, nbins=60):
    """Pooled per-cell E_i / E_tot spectrum (log-log) with ratio panel. Returns sep."""
    bins = make_bins(ref_vals, gen_vals, nbins=nbins, log=True)
    ref_d, gen_d = _hist_densities(ref_vals, gen_vals, bins)
    sep = separation_power(ref_d, gen_d, bins)

    fig, ax, axr = _fig_with_ratio()
    _draw_overlay(ax, ref_d, gen_d, bins)
    _draw_ratio(axr, ref_d, gen_d, bins)
    ax.set_xscale("log"); ax.set_yscale("log")
    axr.set_xscale("log")
    ax.set_ylabel("normalized counts")
    ax.set_title("pixel-level energy spectrum")
    _sep_legend(ax, sep)
    axr.set_xlabel(r"$E_i / E_{\mathrm{tot}}$ (per crystal)")
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return sep


# CaloChallenge uses matplotlib's default colormap (viridis) for shower images.
_SHOWER_CMAP = "viridis"


def average_shower_grid(per_layer_mean, summed_mean, vmax, title, out_path):
    """12-panel mean-energy heatmaps: 11 per-layer (21x21) + the layer sum."""
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    axes = axes.ravel()
    for z in range(NZ):
        im = axes[z].imshow(per_layer_mean[z], origin="lower", cmap=_SHOWER_CMAP,
                            vmin=0, vmax=vmax[z])
        axes[z].set_title(f"layer {z}", fontsize=9)
        axes[z].tick_params(labelsize=6); axes[z].grid(False)
        fig.colorbar(im, ax=axes[z], fraction=0.046, pad=0.04)
    im = axes[11].imshow(summed_mean, origin="lower", cmap=_SHOWER_CMAP,
                         vmin=0, vmax=vmax[11])
    axes[11].set_title("all layers (sum)", fontsize=9)
    axes[11].tick_params(labelsize=6); axes[11].grid(False)
    fig.colorbar(im, ax=axes[11], fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=_DPI)
    plt.close(fig)


def _shower_image(data, vmax, title, out_path, clabel="mean E [MeV]"):
    """Single 21x21 mean-energy heatmap (one layer or the layer sum)."""
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(data, origin="lower", cmap=_SHOWER_CMAP, vmin=0, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("x [crystal]"); ax.set_ylabel("y [crystal]")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=clabel)
    fig.tight_layout()
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


def average_shower_layers(per_layer_mean, summed_mean, vmax, sample, out_dir, prefix):
    """Write each layer (0..10) and the layer-sum as separate PNGs.

    Filenames: <prefix>_layerNN.png (NN=00..10) and <prefix>_sum.png. `sample`
    labels the source (e.g. 'Geant4 ref') in each title. Returns the path list.
    """
    import os
    paths = []
    for z in range(NZ):
        p = os.path.join(out_dir, f"{prefix}_layer{z:02d}.png")
        _shower_image(per_layer_mean[z], vmax[z], f"{sample} — layer {z}", p)
        paths.append(p)
    p = os.path.join(out_dir, f"{prefix}_sum.png")
    _shower_image(summed_mean, vmax[NZ], f"{sample} — all layers (sum)", p)
    paths.append(p)
    return paths
