"""DarkSHINE calorimeter-shower validation metrics.

Replaces the removed ATLAS HLF metric system (calMstrics/getMetrics/ImageLogger,
all cylindrical-specific) with an xyz-geometry version. Compares predicted showers
(stage-1 reconstruction or stage-2 generation) against truth and emits:

  * scalar summaries  -> pl_module.log_dict (cheap, logged every val epoch)
  * distributions     -> wandb.Histogram (pre-binned, compact)
  * one combined PNG  -> truth-vs-pred overlay of every observable in a single
                         figure (one image per metric epoch per tag)

Both wandb and non-wandb (CSV) loggers are supported: histograms/figures go to
wandb when available, otherwise the combined figure is written to disk under the
logger's log dir. Scalars always go through pl_module.log so they land in
whatever logger is active.

Shower tensors are channels-first (N, depth=11, x=43, y=43); energy lives only on
the real crystals selected by the fixed geometry mask (depth, x, y).
"""

import os

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from calo_ldm.geometry import downsample_to_crystals, NZ, NY, NX

try:
    import wandb
    _HAS_WANDB = True
except Exception:  # pragma: no cover - wandb optional
    _HAS_WANDB = False

_EPS = 1e-12

# Radial transverse profile: integer-cell bins from the per-shower (x,y) centroid.
_RADIAL_BINS = 22

# Per-shower scalar observables -> 1D distribution + scalar summary.
# label: human-readable axis label for the combined figure.
SCALAR_OBSERVABLES = {
    "E_tot":      "E_tot [MeV]",
    "R":          "R = E_tot/E_inc",
    "E_max_frac": "E_max / E_tot",
    "z_cog":      "long. centroid <z> [layer]",
    "z_sigma":    "long. width sigma_z [layer]",
    "x_cog":      "lat. centroid <x> [crystal]",
    "y_cog":      "lat. centroid <y> [crystal]",
    "x_sigma":    "lat. width sigma_x [crystal]",
    "y_sigma":    "lat. width sigma_y [crystal]",
    "hits":       "hit multiplicity",
    "sparsity":   "occupancy (hits / crystals)",
}

# Profile observables -> mean curve (truth vs pred) on the combined figure.
PROFILE_OBSERVABLES = {
    "E_layer":   "longitudinal profile (E per layer) [MeV]",
    "layer_frac": "per-layer energy fraction",
    "radial_E":  "transverse profile (E vs radius) [MeV]",
}

# Scalars logged to the active logger every val epoch (kept small on purpose).
_SUMMARY_KEYS = ("E_tot", "R", "hits", "z_cog", "z_sigma")


@torch.no_grad()
def compute_observables(E, E_inc, mask, hit_threshold=0.1):
    """Per-shower observables for a batch of showers.

    Args:
      E:    (N, D, H, W) energy, channels-first (depth, x, y).
      E_inc:(N, 1) or (N,) incident energy [MeV].
      mask: (D, H, W) bool geometry mask (real crystals).
      hit_threshold: energy [MeV] above which a cell counts as a hit.

    Returns a dict of per-shower tensors plus the pooled hit-cell energies
    ('cell_E', 1D) for the energy spectrum.
    """
    mask_b = mask.to(dtype=torch.bool, device=E.device)
    m = mask_b[None].to(E.dtype)                          # (1, D, H, W)
    Em = E * m
    N, D, H, W = Em.shape

    z = torch.arange(D, device=E.device, dtype=E.dtype)
    x = torch.arange(H, device=E.device, dtype=E.dtype)
    y = torch.arange(W, device=E.device, dtype=E.dtype)

    E_tot = Em.sum(dim=(1, 2, 3))                         # (N,)
    denom = E_tot.clamp(min=_EPS)
    E_inc_flat = E_inc.reshape(N).to(E.dtype)
    R = E_tot / E_inc_flat.clamp(min=_EPS)

    # longitudinal: energy per depth layer, centroid, width
    E_layer = Em.sum(dim=(2, 3))                          # (N, D)
    z_cog = (E_layer * z[None]).sum(1) / denom
    z_sigma = ((E_layer * (z[None] - z_cog[:, None]) ** 2).sum(1) / denom).clamp(min=0).sqrt()
    layer_frac = E_layer / denom[:, None]

    # transverse: collapse depth, then per-axis marginals
    E_xy = Em.sum(dim=1)                                  # (N, H, W)
    Ex = E_xy.sum(dim=2)                                  # (N, H)
    Ey = E_xy.sum(dim=1)                                  # (N, W)
    x_cog = (Ex * x[None]).sum(1) / denom
    y_cog = (Ey * y[None]).sum(1) / denom
    x_sigma = ((Ex * (x[None] - x_cog[:, None]) ** 2).sum(1) / denom).clamp(min=0).sqrt()
    y_sigma = ((Ey * (y[None] - y_cog[:, None]) ** 2).sum(1) / denom).clamp(min=0).sqrt()

    # radial transverse profile about each shower's (x_cog, y_cog)
    dx = x[None, :, None] - x_cog[:, None, None]          # (N, H, 1)
    dy = y[None, None, :] - y_cog[:, None, None]          # (N, 1, W)
    r = torch.sqrt(dx ** 2 + dy ** 2)                    # (N, H, W)
    r = torch.nan_to_num(r, nan=0.0, posinf=float(_RADIAL_BINS - 1), neginf=0.0)
    rbin = r.long().clamp(min=0, max=_RADIAL_BINS - 1)
    radial_E = torch.zeros(N, _RADIAL_BINS, device=E.device, dtype=E.dtype)
    radial_E.scatter_add_(1, rbin.reshape(N, -1), E_xy.reshape(N, -1))

    # peak cell
    E_max = Em.amax(dim=(1, 2, 3))
    E_max_frac = E_max / denom

    # topology
    hit_mask = (Em > hit_threshold) & mask_b[None]
    hits = hit_mask.sum(dim=(1, 2, 3)).to(E.dtype)
    n_real = mask_b.sum().to(E.dtype).clamp(min=1)
    sparsity = hits / n_real

    cell_E = Em[hit_mask]                                 # 1D pooled hit energies

    return {
        "E_tot": E_tot, "R": R, "E_max_frac": E_max_frac,
        "z_cog": z_cog, "z_sigma": z_sigma,
        "x_cog": x_cog, "y_cog": y_cog, "x_sigma": x_sigma, "y_sigma": y_sigma,
        "hits": hits, "sparsity": sparsity,
        "E_layer": E_layer, "layer_frac": layer_frac, "radial_E": radial_E,
        "cell_E": cell_E,
    }


class ShowerMetrics:
    """Accumulate truth/pred observables over a validation epoch, then emit.

    Usage:
        m = ShowerMetrics(downsample_mode="sum", hit_threshold=0.1)
        for batch: m.update(truth_E, pred_E, E_inc)   # E ~ (N,11,43,43) or (N,11,21,21)
        m.compute_and_log(pl_module, tag="reco", epoch=e, paired=True)

    All shower-shape observables are computed at REAL DETECTOR (crystal)
    resolution: inputs on the 43x43 half-cell grid are first downsampled to
    21x21 crystals via `downsample_to_crystals` (mode `sum`/`min`); inputs already
    at 21x21 are used as-is. Computing shape on the half-cell grid would be
    physically meaningless (each crystal is split across a 2x2 cell block).

    `paired=True` (stage-1 reconstruction: pred is the reco of the same shower)
    additionally logs a per-shower relative error; `paired=False` (stage-2
    generation: independent samples) compares distributions only.

    To bound memory the pooled cell-energy spectrum is capped at `max_cells`.
    """

    def __init__(self, downsample_mode="sum", hit_threshold=0.1, max_cells=200_000):
        self.downsample_mode = downsample_mode
        self.hit_threshold = hit_threshold
        self.max_cells = max_cells
        # crystal-resolution mask: every one of the 11x21x21 crystals is real.
        self.mask = torch.ones((NZ, NY, NX), dtype=torch.bool)
        self.reset()

    def _to_crystals(self, E):
        # accept both the 43x43 half-cell grid and the 21x21 crystal grid
        if E.shape[-1] == NX:          # already crystal resolution
            return E
        return downsample_to_crystals(E, self.downsample_mode)

    def reset(self):
        self._t = {k: [] for k in SCALAR_OBSERVABLES}
        self._p = {k: [] for k in SCALAR_OBSERVABLES}
        self._tprof = {k: None for k in PROFILE_OBSERVABLES}
        self._pprof = {k: None for k in PROFILE_OBSERVABLES}
        self._tcell, self._pcell = [], []
        self._tcell_n = self._pcell_n = 0
        self._n = 0

    @property
    def n(self):
        return self._n

    @torch.no_grad()
    def update(self, truth_E, pred_E, E_inc, pred_E_inc=None):
        # pred_E_inc lets eval compare independently-conditioned gen vs truth
        # (each set's R uses its own incident energy); defaults to E_inc.
        truth_c = self._to_crystals(truth_E)
        pred_c = self._to_crystals(pred_E)
        mask = self.mask.to(truth_c.device)
        ot = compute_observables(truth_c, E_inc, mask, self.hit_threshold)
        op = compute_observables(pred_c, E_inc if pred_E_inc is None else pred_E_inc,
                                 mask, self.hit_threshold)
        self._n += truth_c.shape[0]

        for k in SCALAR_OBSERVABLES:
            self._t[k].append(ot[k].detach().cpu())
            self._p[k].append(op[k].detach().cpu())

        for k in PROFILE_OBSERVABLES:
            t_sum = ot[k].sum(0).detach().cpu()
            p_sum = op[k].sum(0).detach().cpu()
            self._tprof[k] = t_sum if self._tprof[k] is None else self._tprof[k] + t_sum
            self._pprof[k] = p_sum if self._pprof[k] is None else self._pprof[k] + p_sum

        self._collect_cells(self._tcell, ot["cell_E"], which="t")
        self._collect_cells(self._pcell, op["cell_E"], which="p")

    def _collect_cells(self, store, cells, which):
        n_attr = "_tcell_n" if which == "t" else "_pcell_n"
        have = getattr(self, n_attr)
        room = self.max_cells - have
        if room <= 0:
            return
        c = cells.detach().cpu()
        if c.numel() > room:
            c = c[:room]
        store.append(c)
        setattr(self, n_attr, have + c.numel())

    # --------------------------------------------------------------- results
    def results(self):
        """Concatenate accumulated batches into truth/pred distributions + profiles."""
        t = {k: torch.cat(self._t[k]) for k in SCALAR_OBSERVABLES}
        p = {k: torch.cat(self._p[k]) for k in SCALAR_OBSERVABLES}
        tprof = {k: self._tprof[k] / self._n for k in PROFILE_OBSERVABLES}
        pprof = {k: self._pprof[k] / self._n for k in PROFILE_OBSERVABLES}
        tcell = torch.cat(self._tcell) if self._tcell else torch.zeros(0)
        pcell = torch.cat(self._pcell) if self._pcell else torch.zeros(0)
        return t, p, tprof, pprof, tcell, pcell

    def scalar_summary(self, t, p, tag, paired):
        logs = {}
        for k in _SUMMARY_KEYS:
            tm = t[k].mean().item()
            pm = p[k].mean().item()
            logs[f"{tag}/{k}_truth"] = tm
            logs[f"{tag}/{k}_pred"] = pm
            logs[f"{tag}/{k}_ratio"] = pm / tm if abs(tm) > _EPS else 0.0
            if paired:
                rel = ((p[k] - t[k]).abs() / t[k].abs().clamp(min=_EPS)).mean().item()
                logs[f"{tag}/{k}_relerr"] = rel
        return logs

    # ------------------------------------------------------------------ emit
    @torch.no_grad()
    def compute_and_log(self, pl_module, tag, epoch, paired=False):
        if self._n == 0:
            return
        t, p, tprof, pprof, tcell, pcell = self.results()
        pl_module.log_dict(self.scalar_summary(t, p, tag, paired),
                           on_step=False, on_epoch=True, sync_dist=True)
        self._log_wandb_histograms(pl_module, tag, t, p, tcell, pcell)
        fig = self.build_overview_figure(tag, epoch, t, p, tprof, pprof, tcell, pcell)
        run = self._wandb_run(pl_module)
        if run is not None:
            run.log({f"{tag}/overview": wandb.Image(fig)}, commit=False)
        else:
            self._save_figure(pl_module, fig, tag, epoch)
        plt.close(fig)

    def _wandb_run(self, pl_module):
        if not _HAS_WANDB:
            return None
        logger = getattr(pl_module, "logger", None)
        exp = getattr(logger, "experiment", None)
        # WandbLogger.experiment is the wandb Run; guard against other loggers.
        if exp is not None and exp.__class__.__module__.startswith("wandb"):
            return exp
        return None

    @staticmethod
    def _safe_hist(values, bins=64):
        """wandb.Histogram that tolerates empty/constant/degenerate ranges.

        wandb's default auto-binning raises when the data range is zero (e.g. a
        near-constant observable in a smoke run), so we pre-bin with an explicit,
        always-finite range and pass it via np_histogram.
        """
        v = np.asarray(values, dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return None
        lo, hi = float(v.min()), float(v.max())
        if not (hi > lo):
            lo, hi = lo - 0.5, hi + 0.5
        hist, edges = np.histogram(v, bins=bins, range=(lo, hi))
        return wandb.Histogram(np_histogram=(hist.tolist(), edges.tolist()))

    def _log_wandb_histograms(self, pl_module, tag, t, p, tcell, pcell):
        run = self._wandb_run(pl_module)
        if run is None:
            return
        payload = {}
        for k in SCALAR_OBSERVABLES:
            for name, arr in ((f"{k}_truth", t[k].numpy()), (f"{k}_pred", p[k].numpy())):
                h = self._safe_hist(arr)
                if h is not None:
                    payload[f"{tag}/hist/{name}"] = h
        for name, cells in (("cell_E_truth", tcell), ("cell_E_pred", pcell)):
            if cells.numel():
                h = self._safe_hist(np.log10(cells.numpy() + _EPS))
                if h is not None:
                    payload[f"{tag}/hist/{name}"] = h
        if payload:
            run.log(payload, commit=False)

    def build_overview_figure(self, tag, epoch, t, p, tprof, pprof, tcell, pcell):
        """Truth-vs-pred overview: every scalar observable + cell spectrum + profiles."""
        panels = list(SCALAR_OBSERVABLES) + ["cell_E"] + list(PROFILE_OBSERVABLES)
        ncol = 4
        nrow = (len(panels) + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
        axes = np.atleast_1d(axes).ravel()

        for ax, key in zip(axes, panels):
            if key in SCALAR_OBSERVABLES:
                self._panel_hist(ax, t[key].numpy(), p[key].numpy(), SCALAR_OBSERVABLES[key])
            elif key == "cell_E":
                self._panel_cell_spectrum(ax, tcell.numpy(), pcell.numpy())
            else:
                self._panel_profile(ax, tprof[key].numpy(), pprof[key].numpy(),
                                    PROFILE_OBSERVABLES[key], key)
        for ax in axes[len(panels):]:
            ax.axis("off")

        axes[0].legend(["truth", "pred"], loc="upper right", fontsize=8)
        ttl = f"{tag} — epoch {epoch} — N={self._n}" if epoch is not None else f"{tag} — N={self._n}"
        fig.suptitle(ttl, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        return fig

    @staticmethod
    def _panel_hist(ax, tv, pv, label):
        finite = np.concatenate([tv[np.isfinite(tv)], pv[np.isfinite(pv)]])
        if finite.size == 0:
            ax.axis("off"); return
        lo, hi = np.percentile(finite, [0.5, 99.5])
        if hi <= lo:
            hi = lo + 1.0
        bins = np.linspace(lo, hi, 41)
        ax.hist(tv, bins=bins, histtype="step", color="k", density=True)
        ax.hist(pv, bins=bins, histtype="step", color="tab:red", density=True)
        ax.set_xlabel(label, fontsize=8)
        ax.tick_params(labelsize=7)

    @staticmethod
    def _panel_cell_spectrum(ax, tcell, pcell):
        if tcell.size == 0 and pcell.size == 0:
            ax.axis("off"); return
        lt = np.log10(tcell + _EPS) if tcell.size else np.zeros(0)
        lp = np.log10(pcell + _EPS) if pcell.size else np.zeros(0)
        finite = np.concatenate([lt, lp])
        lo, hi = np.percentile(finite, [0.5, 99.5])
        if hi <= lo:
            hi = lo + 1.0
        bins = np.linspace(lo, hi, 41)
        if lt.size:
            ax.hist(lt, bins=bins, histtype="step", color="k", density=True)
        if lp.size:
            ax.hist(lp, bins=bins, histtype="step", color="tab:red", density=True)
        ax.set_xlabel("log10 cell energy [MeV]", fontsize=8)
        ax.set_yscale("log")
        ax.tick_params(labelsize=7)

    @staticmethod
    def _panel_profile(ax, tprof, pprof, label, key):
        xs = np.arange(len(tprof))
        ax.plot(xs, tprof, color="k", marker="o", ms=3)
        ax.plot(xs, pprof, color="tab:red", marker="o", ms=3)
        ax.set_xlabel(label, fontsize=8)
        if key == "E_layer":
            ax.set_yscale("log")
        ax.tick_params(labelsize=7)

    @staticmethod
    def _save_figure(pl_module, fig, tag, epoch):
        logger = getattr(pl_module, "logger", None)
        base = getattr(logger, "log_dir", None) or getattr(logger, "save_dir", None) or "."
        out_dir = os.path.join(base, "metrics", tag)
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, f"epoch_{epoch:04d}.png"), dpi=90)
