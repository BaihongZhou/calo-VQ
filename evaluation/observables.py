"""Per-shower observables and quantitative scores for DarkSHINE evaluation.

Shower-shape observables (E_tot, R, centroids, widths, longitudinal profile,
hits, occupancy, cell spectrum, ...) are computed by the already-validated
`calo_ldm.metrics.compute_observables`, so the evaluation and the training-time
validation metrics share one code path (same definitions, same units). The
crystal grid is fully real (no padding at 21x21), so the mask passed in is
all-ones.

This module adds the pieces the offline evaluation needs on top:
  * `separation_power` -- the CaloChallenge triangular-discrimination chi2
    (eq. 15 of 2009.03796), a quantitative gen-vs-G4 distance per histogram.
  * `low_deposit_mask` -- the DarkSHINE-specific E_tot < frac*E_inc selection
    (low-deposition / partially-contained events).
  * `pixel_norm_energies` -- pooled per-cell E_i / E_tot for the pixel spectrum.
  * `make_bins` -- shared, robust bin edges so ref and gen are histogrammed
    identically (required for a meaningful separation power).
"""
import numpy as np
import torch

from calo_ldm.metrics import compute_observables  # noqa: F401  (re-exported)
from calo_ldm.geometry import NZ, NY, NX

_EPS = 1e-12


def crystal_mask(device=None):
    """All 11x21x21 crystals are real at detector resolution -> all-ones mask."""
    return torch.ones((NZ, NY, NX), dtype=torch.bool, device=device)


def separation_power(counts_ref, counts_gen, bins):
    """Triangular discrimination (CaloChallenge `_separation_power`, eq.15 2009.03796).

    counts_ref/counts_gen are density histogram heights (from `density=True`);
    they are converted back to probabilities by multiplying with the bin widths.
    Returns a scalar in [0, 1]; 0 = identical distributions.
    """
    cr = np.asarray(counts_ref, dtype=np.float64) * np.diff(bins)
    cg = np.asarray(counts_gen, dtype=np.float64) * np.diff(bins)
    num = (cr - cg) ** 2
    den = cr + cg + 1e-16
    return float(0.5 * (num / den).sum())


def low_deposit_mask(E_tot, E_inc, frac=0.5):
    """Boolean selection of partially-contained events: E_tot < frac * E_inc.

    E_tot, E_inc: (N,) or (N,1) tensors in matched units (MeV).
    """
    et = E_tot.reshape(-1)
    ei = E_inc.reshape(-1)
    return (et < frac * ei)


@torch.no_grad()
def pixel_norm_energies(E_crystal, hit_threshold=0.0, max_cells=2_000_000):
    """Pooled per-cell normalized energy E_i / E_tot over all showers.

    E_crystal: (N, 11, 21, 21) detector-resolution energy. Only cells with
    E_i > hit_threshold contribute (zeros excluded -- the spectrum is shown on a
    log axis). Capped at `max_cells` entries to bound memory.

    Returns a 1D numpy array of E_i / E_tot values.
    """
    N = E_crystal.shape[0]
    flat = E_crystal.reshape(N, -1)
    E_tot = flat.sum(dim=1, keepdim=True).clamp(min=_EPS)       # (N,1)
    norm = flat / E_tot
    sel = flat > hit_threshold
    vals = norm[sel].detach().cpu().numpy()
    if vals.size > max_cells:
        vals = vals[:max_cells]
    return vals


@torch.no_grad()
def per_layer_observables(E_crystal, eps=1e-9):
    """Per-layer shower-shape observables at crystal resolution.

    Mirrors the CaloChallenge per-layer features (`HighLevelFeatures`:
    `GetElayers`/`GetECEtas`/`GetECPhis`/`GetWidthEtas`/`GetWidthPhis`), which
    are all computed *per layer*, adapted from (r, eta, phi) to (z; x, y).

    E_crystal: (N, NZ, NY, NX) detector-resolution energy [MeV]. Returns a dict
    of (N, NZ) numpy arrays:
      E_layer            energy deposited in each layer            [MeV]
      x_cog, y_cog       transverse centre of energy per layer     [crystal]
      x_sigma, y_sigma   transverse RMS width per layer            [crystal]

    Centre/width are undefined for a layer with no deposited energy, so those
    entries are returned as NaN (E_layer < eps) and dropped before histogramming
    -- this avoids the artificial spike at 0 you get from the (E+eps) denominator.
    """
    E = E_crystal.float()
    _, nz, ny, nx = E.shape
    xs = torch.arange(nx, dtype=E.dtype).view(1, 1, 1, nx)
    ys = torch.arange(ny, dtype=E.dtype).view(1, 1, ny, 1)

    E_layer = E.sum(dim=(2, 3))                              # (N, NZ)
    denom = E_layer.clamp(min=eps)
    xc = (E * xs).sum(dim=(2, 3)) / denom
    yc = (E * ys).sum(dim=(2, 3)) / denom
    xw = ((E * xs * xs).sum(dim=(2, 3)) / denom - xc ** 2).clamp(min=0.0).sqrt()
    yw = ((E * ys * ys).sum(dim=(2, 3)) / denom - yc ** 2).clamp(min=0.0).sqrt()

    valid = E_layer > eps
    nan = torch.full_like(E_layer, float("nan"))
    masked = lambda a: torch.where(valid, a, nan).cpu().numpy()
    return {
        "E_layer": E_layer.cpu().numpy(),
        "x_cog": masked(xc), "y_cog": masked(yc),
        "x_sigma": masked(xw), "y_sigma": masked(yw),
    }


def make_bins(ref, gen, nbins=60, log=False, lo_pct=0.5, hi_pct=99.5):
    """Shared bin edges from the combined ref+gen range (robust percentiles).

    log=True returns geometric bins (positive values only). Using one binning
    for both samples is required for the separation power to be well-defined.
    """
    r = np.asarray(ref, dtype=np.float64).ravel()
    g = np.asarray(gen, dtype=np.float64).ravel()
    both = np.concatenate([r[np.isfinite(r)], g[np.isfinite(g)]])
    if both.size == 0:
        return np.linspace(0.0, 1.0, nbins + 1)
    if log:
        both = both[both > 0]
        if both.size == 0:
            return np.logspace(-3, 0, nbins + 1)
        lo, hi = np.percentile(both, [lo_pct, hi_pct])
        lo = max(lo, both.min())
        if not (hi > lo):
            hi = lo * 10 if lo > 0 else 1.0
        return np.logspace(np.log10(lo), np.log10(hi), nbins + 1)
    lo, hi = np.percentile(both, [lo_pct, hi_pct])
    if not (hi > lo):
        hi = lo + 1.0
    return np.linspace(lo, hi, nbins + 1)
