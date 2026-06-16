"""I/O for DarkSHINE offline evaluation.

Loads showers from an `export.h5`-schema HDF5 (the same schema produced by
`gen-tools.py`): `condition` (N,1) incident energy in **keV**, `energy`
(N,43,43,11) per-cell deposited energy stored channels-last. We convert
`condition` to **MeV** (condition_scale=1e-3) so it matches the per-cell energy
units and R = E_tot/E_inc is physical (~0.97); the same scale is applied to both
gen and reference (both use the keV schema). Mirrors `eval-tools.py:load_showers`
and `CaloDarkSHINE`'s condition handling.

Everything downstream is evaluated at REAL DETECTOR (21x21x11 crystal)
resolution: the 43x43 half-cell training grid is collapsed with
`downsample_to_crystals` (every crystal = its 2x2 cell block). At crystal
resolution there is NO padding -- all 11x21x21 = 4851 crystals are real -- so
sums/means/histograms over the full tensor are already restricted to valid
crystals (no per-event `label` mask needed; `label` is a hit flag, a strict
subset of energy>0, and is absent from generated files anyway).
"""
import h5py
import torch

from calo_ldm.geometry import downsample_to_crystals, NX


def load_showers(path, n=None, key_e="energy", key_c="condition", condition_scale=1e-3):
    """Read an HDF5 file -> (E channels-first, C in MeV).

    Returns:
      E: (N, 11, 43, 43) float tensor, channels-first (depth, x, y).
      C: (N, 1) float tensor, incident energy in MeV.
    """
    with h5py.File(path, "r") as f:
        E = torch.from_numpy(f[key_e][:n]).float()             # (N, 43, 43, 11) channels-last
        C = torch.from_numpy(f[key_c][:n]).float() * condition_scale   # keV -> MeV
    if E.shape[-1] == 11:                                       # channels-last -> channels-first
        E = E.permute(0, 3, 1, 2).contiguous()                 # (N, 11, 43, 43)
    if C.dim() == 1:
        C = C.unsqueeze(-1)
    return E, C


def to_crystals(E, mode="sum"):
    """43x43 half-cell grid -> 21x21 crystal grid; already-21x21 passes through."""
    return E if E.shape[-1] == NX else downsample_to_crystals(E, mode)
