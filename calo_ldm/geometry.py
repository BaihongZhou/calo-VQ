"""DarkSHINE ECAL geometry: crystal <-> half-cell-grid mapping.

Single source of truth for how the 21x21x11 = 4851 physical LYSO crystals map
onto the 43x43x11 half-cell training grid, taken from the data producer's
`ML_IO.cpp` (`FillECALEventEnergy`, `StaggeredUpsample=1`). Each crystal occupies
a 2x2 (y,x) block, offset by layer parity so adjacent staggered layers stay
center-aligned:

    layer parity (0-based depth pz):
      pz even  -> crystal block spans grid indices [1:43]  (1..42)
      pz odd   -> crystal block spans grid indices [0:42]  (0..41)

This module provides:
  * NZ, NY, NX, H, W            -- geometry constants
  * build_crystal_mask()        -- (43, 43, 11) bool mask of real crystal cells
  * downsample_to_crystals(E, mode) -- (N, 11, 43, 43) -> (N, 11, 21, 21),
        inverting the 2x2 upsample to real detector resolution.

The downsample is ONNX-friendly (static per-layer slices + reshape + reduce).
"""

import torch

NZ, NY, NX = 11, 21, 21          # crystals along depth, y, x
H, W = NY * 2 + 1, NX * 2 + 1    # 43, 43 (half-cell grid + 1 for the parity stagger)


def layer_slice(pz):
    """Half-open [lo, hi) grid range covering the real crystal block of layer pz.

    pz even -> [1, 43); pz odd -> [0, 42). Same for both the y and x axes.
    """
    return (1, 43) if pz % 2 == 0 else (0, 42)


def build_crystal_mask():
    """Fixed geometry mask, channels-LAST (43, 43, 11) bool = (y, x, depth).

    Marks the real crystal cells (True). All 21x21 crystals tile a contiguous
    42x42 block per layer, so the mask has 11*42*42 = 19404 True cells.
    """
    import numpy as np
    mZYX = np.zeros((NZ, H, W), dtype=bool)            # ML_IO.cpp native [Z][Y][X]
    for pz in range(NZ):
        lo, hi = layer_slice(pz)
        mZYX[pz, lo:hi, lo:hi] = True
    return np.transpose(mZYX, (1, 2, 0)).copy()        # -> (43, 43, 11)


def downsample_to_crystals(E, mode="sum"):
    """Map a half-cell-grid shower to real detector (crystal) resolution.

    Args:
      E:    (N, 11, 43, 43) energy, channels-first (depth, y, x).
      mode: 'sum' -> crystal = sum of its 2x2 block (energy-conserving);
            'min' -> crystal = 4 * min of its 2x2 block.
            For truth (4 equal cells = E/4) both recover the crystal energy E;
            for VQ-VAE reconstructions the 4 cells differ, so the two diverge.

    Returns:
      (N, 11, 21, 21) crystal-resolution energy, channels-first (depth, y, x).
    """
    if E.dim() != 4 or E.shape[1] != NZ or E.shape[2] != H or E.shape[3] != W:
        raise ValueError(f"expected (N,{NZ},{H},{W}), got {tuple(E.shape)}")
    N = E.shape[0]
    layers = []
    for pz in range(NZ):
        lo, hi = layer_slice(pz)
        blk = E[:, pz, lo:hi, lo:hi]                   # (N, 42, 42)
        blk = blk.reshape(N, NY, 2, NX, 2)             # (N, 21, 2, 21, 2)
        if mode == "sum":
            cz = blk.sum(dim=(2, 4))
        elif mode == "min":
            cz = 4.0 * blk.amin(dim=(2, 4))
        else:
            raise ValueError(f"unknown downsample mode {mode!r} (use 'sum' or 'min')")
        layers.append(cz)                              # (N, 21, 21)
    return torch.stack(layers, dim=1)                  # (N, 11, 21, 21)
