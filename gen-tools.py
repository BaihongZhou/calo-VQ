#!/usr/bin/env python
"""Generate DarkSHINE ECAL showers from a trained stage-2 (CondGPT) model.

The model autoregressively samples a code grid (and the quantised energy ratio R)
conditioned on the incident energy, then decodes it through the frozen stage-1
VQ-VAE. Output is written in the same schema as the training file
`DarkSHINE_data/export.h5`:

    condition  (N, 1)         incident energy [MeV]
    energy     (N, 43, 43, 11) deposited energy per cell, channels-last

so generated files are drop-in comparable with the real data. (The truth-only
`label` hit flag is not produced -- it does not exist at generation time.)

Incident energy source:
  --energy E [--nevts N]   N showers at a constant E_inc = E MeV (default;
                           DarkSHINE is currently a single 4 GeV point).
  --cond-file f.h5         draw E_inc by sampling the `condition` column of an
                           existing HDF5 file (use for multi-energy datasets).
"""
import os
import sys
import time
import argparse
from glob import glob

import torch
import h5py
import numpy as np
from omegaconf import OmegaConf

from calo_ldm.util import instantiate_from_config

torch.set_grad_enabled(False)


def build_cond_batch(model, e_inc):
    """e_inc: (N,) or (N,1) tensor [MeV] -> batch dict ready for sample_fullchain."""
    e_inc = e_inc.reshape(-1, 1).float().to(model.device)
    batch = {"E_inc": e_inc}
    batch = model.vq_model.preprocess_cond(batch)   # -> log_E_inc, cond
    batch = model.preprocess_cond(batch)            # -> gpt_cond
    return batch


def sample_incident_energies(args, n):
    if args.cond_file:
        with h5py.File(args.cond_file, "r") as f:
            pool = torch.from_numpy(f[args.cond_key][:]).float().reshape(-1)
        idx = torch.randint(0, pool.shape[0], (n,))
        return pool[idx]
    return torch.full((n,), float(args.energy))


def generate(model, args):
    showers, incident = [], []
    n_done = 0
    while n_done < args.nevts:
        bs = min(args.batch_size, args.nevts - n_done)
        e_inc = sample_incident_energies(args, bs)
        batch = build_cond_batch(model, e_inc)
        gen = model.sample_fullchain(batch)                 # 'pixels_E_pred' (bs,11,H,H)
        # channels-first (depth,y,x) -> channels-last (y,x,depth) to match export.h5.
        # H is 43 (half-cell grid) or 21 (detector shape) per --detector-shape.
        E = gen["pixels_E_pred"].permute(0, 2, 3, 1).cpu()
        showers.append(E)
        incident.append(batch["E_inc"].cpu())
        n_done += bs
        print(f"  generated {n_done}/{args.nevts}")
    return {
        "condition": torch.cat(incident, 0),                # (N,1)
        "energy": torch.cat(showers, 0),                    # (N,43,43,11) or (N,21,21,11)
    }


def sanity_check(data):
    E = data["energy"]
    C = data["condition"]
    Etot = E.flatten(1).sum(1)
    print(f"--> events {E.shape[0]}, energy shape {tuple(E.shape)}, condition shape {tuple(C.shape)}")
    print(f"--> E_inc [{C.min():.1f}, {C.max():.1f}] MeV")
    print(f"--> E_tot [{Etot.min():.1f}, {Etot.max():.1f}] MeV, R=E_tot/E_inc "
          f"[{(Etot / C.reshape(-1)).min():.2e}, {(Etot / C.reshape(-1)).max():.2e}]")
    print(f"--> E_min {E.min():.3e}, E_max {E.max():.3e} (energies non-negative: {bool(E.min() >= 0)})")


def load_model(model_dir, checkpoint, device):
    config_files = sorted(glob(os.path.join(model_dir, "configs", "*.yaml")))
    print("Loading config files:", config_files)
    config = OmegaConf.merge(*[OmegaConf.load(c) for c in config_files])
    model = instantiate_from_config(config["model"])

    ckpt_path = (os.path.join(model_dir, "checkpoints", "last.ckpt")
                 if checkpoint == "auto"
                 else os.path.join(model_dir, "checkpoints", checkpoint))
    print("Loading checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device).eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="output HDF5 file")
    parser.add_argument("--model", required=True, help="stage-2 model log directory (has configs/ and checkpoints/)")
    parser.add_argument("--checkpoint", default="auto", help="checkpoint name under checkpoints/ (default: last.ckpt)")
    parser.add_argument("--energy", type=float, default=4000.0, help="constant incident energy [MeV] when --cond-file is not given")
    parser.add_argument("--cond-file", default=None, help="HDF5 file to sample incident energies from")
    parser.add_argument("--cond-key", default="condition", help="dataset key for incident energy in --cond-file")
    parser.add_argument("--nevts", type=int, default=10000, help="number of showers to generate")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default=None, help="torch device (default: cuda if available else cpu)")
    parser.add_argument("--detector-shape", action="store_true",
                        help="export at real detector resolution (N,21,21,11) instead of the "
                             "(N,43,43,11) half-cell training grid")
    parser.add_argument("--mode", default="sum", choices=["sum", "min"],
                        help="crystal aggregation when --detector-shape: 'sum' or 'min' (4*min)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on device:", device)

    if os.path.exists(args.out):
        print(f"Output file {args.out} already exists! Abort.")
        sys.exit(1)

    model = load_model(args.model, args.checkpoint, device)
    if args.detector_shape:
        # postprocess (inside sample_fullchain) reads vq_model's flags
        model.convert_to_detector_shape = True
        model.downsample_mode = args.mode
        model.vq_model.convert_to_detector_shape = True
        model.vq_model.downsample_mode = args.mode
        print(f"Output: detector shape (N,21,21,11), downsample mode={args.mode}")
    else:
        print("Output: half-cell grid (N,43,43,11)")

    start = time.time()
    result = generate(model, args)
    dt = time.time() - start
    n = result["condition"].shape[0]
    print(f"Generation time {dt:.2f}s total, {dt / n * 1000:.3f} ms/shower")

    sanity_check(result)

    print("Saving to", args.out)
    with h5py.File(args.out, "w") as fout:
        for k, v in result.items():
            print(f"\t{k}", tuple(v.shape))
            fout[k] = v.numpy()
    print("Done.")
