import torch
from torch import nn
import torch.nn.functional as F

from .layers.misc import (
        VoxelSoftmax, FlatVoxelSoftmax,
        VoxelReluExpm1Max, FlatVoxelReluExpm1Max,
        FlatVoxelCeluExpm1Max,
        VoxelReluExpm1MaxD6,
        )

import importlib
import sys

def load_geom_mask(path):
    """Load the DarkSHINE fixed geometry mask.

    The mask is stored channels-last as (x=43, y=43, depth=11) bool, marking the
    real LYSO crystal cells (True) vs padding (False). It is the SAME for every
    sample. We return it channels-first as (depth=11, x=43, y=43) so it lines up
    with the network's (N, C=depth, H=x, W=y) tensor layout.
    """
    import numpy as np
    m = np.load(path)
    t = torch.from_numpy(np.ascontiguousarray(m)).bool()  # (43, 43, 11)
    t = t.permute(2, 0, 1).contiguous()                   # (11, 43, 43)
    return t


def recursive_to(obj, device):
    if isinstance(obj, dict):
        new = {k: recursive_to(v, device) for k,v in obj.items()}
        return new
    elif isinstance(obj, (list, tuple)):
        new = [recursive_to(o, device) for o in obj]
        return new
    else:
        return obj.to(device)

_activation_aliases = {
    'relu': nn.ReLU,
    'sigmoid': nn.Sigmoid,
    'swish': nn.SiLU,
    'silu': nn.SiLU,
    'softplus': nn.Softplus,
    'voxel_softmax': VoxelSoftmax,
    'flat_voxel_softmax': FlatVoxelSoftmax,
    'voxel_relu_expm1_max': VoxelReluExpm1Max,
    'voxel_relu_expm1_max_dyn6': VoxelReluExpm1MaxD6,
    'flat_voxel_relu_expm1_max': FlatVoxelReluExpm1Max,
    'flat_voxel_celu_expm1_max': FlatVoxelCeluExpm1Max,
}
def get_activation_by_name(name):
    if name is None:
        return None
    return _activation_aliases[name.lower()]


def instantiate_from_config(config, passthru={}, overwrite=False):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    params = config.get("params", {})
    for k,v in passthru.items():
        if not k in params:
            print(f">> Using passthru value for {k} in {config['target']}")
            params[k] = v
        else:
            if overwrite:
                params[k] = v
                print(f"Warning inherited option would overwrite the config!! Now {k}={params[k]}",file=sys. stderr)
            else:
                print(f"Warning inherited option would not overwrite the config!! Still use {k}={params[k]}",file=sys. stderr)
    return get_obj_from_str(config["target"])(**params)

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def parse_conv_spec(spec, w_out_prev=None):
    # spec format:
    #   :kAB, :kA -> k=(A,B), k=(A,A)
    #   :sAB, :sA -> stride=(A,B), stride=(A,A)
    #   :p        -> pad_z=True
    #   :*N       -> w_out -> w_out*N
    #   :/N       -> w_out -> w_out//N
    #   :+N       -> w_out -> w_out+N
    #   :-N       -> w_out -> w_out-N
    #   :cN       -> w_out -> N
    # order and case insensitive. you can add extra colons and/or
    # spaces e.g. for readability, they will be ignored.
    # defaults: k=(3,3), stride=(1,1), pad_z=False, w_out=w_out
    # examples:
    #   s12:k53:p -> stride(z=2,a=1), k(z=5,a=3), pad_z=True
    #   k3:s13   -> stride(1,3), k(3,3), pad_z=False
    #   p:k35    -> stride(1,1) k(3,5), pad_z=True
    if w_out_prev is not None:
        w_out = w_out_prev

    s = (1,1)
    k = (3,3)
    pad_z = False
    for x in spec.lower().split(':'):
        x = x.strip()
        if x == '': continue
        elif x.startswith('s'): s=tuple(map(int, x[1:]))
        elif x.startswith('k'): k=tuple(map(int, x[1:]))
        elif x.startswith('*'): w_out *= int(x[1:])
        elif x.startswith('/'): w_out //= int(x[1:])
        elif x.startswith('+'): w_out += int(x[1:])
        elif x.startswith('-'): w_out -= int(x[1:])
        elif x.startswith('c'): w_out = int(x[1:])
        elif x == 'p': pad_z = True
        else:
            raise ValueError(f"Unkown specification key: {x}")

    if len(s) == 1: s = 2*s
    if len(k) == 1: k = 2*k

    if w_out_prev is None:
        return k, s, pad_z
    return k, s, pad_z, w_out


def conv_padding(k, pad):
    """Translate the conv-spec `pad_z` flag into an nn.Conv2d `padding` arg.

    `pad=True` gives "same"-style zero padding of (k-1)//2 on each image dim
    (so stride-1 preserves size and k=4/stride=2 halves it); `pad=False` -> 0.
    `k` is the (k_x, k_y) kernel tuple from `parse_conv_spec`.
    """
    return ((k[0] - 1) // 2, (k[1] - 1) // 2) if pad else 0


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    elif global_step==threshold:
        print("DISC enabled!")
        weight = value
    return weight


def measure_perplexity(predicted_indices, n_embed):
    # src: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py
    # eval cluster perplexity. when perplexity == num_embeddings then all clusters are used exactly equally
    encodings = F.one_hot(predicted_indices, n_embed).float().reshape(-1, n_embed)
    avg_probs = encodings.mean(0)
    perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()  # consider probs
    cluster_use = torch.sum(avg_probs > 0) / n_embed # count all non zero embed
    return perplexity, cluster_use

