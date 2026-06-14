import torch
from torch import nn
import torch.nn.functional as F


def _safe_groups(ch, max_groups=32):
    # largest divisor of ch that is <= max_groups (GroupNorm needs ch % groups == 0).
    for g in range(min(max_groups, ch), 0, -1):
        if ch % g == 0:
            return g
    return 1


class AttnBlock(nn.Module):
    """Self-attention block over the (h, w) plane (taming-transformers / VQGAN style).

    Operates on a (N, C, H, W) feature map: GroupNorm -> 1x1 q/k/v -> full
    self-attention over the H*W positions -> 1x1 projection, added as a residual.
    Intended for the VQ-VAE bottleneck (6x6 = 36 tokens), where global mixing is
    cheap and the CNN's limited receptive field cannot reach across the plane.

    The output projection is zero-initialised so the block is an EXACT identity at
    initialisation (out = x + 0); it can only help relative to the pure-CNN model
    and never perturbs a warm-started checkpoint -> low-risk drop-in.
    """
    def __init__(self, ch, num_groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(_safe_groups(ch, num_groups), ch, eps=1e-6, affine=True)
        self.q = nn.Conv2d(ch, ch, kernel_size=1)
        self.k = nn.Conv2d(ch, ch, kernel_size=1)
        self.v = nn.Conv2d(ch, ch, kernel_size=1)
        self.proj = nn.Conv2d(ch, ch, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        h = self.norm(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        N, C, H, W = q.shape
        q = q.reshape(N, C, H * W).permute(0, 2, 1)             # (N, HW, C)
        k = k.reshape(N, C, H * W)                              # (N, C, HW)
        attn = torch.bmm(q, k) * (C ** -0.5)                    # (N, HW, HW)
        attn = F.softmax(attn, dim=2)
        v = v.reshape(N, C, H * W)                              # (N, C, HW)
        out = torch.bmm(v, attn.permute(0, 2, 1))              # (N, C, HW)
        out = out.reshape(N, C, H, W)
        return x + self.proj(out)
