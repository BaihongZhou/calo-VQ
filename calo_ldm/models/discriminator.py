import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F

from ..layers.misc import LogScale
from ..util import get_activation_by_name, parse_conv_spec, conv_padding


class Discriminator(pl.LightningModule):
    """DarkSHINE xyz GAN discriminator (PatchGAN-style over the x-y plane).

    Kept as a load-bearing component: the paper shows L1/L2 reconstruction alone
    is insufficient for fidelity, so the adversarial term is essential. This is
    the plain-2D-conv (xyz) version of the original cylindrical discriminator;
    cyclic padding and FFT downsampling have been removed.
    """
    def __init__(self, *,
            ch_in,                 # depth = 11
            conv_spec,
            cond_dim=0,
            activation='swish',
            ch_init=-1,
            log_scale_params=None,
            pad_to=48,
            pooling=None,
            ):
        super().__init__()

        if cond_dim not in (0, 1):
            raise NotImplementedError("cond_dim>1 not handled yet.")
        self.register_buffer('cond_dim', torch.tensor(cond_dim))
        self.pad_to = pad_to

        activation_class = get_activation_by_name(activation)

        w_in = ch_in + cond_dim
        w_out = ch_init
        self.layers = nn.Sequential()
        for spec in conv_spec:
            ltype, *spec = spec.split(':')
            ltype = ltype.strip()
            assert ltype == 'pconv', f"xyz discriminator only supports 'pconv', got {ltype!r}"
            k, s, p, w_out = parse_conv_spec(':'.join(spec), w_out)
            self.layers.append(nn.Conv2d(w_in, w_out, kernel_size=k, stride=s, padding=conv_padding(k, p)))
            w_in = w_out
            self.layers.append(activation_class())

        if pooling is None:
            self.layers.append(nn.Conv2d(w_in, 1, kernel_size=(1, 1)))
        elif pooling == 'max':
            self.layers.append(nn.AdaptiveMaxPool2d((1, 1)))
        elif pooling == 'avg':
            self.layers.append(nn.AdaptiveAvgPool2d((1, 1)))
            self.layers.append(nn.Conv2d(w_in, w_in, kernel_size=(1, 1), stride=1))
            self.layers.append(activation_class())
            self.layers.append(nn.Conv2d(w_in, 1, kernel_size=(1, 1), stride=1))

        if log_scale_params:
            self.log_scale = LogScale(*log_scale_params)
        else:
            self.register_module('log_scale', None)

    def forward(self, x, cond=None):
        if self.log_scale is not None:
            x = self.log_scale(x)

        if self.cond_dim > 0:
            xc = cond[:, None, None].expand((-1, -1, x.shape[-2], x.shape[-1]))
            x = torch.concat([x, xc], axis=-3)

        pad = self.pad_to - x.shape[-1]
        if pad > 0:
            x = F.pad(x, (0, pad, 0, pad))

        return self.layers(x)
