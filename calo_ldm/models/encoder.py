import torch
from torch import nn
import torch.nn.functional as F

from ..layers import PlaneConv
from ..layers.misc import LogScale
from ..util import get_activation_by_name, parse_conv_spec


class Encoder(nn.Module):
    """DarkSHINE xyz encoder.

    Input  : pixels (N, C_in=depth=11, H=x=43, W=y=43)  + 1 conditioning channel.
    The (x, y) plane is zero-padded from 43 -> 48 so the stride-2 plane-convs
    downsample cleanly (48 -> 24 -> 12 -> 6). Output: (N, ch_out, h, w) latent
    feature map. There is no periodic dimension and no FFT resampling (those were
    cylindrical-geometry features and have been removed).
    """
    def __init__(self, *,
            ch_in,                 # number of calorimeter layers (depth) = input channels
            ch_out,                # channel dim of the output feature map
            conv_spec,             # 'pconv' layer specifier strings
            cond_dim=1,
            activation='swish',
            output_activation=None,
            init_ch=-1,
            log_scale_params=None,
            pad_to=48,             # pad the (x, y) plane up to this size
            input_hw=43,
            # accepted for config-passthru compatibility but unused in xyz:
            z_pad=None, z_padding_strategy=None,
            ):
        super().__init__()

        if cond_dim not in (0, 1):
            raise NotImplementedError("cond_dim>1 not handled yet.")
        self.register_buffer('cond_dim', torch.tensor(cond_dim))

        self.ch_in = ch_in
        self.ch_out = ch_out
        self.input_hw = input_hw
        self.pad_to = pad_to

        activation_class = get_activation_by_name(activation)
        output_activation_class = get_activation_by_name(output_activation)

        if log_scale_params is not None:
            self.log_scale = LogScale(*log_scale_params)
        else:
            self.register_module('log_scale', None)

        w_in = ch_in + cond_dim
        w_out = init_ch
        self.layers = nn.Sequential()
        for spec in conv_spec:
            ltype, *spec = spec.split(':')
            ltype = ltype.strip()
            assert ltype == 'pconv', f"xyz encoder only supports 'pconv' layers, got {ltype!r}"
            k, s, p, w_out = parse_conv_spec(':'.join(spec), w_out)
            self.layers.append(PlaneConv(w_in, w_out, k=k, stride=s, pad_z=p))
            w_in = w_out
            self.layers.append(activation_class())

        # final 1x1 conv to the latent channel dim
        self.layers.append(PlaneConv(w_in, ch_out, k=(1, 1), stride=(1, 1), pad_z=False))
        if output_activation_class is not None:
            self.layers.append(output_activation_class())

    def forward(self, x, cond=None):
        # x ~ (N, depth, x, y); cond ~ (N, 1)
        if self.log_scale:
            x = self.log_scale(x)

        if self.cond_dim > 0:
            # broadcast the (constant-per-shower) condition across the image plane
            xc = cond[:, None, None].expand((-1, -1, x.shape[-2], x.shape[-1]))  # (N,1,H,W)
            x = torch.concat([x, xc], axis=-3)

        # zero-pad the x,y plane 43 -> pad_to (e.g. 48) for clean strided convs
        pad = self.pad_to - x.shape[-1]
        if pad > 0:
            x = F.pad(x, (0, pad, 0, pad))

        out = x
        for layer in self.layers:
            out = layer(out)
        return out
