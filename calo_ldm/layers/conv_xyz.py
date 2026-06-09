import torch
from torch import nn
import torch.nn.functional as F

# Plain 2D convolutions for the DarkSHINE xyz geometry.
#
# These REPLACE the cylindrical convolutions (calo_ldm/layers/conv2.py,
# CylinderConv2/CylinderConvTranspose2) used for the ATLAS cylindrical geometry.
# The DarkSHINE ECAL is a regular x-y-z crystal grid with NO periodic dimension,
# so the azimuthal cyclic ("wrap-around") padding of the cylindrical convs is
# removed and replaced by ordinary (zero) padding in BOTH the x and y image
# directions. FFT azimuthal resampling is likewise dropped (xyz has no periodic
# axis to resample).
#
# Tensor layout in the network is channels-first (N, C, H, W) where
#   C = calorimeter depth (layer index, 11)   -> "channel" axis
#   H = x cell index, W = y cell index         -> 2D image plane
# i.e. the model input is the channels-last DarkSHINE data (N, 43, 43, 11)
# permuted to (N, 11, 43, 43).


def _check_arg_pair(args, name="kernel size", metavar=None):
    if not metavar:
        metavar = name[0]
    try:
        args = tuple((x for x in args))
    except TypeError:
        args = (args, args)
    if not all((isinstance(x, int) for x in args)):
        raise TypeError(f"{name} must be an integer: {args}")
    if len(args) != 2:
        raise ValueError(f"wrong number of {name}s provided: {args}")
    if any((x < 1 for x in args)):
        raise ValueError(f"{name}s must be positive: {args}")
    return args


class PlaneConv(nn.Module):
    '''
    Ordinary 2D convolution over the (x, y) image plane (no periodic padding).

    Args:
      in_features / out_features: input / output channel counts (channels = depth).
      k:      kernel size, scalar or (k_x, k_y).
      stride: scalar or (s_x, s_y).
      pad:    if True, zero-pad both x and y by (k-1)//2 ("same" padding for
              stride 1; halves the size for stride 2 with k=4, pad=1).
    Shape: (N, C_in, H, W) -> (N, C_out, H', W').
    '''
    def __init__(self, in_features, out_features, k=(3, 3), stride=(1, 1), pad_z=False, bias=True):
        super().__init__()
        k = _check_arg_pair(k, "kernel size")
        stride = _check_arg_pair(stride, "stride")
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.stride = stride
        # `pad_z` keeps the conv_spec DSL flag name; here it pads BOTH image dims.
        self.pad = (0, 0) if not pad_z else ((k[0] - 1) // 2, (k[1] - 1) // 2)

        wshape = (out_features, in_features, k[0], k[1])
        self.weights = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(wshape)))
        self.bias = nn.Parameter(torch.zeros((out_features,))) if bias else None

    def forward(self, x):
        return F.conv2d(x, self.weights, bias=self.bias, stride=self.stride, padding=self.pad)

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"k={self.k}, stride={self.stride}, pad={self.pad}")


class PlaneConvTranspose(nn.Module):
    '''
    Ordinary 2D transposed convolution over the (x, y) image plane.

    With k=4, stride=2, pad=1 this exactly doubles the spatial size; with
    k=3, stride=1, pad=1 it preserves it. No periodic wrap-around.
    Shape: (N, C_in, H, W) -> (N, C_out, H', W').
    '''
    def __init__(self, in_features, out_features, k=(4, 4), stride=(2, 2), pad_z=False, bias=True):
        super().__init__()
        k = _check_arg_pair(k, "kernel size")
        stride = _check_arg_pair(stride, "stride")
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.stride = stride
        self.pad = (0, 0) if not pad_z else ((k[0] - 1) // 2, (k[1] - 1) // 2)

        wshape = (in_features, out_features, k[0], k[1])
        self.weights = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(wshape)))
        self.bias = nn.Parameter(torch.zeros((out_features,))) if bias else None

    def forward(self, x):
        return F.conv_transpose2d(x, self.weights, bias=self.bias,
                                  stride=self.stride, padding=self.pad)

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"k={self.k}, stride={self.stride}, pad={self.pad}")
