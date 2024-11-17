import torch
import torch.nn as nn
import numbers
import math
from typing import List, Optional, Tuple, Union

from torch import Size, Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init


def my_calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    num_input_fmaps = tensor.size(1)
    receptive_field_size = 1
    if tensor.dim() > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size

    return fan_in



class CCLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.compute_weight = torch.zeros_like(self.weight)
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
            self.compute_bias = torch.zeros_like(self.bias)
        else:
            self.register_parameter('bias', None)
            self.register_parameter('compute_bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.update_weight()
        if self.bias is not None:
            fan_in = my_calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
            self.update_bias()


    def update_weight(self):
        column_means = torch.mean(self.weight, dim=0)
        self.compute_weight = torch.sub(self.weight, column_means)

    def update_bias(self):
        bias_mean = torch.mean(self.bias, dim=0)
        self.compute_bias = torch.sub(self.bias, bias_mean)

    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            self.update_weight()
            if self.bias is not None:
                self.update_bias()
        return F.linear(input, self.compute_weight, self.compute_bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

_shape_t = Union[int, List[int], Size]


class RMSNormLayer(nn.Module):
    __constants__ = ["normalized_shape", "eps", "elementwise_affine"]
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
        else:
            self.register_parameter("weight", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)

    def forward(self, x):
        return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
    
    nn.LayerNorm
    

class Centering(nn.Module):
    def __init__(
        self,
        shape: _shape_t,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.shape = shape
    def forward(self, x):
        length = len(self.shape)
        dims_to_mean = [i for i in range (-length, 0)]
        mean = torch.mean(x, dim=dims_to_mean, keepdim=True)
        return x - mean
