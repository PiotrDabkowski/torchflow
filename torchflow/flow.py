import torch
import torch.nn.functional as F
import numpy as np
from typing import List, TypeVar, Any
import copy

FlowData = TypeVar("FlowData", torch.Tensor, List[torch.Tensor], Any)

class Flow:
    def __init__(self, data: FlowData, device=None):
        self.data: FlowData = data
        if device is None:
            if not isinstance(data, torch.Tensor):
                raise ValueError("Device could not be inferred from batch automatically, please provide manually.")
            device = data.device
        # These values should be per example in the data batch.
        self.logpz: torch.Tensor = torch.scalar_tensor(0., dtype=torch.float32, device=device)
        self.logdet: torch.Tensor = torch.scalar_tensor(0., dtype=torch.float32, device=device)
        self.zs: [torch.Tensor] = []

    def get_logp(self):
        return self.logpz + self.logdet

    def get_pix_bits(self, imgs, levels):
        p = self.get_logp().mean()
        num_pix = np.prod(imgs.shape[1:])
        p_pix = p / num_pix + np.log(1. / levels)
        return -p_pix

    def deep_copy(self):
        flow: Flow = copy.copy(self)
        flow.logpz = flow.logpz.detach().clone()
        flow.logdet = flow.logdet.detach().clone()
        flow.data = self._copy_data(flow.data)
        return flow

    @classmethod
    def _copy_data(cls, data: FlowData):
        if data is None:
            return None
        elif isinstance(data, torch.Tensor):
            return data.detach().clone()
        elif isinstance(data, list):
            return [cls._copy_data(e) for e in data]
        else:
            raise ValueError("")

    def sample_like(self, std=0.2):
        sample: Flow = self.deep_copy()
        for e in sample.zs:
            e.normal_(mean=0.0, std=std)
        return sample


class FlowModule(torch.nn.Module):
    def forward(self, flow: Flow) -> Flow:
        return self.encode(flow)

    def encode(self, flow: Flow) -> Flow:
        """Performs the transform of the input flow. Returns the transformed flow."""
        data, logdet = self.encode_(flow.data)
        flow.data = data
        flow.logdet = flow.logdet + logdet
        return flow

    def decode(self, flow: Flow) -> Flow:
        """
        """
        data, logdet = self.decode_(flow.data)
        flow.data = data
        flow.logdet = flow.logdet + logdet
        return flow

    def encode_(self, x: torch.Tensor):
        raise NotImplementedError()

    def decode_(self, x: torch.Tensor):
        raise NotImplementedError()


class FlowInverse(FlowModule):
    def __init__(self, flow_module: FlowModule):
        super().__init__()
        assert isinstance(flow_module, FlowModule)
        self.flow_module = flow_module

    def encode(self, flow: Flow):
        return self.flow_module.decode(flow)

    def decode(self, flow: Flow):
        return self.flow_module.encode(flow)


class FlowSqueeze2D(FlowModule):
    def encode_(self, x):
        b, c, h, w = x.shape
        assert h % 2 == w % 2 == 0
        out_h = h // 2
        out_w = w // 2
        fm_view = x.contiguous().view(b, c, out_h, 2, out_w, 2)
        return fm_view.permute(0, 1, 3, 5, 2, 4).contiguous().view(b, 4 * c, out_h, out_w), 0.

    def decode_(self, x):
        return F.pixel_shuffle(x, 2), 0.


class FlowConv1x1(FlowModule):
    """https://arxiv.org/pdf/1807.03039.pdf"""

    def __init__(self, channels):
        super().__init__()
        # Fixed permutation
        permutation = torch.Tensor(channels, channels)
        torch.nn.init.orthogonal_(permutation, gain=1)
        A_LU, pivots = permutation.lu()
        P, A_L, A_U = torch.lu_unpack(A_LU, pivots)
        self.lower = torch.nn.Parameter(A_L)
        self.upper = torch.nn.Parameter(A_U)
        self.register_buffer('fixed_permutation', P)
        # Permutation must have abs(determinant) equal to 1.
        assert torch.slogdet(self._buffers['fixed_permutation']).logabsdet == 0.0

    def _get_inverse_2d_weight(self):
        # This is fast enough - only needed for decoding. In the final model this can be changed to a constant,
        # maybe jit.trace does it already?
        return torch.inverse(self._get_2d_weight())

    def _get_2d_weight(self):
        return torch.matmul(torch.matmul(self._buffers['fixed_permutation'], torch.tril(self.lower)),
                            torch.triu(self.upper))

    def _get_forward_logdet(self, x):
        element_logdet = torch.sum(torch.log(torch.abs(torch.diag(self.lower))))
        element_logdet += torch.sum(torch.log(torch.abs(torch.diag(self.upper))))
        # Multiply by the number of elements on which the 1x1 conv is performed.
        for dim_size in x.shape[2:]:
            element_logdet *= dim_size
        return element_logdet

    def encode_(self, x):
        dims = len(x.shape)
        if dims == 4:
            return F.conv2d(x, weight=self._get_2d_weight().unsqueeze(2).unsqueeze(2)), self._get_forward_logdet(x)
        elif dims == 3:
            return F.conv1d(x, weight=self._get_2d_weight().unsqueeze(2)), self._get_forward_logdet(x)
        else:
            raise ValueError()

    def decode_(self, x):
        dims = len(x.shape)
        if dims == 4:
            return F.conv2d(x,
                            weight=self._get_inverse_2d_weight().unsqueeze(2).unsqueeze(2)), -self._get_forward_logdet(
                x)
        elif dims == 3:
            return F.conv1d(x, weight=self._get_inverse_2d_weight().unsqueeze(2)), -self._get_forward_logdet(x)
        else:
            raise ValueError()


def mean_for(x, dim):
    for d in reversed(range(len(x.shape))):
        if d != dim:
            x = x.mean(d)
    return x


def sum_for(x, dim):
    for d in reversed(range(len(x.shape))):
        if d != dim:
            x = x.sum(d)
    return x


def adapt_for(x, dim, dims):
    assert len(x.shape) == 1
    shape = dims * [1]
    shape[dim] = x.shape[0]
    return x.view(shape)


class FlowActnorm(FlowModule):
    """https://arxiv.org/pdf/1807.03039.pdf"""

    def __init__(self, channels):
        super().__init__()
        shape = [channels]
        self.log_stds = torch.nn.Parameter(torch.Tensor(*shape))
        self.means = torch.nn.Parameter(torch.Tensor(*shape))
        self.register_buffer('is_initialized', torch.tensor(False))

    def _do_init(self, x):
        means = mean_for(x, dim=1)
        vars = mean_for(x * x, dim=1)
        stds = (vars - means ** 2) ** 0.5
        with torch.no_grad():
            torch.nn.init.constant_(self.log_stds, 1.0)
            torch.nn.init.constant_(self.means, 0.0)
            self.log_stds.view_as(stds).copy_(torch.log(stds))
            self.means.view_as(means).copy_(means)
            self._buffers['is_initialized'].copy_(torch.ones_like(self._buffers['is_initialized']))

    def _get_forward_logdet(self, x):
        logdet = torch.sum(self.log_stds)
        for dim_size in x.shape[2:]:
            logdet *= dim_size
        return -logdet

    def encode_(self, x):
        if not self._buffers['is_initialized'].item():
            self._do_init(x)
        dims = len(x.shape)
        means = adapt_for(self.means, dim=1, dims=dims)
        log_stds = adapt_for(self.log_stds, dim=1, dims=dims)
        return (x - means) * torch.exp(-log_stds), self._get_forward_logdet(x)

    def decode_(self, x):
        dims = len(x.shape)
        means = adapt_for(self.means, dim=1, dims=dims)
        log_stds = adapt_for(self.log_stds, dim=1, dims=dims)
        x = x * torch.exp(log_stds) + means
        return x, -self._get_forward_logdet(x)


class FlowAffineCoupling(FlowModule):
    """https://arxiv.org/pdf/1807.03039.pdf"""

    def __init__(self, nonlinear_module, channel_dim=1):
        super().__init__()
        # take the ownership of the nonlinear_module
        self.nonlinear_module = nonlinear_module
        self.channel_dim = channel_dim

    def _get_biases_logscales(self, left, right):
        biases_logscales = self.nonlinear_module(left)
        biases, logscales = torch.chunk(biases_logscales, 2, dim=self.channel_dim)
        assert biases.shape == right.shape, (left.shape, biases.shape, right.shape)
        assert logscales.shape == right.shape
        return biases, logscales

    def encode_(self, x):
        # Left can be 1 elem larger, as long as nonlinear_module takes care of this, then we are fine.
        left, right = torch.chunk(x, 2, dim=self.channel_dim)
        biases, logscales = self._get_biases_logscales(left, right)
        return torch.cat([left, torch.exp(logscales) * right + biases], dim=self.channel_dim), sum_for(logscales, dim=0)

    def decode_(self, x):
        left, right = torch.chunk(x, 2, dim=self.channel_dim)
        biases, logscales = self._get_biases_logscales(left, right)
        return torch.cat([left, torch.exp(-logscales) * (right - biases)], dim=self.channel_dim), -sum_for(logscales,
                                                                                                           dim=0)


class FlowAdditiveCoupling(FlowModule):
    """https://arxiv.org/pdf/1807.03039.pdf"""

    def __init__(self, nonlinear_module, channel_dim=1):
        super().__init__()
        # take the ownership of the nonlinear_module
        self.channel_dim = channel_dim
        self.nonlinear_module = nonlinear_module

    def encode_(self, x):
        left, right = torch.chunk(x, 2, dim=self.channel_dim)
        return torch.cat([left, right + self.nonlinear_module(left)], dim=self.channel_dim), 0.

    def decode_(self, x):
        left, right = torch.chunk(x, 2, dim=self.channel_dim)
        return torch.cat([left, right - self.nonlinear_module(left)], dim=self.channel_dim), 0.


class FlowSequentialModule(FlowModule):
    def __init__(self, *flow_modules):
        assert flow_modules and all(
            [isinstance(e, FlowModule) for e in flow_modules]), "Must provide nonzero number of FlowModules"
        super().__init__()
        for idx, module in enumerate(flow_modules):
            self.add_module(str(idx), module)

    def encode(self, flow: Flow):
        for module in self._modules.values():
            flow = module.encode(flow)
        return flow

    def decode(self, flow: Flow):
        for module in reversed(self._modules.values()):
            flow = module.decode(flow)
        return flow


class FlowSplit(FlowModule):
    def __init__(self, split_fractions=(0.5, 0.5), split_dim=1):
        super().__init__()
        assert sum(split_fractions) == 1.0, "Split fractions must sum to 1.0"
        self.split_fractions = split_fractions
        self.split_dim = split_dim

    def encode(self, flow: Flow):
        data = flow.data
        dim_size = data.shape[self.split_dim]
        slicer = [slice(None)] * len(data.shape)
        pos = 0
        split_data = []
        for split_fraction in self.split_fractions:
            num_take = split_fraction * dim_size
            if not num_take.is_integer():
                raise ValueError(
                    "dim_size * split_fraction must be an integer (got %f instead) - cannot safely split flow otherwise." % num_take)
            num_take = int(num_take)
            if num_take == 0:
                split_data.append(None)
                continue
            slicer[self.split_dim] = slice(pos, pos + num_take)
            split_data.append(data[tuple(slicer)].contiguous())
            pos += num_take
        flow.data = split_data
        return flow

    def decode(self, flow: Flow):
        if not isinstance(flow.data, list):
            raise ValueError("")
        if len(flow.data) != len(self.split_fractions):
            raise ValueError("")
        get_size_fn = lambda x: x.shape[self.split_dim] if x is not None else 0
        dim_size = sum(map(get_size_fn, flow.data))
        to_concat = []
        for split_data, split_fraction in zip(flow.data, self.split_fractions):
            expected_size = split_fraction * dim_size
            actual_size = get_size_fn(split_data)
            if not expected_size.is_integer() or expected_size != actual_size:
                raise ValueError("")
            if actual_size != 0:
                to_concat.append(split_data)

        flow.data = torch.cat(to_concat, dim=self.split_dim)
        return flow

class FlowParallelStep(FlowModule):
    def __init__(self, *flow_modules):
        super().__init__()
        self.flow_modules: [FlowModule] = flow_modules

    def encode(self, flow: Flow) -> Flow:
        result = []
        if len(flow.data) != len(self.flow_modules):
            raise ValueError("")
        data_items = flow.data
        for data_item, flow_module in zip(data_items, self.flow_modules):
            flow.data = data_item
            flow = flow_module.encode(flow)
            result.append(flow.data)
        flow.data = result
        return flow

    def decode(self, flow: Flow) -> Flow:
        result = []
        if len(flow.data) != len(self.flow_modules):
            raise ValueError("")
        data_items = flow.data
        for data_item, flow_module in zip(reversed(data_items), reversed(self.flow_modules)):
            flow.data = data_item
            flow = flow_module.decode(flow)
            result.append(flow.data)
        result.reverse()
        flow.data = result
        return flow

# class FlowDropIndices(FlowModule):
#     def __init__(self, *data_indices_to_drop):
#         super().__init__()
#         self.data_indices_to_drop = set(data_indices_to_drop)
#
#     def encode(self, flow: Flow) -> Flow:
#         if not isinstance(flow.data, list) or max(self.data_indices_to_drop) >= len(flow.data):
#             raise ValueError("")
#         result = []
#         for i, data_item in enumerate(flow.data):
#             if i in self.data_indices_to_drop:
#                 continue
#             result.append(data_item)
#         flow.data = result
#         return flow
#
#     def decode(self, flow: Flow) -> Flow:
#         if not isinstance(flow.data, list):
#             raise ValueError("")
#         result_size = len(self.data_indices_to_drop) + len(flow.data)
#         if max(self.data_indices_to_drop) >= result_size:
#             raise ValueError("")
#         data_iter = iter(flow.data)
#         result = []
#         for i in range(result_size):
#             if i in self.data_indices_to_drop:
#                 result.append(None)
#             else:
#                 result.append(next(data_iter))
#         assert len(result) == result_size
#         flow.data = result
#         return flow

class FlowNoopStep(FlowModule):
    def encode(self, flow: Flow) -> Flow:
        return flow

    def decode(self, flow: Flow) -> Flow:
        return flow

class FlowTerminateGaussianPrior(FlowModule):
    Log2PI = float(np.log(2 * np.pi))

    def __init__(self):
        super().__init__()

    def encode(self, flow: Flow) -> Flow:
        flow.zs.append(flow.data)
        flow.logpz = flow.logpz + sum_for(self._logpz(flow.data), dim=0)
        flow.data = None
        return flow

    def _logpz(self, zs):
        return -0.5 * (self.Log2PI + zs * zs)

    def decode(self, flow: Flow) -> Flow:
        if flow.data is not None:
            raise ValueError("")
        flow.data = flow.zs.pop()
        flow.logpz = flow.logpz - sum_for(self._logpz(flow.data), dim=0)
        return flow



class FlowSplitGaussianPrior(FlowSequentialModule):
    def __init__(self, split_fraction=0.5, split_dim=1):
        assert 0 < split_fraction <= 1, "Invalid split fraction."
        if split_fraction == 1.0:
            super().__init__(FlowTerminateGaussianPrior())
        else:
            super().__init__(FlowSplit(split_fractions=(1.0 - split_fraction, split_fraction), split_dim=split_dim),
                   FlowParallelStep(
                       FlowNoopStep(),
                       FlowTerminateGaussianPrior()
                   ),
                   FlowInverse(FlowSplit(split_fractions=(1.0, 0.0), split_dim=split_dim))
                )


class FlowGlowStep(FlowSequentialModule):
    """https://arxiv.org/pdf/1807.03039.pdf
    Combines IActnorm -> IConv1x1 -> IAdditiveCoupling
    """

    def __init__(self, channels, conv_hidden_dim=None, use_affine_coupling=True):
        conv_hidden_dim = conv_hidden_dim if conv_hidden_dim is not None else min(channels * 6, 512)
        nonlinear_module = torch.nn.Sequential(
            torch.nn.Conv2d(channels // 2, out_channels=conv_hidden_dim, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(conv_hidden_dim, out_channels=conv_hidden_dim, kernel_size=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(conv_hidden_dim, out_channels=channels if use_affine_coupling else channels // 2,
                            kernel_size=3, padding=1),
        )
        torch.nn.init.constant_(list(nonlinear_module._modules.values())[-1].weight, 0.)
        coupling_module = FlowAffineCoupling if use_affine_coupling else FlowAdditiveCoupling
        super().__init__(
            FlowActnorm(channels=channels),
            FlowConv1x1(channels=channels),
            coupling_module(nonlinear_module=nonlinear_module)
        )


class FlowGlowNetwork(FlowSequentialModule):
    """Glow Network for NCHW images."""

    def __init__(self, glow_step_repeats: [int], channels=3):
        assert len(glow_step_repeats) > 0
        modules = []
        for glow_step_repeat in glow_step_repeats:
            modules.append(FlowSqueeze2D())
            channels *= 4
            modules.append(FlowSequentialModule(*[FlowGlowStep(channels) for _ in range(glow_step_repeat)]))
            modules.append(FlowSplitGaussianPrior(split_fraction=0.5, split_dim=1))
            channels //= 2
        modules[-1] = FlowTerminateGaussianPrior()
        super().__init__(*modules)
