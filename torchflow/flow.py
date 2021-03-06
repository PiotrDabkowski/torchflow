import torch
import torch.nn.functional as F
import numpy as np
from typing import List, TypeVar, Any, Optional, Union, List
import copy

FlowData = TypeVar("FlowData", torch.Tensor, List[torch.Tensor], Any)


def level_encode(data: torch.Tensor, num_levels=32, input_range=(0.0, 1.0), eps=1e-6):
    scale = input_range[1] - input_range[0]
    scaled = (data - input_range[0]) / scale
    leveled = (scaled * (1.0 - eps) * num_levels).floor() + torch.empty_like(
        data
    ).uniform_()
    return leveled / num_levels * scale + input_range[0]


class Flow:
    def __init__(self, data: FlowData, device=None):
        self.data: FlowData = data
        if device is None:
            if not isinstance(data, torch.Tensor):
                raise ValueError(
                    "Device could not be inferred from batch automatically, please provide manually."
                )
            device = data.device
        # These values should be per example in the data batch.
        self.logpz: torch.Tensor = torch.scalar_tensor(
            0.0, dtype=torch.float32, device=device
        )
        self.logdet: torch.Tensor = torch.scalar_tensor(
            0.0, dtype=torch.float32, device=device
        )
        self.zs: [torch.Tensor] = []

    def __repr__(self):
        return "Flow high level info (means): logpz: %.3f  logdet: %.3f" % (
            self.logpz.mean().item(),
            self.logdet.mean().item(),
        )

    def get_logp(self):
        return self.logpz + self.logdet

    def get_elem_bits(self, input_data_shape, num_data_levels):
        p_log2 = self.get_logp() / np.log(2.0)
        p_elem_log2 = p_log2 / np.prod(input_data_shape[1:]) - np.log2(num_data_levels)
        return -p_elem_log2

    def deep_copy(self):
        flow: Flow = copy.copy(self)
        flow.logpz = flow.logpz.detach().clone()
        flow.logdet = flow.logdet.detach().clone()
        flow.data = self._copy_data(flow.data)
        flow.zs = [self._copy_data(e) for e in flow.zs]
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

    def sample_like(
        self,
        distributions_or_temperatures: Union[
            float,
            List[float],
            torch.distributions.Distribution,
            List[torch.distributions.Distribution],
        ] = 0.66,
    ):
        sample: Flow = self.deep_copy()
        if len(sample.zs) == 0:
            return sample
        if isinstance(distributions_or_temperatures, float) or isinstance(
            distributions_or_temperatures, torch.distributions.Distribution
        ):
            distributions_or_temperatures = [distributions_or_temperatures] * len(
                self.zs
            )
        if isinstance(distributions_or_temperatures[0], float):
            distributions_or_temperatures = [
                torch.distributions.Normal(0.0, scale=temperature)
                for temperature in distributions_or_temperatures
            ]
        distributions: List[
            torch.distributions.Distribution
        ] = distributions_or_temperatures
        if len(distributions) != len(sample.zs):
            raise ValueError(
                "Invalid number of distributions provided, expected %d, got %d"
                % (len(sample.zs), len(distributions))
            )
        for z, distribution in zip(sample.zs, distributions):
            z.copy_(distribution.sample(z.shape))
        return sample


class FlowModule(torch.nn.Module):
    CHECK_FLOW_EXPLOSION = True
    # If enabled, FlowSequential modules will automatically verify whether logdet returned by the FlowModule is
    # accurate. The jacobian is calculated automatically, by treating the module as a black box.
    # Only enable to debug logdet issues, numerically calculating logdet is extremely slow.
    VERIFY_LOGDET_ACCURACY = False

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


def get_flow_flat_output(flow: Flow, batch_size: int, skip_num_zs=0) -> torch.Tensor:
    components = (flow.data if isinstance(flow.data, list) else [flow.data]) + flow.zs[
        skip_num_zs:
    ]
    components = [c.view(batch_size, -1) for c in components if c is not None]
    return torch.cat(components, dim=1)


def calculate_jacobian(flow: Flow, module: FlowModule):
    assert isinstance(flow.data, torch.Tensor)
    in_data: torch.Tensor = flow.data
    in_data.requires_grad = True
    batch_size = in_data.shape[0]
    num_initial_zs = len(flow.zs)
    out_flow = module.encode(flow)
    flat_out = get_flow_flat_output(out_flow, batch_size, skip_num_zs=num_initial_zs)
    assert flat_out.numel() == in_data.numel()
    grads = []
    for i in range(flat_out.shape[1]):
        if in_data.grad is not None:
            in_data.grad.zero_()
        flat_out[:, i].sum().backward(retain_graph=True)
        assert in_data.grad is not None
        grads.append(in_data.grad.view(batch_size, -1).detach().clone().unsqueeze(2))
    jacobian = torch.cat(grads, dim=2)
    return jacobian


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
        return (
            fm_view.permute(0, 1, 3, 5, 2, 4).contiguous().view(b, 4 * c, out_h, out_w),
            0.0,
        )

    def decode_(self, x):
        return F.pixel_shuffle(x, 2), 0.0


class FlowSqueeze1D(FlowModule):
    def encode_(self, x):
        bs, c, length = x.shape
        assert length % 2 == 0
        out_length = length // 2
        out_c = 2 * c
        return (
            x.permute(0, 2, 1)
            .contiguous()
            .view(bs, out_length, out_c)
            .permute(0, 2, 1)
            .contiguous(),
            0.0,
        )

    def decode_(self, x):
        bs, c, length = x.shape
        assert c % 2 == 0
        out_length = length * 2
        out_c = c // 2
        return (
            x.permute(0, 2, 1)
            .contiguous()
            .view(bs, out_length, out_c)
            .permute(0, 2, 1)
            .contiguous(),
            0.0,
        )


class _FlowConv1x1(FlowModule):
    def __init__(self):
        super().__init__()

    def _get_inverse_2d_weight(self):
        # This is fast enough - only needed for decoding. In the final model this can be changed to a constant,
        # maybe jit.trace does it already?
        return torch.inverse(self._get_2d_weight())

    def _get_2d_weight(self):
        raise NotImplementedError()

    def _get_forward_logdet(self):
        return torch.slogdet(self._get_2d_weight()).logabsdet

    def __get_forward_logdet(self, x):
        # Multiply by the number of elements on which the 1x1 conv is performed.
        return self._get_forward_logdet() * np.prod(x.shape[2:])

    def encode_(self, x):
        dims = len(x.shape)
        if dims == 4:
            return (
                F.conv2d(x, weight=self._get_2d_weight().unsqueeze(2).unsqueeze(2)),
                self.__get_forward_logdet(x),
            )
        elif dims == 3:
            return (
                F.conv1d(x, weight=self._get_2d_weight().unsqueeze(2)),
                self.__get_forward_logdet(x),
            )
        else:
            raise ValueError()

    def decode_(self, x):
        dims = len(x.shape)
        if dims == 4:
            return (
                F.conv2d(
                    x, weight=self._get_inverse_2d_weight().unsqueeze(2).unsqueeze(2)
                ),
                -self.__get_forward_logdet(x),
            )
        elif dims == 3:
            return (
                F.conv1d(x, weight=self._get_inverse_2d_weight().unsqueeze(2)),
                -self.__get_forward_logdet(x),
            )
        else:
            raise ValueError()


class FlowConv1x1(_FlowConv1x1):
    def __init__(self, channels, orthogonal_init=True):
        super().__init__()
        permutation = torch.Tensor(channels, channels)
        if orthogonal_init:
            torch.nn.init.orthogonal_(permutation, gain=1)
        else:
            torch.nn.init.xavier_normal_(permutation, gain=1)
        self.weight = torch.nn.Parameter(permutation)

    def _get_2d_weight(self):
        return self.weight


class FlowConv1x1Fixed(_FlowConv1x1):
    def __init__(self, channels, orthogonal_init=True):
        super().__init__()
        permutation = torch.Tensor(channels, channels)
        if orthogonal_init:
            torch.nn.init.orthogonal_(permutation, gain=1)
        else:
            torch.nn.init.xavier_normal_(permutation, gain=1)
        self.register_buffer(
            "fixed_permutation_logdet", torch.slogdet(permutation).logabsdet
        )
        self.register_buffer("fixed_permutation_inverse", torch.inverse(permutation))
        self.register_buffer("fixed_permutation", permutation)

    def _get_2d_weight(self):
        return self._buffers["fixed_permutation"]

    def _get_forward_logdet(self):
        return self._buffers["fixed_permutation_logdet"]

    def _get_inverse_2d_weight(self):
        return self._buffers["fixed_permutation_inverse"]


class FlowConv1x1LU(_FlowConv1x1):
    """https://arxiv.org/pdf/1807.03039.pdf"""

    def __init__(self, channels, orthogonal_init=True):
        super().__init__()
        # Fixed permutation
        permutation = torch.Tensor(channels, channels)
        if orthogonal_init:
            torch.nn.init.orthogonal_(permutation, gain=1)
        else:
            torch.nn.init.xavier_normal_(permutation, gain=1)
        A_LU, pivots = permutation.lu()
        P, A_L, A_U = torch.lu_unpack(A_LU, pivots)
        self.lower = torch.nn.Parameter(A_L)
        self.upper = torch.nn.Parameter(A_U)
        self.register_buffer("fixed_permutation", P)
        # Permutation must have abs(determinant) equal to 1.
        assert torch.slogdet(self._buffers["fixed_permutation"]).logabsdet == 0.0

    def _get_2d_weight(self):
        return torch.matmul(
            torch.matmul(self._buffers["fixed_permutation"], torch.tril(self.lower)),
            torch.triu(self.upper),
        )

    def _get_forward_logdet(self):
        element_logdet = torch.sum(torch.log(torch.abs(torch.diag(self.lower))))
        element_logdet += torch.sum(torch.log(torch.abs(torch.diag(self.upper))))
        return element_logdet


def mean_for(x, dim):
    return x.mean([e for e in range(len(x.shape)) if e != dim])


def sum_for(x, dim):
    return x.sum([e for e in range(len(x.shape)) if e != dim])


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
        self.register_buffer("is_initialized", torch.tensor(False))

    def _do_init(self, x):
        means = mean_for(x, dim=1)
        vars = mean_for(x * x, dim=1)
        stds = (vars - means ** 2) ** 0.5
        with torch.no_grad():
            torch.nn.init.constant_(self.log_stds, 1.0)
            torch.nn.init.constant_(self.means, 0.0)
            self.log_stds.view_as(stds).copy_(torch.log(stds))
            self.means.view_as(means).copy_(means)
            self._buffers["is_initialized"].copy_(
                torch.ones_like(self._buffers["is_initialized"])
            )

    def _get_forward_logdet(self, x):
        logdet = torch.sum(self._get_final_log_stds())
        for dim_size in x.shape[2:]:
            logdet *= dim_size
        return -logdet

    def _get_final_log_stds(self):
        return soft_abs_limit(self.log_stds, limit=2.0)

    def encode_(self, x):
        if not self._buffers["is_initialized"].item():
            print("Initializing...")
            self._do_init(x)
        dims = len(x.shape)
        means = adapt_for(self.means, dim=1, dims=dims)
        log_stds = adapt_for(self._get_final_log_stds(), dim=1, dims=dims)
        return (x - means) * torch.exp(-log_stds), self._get_forward_logdet(x)

    def decode_(self, x):
        dims = len(x.shape)
        means = adapt_for(self.means, dim=1, dims=dims)
        log_stds = adapt_for(self._get_final_log_stds(), dim=1, dims=dims)
        x = x * torch.exp(log_stds) + means
        return x, -self._get_forward_logdet(x)


class FlowActivenorm(FlowModule):
    def __init__(self, channels, momentum=0.8):
        super().__init__()
        shape = [channels]
        self.desired_log_scales = torch.nn.Parameter(torch.Tensor(*shape).zero_())
        self.desired_mean = torch.nn.Parameter(torch.Tensor(*shape).zero_())
        self.register_buffer("means", torch.Tensor(*shape).zero_())
        self.register_buffer("vars", torch.Tensor(*shape).zero_() + 1.0)
        self.register_buffer("num_stat_updates", torch.tensor(0))
        self.momentum = momentum

    def _get_actual_log_scales(self):
        return soft_abs_limit(self.desired_log_scales, limit=5.0) - 0.5 * torch.log(
            self._buffers["vars"]
        )

    def _get_actual_mean(self):
        # (-mean*actual_std + desired_mean)
        return self.desired_mean

    def _update_stats(self, x):
        return
        means = mean_for(x, dim=1).detach()
        vars = mean_for(x * x, dim=1).detach() - means ** 2 + 1e-6

        old_weight = 0.0  # (1. - torch.pow(torch.tensor(self.momentum, device=x.device), self._buffers['num_stat_updates'])) * self.momentum
        new_weight = 1 - self.momentum
        means = (means * new_weight + self._buffers["means"] * old_weight) / (
            new_weight + old_weight
        )
        vars = (vars * new_weight + self._buffers["vars"] * old_weight) / (
            new_weight + old_weight
        )

        with torch.no_grad():
            self._buffers["means"].copy_(means)
            self._buffers["vars"].copy_(vars)
            self._buffers["num_stat_updates"].copy_(
                self._buffers["num_stat_updates"] + 1
            )

    def _get_forward_logdet(self, x):
        logdet = torch.sum(self._get_actual_log_scales())
        for dim_size in x.shape[2:]:
            logdet *= dim_size
        return -logdet

    def encode_(self, x):
        if self.training:
            self._update_stats(x)

        # (x - mean) / std * desired_std + desired_mean =
        # = x * actual_scale + (-mean*actual_scale + desired_mean) =
        # = x * actual_scale + actual_mean
        # Where actual_std = desired_std / std
        dims = len(x.shape)
        means = adapt_for(self._get_actual_mean(), dim=1, dims=dims)
        scales = adapt_for(torch.exp(self._get_actual_log_scales()), dim=1, dims=dims)
        return x * scales - means, self._get_forward_logdet(x)

    def decode_(self, x):
        dims = len(x.shape)
        means = adapt_for(self._get_actual_mean(), dim=1, dims=dims)
        inv_scales = adapt_for(
            torch.exp(-self._get_actual_log_scales()), dim=1, dims=dims
        )
        return (x + means) * inv_scales, -self._get_forward_logdet(x)


def soft_abs_limit(x, limit=5.0):
    return limit * torch.tanh(x / limit)


class FlowAffineCoupling(FlowModule):
    """https://arxiv.org/pdf/1807.03039.pdf"""

    def __init__(self, nonlinear_module, channel_dim=1):
        super().__init__()
        # take the ownership of the nonlinear_module
        self.nonlinear_module = nonlinear_module
        self.channel_dim = channel_dim

    def _get_biases_logscales(self, left, right):
        biases_logscales = self.nonlinear_module(right)
        biases, logscales = torch.chunk(biases_logscales, 2, dim=self.channel_dim)
        logscales = soft_abs_limit(logscales, limit=1.0)
        assert biases.shape == left.shape, (left.shape, biases.shape, right.shape)
        assert logscales.shape == left.shape
        return biases, logscales

    def encode_(self, x):
        # Left can be 1 elem larger, as long as nonlinear_module takes care of this, then we are fine.
        left, right = torch.chunk(x, 2, dim=self.channel_dim)
        biases, logscales = self._get_biases_logscales(left, right)
        return (
            torch.cat(
                [torch.exp(logscales) * left + biases, right], dim=self.channel_dim
            ),
            sum_for(logscales, dim=0),
        )

    def decode_(self, x):
        left, right = torch.chunk(x, 2, dim=self.channel_dim)
        biases, logscales = self._get_biases_logscales(left, right)
        return (
            torch.cat(
                [torch.exp(-logscales) * (left - biases), right], dim=self.channel_dim
            ),
            -sum_for(logscales, dim=0),
        )


class FlowAdditiveCoupling(FlowModule):
    """https://arxiv.org/pdf/1807.03039.pdf"""

    def __init__(self, nonlinear_module, channel_dim=1):
        super().__init__()
        # take the ownership of the nonlinear_module
        self.channel_dim = channel_dim
        self.nonlinear_module = nonlinear_module

    def encode_(self, x):
        left, right = torch.chunk(x, 2, dim=self.channel_dim)
        return (
            torch.cat(
                [left + self.nonlinear_module(right), right] , dim=self.channel_dim
            ),
            0.0,
        )

    def decode_(self, x):
        left, right = torch.chunk(x, 2, dim=self.channel_dim)
        return (
            torch.cat(
                [left - self.nonlinear_module(right), right], dim=self.channel_dim
            ),
            0.0,
        )


class FlowSequentialModule(FlowModule):
    def __init__(self, *flow_modules):
        assert flow_modules and all(
            [isinstance(e, FlowModule) for e in flow_modules]
        ), "Must provide nonzero number of FlowModules"
        super().__init__()
        for idx, module in enumerate(flow_modules):
            self.add_module(str(idx), module)

    def encode(self, flow: Flow):
        in_flow = None
        for module in self._modules.values():
            if self.VERIFY_LOGDET_ACCURACY:
                in_flow = flow.deep_copy()
            flow = module.encode(flow)
            if self.CHECK_FLOW_EXPLOSION:
                logdet = flow.logdet.mean().cpu().item()
                logpz = flow.logpz.mean().cpu().item()
                if abs(logdet) > 1e10:
                    raise ValueError("Logdet exploded %f, module %s" % (logdet, module))
                if abs(logpz) > 1e10:
                    raise ValueError("Logpz exploded %f, module %s" % (logpz, module))
            if self.VERIFY_LOGDET_ACCURACY and isinstance(in_flow.data, torch.Tensor):
                numel = in_flow.data[0].numel()
                actual_ele_logdet = (
                    flow.logdet.mean().cpu().item() - in_flow.logdet.mean().cpu().item()
                ) / numel
                ele_logdet_est = (
                    torch.slogdet(calculate_jacobian(in_flow, module))
                    .logabsdet.mean()
                    .cpu()
                    .item()
                    / numel
                )
                is_ok = abs(ele_logdet_est - actual_ele_logdet) < 0.01
                print(
                    "[Module %s] Logdet Actual: %.3f Expected ~%.3f Is OK?: %s"
                    % (
                        module.__class__.__name__,
                        actual_ele_logdet,
                        ele_logdet_est,
                        is_ok,
                    )
                )
                if not is_ok:
                    raise ValueError("!!! Bad logdet in module %s" % repr(module))
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
                    "dim_size * split_fraction must be an integer (got %f instead) - cannot safely split flow otherwise."
                    % num_take
                )
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
        for data_item, flow_module in zip(
            reversed(data_items), reversed(self.flow_modules)
        ):
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


class FlowTerminate(FlowModule):
    Log2PI = float(np.log(2 * np.pi))

    def __init__(self, distribution: Optional[torch.distributions.Distribution] = None):
        super().__init__()
        self.distribution: torch.distributions.Distribution = distribution if distribution is not None else torch.distributions.Normal(
            0.0, 1.0
        )

    def encode(self, flow: Flow) -> Flow:
        flow.zs.append(flow.data)
        flow.logpz = flow.logpz + sum_for(self.distribution.log_prob(flow.data), dim=0)
        flow.data = None
        return flow

    def decode(self, flow: Flow) -> Flow:
        if flow.data is not None:
            raise ValueError("")
        flow.data = flow.zs.pop()
        flow.logpz = flow.logpz - sum_for(self.distribution.log_prob(flow.data), dim=0)
        return flow


class FlowSplitTerminate(FlowSequentialModule):
    def __init__(
        self,
        split_fraction=0.5,
        split_dim=1,
        distribution: Optional[torch.distributions.Distribution] = None,
    ):
        assert 0 < split_fraction <= 1, "Invalid split fraction."
        if split_fraction == 1.0:
            super().__init__(FlowTerminate(distribution=distribution))
        else:
            super().__init__(
                FlowSplit(
                    split_fractions=(1.0 - split_fraction, split_fraction),
                    split_dim=split_dim,
                ),
                FlowParallelStep(
                    FlowNoopStep(), FlowTerminate(distribution=distribution)
                ),
                FlowInverse(FlowSplit(split_fractions=(1.0, 0.0), split_dim=split_dim)),
            )


class _CausalConv1d(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        assert stride == 1, "Not supported."
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self._causal_padding = (kernel_size - 1) * dilation

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        padded_input = F.pad(input, (self._causal_padding, 0), "constant", 0.0)
        result = super().forward(padded_input)
        assert result.shape[2] == input.shape[2], (input.shape, result.shape)
        return result


class FlowQuartzStep(FlowSequentialModule):
    def __init__(self, channels, use_affine_coupling=True, kernel_size=3, dilation=1):
        nn_channels = channels // 2
        nn_out_channels = channels - nn_channels if not use_affine_coupling else 2*(channels - nn_channels)
        affine_mapping_conv = torch.nn.Conv1d(
            nn_channels,
            out_channels=nn_out_channels,
            kernel_size=1,
            padding=0,
        )
        torch.nn.init.constant_(affine_mapping_conv.weight, 0.0)

        nonlinear_module = torch.nn.Sequential(
            _CausalConv1d(
                nn_channels,
                out_channels=nn_channels,
                kernel_size=kernel_size,
                groups=nn_channels,
                dilation=dilation,
            ),
            torch.nn.Conv1d(
                nn_channels, out_channels=nn_channels, kernel_size=1, padding=0
            ),
            torch.nn.BatchNorm1d(nn_channels),
            torch.nn.ReLU(),
            affine_mapping_conv,
        )
        coupling_module = (
            FlowAffineCoupling if use_affine_coupling else FlowAdditiveCoupling
        )
        super().__init__(
            FlowActnorm(channels=channels),
            FlowConv1x1LU(channels=channels),
            coupling_module(nonlinear_module=nonlinear_module),
        )


class FlowGlowStep(FlowSequentialModule):
    """https://arxiv.org/pdf/1807.03039.pdf
    Combines IActnorm -> IConv1x1 -> IAdditiveCoupling
    """

    def __init__(self, channels, conv_hidden_dim=None, use_affine_coupling=True):
        nn_channels = channels // 2
        nn_out_channels = channels - nn_channels if not use_affine_coupling else 2*(channels - nn_channels)
        conv_hidden_dim = (
            conv_hidden_dim if conv_hidden_dim is not None else max(min(channels * 12, 512), nn_channels)
        )

        nonlinear_module = torch.nn.Sequential(
            torch.nn.Conv2d(
                nn_channels, out_channels=conv_hidden_dim, kernel_size=3, padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                conv_hidden_dim, out_channels=conv_hidden_dim, kernel_size=1, padding=0
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                conv_hidden_dim,
                out_channels=nn_out_channels,
                kernel_size=3,
                padding=1,
            ),
        )
        torch.nn.init.constant_(
            list(nonlinear_module._modules.values())[-1].weight, 0.0
        )
        coupling_module = (
            FlowAffineCoupling if use_affine_coupling else FlowAdditiveCoupling
        )
        super().__init__(
            FlowActnorm(channels=channels),
            FlowConv1x1LU(channels=channels),
            coupling_module(nonlinear_module=nonlinear_module),
        )


class FlowGlowNetwork(FlowSequentialModule):
    """Glow Network for NCHW images."""

    def __init__(self, glow_step_repeats: [int], channels=3):
        assert len(glow_step_repeats) > 0
        modules = []
        for glow_step_repeat in glow_step_repeats:
            modules.append(FlowSqueeze2D())
            channels *= 4
            modules.append(
                FlowSequentialModule(
                    *[
                        FlowGlowStep(channels, use_affine_coupling=True)
                        for _ in range(glow_step_repeat)
                    ]
                )
            )
            modules.append(FlowSplitTerminate(split_fraction=0.5, split_dim=1))
            channels //= 2
        modules[-1] = FlowTerminate()
        super().__init__(*modules)


class FlowGlowNetwork2(FlowSequentialModule):
    """Glow Network for NCHW images."""

    def __init__(self, glow_step_repeats: [int], channels=3, separate_after_stage1=True):
        assert len(glow_step_repeats) > 1
        modules = []
        for i, glow_step_repeat in enumerate(glow_step_repeats):
            modules.append(
                FlowSequentialModule(
                    *[
                        FlowGlowStep(channels, use_affine_coupling=True)
                        for _ in range(glow_step_repeat)
                    ]
                )
            )
            if i != 0:
                assert channels % 2 == 0
                modules.append(FlowSplitTerminate(split_fraction=0.5, split_dim=1))
                channels //= 2
            modules.append(FlowSqueeze2D())
            channels *= 4
        modules.pop()
        modules.pop()
        modules.append(FlowTerminate())
        super().__init__(*modules)