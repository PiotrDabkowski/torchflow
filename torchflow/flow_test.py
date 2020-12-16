import pytest
from .flow import *
import torch
import math

torch.manual_seed(11)

CHANNELS = 8
IMG_TENSOR_SHAPE = (3, CHANNELS, 16, 16)


def assert_close(x, y, tolerance=1e-4):
    assert torch.mean(torch.abs(x - y)).item() <= tolerance, (x, y)

class NoSeparationFlowModuleTester:
    def __init__(self, module: FlowModule, input_shape):
        self.module = module
        self.input_shape = input_shape
        self.sample_tensor = 1.1 * torch.randn(input_shape) + 0.11

    def check_encode(self):
        in_flow = Flow(self.sample_tensor)
        flow = self.module.encode(in_flow)
        assert isinstance(flow, Flow)
        assert np.prod(flow.data.shape) == np.prod(self.sample_tensor.shape)
        assert flow.logdet.shape == torch.Size([self.sample_tensor.shape[0]]) or len(flow.logdet.shape) == 0
        assert len(flow.zs) == 0

        # Determinism. Must be deterministic to invert.
        flow2 = self.module.encode(Flow(self.sample_tensor))
        assert_close(flow.data, flow2.data)
        assert_close(flow.logdet, flow2.logdet)

    

    def check_decode(self):
        in_flow = Flow(self.sample_tensor)
        decoded_flow = self.module.decode(self.module.encode(Flow(self.sample_tensor)))

        assert isinstance(decoded_flow, Flow)
        assert decoded_flow.data.shape == self.sample_tensor.shape
        assert decoded_flow.logdet.shape == torch.Size([self.sample_tensor.shape[0]]) or len(
            decoded_flow.logdet.shape) == 0
        assert len(decoded_flow.zs) == 0

        assert_close(decoded_flow.data, in_flow.data)
        assert_close(decoded_flow.logdet.mean(0), in_flow.logdet)

    def check_logdet(self):
        # a much more complicated test, need to approximate logdet...
        flow = self.module.encode(Flow(self.sample_tensor))

        logdet_estimate = self._estimate_logdet()
        logdet = flow.logdet
        if len(logdet.shape) == 0:
            logdet_estimate = logdet_estimate.mean(0)
        mean_abs_logdet = abs(logdet.mean(0).item())
        if mean_abs_logdet < 1:
            assert_close(logdet, logdet_estimate, 0.005)
        else:
            assert_close(logdet, logdet_estimate, mean_abs_logdet * 0.001)

    def _estimate_logdet(self):
        dx = 1e-2
        batch_size = self.sample_tensor.shape[0]
        test_tensor = self.sample_tensor.view(self.sample_tensor.shape[0], -1).clone()
        original_encoding = self.module.encode(Flow(test_tensor.view(self.input_shape))).data.view(batch_size, -1)
        if np.prod(original_encoding.shape) != np.prod(self.sample_tensor.shape):
            return torch.tensor(0., device=self.sample_tensor.device)

        det_rows = []
        for feature in range(test_tensor.shape[1]):
            d_sample = torch.zeros_like(test_tensor)
            d_sample[:, feature] = dx
            sample_encoding = self.module.encode(Flow((test_tensor + d_sample).view(self.input_shape))).data.view(
                batch_size, -1)
            det_row = (sample_encoding - original_encoding).unsqueeze(2)
            det_rows.append(det_row)

        jacobian = torch.cat(det_rows, dim=2) / dx
        pixel_logdet = torch.slogdet(jacobian).logabsdet
        return pixel_logdet


def get_no_sep_modules():
    return [
        FlowActnorm(CHANNELS),
        FlowConv1x1(CHANNELS),
        FlowInverse(FlowConv1x1(CHANNELS)),
        FlowSequentialModule(
            FlowActnorm(channels=CHANNELS),
            FlowConv1x1(channels=CHANNELS)),
        FlowSqueeze2D(),
        FlowAdditiveCoupling(
            torch.nn.Conv2d(CHANNELS // 2, out_channels=CHANNELS // 2, kernel_size=3, padding=1)),
        FlowAffineCoupling(
            torch.nn.Conv2d(CHANNELS // 2, out_channels=CHANNELS, kernel_size=3, padding=1)),
        FlowGlowStep(CHANNELS, use_affine_coupling=False),
        FlowGlowStep(CHANNELS, use_affine_coupling=True),
    ]


@pytest.mark.parametrize("flow_module", get_no_sep_modules())
def test_encode_no_separation(flow_module):
    tester = NoSeparationFlowModuleTester(flow_module, IMG_TENSOR_SHAPE)
    tester.check_encode()


@pytest.mark.parametrize("flow_module", get_no_sep_modules())
def test_decode_no_separation(flow_module):
    tester = NoSeparationFlowModuleTester(flow_module, IMG_TENSOR_SHAPE)
    tester.check_decode()


@pytest.mark.parametrize("flow_module", get_no_sep_modules())
def test_logdet_no_separation(flow_module):
    tester = NoSeparationFlowModuleTester(flow_module, IMG_TENSOR_SHAPE)
    tester.check_logdet()

def normal_prob(x):
    return torch.exp(-x*x / 2.0) / (2*math.pi)**0.5

@pytest.mark.parametrize("split_fraction", [0.25, 0.5, 0.75, 1.0])
def test_separation(split_fraction):
    flow_module = FlowSplitGaussianPrior(split_fraction=split_fraction, split_dim=1)
    sample_tensor = 1.1 * torch.randn(IMG_TENSOR_SHAPE) + 0.11
    flow = flow_module.encode(Flow(sample_tensor))
    assert flow.logdet.item() == 0
    assert len(flow.zs) == 1
    assert_close(flow.logpz, sum_for(torch.log(normal_prob(flow.zs[0])), dim=0))

    assert_close(torch.cat([flow.data, flow.zs[0]], dim=1) if flow.data is not None else flow.zs[0], sample_tensor)

    decoded_flow = flow_module.decode(flow)
    assert_close(decoded_flow.data, sample_tensor)
    assert len(decoded_flow.zs) == 0
    assert flow.logdet.item() == 0
    assert torch.abs(flow.logpz).sum().item() == 0


def test_glow_network():
    sample_tensor = torch.randn(5, 3, 32, 32)
    flow_module = FlowGlowNetwork([2, 2])

    encoded = flow_module.encode(Flow(sample_tensor))
    assert encoded.data is None
    assert encoded.logdet.mean().item() != 0.0
    assert encoded.logpz.mean().item() != 0.0
    decoded = flow_module.decode(encoded)
    assert_close(decoded.data, sample_tensor)
    assert abs(decoded.logdet.mean().item()) <= 1e-3
    assert abs(decoded.logpz.mean().item()) <= 1e-3
    assert len(decoded.zs) == 0




