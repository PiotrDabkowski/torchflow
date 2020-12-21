import pytest
from .flow import *
import torch
import math

torch.manual_seed(11)

CHANNELS = 4
IMG = (3, CHANNELS, 12, 12)
AUDIO = (7, CHANNELS, 122)


def assert_close(
    actual, expected, abs_tolerance=1e-4, rel_tolerance=1e-3, reduction=torch.max
):
    assert reduction(torch.abs(actual - expected)).item() <= abs_tolerance, (
        actual,
        expected,
    )
    assert (
        reduction(torch.abs(actual - expected) / (torch.abs(expected) + 1.0)).item()
        <= rel_tolerance
    ), (actual, expected)


def assert_same_flow(
    actual: Flow,
    expected: Flow,
    abs_tolerance=1e-3,
    rel_tolerance=1e-3,
    reduction=torch.max,
):
    assert isinstance(actual, Flow)

    def cmp(actual, expected):
        if actual is None or expected is None:
            assert actual is None
            assert expected is None
        elif actual.shape == expected.shape:
            assert_close(
                actual, expected, abs_tolerance, rel_tolerance, reduction=reduction
            )
        else:
            assert len(actual.shape) <= 1
            assert len(expected.shape) <= 1
            assert_close(
                actual.mean(),
                expected.mean(),
                abs_tolerance,
                rel_tolerance,
                reduction=reduction,
            )

    cmp(actual.data, expected.data)
    cmp(actual.logdet, expected.logdet)
    cmp(actual.logpz, expected.logpz)
    assert len(actual.zs) == len(expected.zs)
    for actual_z, expected_z in zip(actual.zs, expected.zs):
        cmp(actual_z, expected_z)


def get_sample_tensor(shape=IMG):
    return torch.empty(shape).uniform_(-5.5, 20.5)


def get_modules():
    return [
        (FlowActnorm(CHANNELS), IMG),
        (FlowActnorm(CHANNELS), AUDIO),
        (FlowConv1x1(CHANNELS), IMG),
        (FlowConv1x1(CHANNELS), AUDIO),
        (FlowConv1x1(CHANNELS, orthogonal_init=False), IMG),
        (FlowConv1x1(CHANNELS, orthogonal_init=False), AUDIO),
        (FlowInverse(FlowConv1x1(CHANNELS)), IMG),
        (FlowInverse(FlowConv1x1(CHANNELS)), AUDIO),
        (FlowConv1x1Fixed(CHANNELS), IMG),
        (FlowConv1x1Fixed(CHANNELS), AUDIO),
        (FlowConv1x1Fixed(CHANNELS, orthogonal_init=False), IMG),
        (FlowConv1x1Fixed(CHANNELS, orthogonal_init=False), AUDIO),
        (FlowConv1x1LU(CHANNELS), IMG),
        (FlowConv1x1LU(CHANNELS), AUDIO),
        (FlowConv1x1LU(CHANNELS, orthogonal_init=False), IMG),
        (FlowConv1x1LU(CHANNELS, orthogonal_init=False), AUDIO),
        (
            FlowSequentialModule(
                FlowActnorm(channels=CHANNELS), FlowConv1x1(channels=CHANNELS)
            ),
            IMG,
        ),
        (
            FlowSequentialModule(
                FlowActnorm(channels=CHANNELS), FlowConv1x1(channels=CHANNELS)
            ),
            AUDIO,
        ),
        (FlowSqueeze2D(), IMG),
        (FlowSqueeze1D(), AUDIO),
        (
            FlowAdditiveCoupling(
                torch.nn.Conv2d(
                    CHANNELS // 2, out_channels=CHANNELS // 2, kernel_size=3, padding=1
                )
            ),
            IMG,
        ),
        (
            FlowAffineCoupling(
                torch.nn.Conv2d(
                    CHANNELS // 2, out_channels=CHANNELS, kernel_size=3, padding=1
                )
            ),
            IMG,
        ),
        (FlowGlowStep(CHANNELS, use_affine_coupling=False), IMG),
        (FlowGlowStep(3, use_affine_coupling=True), (5, 3, 16, 16)),
        (FlowGlowStep(3, use_affine_coupling=False), (5, 3, 16, 16)),
        (FlowGlowStep(CHANNELS, use_affine_coupling=True), IMG),
        (FlowQuartzStep(CHANNELS, use_affine_coupling=False), AUDIO),
        (FlowQuartzStep(CHANNELS, use_affine_coupling=True), AUDIO),
        (FlowTerminate(), IMG),
        (FlowTerminate(), AUDIO),
        (FlowSplitTerminate(split_fraction=0.5), IMG),
        (FlowSplitTerminate(split_fraction=0.5), AUDIO),
    ] + get_networks() # networks are also modules.


def get_networks():
    return [
        (FlowGlowNetwork([1, 1], channels=CHANNELS), IMG),
        (FlowGlowNetwork2([1, 1, 1], channels=CHANNELS), IMG),
    ]


@pytest.mark.parametrize("module_and_inp_shape", get_modules())
def test_encode_deterministic(module_and_inp_shape):
    module, inp_shape = module_and_inp_shape
    print(module.__class__.__name__, inp_shape)

    sample_tensor = get_sample_tensor(inp_shape)
    assert_same_flow(
        module.encode(Flow(sample_tensor)),
        module.encode(Flow(sample_tensor)),
        abs_tolerance=1e-7,
        rel_tolerance=1e-7,
        reduction=torch.max,
    )


@pytest.mark.parametrize("module_and_inp_shape", get_modules())
def test_decode(module_and_inp_shape):
    module, inp_shape = module_and_inp_shape
    print(module.__class__.__name__, inp_shape)
    in_flow = Flow(get_sample_tensor(inp_shape))
    decoded_flow = module.decode(module.encode(in_flow.deep_copy()))
    assert_same_flow(
        in_flow,
        decoded_flow,
        abs_tolerance=1e-3,
        rel_tolerance=1e-3,
        reduction=torch.max,
    )


@pytest.mark.parametrize("module_and_inp_shape", get_modules())
def test_logdet_2d(module_and_inp_shape):
    module, inp_shape = module_and_inp_shape
    print(module.__class__.__name__, inp_shape)
    sample_tensor = get_sample_tensor(inp_shape)
    flow = module.encode(Flow(sample_tensor))

    logdet_estimate = torch.slogdet(
        calculate_jacobian(Flow(sample_tensor), module)
    ).logabsdet

    logdet = flow.logdet
    if len(logdet.shape) == 0:
        logdet_estimate = logdet_estimate.mean(0)
    max_abs_logdet = abs(logdet.max(0)[0].item())
    if max_abs_logdet > 1:
        assert_close(
            logdet,
            logdet_estimate,
            abs_tolerance=5e-3 * max_abs_logdet,
            rel_tolerance=5e-3,
            reduction=torch.max,
        )
    else:
        assert_close(
            logdet,
            logdet_estimate,
            abs_tolerance=1e-3,
            rel_tolerance=0.5,
            reduction=torch.max,
        )


def normal_prob(x):
    return torch.exp(-x * x / 2.0) / (2 * math.pi) ** 0.5


@pytest.mark.parametrize("split_fraction", [0.25, 0.5, 0.75, 1.0])
def test_separation(split_fraction):
    flow_module = FlowSplitTerminate(split_fraction=split_fraction, split_dim=1)
    sample_tensor = 1.1 * torch.randn(IMG) + 0.11
    flow = flow_module.encode(Flow(sample_tensor))
    assert flow.logdet.item() == 0
    assert len(flow.zs) == 1
    assert_close(
        flow.logpz,
        sum_for(torch.log(normal_prob(flow.zs[0])), dim=0),
        abs_tolerance=1e-3,
    )

    assert_close(
        torch.cat([flow.data, flow.zs[0]], dim=1)
        if flow.data is not None
        else flow.zs[0],
        sample_tensor,
    )

    decoded_flow = flow_module.decode(flow)
    assert_close(decoded_flow.data, sample_tensor)
    assert len(decoded_flow.zs) == 0
    assert flow.logdet.item() == 0
    assert torch.abs(flow.logpz).sum().item() == 0


def test_logpz_correct():
    sample_tensor = get_sample_tensor(IMG)
    flow = Flow(sample_tensor)
    module = FlowGlowNetwork([1, 1], channels=CHANNELS)
    flow = module.encode(flow)
    assert (
        flow.data is None
    ), "Flow must be fully terminated at the end of the flow network."
    expected_logpz = (
        torch.distributions.Normal(0.0, 1.0)
        .log_prob(get_flow_flat_output(flow, sample_tensor.shape[0]))
        .sum(1)
    )
    assert_close(flow.logpz, expected_logpz, abs_tolerance=1e-2)
    data_bits = flow.get_elem_bits(sample_tensor.shape, 256)
    assert_close(
        data_bits, torch.zeros_like(data_bits) + 11, abs_tolerance=4, rel_tolerance=20
    )


def test_sampling():
    sample_tensor = get_sample_tensor()
    flow = Flow(sample_tensor)
    module = FlowGlowNetwork([1, 1], channels=CHANNELS)
    flow = module.encode(flow)
    sample_flow = flow.sample_like(distributions_or_temperatures=0.66)
    sample_flow = module.decode(sample_flow)
    assert len(sample_flow.zs) == 0
