## Easy Normalizing Flows in PyTorch

[![Build Status](https://travis-ci.com/PiotrDabkowski/torchflow.svg?branch=master)](https://travis-ci.com/PiotrDabkowski/torchflow)

```
pip install torchflow
```

Example:

```python
# Simple Glow-style network.
flow_module = FlowSequentialModule(
    FlowSqueeze2D(),
    FlowSplitTerminate(split_fraction=0.5),
    FlowGlowStep(channels=6),
    FlowSqueeze2D(),
    FlowSplitTerminate(split_fraction=0.5),
    FlowGlowStep(channels=12),
    FlowTerminate()
)

imgs = torch.randn(64, 3, 32, 32)
# Encode the images using the flow network.
encoded_flow = flow_module.encode(Flow(imgs))
# The log probabilities and log determinants are now available:
print(encoded_flow.logdet, encoded_flow.logpz)
# Sampling is also possible, here with temperature of 0.7:
sampled_imgs = flow_module.decode(encoded_flow.sample_like(0.7)).data
# The flow can easily be reversed, restored_imgs will be identical to imgs:
restored_imgs = flow_module.decode(encoded_flow).data
```

### Normalizing Flows Background

For intro to Normalizing Flows, please check the amazing [Normalizing Flows tutorial](https://blog.evjang.com/2018/01/nf1.html) by Eric Jang.

The code in this repo has been inspired by [Variational Inference with Normalizing Flows](https://arxiv.org/abs/1505.05770) and 
[Glow: Generative Flow with Invertible 1×1 Convolution](https://arxiv.org/abs/1807.03039). 
