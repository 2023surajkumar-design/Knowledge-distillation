# Task 3 validated components

- Frozen 45k/5k Task 1 split and test firewall are reused unchanged.
- The canonical Task 2 seed-43 best checkpoint strictly loads into ResNet18
  latent weights before ternary conversion.
- The compliant conversion covers 20 convolutions (including `conv1`) and the
  final `fc` layer.
- Forward weights are hard ternary; FP32 latent weights receive finite STE
  gradients and are distinct from deployed weights.
- Weight decay is applied only to latent convolution/linear weights; BatchNorm,
  biases, and learned scales are excluded.
- Checkpoint reconstruction restores the exact quantizer configuration and
  verifies deployed weights without test access.
