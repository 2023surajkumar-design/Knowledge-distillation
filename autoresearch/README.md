# CIFAR-specific autoresearch harness

This will be a constrained, CIFAR-10-specific harness—not a copy of the language-model harness. Its immutable preparation layer must own the frozen 45k/5k split and forbid test access, architecture edits, teacher unfreezing, external data, and non-ternary student inference.

It is deferred until the relevant research stage is explicitly invoked.
