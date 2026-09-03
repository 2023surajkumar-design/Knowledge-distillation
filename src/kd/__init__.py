"""Task 4 vanilla-logit distillation utilities only."""

from .losses import vanilla_kd_loss
from .teacher import assert_teacher_frozen, load_frozen_teacher

__all__ = ["vanilla_kd_loss", "assert_teacher_frozen", "load_frozen_teacher"]
