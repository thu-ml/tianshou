from contextlib import contextmanager

from torch import nn


@contextmanager
def in_eval_mode(module: nn.Module) -> None:
    """Temporarily switch to evaluation mode."""
    train = module.training
    try:
        module.eval()
        yield
    finally:
        module.train(train)


@contextmanager
def in_train_mode(module: nn.Module) -> None:
    """Temporarily switch to training mode."""
    train = module.training
    try:
        module.train()
        yield
    finally:
        module.train(train)
