import torch


def huber_loss(x: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    """Calculate huber loss element-wisely depending on kappa k.

    See https://en.wikipedia.org/wiki/Huber_loss
    """
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))
