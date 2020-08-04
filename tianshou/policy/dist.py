import torch


class DiagGaussian(torch.distributions.Normal):
    """Diagonal Gaussian distribution."""

    def log_prob(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)
