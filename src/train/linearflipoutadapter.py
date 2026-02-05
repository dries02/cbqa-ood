import torch
from bayesian_torch.layers.flipout_layers import LinearFlipout
from torch import nn


class LinearFlipoutAdapter(nn.Module):
    """Wrap LinearFlipout so forward(x) -> Tensor, while KL is stored in self.last_kl."""

    def __init__(self, linear: nn.Linear, rho: float = -3.0) -> None:
        super().__init__()
        self.flip = LinearFlipout(
            in_features=linear.in_features,
            out_features=linear.out_features,
            posterior_rho_init=rho,
            bias=(linear.bias is not None),
        )
        self.last_kl = None

        with torch.no_grad():
            self.flip.mu_weight.copy_(linear.weight)
            self.flip.prior_weight_mu.copy_(linear.weight)

            if linear.bias is not None:
                self.flip.mu_bias.copy_(linear.bias)
                self.flip.bias.copy_(linear.bias)

    @property
    def weight(self) -> torch.Tensor:
        return self.flip.mu_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, kl = self.flip(x)
        self.last_kl = kl
        return y
