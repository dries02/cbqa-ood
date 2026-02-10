import torch
from bayesian_torch.layers.flipout_layers import LinearFlipout
from torch import nn


class LinearFlipoutAdapter(nn.Module):
    """Thin wrapper around bayesian-torch `LinearFlipout`, MOPED-style (https://arxiv.org/abs/1906.05323).

    - Forces `forward() -> Tensor` (avoids tuple default).
    - prior mean and posterior mean initialized from pretrained weights.
    - bias currently not supported.
    
    Remark: the bayesian-torch implementation seems to contain mistakes.
    - prior_variance should be prior_sigma.
    - MOPED looks wrong
    """

    def __init__(self, pretrained_linear: nn.Linear, rho: float) -> None:
        """Create a LinearFlipoutAdapter."""
        super().__init__()
        if pretrained_linear.bias is not None:
            msg = "Bias currently not supported."
            raise ValueError(msg)

        prior_sigma = 0.1 #0.1 * torch.std(pretrained_linear.weight.data).item()
        self.flip = LinearFlipout(                  # prior mean and posterior mean overwritten with w_pretrained
            in_features=pretrained_linear.in_features,
            out_features=pretrained_linear.out_features,
            prior_variance=prior_sigma,             # set prior sigma. typo in library...
            bias=False,
        )

        delta = 0.1

        with torch.no_grad():
            self.flip.mu_weight.copy_(pretrained_linear.weight)
            self.flip.prior_weight_mu.copy_(pretrained_linear.weight)
                            # numerically stable version of log(exp(delta |w| - 1))
            rho_weight = torch.log(torch.expm1(torch.clamp(delta * pretrained_linear.weight.abs(), min=1e-7)))
            self.flip.rho_weight.copy_(rho_weight)

    @property
    def weight(self) -> torch.Tensor:
        return self.flip.mu_weight      # added to appease PyTorch, it expects this (not sure why)
                                        # TODO CHECK WHY
    def kl_loss(self) -> torch.Tensor:
        """Compute KL loss with LinearFlipout layer."""
        n_w = self.flip.in_features * self.flip.out_features
        return n_w * self.flip.kl_loss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Make a forward pass with LinearFlipout layer."""
        return self.flip(x, return_kl=False)            # by default returns a tuple!
