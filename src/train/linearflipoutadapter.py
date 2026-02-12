import torch
from bayesian_torch.layers.flipout_layers import LinearFlipout
from torch import nn


class LinearFlipoutAdapter(nn.Module):
    """Thin wrapper around bayesian-torch `LinearFlipout`, MOPED-style (https://arxiv.org/abs/1906.05323).

    - Forces `forward() -> Tensor` (avoids tuple default).
    - prior mean and posterior mean initialized from pretrained weights.
    - bias currently not supported.

    Remark: the bayesian-torch implementation seems to contain mistakes.
    - `prior_variance` should be `prior_sigma`.
    - MOPED looks wrong
    """

    def __init__(self, pretrained_linear: nn.Linear, rho: float) -> None:
        """Create a LinearFlipoutAdapter."""
        super().__init__()
        if pretrained_linear.bias is not None:
            msg = "Bias currently not supported."
            raise ValueError(msg)

        prior_sigma = 1.0 #2 * torch.std(pretrained_linear.weight.data).item()
        self.flip = LinearFlipout(                  # prior mean and posterior mean overwritten with w_pretrained
            in_features=pretrained_linear.in_features,
            out_features=pretrained_linear.out_features,
            prior_variance=prior_sigma,             # set prior sigma. typo in library...

            posterior_rho_init=-3,
            bias=False,
        )

        # delta = 0.2

        with torch.no_grad():
            self.flip.mu_weight.copy_(pretrained_linear.weight)
            self.flip.prior_weight_mu.copy_(pretrained_linear.weight)
                            # numerically stable version of log(exp(delta |w| - 1))
            # rho_weight = torch.log(torch.expm1(torch.clamp(delta * pretrained_linear.weight.abs(), min=1e-6)))
            # self.flip.rho_weight.copy_(rho_weight)

            # target_sigma = torch.full_like(pretrained_linear.weight, prior_sigma)
            # rho = torch.log(torch.expm1(target_sigma))  # inverse softplus
            # self.flip.rho_weight.copy_(rho)

    @property
    def weight(self) -> torch.Tensor:
            # T5DenseActDense.forward() wants to check if this data member exists and is a tensor
        return self.flip.mu_weight

    def kl_loss(self) -> torch.Tensor:
        """Compute KL loss with LinearFlipout layer."""
        return self.flip.kl_loss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Make a forward pass with LinearFlipout layer."""
        return self.flip(x, return_kl=False)            # by default returns a tuple!
