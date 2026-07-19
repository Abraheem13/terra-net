"""Exact GP over descriptors with ARD-RBF kernel (gpytorch).

Gives (i) a strong small-data baseline, (ii) native predictive variance for the
UQ comparison, (iii) learned ARD lengthscales = built-in feature relevance.
"""
from __future__ import annotations

import gpytorch
import torch


class DescriptorGP(gpytorch.models.ExactGP):
    def __init__(self, x, y, likelihood, ard_dims: int = 116):
        super().__init__(x, y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=ard_dims)
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def train_gp(x: torch.Tensor, y: torch.Tensor, iters: int = 300, lr: float = 0.05):
    lik = gpytorch.likelihoods.GaussianLikelihood()
    model = DescriptorGP(x, y, lik)
    model.train(); lik.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
    for _ in range(iters):
        opt.zero_grad()
        loss = -mll(model(x), y)
        loss.backward(); opt.step()
    model.eval(); lik.eval()
    return model, lik
