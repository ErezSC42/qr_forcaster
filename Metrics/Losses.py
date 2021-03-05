import torch
from torch.nn import Module


class QuantileLoss(Module):
    '''source: https://medium.com/the-artificial-impostor/quantile-regression-part-2-6fdbc26b2629'''

    def __init__(self, quantiles, device):
        super().__init__()
        self.quantiles = quantiles
        self.device = device

    def forward(self, preds, target):
        """preds: tensor of shape (batch, num_horizons, num_quantiles)
        target: tensor of shape (batch, num_horizons"""
        if self.device == "gpu":
            preds = preds.to("cuda")
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        preds = preds.to("cuda")
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, :, i]
            losses.append(
                torch.max(
                    (q - 1) * errors,
                    q * errors
                ).unsqueeze(1))
        loss = torch.mean(
            torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss
