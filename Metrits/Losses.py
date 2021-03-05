import torch
from torch.nn import functional as F, Module


class DummyLoss:
    def __init__(self):
        pass

    def calc_loss(self, pred, y):  # placeholder
        loss = F.mse_loss(pred, y)
        return loss


class QuantileLoss(Module):
    '''source: https://medium.com/the-artificial-impostor/quantile-regression-part-2-6fdbc26b2629'''

    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        '''Predictions: tensor of shape (num_horizons, num_quantiles)'''
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(
                torch.max(
                    (q - 1) * errors,
                    q * errors
                ).unsqueeze(1))
        loss = torch.mean(
            torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss
