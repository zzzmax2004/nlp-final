import torch
import torch.nn as nn

class SpearmanLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        # avoid div by 0
        eps = 1e-8

        # normalize
        preds_diff = preds - preds.mean(dim=0, keepdim=True)
        targets_diff = targets - targets.mean(dim=0, keepdim=True)

        # calculate the covariance (numerator)
        cov = (preds_diff * targets_diff).sum(dim=0)

        # calculate variances
        preds_var = torch.sqrt((preds_diff ** 2).sum(dim=0) + eps)
        targets_var = torch.sqrt((targets_diff ** 2).sum(dim=0) + eps)

        # calculate Pearson Correlation
        pearson_corr = cov / (preds_var * targets_var)

        loss = 1 - pearson_corr.mean()

        return loss


