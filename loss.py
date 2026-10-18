import torch
import torch.nn as nn

class MSEBCELoss(nn.Module):

    def __init__(self, alpha: float, tau: float, soft_targets: bool, wet_threshold: float = 0.03):
        super().__init__()
        self.alpha = alpha
        self.tau = tau
        self.wet_threshold = wet_threshold
        self.soft_targets = soft_targets
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_t):
        mse = self.mse(y_pred, y_t)
        logits = (y_pred - self.wet_threshold) / self.tau
        if self.soft_targets:
            target = torch.sigmoid((y_t - self.wet_threshold) / self.tau)
        else:
            target = (y_t > self.wet_threshold).float()
        bce = self.bce(logits, target)
        return mse + self.alpha * bce


class HuberBCELoss(nn.Module):

    def __init__(self, delta: float, alpha: float, tau: float, soft_targets: bool, wet_threshold: float = 0.03):
        super().__init__()
        self.alpha = alpha
        self.tau = tau
        self.wet_threshold = wet_threshold
        self.soft_targets = soft_targets
        self.huber = nn.HuberLoss(delta=delta)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, y_t):
        huber = self.huber(y_pred, y_t)
        logits = (y_pred - self.wet_threshold) / self.tau
        if self.soft_targets:
            target = torch.sigmoid((y_t - self.wet_threshold) / self.tau)
        else:
            target = (y_t > self.wet_threshold).float()
        bce = self.bce(logits, target)
        return huber + self.alpha * bce
    