import torch
from torch import nn


class WassersteinLoss(nn.Module):
    """
    Wasserstein loss
    """
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_target):
        loss = - torch.mean(y_pred * y_target)
        return loss


class GradientPenalty(nn.Module):
    """
    Gradient Penalty
    """
    def __init__(self):
        super().__init__()

    def forward(self, inputs, outputs):
        grad = torch.autograd.grad(
            inputs=inputs,
            outputs=outputs,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
            retain_graph=True, )[0]
        grad_ = torch.norm(grad.view(grad.size(0), -1), p=2, dim=1)
        penalty = torch.mean((1. - grad_) ** 2)
        return penalty