import torch
from torch import nn


class HardestTripletRankingLoss(nn.Module):
    def __init__(self, margin: float = 0.2) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
            raise ValueError("Training scores must be a square paired batch matrix")
        positive = scores.diagonal()
        eye = torch.eye(scores.shape[0], dtype=torch.bool, device=scores.device)
        caption_cost = (self.margin - positive[:, None] + scores).masked_fill(eye, 0.0)
        image_cost = (self.margin - positive[None, :] + scores).masked_fill(eye, 0.0)
        return caption_cost.clamp_min(0).max(dim=1).values.sum() + image_cost.clamp_min(0).max(dim=0).values.sum()
