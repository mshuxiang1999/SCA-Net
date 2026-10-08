from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel


class ImageRegionEncoder(nn.Module):
    """Project Faster R-CNN region features to the paper's shared d-dimensional space."""

    def __init__(self, input_dim: int = 1024, embed_dim: int = 1024) -> None:
        super().__init__()
        self.projection = nn.Identity() if input_dim == embed_dim else nn.Linear(input_dim, embed_dim)

    def forward(self, regions: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(regions), dim=-1)


class BertWordEncoder(nn.Module):
    """BERT/Bi-GRU is interchangeable in the paper; this is the BERT setting."""

    def __init__(self, model_name_or_path: str, embed_dim: int = 1024) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name_or_path)
        self.projection = nn.Linear(self.bert.config.hidden_size, embed_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return F.normalize(self.projection(hidden), dim=-1)
