from __future__ import annotations

import torch
from torch import nn

from .artifacts import SemanticArtifacts
from .encoders import BertWordEncoder, ImageRegionEncoder
from .similarity import SCASimilarity


class SCANet(nn.Module):
    def __init__(
        self,
        artifacts: SemanticArtifacts,
        bert_model: str,
        image_dim: int = 1024,
        embed_dim: int = 1024,
        **similarity_options,
    ) -> None:
        super().__init__()
        if artifacts.embedding_dim != embed_dim:
            raise ValueError(
                f"Artifact dimension {artifacts.embedding_dim} does not match embed_dim {embed_dim}"
            )
        self.image_encoder = ImageRegionEncoder(image_dim, embed_dim)
        self.text_encoder = BertWordEncoder(bert_model, embed_dim)
        self.similarity = SCASimilarity(artifacts, **similarity_options)

    def encode_images(self, regions: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(regions)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(input_ids, attention_mask)

    def forward(self, batch: dict[str, torch.Tensor], return_parts: bool = False):
        image_features = self.encode_images(batch["images"])
        text_features = self.encode_text(batch["input_ids"], batch["attention_mask"])
        return self.similarity(
            image_features=image_features,
            text_features=text_features,
            image_lengths=batch["image_lengths"],
            text_lengths=batch["text_lengths"],
            image_theta=batch["image_theta"],
            text_theta=batch["text_theta"],
            word_topic_ids=batch["word_topic_ids"],
            return_parts=return_parts,
        )
