from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch


@dataclass
class SemanticArtifacts:
    visual_concept_features: torch.Tensor
    text_concept_features: torch.Tensor
    visual_topic_concept: torch.Tensor
    text_topic_concept: torch.Tensor
    visual_doc_topic: torch.Tensor
    text_doc_topic: torch.Tensor
    visual_key_dims: torch.Tensor
    text_key_dims: torch.Tensor
    visual_concept_to_topic: torch.Tensor
    text_concept_to_topic: torch.Tensor
    text_vocab: Dict[str, int]

    @classmethod
    def load(cls, path: str | Path, device: torch.device | str = "cpu") -> "SemanticArtifacts":
        data = np.load(path, allow_pickle=True)
        required = {
            "visual_concept_features", "text_concept_features",
            "visual_topic_concept", "text_topic_concept",
            "visual_doc_topic", "text_doc_topic",
            "visual_key_dims", "text_key_dims",
            "visual_concept_to_topic", "text_concept_to_topic",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Semantic artifact is missing arrays: {sorted(missing)}")

        def tensor(name: str, dtype: torch.dtype) -> torch.Tensor:
            return torch.as_tensor(data[name], dtype=dtype, device=device)

        vocab: Dict[str, int] = {}
        if "text_vocab_tokens" in data.files:
            vocab = {str(token): i for i, token in enumerate(data["text_vocab_tokens"].tolist())}

        return cls(
            visual_concept_features=tensor("visual_concept_features", torch.float32),
            text_concept_features=tensor("text_concept_features", torch.float32),
            visual_topic_concept=tensor("visual_topic_concept", torch.float32),
            text_topic_concept=tensor("text_topic_concept", torch.float32),
            visual_doc_topic=tensor("visual_doc_topic", torch.float32),
            text_doc_topic=tensor("text_doc_topic", torch.float32),
            visual_key_dims=tensor("visual_key_dims", torch.long),
            text_key_dims=tensor("text_key_dims", torch.long),
            visual_concept_to_topic=tensor("visual_concept_to_topic", torch.long),
            text_concept_to_topic=tensor("text_concept_to_topic", torch.long),
            text_vocab=vocab,
        )

    def to(self, device: torch.device | str) -> "SemanticArtifacts":
        values = {}
        for name, value in self.__dict__.items():
            values[name] = value.to(device) if torch.is_tensor(value) else value
        return SemanticArtifacts(**values)

    @property
    def embedding_dim(self) -> int:
        return int(self.visual_concept_features.shape[-1])

    def topic_ids_for_tokens(self, tokens: Iterable[str]) -> torch.Tensor:
        ids = []
        for token in tokens:
            concept_id = self.text_vocab.get(token.lower().removeprefix("##"), -1)
            topic_id = -1 if concept_id < 0 else int(self.text_concept_to_topic[concept_id])
            ids.append(topic_id)
        return torch.tensor(ids, dtype=torch.long)
