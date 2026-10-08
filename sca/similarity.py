from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .artifacts import SemanticArtifacts


def _cosine_matrix(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return F.normalize(left, dim=-1) @ F.normalize(right, dim=-1).transpose(-1, -2)


def _masked_cosine(left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(left * mask, right * mask, dim=-1, eps=1e-8)


@dataclass
class ScoreParts:
    initial: torch.Tensor
    instance: torch.Tensor
    fragment: torch.Tensor
    final: torch.Tensor


class SCASimilarity(nn.Module):
    def __init__(
        self,
        artifacts: SemanticArtifacts,
        top_prototypes: int = 3,
        lambda_initial: float = 0.5,
        lambda_instance: float = 0.1,
        lambda_fragment: float = 0.1,
        attention_temperature: float = 9.0,
    ) -> None:
        super().__init__()
        self.top_prototypes = top_prototypes
        self.lambda_initial = lambda_initial
        self.lambda_instance = lambda_instance
        self.lambda_fragment = lambda_fragment
        self.log_temperature = nn.Parameter(torch.tensor(attention_temperature).log())

        semantic_buffers = (
            "visual_concept_features", "text_concept_features",
            "visual_key_dims", "text_key_dims",
            "visual_concept_to_topic", "text_concept_to_topic",
        )
        for name in semantic_buffers:
            self.register_buffer(name, getattr(artifacts, name).clone())

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def _union_mask(self, visual_topic: torch.Tensor, text_topic: torch.Tensor) -> torch.Tensor:
        shape = torch.broadcast_shapes(visual_topic.shape, text_topic.shape)
        visual_topic = visual_topic.expand(shape)
        text_topic = text_topic.expand(shape)
        dim = self.visual_concept_features.shape[-1]
        mask = torch.zeros(*shape, dim, dtype=self.visual_concept_features.dtype, device=visual_topic.device)
        mask.scatter_(-1, self.visual_key_dims[visual_topic], 1.0)
        mask.scatter_(-1, self.text_key_dims[text_topic], 1.0)
        return mask

    def _region_topics(self, regions: torch.Tensor) -> torch.Tensor:
        concept_ids = _cosine_matrix(regions, self.visual_concept_features).argmax(dim=-1)
        return self.visual_concept_to_topic[concept_ids]

    @staticmethod
    def _word_topics(word_topic_ids: torch.Tensor, text_theta: torch.Tensor) -> torch.Tensor:
        fallback = text_theta.argmax().expand_as(word_topic_ids)
        return torch.where(word_topic_ids >= 0, word_topic_ids, fallback)

    def initial_similarity(self, regions: torch.Tensor, words: torch.Tensor) -> torch.Tensor:
        raw = _cosine_matrix(regions, words).clamp_min(0)
        normalized = F.normalize(raw, p=2, dim=1, eps=1e-8)
        beta = F.softmax(self.temperature * normalized, dim=0)
        visual_context = beta.transpose(0, 1) @ regions
        return F.cosine_similarity(words, visual_context, dim=-1, eps=1e-8).mean()

    def fragment_similarity(
        self,
        regions: torch.Tensor,
        words: torch.Tensor,
        word_topic_ids: torch.Tensor,
        text_theta: torch.Tensor,
    ) -> torch.Tensor:
        base = _cosine_matrix(regions, words)
        region_topics = self._region_topics(regions)
        word_topics = self._word_topics(word_topic_ids, text_theta)

        best_word = base.argmax(dim=1)
        best_region = base.argmax(dim=0)
        region_masks = self._union_mask(region_topics, word_topics[best_word])
        word_masks = self._union_mask(region_topics[best_region], word_topics)
        filtered_regions = regions * region_masks
        filtered_words = words * word_masks

        filtered_similarity = _cosine_matrix(filtered_regions, filtered_words).clamp_min(0)

        region_normalized = F.normalize(filtered_similarity, p=2, dim=0, eps=1e-8)
        alpha = F.softmax(self.temperature * region_normalized, dim=1)
        text_context = alpha @ filtered_words
        image_to_text = F.cosine_similarity(filtered_regions, text_context, dim=-1, eps=1e-8).mean()

        word_normalized = F.normalize(filtered_similarity, p=2, dim=1, eps=1e-8)
        beta = F.softmax(self.temperature * word_normalized, dim=0)
        image_context = beta.transpose(0, 1) @ filtered_regions
        text_to_image = F.cosine_similarity(filtered_words, image_context, dim=-1, eps=1e-8).mean()
        return 0.5 * (image_to_text + text_to_image)

    def instance_similarity(
        self,
        regions: torch.Tensor,
        words: torch.Tensor,
        image_theta: torch.Tensor,
        text_theta: torch.Tensor,
    ) -> torch.Tensor:
        image_global = regions.mean(dim=0)
        text_global = words.mean(dim=0)
        image_values, image_topics = image_theta.topk(self.top_prototypes)
        text_values, text_topics = text_theta.topk(self.top_prototypes)
        image_weights = F.softmax(image_values, dim=0)
        text_weights = F.softmax(text_values, dim=0)

        visual = image_topics[:, None].expand(-1, self.top_prototypes)
        textual = text_topics[None, :].expand(self.top_prototypes, -1)
        masks = self._union_mask(visual, textual)
        similarities = _masked_cosine(image_global.expand_as(masks), text_global.expand_as(masks), masks)
        pair_weights = image_weights[:, None] * text_weights[None, :]
        return (pair_weights * similarities).sum() / (self.top_prototypes ** 2)

    def pair(
        self,
        regions: torch.Tensor,
        words: torch.Tensor,
        image_theta: torch.Tensor,
        text_theta: torch.Tensor,
        word_topic_ids: torch.Tensor,
    ) -> ScoreParts:
        initial = self.initial_similarity(regions, words)
        instance = self.instance_similarity(regions, words, image_theta, text_theta)
        fragment = self.fragment_similarity(regions, words, word_topic_ids, text_theta)
        final = (
            self.lambda_initial * initial
            + self.lambda_instance * instance
            + self.lambda_fragment * fragment
        )
        return ScoreParts(initial, instance, fragment, final)

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        image_lengths: torch.Tensor,
        text_lengths: torch.Tensor,
        image_theta: torch.Tensor,
        text_theta: torch.Tensor,
        word_topic_ids: torch.Tensor,
        return_parts: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        rows = []
        initial_rows, instance_rows, fragment_rows = [], [], []
        for image_index in range(image_features.shape[0]):
            row, row_ini, row_ins, row_fra = [], [], [], []
            regions = image_features[image_index, : int(image_lengths[image_index])]
            for text_index in range(text_features.shape[0]):
                length = int(text_lengths[text_index])
                parts = self.pair(
                    regions,
                    text_features[text_index, :length],
                    image_theta[image_index],
                    text_theta[text_index],
                    word_topic_ids[text_index, :length],
                )
                row.append(parts.final)
                row_ini.append(parts.initial)
                row_ins.append(parts.instance)
                row_fra.append(parts.fragment)
            rows.append(torch.stack(row))
            initial_rows.append(torch.stack(row_ini))
            instance_rows.append(torch.stack(row_ins))
            fragment_rows.append(torch.stack(row_fra))
        final = torch.stack(rows)
        if not return_parts:
            return final
        return {
            "initial": torch.stack(initial_rows),
            "instance": torch.stack(instance_rows),
            "fragment": torch.stack(fragment_rows),
            "final": final,
        }
