from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .artifacts import SemanticArtifacts

_WORD_RE = re.compile(r"[A-Za-z]+")


def _paper_words(caption: str, vocab: set[str]) -> list[str]:
    words = [match.group(0).lower() for match in _WORD_RE.finditer(caption)]
    if vocab:
        return [word for word in words if word in vocab]
    return words


class PrecomputedRetrievalDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        tokenizer: Any,
        artifacts: SemanticArtifacts,
        max_words: int = 64,
    ) -> None:
        root = Path(root)
        self.images = np.load(root / f"{split}_ims.npy", mmap_mode="r")
        self.captions = (root / f"{split}_caps.txt").read_text(encoding="utf-8").splitlines()
        self.tokenizer = tokenizer
        self.artifacts = artifacts
        self.max_words = max_words
        self.vocab_words = set(artifacts.text_vocab)
        self.captions_per_image = len(self.captions) // len(self.images)
        if self.captions_per_image * len(self.images) != len(self.captions):
            raise ValueError("Caption count must be an integer multiple of image count")
        if len(artifacts.visual_doc_topic) != len(self.images):
            raise ValueError("visual_doc_topic rows do not match split image count")
        if len(artifacts.text_doc_topic) != len(self.captions):
            raise ValueError("text_doc_topic rows do not match split caption count")

    def __len__(self) -> int:
        return len(self.captions)

    def __getitem__(self, caption_id: int) -> dict[str, Any]:
        image_id = caption_id // self.captions_per_image
        filtered_caption = " ".join(_paper_words(self.captions[caption_id], self.vocab_words))
        if not filtered_caption:
            filtered_caption = self.captions[caption_id]
        encoded = self.tokenizer(
            filtered_caption,
            truncation=True,
            max_length=self.max_words,
            add_special_tokens=False,
        )
        tokens = self.tokenizer.convert_ids_to_tokens(encoded["input_ids"])
        return {
            "image": torch.from_numpy(np.asarray(self.images[image_id]).copy()).float(),
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "word_topic_ids": self.artifacts.topic_ids_for_tokens(tokens),
            "image_theta": self.artifacts.visual_doc_topic[image_id].cpu(),
            "text_theta": self.artifacts.text_doc_topic[caption_id].cpu(),
            "image_id": image_id,
            "caption_id": caption_id,
        }


def collate_batch(items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    images = [item["image"] for item in items]
    image_lengths = torch.tensor([len(image) for image in images], dtype=torch.long)
    input_ids = [item["input_ids"] for item in items]
    text_lengths = torch.tensor([len(ids) for ids in input_ids], dtype=torch.long)
    return {
        "images": pad_sequence(images, batch_first=True),
        "image_lengths": image_lengths,
        "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=0),
        "attention_mask": pad_sequence(
            [torch.ones_like(ids) for ids in input_ids], batch_first=True, padding_value=0
        ),
        "text_lengths": text_lengths,
        "word_topic_ids": pad_sequence(
            [item["word_topic_ids"] for item in items], batch_first=True, padding_value=-1
        ),
        "image_theta": torch.stack([item["image_theta"] for item in items]),
        "text_theta": torch.stack([item["text_theta"] for item in items]),
        "image_ids": torch.tensor([item["image_id"] for item in items], dtype=torch.long),
        "caption_ids": torch.tensor([item["caption_id"] for item in items], dtype=torch.long),
    }


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
        drop_last=shuffle,
    )
