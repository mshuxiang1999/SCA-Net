from __future__ import annotations

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def retrieval_metrics(scores: np.ndarray, captions_per_image: int = 5) -> dict[str, float]:
    n_images, n_captions = scores.shape
    if n_captions != n_images * captions_per_image:
        raise ValueError("Score matrix shape is inconsistent with captions_per_image")
    i2t_ranks = np.empty(n_images)
    for image_id in range(n_images):
        order = np.argsort(-scores[image_id])
        positives = range(image_id * captions_per_image, (image_id + 1) * captions_per_image)
        i2t_ranks[image_id] = min(np.where(order == positive)[0][0] for positive in positives)

    t2i_ranks = np.empty(n_captions)
    transposed = scores.T
    for caption_id in range(n_captions):
        order = np.argsort(-transposed[caption_id])
        t2i_ranks[caption_id] = np.where(order == caption_id // captions_per_image)[0][0]

    result = {}
    for prefix, ranks in (("i2t", i2t_ranks), ("t2i", t2i_ranks)):
        result[f"{prefix}_r1"] = float(100 * np.mean(ranks < 1))
        result[f"{prefix}_r5"] = float(100 * np.mean(ranks < 5))
        result[f"{prefix}_r10"] = float(100 * np.mean(ranks < 10))
        result[f"{prefix}_medr"] = float(np.median(ranks) + 1)
    result["rsum"] = sum(result[key] for key in (
        "i2t_r1", "i2t_r5", "i2t_r10", "t2i_r1", "t2i_r5", "t2i_r10"
    ))
    return result


@torch.no_grad()
def encode_split(model, loader, device: torch.device):
    model.eval()
    images: dict[int, tuple[torch.Tensor, int, torch.Tensor]] = {}
    texts: dict[int, tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]] = {}
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        encoded_images = model.encode_images(batch["images"].to(device)).cpu()
        encoded_text = model.encode_text(ids, mask).cpu()
        for row in range(len(batch["caption_ids"])):
            image_id = int(batch["image_ids"][row])
            caption_id = int(batch["caption_ids"][row])
            image_length = int(batch["image_lengths"][row])
            text_length = int(batch["text_lengths"][row])
            images.setdefault(image_id, (
                encoded_images[row, :image_length], image_length, batch["image_theta"][row]
            ))
            texts[caption_id] = (
                encoded_text[row, :text_length], text_length,
                batch["text_theta"][row], batch["word_topic_ids"][row, :text_length],
            )

    image_rows = [images[index] for index in sorted(images)]
    text_rows = [texts[index] for index in sorted(texts)]
    return {
        "images": pad_sequence([row[0] for row in image_rows], batch_first=True),
        "image_lengths": torch.tensor([row[1] for row in image_rows]),
        "image_theta": torch.stack([row[2] for row in image_rows]),
        "texts": pad_sequence([row[0] for row in text_rows], batch_first=True),
        "text_lengths": torch.tensor([row[1] for row in text_rows]),
        "text_theta": torch.stack([row[2] for row in text_rows]),
        "word_topic_ids": pad_sequence([row[3] for row in text_rows], batch_first=True, padding_value=-1),
    }


@torch.no_grad()
def score_split(model, encoded: dict[str, torch.Tensor], device: torch.device, shard_size: int = 64):
    model.eval()
    rows = []
    for image_start in range(0, len(encoded["images"]), shard_size):
        image_end = image_start + shard_size
        columns = []
        for text_start in range(0, len(encoded["texts"]), shard_size):
            text_end = text_start + shard_size
            columns.append(model.similarity(
                encoded["images"][image_start:image_end].to(device),
                encoded["texts"][text_start:text_end].to(device),
                encoded["image_lengths"][image_start:image_end].to(device),
                encoded["text_lengths"][text_start:text_end].to(device),
                encoded["image_theta"][image_start:image_end].to(device),
                encoded["text_theta"][text_start:text_end].to(device),
                encoded["word_topic_ids"][text_start:text_end].to(device),
            ).cpu())
        rows.append(torch.cat(columns, dim=1))
    return torch.cat(rows, dim=0).numpy()
