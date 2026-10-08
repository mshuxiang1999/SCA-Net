from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoTokenizer

from sca.artifacts import SemanticArtifacts
from sca.data import PrecomputedRetrievalDataset, make_loader
from sca.evaluation import encode_split, retrieval_metrics, score_split
from sca.model import SCANet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    artifacts = SemanticArtifacts.load(args.artifacts)
    tokenizer = AutoTokenizer.from_pretrained(config["bert_model"])
    dataset = PrecomputedRetrievalDataset(args.data_root, args.split, tokenizer, artifacts)
    loader = make_loader(dataset, args.batch_size, False, args.workers)
    model = SCANet(
        artifacts, config["bert_model"], image_dim=config["image_dim"], embed_dim=config["embed_dim"],
        top_prototypes=3, lambda_initial=0.5, lambda_instance=0.1, lambda_fragment=0.1,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    encoded = encode_split(model, loader, device)
    scores = score_split(model, encoded, device, args.shard_size)
    print(json.dumps(retrieval_metrics(scores, dataset.captions_per_image), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
