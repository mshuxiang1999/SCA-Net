from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer

from sca.artifacts import SemanticArtifacts
from sca.data import PrecomputedRetrievalDataset, make_loader
from sca.evaluation import encode_split, retrieval_metrics, score_split
from sca.losses import HardestTripletRankingLoss
from sca.model import SCANet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-artifacts", required=True)
    parser.add_argument("--val-artifacts", required=True)
    parser.add_argument("--bert-model", default="bert-base-uncased")
    parser.add_argument("--output-dir", default="runs/sca")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--bert-learning-rate", type=float, default=5e-5)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--embed-dim", type=int, default=1024)
    parser.add_argument("--image-dim", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-shard-size", type=int, default=64)
    return parser.parse_args()


def move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    train_artifacts = SemanticArtifacts.load(args.train_artifacts)
    val_artifacts = SemanticArtifacts.load(args.val_artifacts)
    train_data = PrecomputedRetrievalDataset(args.data_root, "train", tokenizer, train_artifacts)
    val_data = PrecomputedRetrievalDataset(args.data_root, "dev", tokenizer, val_artifacts)
    train_loader = make_loader(train_data, args.batch_size, True, args.workers)
    val_loader = make_loader(val_data, args.batch_size, False, args.workers)

    model = SCANet(
        train_artifacts, args.bert_model, image_dim=args.image_dim, embed_dim=args.embed_dim,
        top_prototypes=3, lambda_initial=0.5, lambda_instance=0.1, lambda_fragment=0.1,
    ).to(device)
    bert_parameters = list(model.text_encoder.bert.parameters())
    bert_ids = {id(parameter) for parameter in bert_parameters}
    other_parameters = [parameter for parameter in model.parameters() if id(parameter) not in bert_ids]
    optimizer = torch.optim.Adam([
        {"params": other_parameters, "lr": args.learning_rate},
        {"params": bert_parameters, "lr": args.bert_learning_rate},
    ])
    criterion = HardestTripletRankingLoss(args.margin)
    best_rsum = -1.0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            scores = model(batch)
            loss = criterion(scores)
            loss.backward()
            clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            running_loss += float(loss.detach())

        encoded = encode_split(model, val_loader, device)
        scores = score_split(model, encoded, device, args.eval_shard_size)
        metrics = retrieval_metrics(scores, val_data.captions_per_image)
        record = {"epoch": epoch + 1, "loss": running_loss / max(len(train_loader), 1), **metrics}
        print(json.dumps(record, sort_keys=True))
        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "config": vars(args),
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if metrics["rsum"] > best_rsum:
            best_rsum = metrics["rsum"]
            torch.save(checkpoint, output_dir / "best.pt")


if __name__ == "__main__":
    main()
