from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def idf_key_dimensions(
    prototype_features: np.ndarray,
    global_features: np.ndarray,
    doc_topic: np.ndarray,
    per_document_top_k: int,
    dimensions_per_topic: int,
) -> np.ndarray:
    assignments = doc_topic.argmax(axis=1)
    result = np.empty((prototype_features.shape[0], dimensions_per_topic), dtype=np.int64)
    for topic in range(prototype_features.shape[0]):
        documents = global_features[assignments == topic]
        counts = np.zeros(prototype_features.shape[1], dtype=np.int64)
        if len(documents):
            similarity_vectors = documents * prototype_features[topic][None, :]
            top_dims = np.argpartition(similarity_vectors, -per_document_top_k, axis=1)[:, -per_document_top_k:]
            np.add.at(counts, top_dims.reshape(-1), 1)
        order = np.lexsort((np.arange(len(counts)), -counts))
        result[topic] = order[:dimensions_per_topic]
    return result


def fit(args: argparse.Namespace) -> None:
    import joblib
    from sklearn.decomposition import LatentDirichletAllocation

    visual_counts = np.load(args.visual_counts)
    text_counts = np.load(args.text_counts)
    visual_concepts = np.load(args.visual_concept_features)
    text_concepts = np.load(args.text_concept_features)
    image_global = np.load(args.image_global_features)
    text_global = np.load(args.text_global_features)

    lda_kwargs = dict(
        n_components=args.num_topics,
        random_state=args.random_state,
        learning_method="batch",
        max_iter=args.lda_iterations,
    )
    visual_lda = LatentDirichletAllocation(**lda_kwargs).fit(visual_counts)
    text_lda = LatentDirichletAllocation(**lda_kwargs).fit(text_counts)
    visual_theta = visual_lda.transform(visual_counts)
    text_theta = text_lda.transform(text_counts)
    visual_phi = visual_lda.components_ / visual_lda.components_.sum(axis=1, keepdims=True)
    text_phi = text_lda.components_ / text_lda.components_.sum(axis=1, keepdims=True)
    visual_prototypes = visual_phi @ visual_concepts
    text_prototypes = text_phi @ text_concepts

    visual_keys = idf_key_dimensions(
        visual_prototypes, image_global, visual_theta, args.idf_top_k, args.key_dimensions
    )
    text_keys = idf_key_dimensions(
        text_prototypes, text_global, text_theta, args.idf_top_k, args.key_dimensions
    )
    vocab_tokens = []
    if args.text_vocab:
        vocab_data = json.loads(Path(args.text_vocab).read_text(encoding="utf-8"))
        if isinstance(vocab_data, dict):
            vocab_tokens = [token for token, _ in sorted(vocab_data.items(), key=lambda item: item[1])]
        else:
            vocab_tokens = list(vocab_data)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        visual_concept_features=visual_concepts,
        text_concept_features=text_concepts,
        visual_topic_concept=visual_phi,
        text_topic_concept=text_phi,
        visual_doc_topic=visual_theta,
        text_doc_topic=text_theta,
        visual_key_dims=visual_keys,
        text_key_dims=text_keys,
        visual_concept_to_topic=visual_phi.argmax(axis=0),
        text_concept_to_topic=text_phi.argmax(axis=0),
        text_vocab_tokens=np.asarray(vocab_tokens),
    )
    joblib.dump({"visual": visual_lda, "text": text_lda}, output.with_suffix(".joblib"))


def transform(args: argparse.Namespace) -> None:
    import joblib

    source = np.load(args.source_artifact, allow_pickle=True)
    models = joblib.load(args.lda_models)
    visual_theta = models["visual"].transform(np.load(args.visual_counts))
    text_theta = models["text"].transform(np.load(args.text_counts))
    payload = {name: source[name] for name in source.files}
    payload["visual_doc_topic"] = visual_theta
    payload["text_doc_topic"] = text_theta
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    fit_parser = sub.add_parser("fit")
    for name in (
        "visual_counts",
        "text_counts",
        "visual_concept_features",
        "text_concept_features",
        "image_global_features",
        "text_global_features",
        "output",
    ):
        fit_parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    fit_parser.add_argument("--text-vocab")
    fit_parser.add_argument("--num-topics", type=int, default=100)
    fit_parser.add_argument("--idf-top-k", type=int, default=3)
    fit_parser.add_argument("--key-dimensions", type=int, default=100)
    fit_parser.add_argument("--random-state", type=int, default=42)
    fit_parser.add_argument("--lda-iterations", type=int, default=30)
    fit_parser.set_defaults(function=fit)

    transform_parser = sub.add_parser("transform")
    for name in ("source_artifact", "lda_models", "visual_counts", "text_counts", "output"):
        transform_parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    transform_parser.set_defaults(function=transform)
    return root


if __name__ == "__main__":
    parsed = parser().parse_args()
    parsed.function(parsed)
