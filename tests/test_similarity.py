import torch

from sca.artifacts import SemanticArtifacts
from sca.losses import HardestTripletRankingLoss
from sca.similarity import SCASimilarity


def artifacts() -> SemanticArtifacts:
    return SemanticArtifacts(
        visual_concept_features=torch.eye(4)[:2],
        text_concept_features=torch.eye(4)[:2],
        visual_topic_concept=torch.eye(2),
        text_topic_concept=torch.eye(2),
        visual_doc_topic=torch.tensor([[0.8, 0.2], [0.1, 0.9]]),
        text_doc_topic=torch.tensor([[0.7, 0.3], [0.2, 0.8]]),
        visual_key_dims=torch.tensor([[0, 2], [1, 3]]),
        text_key_dims=torch.tensor([[0, 3], [1, 2]]),
        visual_concept_to_topic=torch.tensor([0, 1]),
        text_concept_to_topic=torch.tensor([0, 1]),
        text_vocab={"first": 0, "second": 1},
    )


def test_equation_2_uses_union_of_key_dimensions():
    module = SCASimilarity(artifacts(), top_prototypes=2)
    mask = module._union_mask(torch.tensor(0), torch.tensor(1))
    assert torch.equal(mask, torch.tensor([1.0, 1.0, 1.0, 0.0]))


def test_equation_11_applies_pair_weights_and_pair_count():
    module = SCASimilarity(artifacts(), top_prototypes=2)
    regions = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    words = regions.clone()
    value = module.instance_similarity(
        regions, words, torch.tensor([0.8, 0.2]), torch.tensor([0.7, 0.3])
    )
    # Every masked cosine is one; softmax weights each sum to one; Eq. (11) divides by 2*2.
    assert torch.allclose(value, torch.tensor(0.25), atol=1e-6)


def test_equation_14_is_used_in_the_only_forward_path():
    module = SCASimilarity(artifacts(), top_prototypes=2)
    images = torch.tensor([[[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]])
    texts = images.clone()
    parts = module(
        images, texts, torch.tensor([2]), torch.tensor([2]),
        torch.tensor([[0.8, 0.2]]), torch.tensor([[0.7, 0.3]]),
        torch.tensor([[0, 1]]), return_parts=True,
    )
    expected = 0.5 * parts["initial"] + 0.1 * parts["instance"] + 0.1 * parts["fragment"]
    assert torch.allclose(parts["final"], expected)


def test_equation_15_selects_hardest_negatives():
    scores = torch.tensor([[1.0, 0.9, 0.1], [0.2, 1.0, 0.8], [0.7, 0.2, 1.0]])
    loss = HardestTripletRankingLoss(margin=0.2)(scores)
    # Caption-side costs: .1, 0, 0; image-side costs: 0, .1, 0.
    assert torch.allclose(loss, torch.tensor(0.2), atol=1e-6)
