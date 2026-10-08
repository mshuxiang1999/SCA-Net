import numpy as np

from preprocess.build_semantics import idf_key_dimensions


def test_algorithm_1_counts_top_dimensions_per_assigned_document():
    prototypes = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    documents = np.array([[9.0, 8.0, 1.0, 0.0], [7.0, 6.0, 1.0, 0.0], [0.0, 1.0, 8.0, 9.0]])
    theta = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]])
    keys = idf_key_dimensions(prototypes, documents, theta, per_document_top_k=2, dimensions_per_topic=2)
    assert set(keys[0]) == {0, 1}
    assert set(keys[1]) == {2, 3}
