import numpy as np

from preprocess.build_semantics import idf_key_dimensions


def test_idf_key_dimensions_selects_frequent_top_dimensions():
    prototypes = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    documents = np.array([[5.0, 4.0, 0.0, 0.0], [4.0, 3.0, 0.0, 0.0], [0.0, 0.0, 7.0, 6.0]])
    theta = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]])
    keys = idf_key_dimensions(prototypes, documents, theta, per_document_top_k=2, dimensions_per_topic=2)
    assert np.array_equal(keys[0], np.array([0, 1]))
    assert np.array_equal(keys[1], np.array([2, 3]))
