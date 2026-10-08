# SCA-Net

PyTorch implementation for **Semantic-aware Cross-modal Alignment via Adaptive Dimension Filtering and Expanding for Image-Text Retrieval**.

SCA-Net is an image-text retrieval model that aligns visual regions and textual words by modeling semantic prototypes at both fragment and instance levels. The method introduces two core modules:

- **Intra-modality Dimension Filtering (IDF)** selects prototype-specific key dimensions inside each modality.
- **Inter-modality Dimension Expanding (IDE)** builds cross-modal compact feature spaces by taking the union of key dimensions from visual and textual semantic prototypes.

The released code follows the equations, algorithms, and implementation settings of the paper.


## Requirements

```bash
conda create -n sca python=3.10
conda activate sca
pip install -r requirements.txt
```

Main dependencies:

```text
torch
transformers
numpy
scikit-learn
joblib
pytest
```

## Data Preparation

This repository uses the common precomputed-feature layout for image-text retrieval:

```text
data/f30k_precomp/
  train_caps.txt
  train_ims.npy
  dev_caps.txt
  dev_ims.npy
  test_caps.txt
  test_ims.npy
```

Each `*_ims.npy` file stores Faster R-CNN region features. Captions should be ordered by image. Flickr30K and MS-COCO normally contain five captions per image.

## Training

```bash
bash scripts/train_f30k.sh
```

The paper settings are used by default:

```text
feature dimension d = 1024
number of regions M = 36
optimizer = Adam
learning rate = 0.0005
batch size = 128
epochs = 20
margin gamma = 0.2
visual/textual prototypes = 100 / 100
LDA random seed = 42
LDA passes = 30
IDF top-k = 3
IDF retained dimensions tau = 100
selected instance-level prototypes = 3 / 3
lambda_1, lambda_2, lambda_3 = 0.5, 0.1, 0.1
```

## Evaluation

```bash
bash scripts/test_f30k.sh
```

The evaluation code reports standard image-text retrieval metrics including image-to-text and text-to-image Recall@1, Recall@5, Recall@10, and Rsum.

## Testing

```bash
pytest -q
```

The tests cover IDF key dimension selection, IDE union masks, instance-level aggregation, score fusion, and hardest-negative ranking loss.
