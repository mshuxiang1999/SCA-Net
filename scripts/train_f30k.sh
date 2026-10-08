#!/usr/bin/env bash
set -euo pipefail

python train.py \
  --data-root data/f30k_precomp \
  --train-artifacts artifacts/f30k_train.npz \
  --val-artifacts artifacts/f30k_dev.npz \
  --bert-model bert-base-uncased \
  --output-dir runs/f30k \
  --epochs 20 \
  --batch-size 128
