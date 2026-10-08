#!/usr/bin/env bash
set -euo pipefail

python evaluate.py \
  --checkpoint runs/f30k/best.pt \
  --data-root data/f30k_precomp \
  --artifacts artifacts/f30k_test.npz \
  --split test
