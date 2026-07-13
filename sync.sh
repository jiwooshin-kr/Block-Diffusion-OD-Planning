#!/bin/bash
rsync -av --delete \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pt' \
  --exclude '*.pth' \
  --exclude 'sets_model' \
  --exclude 'sets_res' \
  --exclude 'sets_data' \
  --exclude 'porto_data' \
  --exclude '.claude' \
  --exclude 'CLAUDE.md' \
  --exclude '*.pdf' \
  --exclude '.DS_Store' \
  ./ wp03052@143.248.80.20:/home/aailab/data/wp03052/Synthetic-Data-DiffPlan/
