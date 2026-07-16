#!/bin/bash
rsync -av --delete \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pt' \
  --exclude '*.pth' \
  --exclude 'sets_model' \
  --exclude 'sets_res' \
  --exclude 'sets_data' \
  --exclude 'sets_log' \
  --exclude 'sets_disc' \
  --exclude 'figs' \
  --exclude 'porto_data' \
  --exclude '.claude' \
  --exclude 'CLAUDE.md' \
  --exclude '*.pdf' \
  --exclude 'pdfs' \
  --exclude '.DS_Store' \
  ./ wp03052@143.248.84.179:/home/aailab/wp03052/Synthetic-Data/Block-Diffusion-OD-Planning
