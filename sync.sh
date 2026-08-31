#!/bin/bash
# 로컬 -> 서버 코드 동기화. --delete 를 쓰므로 서버에만 존재하는 산출물/데이터
# 디렉터리는 반드시 아래 exclude 에 있어야 한다. 새 산출물 폴더를 만들 때마다
# 이 목록에 추가할 것 (누락 시 rsync 가 서버에서 지워버린다).
rsync -av --delete \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pt' \
  --exclude '*.pth' \
  --exclude 'sets_*' \
  --exclude 'figs' \
  --exclude 'porto_data' \
  --exclude 'porto_data_v6' \
  --exclude '.claude' \
  --exclude 'CLAUDE.md' \
  --exclude '*.pdf' \
  --exclude 'pdfs' \
  --exclude 'report_src' \
  --exclude '.DS_Store' \
  ./ wp03052@143.248.84.179:/home/aailab/wp03052/Synthetic-Data/Block-Diffusion-OD-Planning
