#!/bin/bash
# 서버 -> 로컬 결과 회수 (보고서 빌드용). sync.sh 의 반대 방향이며 --delete 를 쓰지 않는다.
# 체크포인트(.pth)와 pool 은 크고 보고서에 필요 없으므로 제외한다.
#   usage: bash pull_v6.sh [LE|LP]   (기본 LE)
set -u
V="${1:-LE}"
SRV=wp03052@143.248.84.179
REMOTE=/home/aailab/wp03052/Synthetic-Data/Block-Diffusion-OD-Planning
rsync -av \
  --exclude '*.pth' --exclude '*.pt' \
  "$SRV:$REMOTE/sets_v6/$V/res/" "./sets_v6/$V/res/"
rsync -av \
  --include '*.log' --include '*.marker' --include '*.txt' --include '*/' --exclude '*' \
  "$SRV:$REMOTE/sets_v6/$V/log/" "./sets_v6/$V/log/"
echo "pulled sets_v6/$V/{res,log}"
