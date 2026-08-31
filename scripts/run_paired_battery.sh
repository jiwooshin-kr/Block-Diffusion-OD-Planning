#!/bin/bash
# ============================================================================
# 짝지은 검정 배터리 — 한 arm 쌍에 대해 4개 지표를 모두 돌린다.
#
#   usage: VARIANT=LE A=adj+modelD B=dcbg+adjp SFX=adjIW_vs_dcbgadjp \
#            bash scripts/run_paired_battery.sh
#          VARIANT=LE A=adj+modelD B=adjonly SFX=adjIW_vs_adjonly REGIMES=seen \
#            bash scripts/run_paired_battery.sh          # adjonly 는 unseen 레코드가 없다
#          VARIANT=LE A=adj+modelD B='dcbg+adjp@g3' SFX=adjIW_vs_dcbgadjp_g3 \
#            bash scripts/run_paired_battery.sh          # gamma 스윕 arm
#
#   METRICS 기본값 = "va em lcs_norm dtw".  DTW 가 부담되면 METRICS="va em lcs_norm".
#   출력: sets_v6/$VARIANT/res/paired_{metric}_{SFX}.json
#         (va 는 과거 이름 호환을 위해 paired_{SFX}.json 으로 쓴다)
#
#   왜: 두 arm 은 같은 1,000 OD 쌍에서 평가되므로 독립 표본이 아니다. 이진 지표는
#   정확 McNemar, 연속 지표(DTW/LCS)는 Wilcoxon signed-rank + 짝지은 부트스트랩.
#   v&a 는 마스킹만으로 validity 가 0.996 에 포화하므로 가이던스 기여를 못 보여준다 —
#   DTW/LCS 가 가장 민감하다(실측: blk2 에서 v&a p=0.113, EM p=0.095, DTW p=0.0076).
# ============================================================================
set -u
cd /home/aailab/wp03052/Synthetic-Data/Block-Diffusion-OD-Planning
source /home/aailab/wp03052/venvs/od_planning_venv/bin/activate

VARIANT="${VARIANT:-LE}"
A="${A:?arm A 를 지정할 것 (예: adj+modelD)}"
B="${B:?arm B 를 지정할 것 (예: dcbg+adjp)}"
SFX="${SFX:?출력 접미사를 지정할 것}"
METRICS="${METRICS:-va dtw_sum lcs}"
REGIMES="${REGIMES:-seen unseen}"
BLOCKS="${BLOCKS:-1,2,4,8,16,32,64}"
BOOT="${BOOT:-5000}"

FAM="${VARIANT}1.0-0.05"
S=sets_v6/$VARIANT/res

PAIRS=""
for R in $REGIMES; do
  case "$R" in
    seen)   PAIRS+="seen:v6${VARIANT}{B}seen," ;;
    unseen) PAIRS+="unseen:v6${VARIANT}{B}uns," ;;
  esac
done
PAIRS="${PAIRS%,}"

echo "########## paired battery  $VARIANT  A=$A  B=$B  metrics=[$METRICS]  ($(date)) ##########"
for M in $METRICS; do
  if [ "$M" = "va" ]; then OUT="paired_${SFX}.json"; else OUT="paired_${M}_${SFX}.json"; fi
  echo "----- metric=$M -> $S/$OUT"
  python -u tools/collect/paired_test.py \
    -res_path $S -porto ./porto_data_v6 -dver v6 -family $FAM \
    -blocks "$BLOCKS" -pairs "$PAIRS" -a "$A" -b "$B" \
    -metric "$M" -boot $BOOT -out "$OUT" || echo "PAIRED $M FAILED"
done
echo "########## paired battery done ($(date)) ##########"
