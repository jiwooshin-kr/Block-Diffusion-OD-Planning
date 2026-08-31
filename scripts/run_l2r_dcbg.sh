#!/bin/bash
# D-CBG (Schiff et al.) under LEFT-TO-RIGHT within-block reveal order, so the
# published baseline is measured under the same sampler that our IW method was
# measured under in scripts/run_l2r_v4.sh. Both variants:
#   exact : enumerate all V tokens at the reveal position (their _cbg_denoise)
#   fo    : first-order Taylor, one classifier fwd+bwd per reveal
# Everything else matches §5.1-§5.4 of the v4 report: same v4 backbones, same
# v4 adjacency-bounded model-negative classifiers, gamma = 1.0 (no tempering,
# matching the IW side's w_gamma = 1.0), same 1,000 reserved OD pairs, batch
# 100, seed 7, RAW scoring.
#
# Phase 1 = seen (e0 classifier), phase 2 = unseen (e99 -> except_0 zero-shot).
# The collector runs after EACH phase so partial results are always usable.
#
# Tags: DCBG_mask{B}_g1.0_adj[_fo]_v4l2r{uns}
set -u
cd /home/aailab/wp03052/Synthetic-Data/Block-Diffusion-OD-Planning
source /home/aailab/wp03052/venvs/od_planning_venv/bin/activate
D=sets_disc; M=sets_model; R=sets_res; L=sets_log
mkdir -p $L
CK(){ echo $M/BD_porto_v3_normal_mask_blk$1_v4_bd.pth; }
BLOCKS="${BLOCKS:-1 2 4 8 16 32 64}"
FAM="${FAM:-0.05}"
F01=""; [ "$FAM" = "0.1" ] && F01="f01"
OUT=dcbg_table_l2r_f005.json; [ "$FAM" = "0.1" ] && OUT=dcbg_table_l2r_f01.json
GAMMA="${GAMMA:-1.0}"
# gamma != 1 gets its own table/markers/logs so the gamma=1 record is never clobbered
GTAG=""; [ "$GAMMA" != "1.0" ] && { GTAG="_g${GAMMA}"; OUT="${OUT%.json}${GTAG}.json"; }
# ADJP=1 adds the Lemma-3 adjacency mask to the guided reveal (adj+D-CBG arm)
ADJP="${ADJP:-0}"
APTAG=""; [ "$ADJP" = "1" ] && { APTAG="_adjp"; OUT="${OUT%.json}${APTAG}.json"; }

run_phase () {           # $1 = regime (e0|e99), $2 = log/tag suffix
  local REG=$1 SFX=$2
  local E99="" ; [ "$REG" = "e99" ] && E99="_e99"
  JOBS=()
  # first-order first (cheap) so the fast half of the table lands early
  for V in fo exact; do
    AP=0; [ "$V" = "fo" ] && AP=1
    for B in $BLOCKS; do
      CLF=$D/DCBGclf_mask_blk${B}_f${FAM}_p1_adj${E99}_modelneg_v4.pth
      [ -f "$CLF" ] || { echo "MISSING CLF $CLF -- skipping"; continue; }
      JOBS+=("python -u tools/eval/eval_dcbg.py -kernel mask -blk $B -family $FAM -gamma $GAMMA -adj 1 \
        -approx $AP -adj_prop $ADJP -order l2r -ckpt $(CK $B) -clf $CLF -res_suffix _v4l2r${F01}${SFX}")
    done
  done
  echo "=== phase $REG: ${#JOBS[@]} jobs ($(date)) ==="
  for g in 0 1 2 3; do
    ( for i in "${!JOBS[@]}"; do
        if (( i % 4 == g )); then
          echo "[gpu$g] $REG job $i: ${JOBS[$i]}"
          CUDA_VISIBLE_DEVICES=$g ${JOBS[$i]} > $L/l2rdcbg${F01}${GTAG}${APTAG}_${REG}_job$i.log 2>&1 \
            || echo "L2R DCBG $REG JOB $i FAILED"
        fi
      done ) &
  done
  wait
  echo "=== phase $REG done ($(date)) ==="
}

run_phase e0 ""
python -u tools/collect/collect_dcbg_table.py -family $FAM -gamma $GAMMA -out $OUT \
  -sfx_seen ${APTAG}_v4l2r${F01} -sfx_uns ${APTAG}_v4l2r${F01}uns > $R/${OUT%.json}_seen.log 2>&1 || echo "COLLECT(seen) FAILED"
tail -20 $R/${OUT%.json}_seen.log
echo L2R_DCBG_SEEN_DONE > $L/l2r_dcbg${F01}${GTAG}${APTAG}_seen.marker

run_phase e99 "uns"
python -u tools/collect/collect_dcbg_table.py -family $FAM -gamma $GAMMA -out $OUT \
  -sfx_seen ${APTAG}_v4l2r${F01} -sfx_uns ${APTAG}_v4l2r${F01}uns > $R/${OUT%.json}.log 2>&1 || echo "COLLECT FAILED"
tail -40 $R/${OUT%.json}.log

echo L2R_DCBG_DONE > $L/l2r_dcbg${F01}${GTAG}${APTAG}_done.marker
echo "L2R_DCBG_DONE ($(date))"
