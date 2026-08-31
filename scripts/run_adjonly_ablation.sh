#!/bin/bash
# Three-arm ablation: adj-only / IW-only / adj+IW, under BOTH reveal orders.
#
#   adj-only : Lemma-3 adjacency-constrained proposals, NO discriminator
#              (plan_guided(disc=None, adj_prop=True))  -> isolates the mask
#   IW-only  : discriminator reweighting, masking OFF                 (modelD)
#   adj+IW   : both                                              (adj+modelD)
# base is produced by the same run as the unguided reference.
#
# adj-only needs no discriminator, so there is no seen/unseen split for that
# arm; the e0 disc is passed only because the other arms in the same process
# need one, and the adj-only arm ignores it. 7 blocks x 2 orders = 14 jobs.
#
# Tags: abl{B}fh (first_hit) / abl{B}l2r  -- each writes 4 cfg rows:
#       base, modelD, adj+modelD, adjonly
set -u
cd /home/aailab/wp03052/Synthetic-Data/Block-Diffusion-OD-Planning
source /home/aailab/wp03052/venvs/od_planning_venv/bin/activate
D=sets_disc; M=sets_model; R=sets_res; L=sets_log
mkdir -p $L
CK(){ echo $M/BD_porto_v3_normal_mask_blk$1_v4_bd.pth; }
FAM="${FAM:-0.05}"
F01=""; [ "$FAM" = "0.1" ] && F01="f01"
OUT=va_table_abl_f005.json; [ "$FAM" = "0.1" ] && OUT=va_table_abl_f01.json

echo "=== adj-only ablation: 14 jobs  fam=$FAM ($(date)) ==="
JOBS=()
for B in 1 2 4 8 16 32 64; do
  for O in first_hit l2r; do
    T=fh; [ "$O" = "l2r" ] && T=l2r
    JOBS+=("python -u tools/eval/three_way_postproc.py -family $FAM -adjonly 1 -order $O -ckpt $(CK $B) \
      -disc $D/BDdisc_f${FAM}_p1_e0_model_blk${B}_v4.pth -tag abl${F01}${B}${T}")
  done
done
for g in 0 1 2 3; do
  ( for i in "${!JOBS[@]}"; do
      if (( i % 4 == g )); then
        echo "[gpu$g] job $i: ${JOBS[$i]}"
        CUDA_VISIBLE_DEVICES=$g ${JOBS[$i]} > $L/abl${F01}_job$i.log 2>&1 || echo "ABL JOB $i FAILED"
      fi
    done ) &
done
wait
echo "=== evals done ($(date)) ==="

# collect: reuse collect_va_table's reconstruction, one call per order.
python -u tools/collect/collect_va_table.py -family $FAM -pairs "fh:abl${F01}{B}fh,l2r:abl${F01}{B}l2r" \
  -cfgs base,modelD,adj+modelD,adjonly -out $OUT \
  > $R/${OUT%.json}.log 2>&1 || echo "COLLECT FAILED"
tail -60 $R/${OUT%.json}.log

echo ABL_DONE > $L/abl${F01}_done.marker
echo "ABL_DONE ($(date))"
