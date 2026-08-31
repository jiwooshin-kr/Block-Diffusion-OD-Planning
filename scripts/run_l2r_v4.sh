#!/bin/bash
# Reveal-ORDER ablation on the v4 guidance table (fam 0.05, except_0).
# Identical protocol to STAGE E of scripts/run_v4_all.sh -- same v4 backbones, same v4
# discriminators, same 1,000 reserved OD pairs, batch 100, seed 7, n_is 100 --
# with the ONLY change being the within-block reveal order:
#   first_hit : reveal a uniformly-chosen masked position each step (default)
#   l2r       : reveal the left-most masked position each step (deterministic)
# The first-hitting TIME schedule is unchanged in both; only the position
# choice differs. At blk1 the two coincide by construction (one position per
# block), which is the built-in control.
#   14 jobs = 7 block sizes x {seen (e0 disc), unseen (e99 disc)}, 4 GPU lanes
set -u
cd /home/aailab/wp03052/Synthetic-Data/Block-Diffusion-OD-Planning
source /home/aailab/wp03052/venvs/od_planning_venv/bin/activate
D=sets_disc; M=sets_model; R=sets_res; L=sets_log
mkdir -p $L
CK(){ echo $M/BD_porto_v3_normal_mask_blk$1_v4_bd.pth; }
FAM="${FAM:-0.05}"                      # edge-remove ratio
F01=""; [ "$FAM" = "0.1" ] && F01="f01" # tag / filename discriminator
OUT=va_table_l2r_f005.json; [ "$FAM" = "0.1" ] && OUT=va_table_l2r_f01.json

echo "=== l2r three_way evals x14  fam=$FAM ($(date)) ==="
JOBS=()
for B in 1 2 4 8 16 32 64; do
  JOBS+=("python -u tools/eval/three_way_postproc.py -family $FAM -order l2r -ckpt $(CK $B) -disc $D/BDdisc_f${FAM}_p1_e0_model_blk${B}_v4.pth  -tag l2r${F01}m${B}seen")
  JOBS+=("python -u tools/eval/three_way_postproc.py -family $FAM -order l2r -ckpt $(CK $B) -disc $D/BDdisc_f${FAM}_p1_e99_model_blk${B}_v4.pth -tag l2r${F01}m${B}uns")
done
for g in 0 1 2 3; do
  ( for i in "${!JOBS[@]}"; do
      if (( i % 4 == g )); then
        echo "[gpu$g] job $i: ${JOBS[$i]}"
        CUDA_VISIBLE_DEVICES=$g ${JOBS[$i]} > $L/l2r${F01}_eval_job$i.log 2>&1 || echo "L2R JOB $i FAILED"
      fi
    done ) &
done
wait
echo "=== evals done ($(date)) ==="

python -u tools/collect/collect_va_table.py -family $FAM -tag_seen "l2r${F01}m{B}seen" -tag_uns "l2r${F01}m{B}uns" \
  -out $OUT > $R/${OUT%.json}.log 2>&1 || echo "COLLECT FAILED"
tail -50 $R/${OUT%.json}.log

echo L2R_DONE > $L/l2r${F01}_done.marker
echo "L2R_DONE ($(date))"
