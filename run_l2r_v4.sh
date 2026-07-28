#!/bin/bash
# Reveal-ORDER ablation on the v4 guidance table (fam 0.05, except_0).
# Identical protocol to STAGE E of run_v4_all.sh -- same v4 backbones, same v4
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
source /home/aailab/wp03052/syn_data/bin/activate
D=sets_disc; M=sets_model; R=sets_res; L=sets_log
mkdir -p $L
CK(){ echo $M/BD_porto_v3_normal_mask_blk$1_v4_bd.pth; }

echo "=== l2r three_way evals x14 ($(date)) ==="
JOBS=()
for B in 1 2 4 8 16 32 64; do
  JOBS+=("python -u three_way_postproc.py -family 0.05 -order l2r -ckpt $(CK $B) -disc $D/BDdisc_f0.05_p1_e0_model_blk${B}_v4.pth  -tag l2rm${B}seen")
  JOBS+=("python -u three_way_postproc.py -family 0.05 -order l2r -ckpt $(CK $B) -disc $D/BDdisc_f0.05_p1_e99_model_blk${B}_v4.pth -tag l2rm${B}uns")
done
for g in 0 1 2 3; do
  ( for i in "${!JOBS[@]}"; do
      if (( i % 4 == g )); then
        echo "[gpu$g] job $i: ${JOBS[$i]}"
        CUDA_VISIBLE_DEVICES=$g ${JOBS[$i]} > $L/l2r_eval_job$i.log 2>&1 || echo "L2R JOB $i FAILED"
      fi
    done ) &
done
wait
echo "=== evals done ($(date)) ==="

python -u collect_va_table.py -family 0.05 -tag_seen 'l2rm{B}seen' -tag_uns 'l2rm{B}uns' \
  -out va_table_l2r_f005.json > $R/va_table_l2r_f005.log 2>&1 || echo "COLLECT FAILED"
tail -50 $R/va_table_l2r_f005.log

echo L2R_DONE > $L/l2r_done.marker
echo "L2R_DONE ($(date))"
