#!/bin/bash
# n_is = 300 replication of the section-4 guidance table (fam 0.05, except_0).
# Identical protocol to STAGE E of run_v4_all.sh -- same v4 backbones, same v4
# discriminators, same 1,000 reserved OD pairs, same batch/seed -- with the
# ONLY change being the importance-sample count 100 -> 300.
#   14 jobs = 7 block sizes x {seen (e0 disc), unseen (e99 disc)}, 4 GPU lanes
# Then assembles the 10-column table into sets_res/va_table_n300_f005.json.
set -u
cd /home/aailab/wp03052/Synthetic-Data/Block-Diffusion-OD-Planning
source /home/aailab/wp03052/syn_data/bin/activate
D=sets_disc; M=sets_model; R=sets_res; L=sets_log
mkdir -p $L
CK(){ echo $M/BD_porto_v3_normal_mask_blk$1_v4_bd.pth; }

echo "=== n_is=300 three_way evals x14 ($(date)) ==="
JOBS=()
for B in 1 2 4 8 16 32 64; do
  JOBS+=("python -u three_way_postproc.py -family 0.05 -n_is 300 -ckpt $(CK $B) -disc $D/BDdisc_f0.05_p1_e0_model_blk${B}_v4.pth  -tag n300m${B}seen")
  JOBS+=("python -u three_way_postproc.py -family 0.05 -n_is 300 -ckpt $(CK $B) -disc $D/BDdisc_f0.05_p1_e99_model_blk${B}_v4.pth -tag n300m${B}uns")
done
for g in 0 1 2 3; do
  ( for i in "${!JOBS[@]}"; do
      if (( i % 4 == g )); then
        echo "[gpu$g] job $i: ${JOBS[$i]}"
        CUDA_VISIBLE_DEVICES=$g ${JOBS[$i]} > $L/n300_eval_job$i.log 2>&1 || echo "N300 JOB $i FAILED"
      fi
    done ) &
done
wait
echo "=== evals done ($(date)) ==="

python -u collect_va_table.py -family 0.05 -tag_seen 'n300m{B}seen' -tag_uns 'n300m{B}uns' \
  -out va_table_n300_f005.json > $R/va_table_n300_f005.log 2>&1 || echo "COLLECT FAILED"
tail -50 $R/va_table_n300_f005.log

echo N300_DONE > $L/n300_done.marker
echo "N300_DONE ($(date))"
