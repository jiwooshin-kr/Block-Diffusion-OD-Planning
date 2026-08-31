#!/bin/bash
# fam 0.1 round (§4.1/§4.2 at edge-remove-ratio 0.1) + disc generalization
# diagnostic (both families). Designed for tmux; writes sets_log/fam01_done.marker
# at the end.
#   STAGE1 (parallel): GPU0/1 e0 (seen) discs f0.1 | GPU2 e99 (unseen) multi f0.1
#                      | GPU3 fresh seed-99 geneval pools + disc_gen_eval f0.05
#   STAGE2: 14 three_way evals f0.1 (base/IW/adj+IW x raw/P1/P3/P1P3), tags
#           f01m{B}seen / f01m{B}uns
#   STAGE3: disc_gen_eval f0.1 (needs the new discs)
set -u
cd /home/aailab/wp03052/Synthetic-Data/Block-Diffusion-OD-Planning
source /home/aailab/wp03052/venvs/od_planning_venv/bin/activate
D=sets_disc; M=sets_model; R=sets_res; L=sets_log
mkdir -p $L
CK(){ echo $M/BD_porto_v3_normal_mask_blk$1_v2_bd.pth; }

echo "=== STAGE1 start ($(date)) ==="
( for B in 1 2 4 8; do
    CUDA_VISIBLE_DEVICES=0 python -u tools/train/train_bd_disc.py -family 0.1 -frac 1 -exp e0 -neg model \
      -pool $D/uncond_pool_blk$B.pth > $L/f01_disc_e0_blk$B.log 2>&1 || echo "e0 blk$B FAIL"
  done ) &
( for B in 16 32 64; do
    CUDA_VISIBLE_DEVICES=1 python -u tools/train/train_bd_disc.py -family 0.1 -frac 1 -exp e0 -neg model \
      -pool $D/uncond_pool_blk$B.pth > $L/f01_disc_e0_blk$B.log 2>&1 || echo "e0 blk$B FAIL"
  done ) &
CUDA_VISIBLE_DEVICES=2 python -u tools/train/train_e99_model_multi.py -family 0.1 \
  -jobs 1:$D/uncond_pool_blk1.pth,2:$D/uncond_pool_blk2.pth,4:$D/uncond_pool_blk4.pth,8:$D/uncond_pool_blk8.pth,16:$D/uncond_pool_blk16.pth,32:$D/uncond_pool_blk32.pth,64:$D/uncond_pool_blk64.pth \
  > $L/f01_disc_e99_multi.log 2>&1 &
( for B in 1 2 4 8 16 32 64; do
    CUDA_VISIBLE_DEVICES=3 python -u tools/gen/gen_bd_uncond_pool.py -ckpt $(CK $B) \
      -out $D/geneval_pool_blk$B.pth -n 5000 -seed 99 > $L/geneval_pool_blk$B.log 2>&1 || echo "pool blk$B FAIL"
  done
  CUDA_VISIBLE_DEVICES=3 python -u tools/eval/disc_gen_eval.py -family 0.05 -pool_pat "$D/geneval_pool_blk{B}.pth" > $R/disc_gen_eval_f005.log 2>&1 || echo "disc_gen f0.05 FAIL"
) &
wait
echo "=== STAGE1 done ($(date)) ==="
ls $D/BDdisc_f0.1_p1_e0_model_blk*.pth $D/BDdisc_f0.1_p1_e99_model_blk*.pth

echo "=== STAGE2 start ($(date)) ==="
JOBS=()
for B in 1 2 4 8 16 32 64; do
  JOBS+=("python -u tools/eval/three_way_postproc.py -family 0.1 -ckpt $(CK $B) -disc $D/BDdisc_f0.1_p1_e0_model_blk$B.pth -tag f01m${B}seen")
  JOBS+=("python -u tools/eval/three_way_postproc.py -family 0.1 -ckpt $(CK $B) -disc $D/BDdisc_f0.1_p1_e99_model_blk$B.pth -tag f01m${B}uns")
done
for g in 0 1 2 3; do
  ( for i in "${!JOBS[@]}"; do
      if (( i % 4 == g )); then
        echo "[gpu$g] job $i: ${JOBS[$i]}"
        CUDA_VISIBLE_DEVICES=$g ${JOBS[$i]} > $L/f01_eval_job$i.log 2>&1 || echo "EVAL JOB $i FAILED"
      fi
    done ) &
done
wait
echo "=== STAGE2 done ($(date)) ==="

echo "=== STAGE3: disc_gen_eval f0.1 ==="
CUDA_VISIBLE_DEVICES=0 python -u tools/eval/disc_gen_eval.py -family 0.1 -pool_pat "$D/geneval_pool_blk{B}.pth" > $R/disc_gen_eval_f01.log 2>&1 || echo "disc_gen f0.1 FAIL"

echo FAM01_ALL_DONE > $L/fam01_done.marker
echo "FAM01_ALL_DONE ($(date))"
