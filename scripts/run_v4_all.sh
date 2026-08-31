#!/bin/bash
# v4 program: retrain ALL mask backbones with the EOS boundary fix
# (build_canvas: +1 END block when dst lands on a block boundary), then redo
# the full evaluation program on the new bases. Old v2 artifacts are left in
# place; every v4 artifact carries a distinct name (ckpt *_v4_bd.pth, discs
# *_v4.pth, pools *_v4_*, tags v4m*/v4f01m*).
#   A: train mask blk 1..64, suffix v4                       (~2h, 4 lanes)
#   B: conditional eval (EM/PC, held-out real pairs, 3 seeds) + uncond pools
#   C: uncond pool eval (EOS-fix verification incl. boundary residue)
#   D: discs -- fam 0.05 & 0.1 x {e0,e99} model per block + {e0,e99} data
#   E: three_way evals x28 (fam 0.05 + 0.1, seen + unseen, blk 1..64)
#   F: disc generalization eval x4 (model/data x fam)
# Run inside tmux. Writes sets_log/v4_done.marker at the end.
set -u
cd /home/aailab/wp03052/Synthetic-Data/Block-Diffusion-OD-Planning
source /home/aailab/wp03052/venvs/od_planning_venv/bin/activate
D=sets_disc; M=sets_model; R=sets_res; L=sets_log
mkdir -p $L
CK(){ echo $M/BD_porto_v3_normal_mask_blk$1_v4_bd.pth; }

echo "=== STAGE A: retrain mask backbones (EOS fix) ($(date)) ==="
(bash scripts/train_porto_bd.sh mask 1  0 v4 > $L/v4_train_mask1.log 2>&1
 bash scripts/train_porto_bd.sh mask 16 0 v4 > $L/v4_train_mask16.log 2>&1) &
(bash scripts/train_porto_bd.sh mask 2  1 v4 > $L/v4_train_mask2.log 2>&1
 bash scripts/train_porto_bd.sh mask 32 1 v4 > $L/v4_train_mask32.log 2>&1) &
(bash scripts/train_porto_bd.sh mask 4  2 v4 > $L/v4_train_mask4.log 2>&1
 bash scripts/train_porto_bd.sh mask 64 2 v4 > $L/v4_train_mask64.log 2>&1) &
(bash scripts/train_porto_bd.sh mask 8  3 v4 > $L/v4_train_mask8.log 2>&1) &
wait
echo "=== STAGE A done ($(date)) ==="
ls $M/BD_porto_v3_normal_mask_blk*_v4_bd.pth || { echo "STAGE A MISSING CKPTS - ABORT"; exit 1; }

echo "=== STAGE B: conditional eval + pools ($(date)) ==="
(CUDA_VISIBLE_DEVICES=0 python -u tools/eval/eval_bd_sweep_controlled.py -kernel mask -suffix v4 \
   > $L/v4_ctrl_mask.log 2>&1 || echo "ctrl eval FAIL") &
(CUDA_VISIBLE_DEVICES=1 python -u tools/gen/gen_bd_uncond_pool.py -ckpt $(CK 1)  -out $D/uncond_pool_v4_blk1.pth  -n 20000 > $L/v4_pool1.log 2>&1
 CUDA_VISIBLE_DEVICES=1 python -u tools/gen/gen_bd_uncond_pool.py -ckpt $(CK 2)  -out $D/uncond_pool_v4_blk2.pth  -n 20000 > $L/v4_pool2.log 2>&1
 CUDA_VISIBLE_DEVICES=1 python -u tools/gen/gen_bd_uncond_pool.py -ckpt $(CK 4)  -out $D/uncond_pool_v4_blk4.pth  -n 20000 > $L/v4_pool4.log 2>&1) &
(CUDA_VISIBLE_DEVICES=2 python -u tools/gen/gen_bd_uncond_pool.py -ckpt $(CK 8)  -out $D/uncond_pool_v4_blk8.pth  -n 20000 > $L/v4_pool8.log 2>&1
 CUDA_VISIBLE_DEVICES=2 python -u tools/gen/gen_bd_uncond_pool.py -ckpt $(CK 16) -out $D/uncond_pool_v4_blk16.pth -n 20000 > $L/v4_pool16.log 2>&1
 CUDA_VISIBLE_DEVICES=2 python -u tools/gen/gen_bd_uncond_pool.py -ckpt $(CK 32) -out $D/uncond_pool_v4_blk32.pth -n 20000 > $L/v4_pool32.log 2>&1) &
(CUDA_VISIBLE_DEVICES=3 python -u tools/gen/gen_bd_uncond_pool.py -ckpt $(CK 64) -out $D/uncond_pool_v4_blk64.pth -n 20000 > $L/v4_pool64.log 2>&1
 for B in 1 2 4 8 16 32 64; do
   CUDA_VISIBLE_DEVICES=3 python -u tools/gen/gen_bd_uncond_pool.py -ckpt $(CK $B) \
     -out $D/geneval_pool_v4_blk$B.pth -n 5000 -seed 99 > $L/v4_gpool$B.log 2>&1
 done) &
wait
echo "=== STAGE B done ($(date)) ==="

echo "=== STAGE C: uncond pool eval (EOS-fix verification) ($(date)) ==="
python -u tools/eval/eval_uncond_pools.py -pool_pat "./sets_disc/uncond_pool_v4_blk{B}.pth" \
  -out_name uncond_pool_eval_v4.json > $R/uncond_pool_eval_v4.log 2>&1 || echo "uncond eval FAIL"
tail -9 $R/uncond_pool_eval_v4.log

echo "=== STAGE D: discriminators ($(date)) ==="
( for F in 0.05 0.1; do for B in 1 2 4 8; do
    CUDA_VISIBLE_DEVICES=0 python -u tools/train/train_bd_disc.py -family $F -frac 1 -exp e0 -neg model \
      -pool $D/uncond_pool_v4_blk$B.pth -outsfx _v4 > $L/v4_disc_e0_f${F}_blk$B.log 2>&1 || echo "e0 f$F blk$B FAIL"
  done; done ) &
( for F in 0.05 0.1; do for B in 16 32 64; do
    CUDA_VISIBLE_DEVICES=1 python -u tools/train/train_bd_disc.py -family $F -frac 1 -exp e0 -neg model \
      -pool $D/uncond_pool_v4_blk$B.pth -outsfx _v4 > $L/v4_disc_e0_f${F}_blk$B.log 2>&1 || echo "e0 f$F blk$B FAIL"
  done; done
  CUDA_VISIBLE_DEVICES=1 python -u tools/train/train_bd_disc.py -family 0.05 -frac 1 -exp e0 -neg data -outsfx _v4 \
    > $L/v4_disc_e0data_f005.log 2>&1 || echo "e0 data f0.05 FAIL"
  CUDA_VISIBLE_DEVICES=1 python -u tools/train/train_bd_disc.py -family 0.1 -frac 1 -exp e0 -neg data -outsfx _v4 \
    > $L/v4_disc_e0data_f01.log 2>&1 || echo "e0 data f0.1 FAIL" ) &
JOBSPEC=""
for B in 1 2 4 8 16 32 64; do JOBSPEC+="$B:$D/uncond_pool_v4_blk$B.pth:_v4,"; done
JOBSPEC=${JOBSPEC%,}
( CUDA_VISIBLE_DEVICES=2 python -u tools/train/train_e99_model_multi.py -family 0.05 -jobs "$JOBSPEC" \
    > $L/v4_disc_e99_f005.log 2>&1 || echo "e99 multi f0.05 FAIL"
  CUDA_VISIBLE_DEVICES=2 python -u tools/train/train_bd_disc.py -family 0.05 -frac 1 -exp e99 -neg data -outsfx _v4 \
    > $L/v4_disc_e99data_f005.log 2>&1 || echo "e99 data f0.05 FAIL" ) &
( CUDA_VISIBLE_DEVICES=3 python -u tools/train/train_e99_model_multi.py -family 0.1 -jobs "$JOBSPEC" \
    > $L/v4_disc_e99_f01.log 2>&1 || echo "e99 multi f0.1 FAIL"
  CUDA_VISIBLE_DEVICES=3 python -u tools/train/train_bd_disc.py -family 0.1 -frac 1 -exp e99 -neg data -outsfx _v4 \
    > $L/v4_disc_e99data_f01.log 2>&1 || echo "e99 data f0.1 FAIL" ) &
wait
echo "=== STAGE D done ($(date)) ==="
ls $D/BDdisc_f0.05_p1_e0_model_blk*_v4.pth $D/BDdisc_f0.1_p1_e99_model_blk*_v4.pth | wc -l

echo "=== STAGE E: three_way evals x28 ($(date)) ==="
JOBS=()
for B in 1 2 4 8 16 32 64; do
  JOBS+=("python -u tools/eval/three_way_postproc.py -family 0.05 -ckpt $(CK $B) -disc $D/BDdisc_f0.05_p1_e0_model_blk${B}_v4.pth  -tag v4m${B}seen")
  JOBS+=("python -u tools/eval/three_way_postproc.py -family 0.05 -ckpt $(CK $B) -disc $D/BDdisc_f0.05_p1_e99_model_blk${B}_v4.pth -tag v4m${B}uns")
  JOBS+=("python -u tools/eval/three_way_postproc.py -family 0.1  -ckpt $(CK $B) -disc $D/BDdisc_f0.1_p1_e0_model_blk${B}_v4.pth   -tag v4f01m${B}seen")
  JOBS+=("python -u tools/eval/three_way_postproc.py -family 0.1  -ckpt $(CK $B) -disc $D/BDdisc_f0.1_p1_e99_model_blk${B}_v4.pth  -tag v4f01m${B}uns")
done
for g in 0 1 2 3; do
  ( for i in "${!JOBS[@]}"; do
      if (( i % 4 == g )); then
        echo "[gpu$g] job $i: ${JOBS[$i]}"
        CUDA_VISIBLE_DEVICES=$g ${JOBS[$i]} > $L/v4_eval_job$i.log 2>&1 || echo "EVAL JOB $i FAILED"
      fi
    done ) &
done
wait
echo "=== STAGE E done ($(date)) ==="

echo "=== STAGE F: disc generalization evals ($(date)) ==="
(CUDA_VISIBLE_DEVICES=0 python -u tools/eval/disc_gen_eval.py -family 0.05 -sfx _v4 \
   -pool_pat "./sets_disc/geneval_pool_v4_blk{B}.pth" > $R/disc_gen_eval_f005_v4.log 2>&1 || echo "gen f0.05 FAIL") &
(CUDA_VISIBLE_DEVICES=1 python -u tools/eval/disc_gen_eval.py -family 0.1 -sfx _v4 \
   -pool_pat "./sets_disc/geneval_pool_v4_blk{B}.pth" > $R/disc_gen_eval_f01_v4.log 2>&1 || echo "gen f0.1 FAIL") &
(CUDA_VISIBLE_DEVICES=2 python -u tools/eval/disc_gen_eval.py -family 0.05 -neg data -sfx _v4 \
   > $R/disc_gen_eval_f005_data_v4.log 2>&1 || echo "gen data f0.05 FAIL") &
(CUDA_VISIBLE_DEVICES=3 python -u tools/eval/disc_gen_eval.py -family 0.1 -neg data -sfx _v4 \
   > $R/disc_gen_eval_f01_data_v4.log 2>&1 || echo "gen data f0.1 FAIL") &
wait
echo "=== STAGE F done ($(date)) ==="

echo V4_ALL_DONE > $L/v4_done.marker
echo "V4_ALL_DONE ($(date))"
