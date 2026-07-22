#!/bin/bash
# §4.1/§4.2 completion:
#  - §4.1/§4.2 blk1,2 base-only (base is scenario/disc-independent; unseen base = seen base)
#  - §4.2 UNSEEN model-negative discs for blk 4/8/16/32/64 (e99, negatives = model gen)
#  - imbalance control: blk16 with a 200k model-neg pool vs the standard 20k
set -u
cd /home/aailab/wp03052/Synthetic-Data/Block-Diffusion-OD-Planning
source /home/aailab/wp03052/syn_data/bin/activate
R=sets_res; D=sets_disc; M=sets_model
CKm(){ echo $M/BD_porto_v3_normal_mask_blk$1_v2_bd.pth; }

echo "=== STAGE1 (parallel): train 5 discs (GPU0) | gen 200k pool (GPU1) | base blk1,2 (GPU2) ==="
CUDA_VISIBLE_DEVICES=0 python -u train_e99_model_multi.py \
  -jobs 4:$D/uncond_pool_blk4.pth,8:$D/uncond_pool_blk8.pth,16:$D/uncond_pool_blk16.pth,32:$D/uncond_pool_blk32.pth,64:$D/uncond_pool_blk64.pth \
  > $R/train_e99_multi.log 2>&1 &
PT=$!
CUDA_VISIBLE_DEVICES=1 python -u gen_bd_uncond_pool.py -ckpt $(CKm 16) \
  -out $D/uncond_pool_blk16_200k.pth -n 200000 -batch 800 > $R/gen200k.log 2>&1 &
PG=$!
( for B in 1 2; do CUDA_VISIBLE_DEVICES=2 python -u base_only.py -ckpt $(CKm $B) -tag v3m${B} > $R/base_blk${B}.log 2>&1; done; echo BASE12_DONE ) &
PB=$!

wait $PT; echo "=== discs (5 blocks) trained ==="

echo "=== STAGE2 (parallel GPU0,2,3): unseen model-neg evals blk4/8/16/32/64 ==="
G=(0 2 3); i=0
for B in 4 8 16 32 64; do
  g=${G[$((i%3))]}; i=$((i+1))
  CUDA_VISIBLE_DEVICES=$g python -u three_way_postproc.py -ckpt $(CKm $B) \
    -disc $D/BDdisc_f0.05_p1_e99_model_blk${B}.pth -tag v3m${B}uns > $R/unseen_m${B}.log 2>&1 &
done

wait $PG; echo "=== 200k pool ready; training control disc on GPU1 (parallel with evals) ==="
CUDA_VISIBLE_DEVICES=1 python -u train_e99_model_multi.py \
  -jobs 16:$D/uncond_pool_blk16_200k.pth:_neg200k > $R/train_ctrl200k.log 2>&1
echo "=== control disc trained ==="

wait; echo "=== all unseen evals done ==="
echo "=== STAGE3: control eval (blk16, 200k model-neg) ==="
CUDA_VISIBLE_DEVICES=0 python -u three_way_postproc.py -ckpt $(CKm 16) \
  -disc $D/BDdisc_f0.05_p1_e99_model_blk16_neg200k.pth -tag v3m16uns200k > $R/unseen_m16_200k.log 2>&1

wait $PB
echo "PIPELINE_ALL_DONE"
