#!/bin/bash
# Fill in §4.4 arrival + unseen:
#  (1) seen  first_hit (e0 model-neg): to record IW/adj+IW arrival
#  (2) unseen l2r       (e99 model-neg): the unseen counterpart of §4.4
#  (3) blk1/2 base, both orders (order-invariance check on except_0)
set -u
cd /home/aailab/wp03052/Synthetic-Data/Block-Diffusion-OD-Planning
source /home/aailab/wp03052/syn_data/bin/activate
D=sets_disc; M=sets_model; R=sets_res
CKm(){ echo $M/BD_porto_v3_normal_mask_blk$1_v2_bd.pth; }

run3(){ # $1=gpu $2=blk $3=disc $4=order $5=tag
  CUDA_VISIBLE_DEVICES=$1 python -u three_way_postproc.py -ckpt $(CKm $2) \
    -disc $3 -order $4 -tag $5 > $R/$5.log 2>&1
}

# (1) seen first_hit  &  (2) unseen l2r  -> 10 runs across 4 GPUs
( run3 0 4  $D/BDdisc_f0.05_p1_e0_model_blk4.pth  first_hit v3m4sfh
  run3 0 32 $D/BDdisc_f0.05_p1_e0_model_blk32.pth first_hit v3m32sfh
  run3 0 64 $D/BDdisc_f0.05_p1_e99_model_blk64.pth l2r      v3m64unsl2r ) &
( run3 1 8  $D/BDdisc_f0.05_p1_e0_model_blk8.pth  first_hit v3m8sfh
  run3 1 64 $D/BDdisc_f0.05_p1_e0_model_blk64.pth first_hit v3m64sfh
  run3 1 4  $D/BDdisc_f0.05_p1_e99_model_blk4.pth  l2r      v3m4unsl2r ) &
( run3 2 16 $D/BDdisc_f0.05_p1_e0_model_blk16.pth first_hit v3m16sfh
  run3 2 8  $D/BDdisc_f0.05_p1_e99_model_blk8.pth  l2r      v3m8unsl2r ) &
( run3 3 16 $D/BDdisc_f0.05_p1_e99_model_blk16.pth l2r      v3m16unsl2r
  run3 3 32 $D/BDdisc_f0.05_p1_e99_model_blk32.pth l2r      v3m32unsl2r ) &
# (3) blk1/2 base, both orders (base is disc-independent) on GPU3 tail
( for B in 1 2; do for O in first_hit l2r; do
    CUDA_VISIBLE_DEVICES=3 python -u base_only.py -ckpt $(CKm $B) -order $O -tag v3m${B}base_${O} > $R/v3m${B}base_${O}.log 2>&1
  done; done ) &
wait

echo "=== SEEN first_hit (IW/adj+IW arrival) ==="
for B in 4 8 16 32 64; do grep -hE "v3m${B}sfh_(base|modelD|adj.modelD)_raw " $R/v3m${B}sfh.log | sed "s/  */ /g"; done
echo "=== UNSEEN l2r ==="
for B in 4 8 16 32 64; do grep -hE "v3m${B}unsl2r_(base|modelD|adj.modelD)_(raw|P1P3) " $R/v3m${B}unsl2r.log | sed "s/  */ /g"; done
echo "=== blk1/2 base both orders ==="
for B in 1 2; do for O in first_hit l2r; do grep -hE "base_raw |base_P1P3 " $R/v3m${B}base_${O}.log | sed "s/  */ /g"; done; done
echo "L2RIW_MORE_DONE"
