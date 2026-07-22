#!/bin/bash
# IW guidance ON TOP of left-to-right reveal order, model-negative disc,
# except_0 (§4.1 seen setting). base / IW(modelD) / adj+IW, all block sizes.
# blk1/2 are order-invariant (block <=2 has no reveal-order freedom) -> base
# equals §4.1; not re-run.
set -u
cd /home/aailab/wp03052/Synthetic-Data/Block-Diffusion-OD-Planning
source /home/aailab/wp03052/syn_data/bin/activate
D=sets_disc; M=sets_model; R=sets_res
CKm(){ echo $M/BD_porto_v3_normal_mask_blk$1_v2_bd.pth; }
GPUS=(0 1 2 3); i=0
for B in 4 8 16 32 64; do
  g=${GPUS[$((i % 4))]}; i=$((i+1))
  CUDA_VISIBLE_DEVICES=$g python -u three_way_postproc.py -ckpt $(CKm $B) \
    -disc $D/BDdisc_f0.05_p1_e0_model_blk${B}.pth -order l2r -tag v3m${B}l2r \
    > $R/l2riw_m${B}.log 2>&1 &
  # keep at most 4 concurrent
  if [ $((i % 4)) -eq 0 ]; then wait; fi
done
wait
echo "=== L2R+IW results (raw + P1P3) ==="
for B in 4 8 16 32 64; do
  echo "-- blk$B --"
  grep -hE "l2r_(base|modelD|adj.modelD)_(raw|P1P3) " $R/l2riw_m$B.log | sed "s/  */ /g"
done
echo "L2R_IW_ALL_DONE"
