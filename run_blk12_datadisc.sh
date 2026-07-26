#!/bin/bash
# blk1/2 only, normal-negative (data) discriminator.
# SEEN  : disc trained on except_0 vs normal real paths (e0_data)
# UNSEEN: disc trained on except_1..99 vs normal real paths (e99_data), zero-shot on except_0
# data-negative disc is block-independent -> one file per regime covers blk1 & blk2.
# three_way_postproc emits base / IW(disc) / adj+IW(disc) x raw/P1/P3/P1P3.
set -u
cd /home/aailab/wp03052/Synthetic-Data/Block-Diffusion-OD-Planning
source /home/aailab/wp03052/syn_data/bin/activate
D=sets_disc; M=sets_model; R=sets_res
G=0
CK() { echo $M/BD_porto_v3_normal_mask_blk$1_v2_bd.pth; }

echo "=== SEEN (e0_data) blk1/2 ==="
for B in 1 2; do
  CUDA_VISIBLE_DEVICES=$G python -u three_way_postproc.py -ckpt $(CK $B) \
    -disc $D/BDdisc_f0.05_p1_e0_data.pth -tag v3d${B}seen > $R/blk12_datadisc_seen_m${B}.log 2>&1
  echo "  blk$B seen done"
done

echo "=== UNSEEN (e99_data) blk1/2 ==="
for B in 1 2; do
  CUDA_VISIBLE_DEVICES=$G python -u three_way_postproc.py -ckpt $(CK $B) \
    -disc $D/BDdisc_f0.05_p1_e99_data.pth -tag v3d${B}unseen > $R/blk12_datadisc_unseen_m${B}.log 2>&1
  echo "  blk$B unseen done"
done

echo "=== RESULTS ==="
echo "--- SEEN (e0_data) blk1/2 ---"
grep -hE "_base_raw |_modelD_raw |adj\+modelD_raw |modelD_P1P3 |adj\+modelD_P1P3 " $R/blk12_datadisc_seen_m*.log | sed 's/  */ /g'
echo "--- UNSEEN (e99_data) blk1/2 ---"
grep -hE "_base_raw |_modelD_raw |adj\+modelD_raw |modelD_P1P3 |adj\+modelD_P1P3 " $R/blk12_datadisc_unseen_m*.log | sed 's/  */ /g'
echo BLK12_DATADISC_ALL_DONE
