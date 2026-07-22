#!/bin/bash
# (1) blk1/2 base + IW on except_0 (seen, e0_data block-independent disc)
# (2) Unseen (e99_data block-independent disc) for blk 1,2,4,8,16,32,64
# three_way_postproc emits base / IW(disc) / adj+IW(disc) x raw/P1/P3/P1P3.
# base is disc-independent, so it doubles as the "unseen base" too.
set -u
cd /home/aailab/wp03052/Synthetic-Data/Block-Diffusion-OD-Planning
source /home/aailab/wp03052/syn_data/bin/activate
D=sets_disc; M=sets_model; R=sets_res
G=3
CK() { echo $M/BD_porto_v3_normal_mask_blk$1_v2_bd.pth; }

echo "=== (1) blk1/2 SEEN (e0_data) ==="
for B in 1 2; do
  CUDA_VISIBLE_DEVICES=$G python -u three_way_postproc.py -ckpt $(CK $B) \
    -disc $D/BDdisc_f0.05_p1_e0_data.pth -tag v3m${B}seen > $R/blk12_seen_m${B}.log 2>&1
  echo "  blk$B seen done"
done

echo "=== (2) UNSEEN (e99_data) blk 1,2,4,8,16,32,64 ==="
for B in 1 2 4 8 16 32 64; do
  CUDA_VISIBLE_DEVICES=$G python -u three_way_postproc.py -ckpt $(CK $B) \
    -disc $D/BDdisc_f0.05_p1_e99_data.pth -tag v3m${B}unseen > $R/unseen_m${B}.log 2>&1
  echo "  blk$B unseen done"
done

echo "=== RESULTS ==="
echo "--- SEEN (e0_data) blk1/2 ---"
grep -hE "_base_raw |_modelD_raw |adj\+modelD_raw |modelD_P1P3 |adj\+modelD_P1P3 " $R/blk12_seen_m*.log | sed 's/  */ /g'
echo "--- UNSEEN (e99_data) blk1..64 ---"
grep -hE "_base_raw |_modelD_raw |adj\+modelD_raw |modelD_P1P3 |adj\+modelD_P1P3 " $R/unseen_m*.log | sed 's/  */ /g'
echo BLK12_UNSEEN_ALL_DONE
