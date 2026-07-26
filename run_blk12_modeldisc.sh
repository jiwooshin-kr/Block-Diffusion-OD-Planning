#!/bin/bash
# blk1/2 model-negative disc pipeline (to fill the model-negative side of §4.1/4.1b).
#   (1) uncond pool from the SAME v2 ckpt used at eval (p_theta match; matches run_v3_all.sh)
#   (2) train e0 (seen) + e99 (unseen) model-negative discs
#   (3) eval with three_way_postproc -> base / IW / adj+IW x raw/P1/P3/P1P3
# Runs blk1 on GPU0 and blk2 on GPU1 in parallel; each chain is sequential.
set -u
cd /home/aailab/wp03052/Synthetic-Data/Block-Diffusion-OD-Planning
source /home/aailab/wp03052/syn_data/bin/activate
D=sets_disc; M=sets_model; R=sets_res; L=sets_log
CK() { echo $M/BD_porto_v3_normal_mask_blk$1_v2_bd.pth; }

chain() {   # $1=block  $2=gpu
  B=$1; G=$2
  echo "[blk$B] (1) uncond pool ..."
  CUDA_VISIBLE_DEVICES=$G python -u gen_bd_uncond_pool.py \
    -ckpt $(CK $B) -out $D/uncond_pool_blk$B.pth -n 20000 > $L/blk12md_pool$B.log 2>&1 || { echo "[blk$B] pool FAIL"; return 1; }

  echo "[blk$B] (2a) e0 (seen) disc ..."
  CUDA_VISIBLE_DEVICES=$G python -u train_bd_disc.py -family 0.05 -frac 1 -exp e0 -neg model \
    -pool $D/uncond_pool_blk$B.pth > $L/blk12md_disc_e0_$B.log 2>&1 || { echo "[blk$B] e0 disc FAIL"; return 1; }

  echo "[blk$B] (2b) e99 (unseen) disc ..."
  CUDA_VISIBLE_DEVICES=$G python -u train_bd_disc.py -family 0.05 -frac 1 -exp e99 -neg model \
    -pool $D/uncond_pool_blk$B.pth > $L/blk12md_disc_e99_$B.log 2>&1 || { echo "[blk$B] e99 disc FAIL"; return 1; }

  echo "[blk$B] (3a) eval SEEN (e0 model) ..."
  CUDA_VISIBLE_DEVICES=$G python -u three_way_postproc.py -ckpt $(CK $B) \
    -disc $D/BDdisc_f0.05_p1_e0_model_blk$B.pth -tag v3mm${B}seen > $R/blk12md_seen_m$B.log 2>&1 || { echo "[blk$B] eval seen FAIL"; return 1; }

  echo "[blk$B] (3b) eval UNSEEN (e99 model) ..."
  CUDA_VISIBLE_DEVICES=$G python -u three_way_postproc.py -ckpt $(CK $B) \
    -disc $D/BDdisc_f0.05_p1_e99_model_blk$B.pth -tag v3mm${B}unseen > $R/blk12md_unseen_m$B.log 2>&1 || { echo "[blk$B] eval unseen FAIL"; return 1; }

  echo "[blk$B] DONE"
}

chain 1 0 &
chain 2 1 &
wait

echo "=== RESULTS (model-negative disc) ==="
echo "--- SEEN (e0 model) blk1/2 ---"
grep -hE "_base_raw |_modelD_raw |adj\+modelD_raw |modelD_P1P3 |adj\+modelD_P1P3 " $R/blk12md_seen_m*.log | sed 's/  */ /g'
echo "--- UNSEEN (e99 model) blk1/2 ---"
grep -hE "_base_raw |_modelD_raw |adj\+modelD_raw |modelD_P1P3 |adj\+modelD_P1P3 " $R/blk12md_unseen_m*.log | sed 's/  */ /g'
echo BLK12_MODELDISC_ALL_DONE
