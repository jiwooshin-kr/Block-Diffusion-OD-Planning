#!/bin/bash
# v2 rigorous retraining: seeded split (shuffle=False + seed 1), 14 models
# (mask x7, graph d=2.0 x7), then controlled held-out evaluation, both kernels.
set -u
cd "$(dirname "$0")"
mkdir -p sets_log
echo "=== v2 retraining ($(date)) ==="
(bash train_porto_bd.sh mask 1  0 v2 2>&1 | tee sets_log/v2_mask1.log
 bash train_porto_bd.sh mask 16 0 v2 2>&1 | tee sets_log/v2_mask16.log
 bash train_porto_bd.sh graph 4  0 v2 -bd_eos_deg 2.0 2>&1 | tee sets_log/v2_graph4.log
 bash train_porto_bd.sh graph 64 0 v2 -bd_eos_deg 2.0 2>&1 | tee sets_log/v2_graph64.log) &
(bash train_porto_bd.sh mask 2  1 v2 2>&1 | tee sets_log/v2_mask2.log
 bash train_porto_bd.sh mask 32 1 v2 2>&1 | tee sets_log/v2_mask32.log
 bash train_porto_bd.sh graph 8  1 v2 -bd_eos_deg 2.0 2>&1 | tee sets_log/v2_graph8.log
 bash train_porto_bd.sh graph 1  1 v2 -bd_eos_deg 2.0 2>&1 | tee sets_log/v2_graph1.log) &
(bash train_porto_bd.sh mask 4  2 v2 2>&1 | tee sets_log/v2_mask4.log
 bash train_porto_bd.sh mask 64 2 v2 2>&1 | tee sets_log/v2_mask64.log
 bash train_porto_bd.sh graph 16 2 v2 -bd_eos_deg 2.0 2>&1 | tee sets_log/v2_graph16.log) &
(bash train_porto_bd.sh mask 8  3 v2 2>&1 | tee sets_log/v2_mask8.log
 bash train_porto_bd.sh graph 2  3 v2 -bd_eos_deg 2.0 2>&1 | tee sets_log/v2_graph2.log
 bash train_porto_bd.sh graph 32 3 v2 -bd_eos_deg 2.0 2>&1 | tee sets_log/v2_graph32.log) &
wait
echo "=== v2 controlled evaluation ($(date)) ==="
(CUDA_VISIBLE_DEVICES=0 python -u eval_bd_sweep_controlled.py -kernel mask  2>&1 | tee sets_log/v2_ctrl_mask.log) &
(CUDA_VISIBLE_DEVICES=1 python -u eval_bd_sweep_controlled.py -kernel graph 2>&1 | tee sets_log/v2_ctrl_graph.log) &
wait
echo V2_ALL_DONE > sets_log/v2_done.marker
