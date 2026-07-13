#!/bin/bash
# =============================================================================
# Diffusion-only OD planning (no autoregressive h(x|ori,dst))
#
#   Stage 1 (od_train): trains the OD-conditioned diffusion model (EPSM_OD /
#                       RestorerOD) with CFG condition dropout, plus the
#                       conditional length predictor p(L|ori,dst) and the
#                       GMM fallback. Saves  ./sets_model/${model_name}_od.pth
#   Stage 2 (auto)    : evaluates planning on held-out OD pairs with both
#                       oracle lengths and predicted lengths
#                       (Hit Ratio / DTW / LCS / length error).
#
#   To re-run evaluation only on a trained model, use the commented
#   "EVAL ONLY" block at the bottom.
# =============================================================================
export CUDA_VISIBLE_DEVICES=0

# ==============================================================================
# Experiment Settings
# ==============================================================================
# NOTE: index must match your file names, e.g. porto_shrink_A_v3-0.1_normal.ts
#       -> shortest_org_idx "v3-0.1_normal" (dots, matching porto_data/)
edge_remove_ratio="0.1"      # 0.05 or 0.1 (matches file naming)
lr="0.0005"                  # 0.0001, 0.0005, 0.001
bs="32"                      # 16 or 32
drop_cond="0.1"              # CFG condition dropout rate during training
cfg_scale="2.0"              # guidance scale at planning time (1.0 = CFG off)

model_name="OD_porto_v3-${edge_remove_ratio}_lr_${lr}_bs_${bs}_drop_${drop_cond}"

# ==============================================================================
# Train + Evaluate (diffusion-only OD planning)
# ==============================================================================
python3 main_od.py \
    -device "default" \
    -path "./sets_data" \
    -model_path "./sets_model" \
    -res_path "./sets_res" \
    -d_name "porto" \
    -model_name "${model_name}" \
    -method "od_train" \
    -shortest_data_path "./porto_data" \
    -shortest_org_idx "v3-${edge_remove_ratio}_normal" \
    -beta_lb 0.0001 \
    -beta_ub 10 \
    -max_T 100 \
    -gmm_comp 5 \
    -dims "[100, 120, 200]" \
    -hidden_dim 32 \
    -n_epoch 1 \
    -bs ${bs} \
    -lr ${lr} \
    -gmm_samples 100000 \
    -drop_cond ${drop_cond} \
    -guidance_scale ${cfg_scale} \
    -length_mode "both" \
    -len_epochs 20 \
    -od_max_len 100 \
    -batch_traj_num 200 \
    -eval_num 1000

# ==============================================================================
# EVAL ONLY (loads ./sets_model/${model_name}_od.pth)
# ==============================================================================
# python3 main_od.py \
#     -device "default" \
#     -path "./sets_data" \
#     -model_path "./sets_model" \
#     -res_path "./sets_res" \
#     -d_name "porto" \
#     -model_name "${model_name}" \
#     -method "od_plan" \
#     -shortest_data_path "./porto_data" \
#     -shortest_org_idx "v3-${edge_remove_ratio}_normal" \
#     -guidance_scale ${cfg_scale} \
#     -length_mode "both" \
#     -batch_traj_num 200 \
#     -eval_num 1000
