# O/D conditional diffusion training
# GPU 0 is often busy - check nvidia-smi and adjust if needed
export CUDA_VISIBLE_DEVICES=1

# ==============================================================================
# Experiment Settings
# ==============================================================================
edge_remove_ratio="0.05"     # 0.05 or 0.1
lr="0.0005"                  # 0.0001, 0.0005, 0.001
bs="32"                      # 16 or 32
od_dropout="0.1"             # independent dropout rate for O and D conditions

# ==============================================================================
# Train
# ==============================================================================
model_name="OD_porto_v3-${edge_remove_ratio}_lr_${lr}_bs_${bs}_oddrop_${od_dropout}"

python3 main.py \
    -device "default" \
    -path "./sets_data" \
    -model_path "./sets_model" \
    -res_path "./sets_res" \
    -d_name "porto" \
    -model_name "${model_name}" \
    -method "seq" \
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
    -eval_num 2000 \
    -od_cond \
    -od_dropout ${od_dropout}
