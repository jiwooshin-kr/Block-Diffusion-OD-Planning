# no plan gen
export CUDA_VISIBLE_DEVICES=0

# ==============================================================================
# Experiment Settings
# ==============================================================================
edge_remove_ratio="0.05"     # 0.05 or 0.1
lr="0.0005"                  # 0.0001, 0.0005, 0.001
bs="32"                      # 16 or 32

# ==============================================================================
# Train
# ==============================================================================
model_name="TEST_porto_v3-${edge_remove_ratio}_lr_${lr}_bs_${bs}_retrain"

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
    -eval_num 2000


# # ==============================================================================
# # Train every setting
# # ==============================================================================
# for ratio in 0.05 0.1
# do
#     for bs in 16 32
#     do
#         for lr in 0.0001 0.0005 0.001
#         do
#             model_name="porto_v3-${ratio}_lr_${lr}_bs_${bs}"

#             echo "Running ${model_name}"

#             python3 main.py \
#                 -device "default" \
#                 -path "./sets_data" \
#                 -model_path "./sets_model" \
#                 -res_path "./sets_res" \
#                 -d_name "porto" \
#                 -model_name "${model_name}" \
#                 -method "seq" \
#                 -shortest_data_path "./porto_data" \
#                 -shortest_org_idx "v3-${ratio}_normal" \
#                 -beta_lb 0.0001 \
#                 -beta_ub 10 \
#                 -max_T 100 \
#                 -gmm_comp 5 \
#                 -dims "[100, 120, 200]" \
#                 -hidden_dim 32 \
#                 -n_epoch 1 \
#                 -bs ${bs} \
#                 -lr ${lr} \
#                 -gmm_samples 100000 \
#                 -eval_num 2000
#         done
#     done
# done