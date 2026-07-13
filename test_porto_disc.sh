export CUDA_VISIBLE_DEVICES=0

# Choose the experiment settings
loss_type="new_loss"          # "bce_loss" or "new_loss"
edge_remove_ratio=0.05        # 0.05 or 0.1

case "${edge_remove_ratio}" in
    0.1)
        iter_list=$(seq 500 500 2500)           # 500, 1000, 1500, 2000, 2500
        ;;
    0.05)
        iter_list=$(seq 1000 1000 4000)         # 1000, 2000, 3000, 4000
        ;;
    *)
        echo "Invalid edge_remove_ratio: ${edge_remove_ratio}. Please set it to either 0.05 or 0.1."
        exit 1
        ;;
esac

# ==============================================================================
# (1) Previous Discriminator
# ==============================================================================
for iter in $iter_list
do
    echo "Running disc iter = $iter"
    python disc_eval.py -device "cuda"  \
            -path "./sets_data" \
            -model_path "./sets_model" \
            -res_path "./sets_res" \
            -disc_path "./sets_disc" \
            -shortest_data_path "./porto_data" \
            -d_name "porto" \
            -method "seq" \
            -eval_num 2000 \
            -shortest_new_idx v3-${edge_remove_ratio}_except_0 \
            -save_name "disc_v3-${edge_remove_ratio}_except_0_${loss_type}_iter_${iter}" \
            -model_name porto_v3-${edge_remove_ratio}_lr_0.0005_bs_32_retrain_epoch_0 \
            -disc_name "disc_v3-${edge_remove_ratio}_except_0_${loss_type}_iter_${iter}" 
done

# ==============================================================================
# (2) GNN - Single Exceptional Case 
# ==============================================================================
for iter in $iter_list
do
    echo "Running disc iter = $iter"
    python disc_eval.py -device "cuda"  \
            -path "./sets_data" \
            -model_path "./sets_model" \
            -res_path "./sets_res" \
            -disc_path "./sets_disc" \
            -shortest_data_path "./porto_data" \
            -d_name "porto" \
            -method "seq" \
            -eval_num 2000 \
            -shortest_new_idx v3-${edge_remove_ratio}_except_0 \
            -save_name "disc_v3-${edge_remove_ratio}_except_0_${loss_type}_use_gnn_iter_${iter}" \
            -model_name porto_v3-${edge_remove_ratio}_lr_0.0005_bs_32_retrain_epoch_0 \
            -disc_name "disc_v3-${edge_remove_ratio}_except_0_${loss_type}_use_gnn_iter_${iter}" 
done

# ==============================================================================
# (3) GNN - Multiple Exceptional Cases
# ==============================================================================
for iter in $iter_list
do
    echo "Running disc iter = $iter"
    python disc_eval.py -device "cuda"  \
            -path "./sets_data" \
            -model_path "./sets_model" \
            -res_path "./sets_res" \
            -disc_path "./sets_disc" \
            -shortest_data_path "./porto_data" \
            -d_name "porto" \
            -method "seq" \
            -eval_num 2000 \
            -shortest_new_idx v3-${edge_remove_ratio}_except_0 \
            -save_name "disc_v3-${edge_remove_ratio}_multiple_${loss_type}_gnn_iter_${iter}" \
            -model_name porto_v3-${edge_remove_ratio}_lr_0.0005_bs_32_retrain_epoch_0 \
            -disc_name "disc_v3-${edge_remove_ratio}_multiple_${loss_type}_gnn_iter_${iter}" 
done