export CUDA_VISIBLE_DEVICES=0

edge_remove_ratio_list=("0.05" "0.1")
# loss_type_list=("bce_loss" "new_loss")
loss_type_list=("new_loss")


for edge_remove_ratio in "${edge_remove_ratio_list[@]}"; do
  for loss_type in "${loss_type_list[@]}"; do

    # Loss option
    if [ "$loss_type" = "new_loss" ]; then
      reg_flag="-use_logit_reg"
    else
      reg_flag=""
    fi

    # Edge remove ratio option
    case "${edge_remove_ratio}" in
      0.05) save_iter=1000 ;;
      0.1) save_iter=500 ;;
      *) echo "Invalid edge_remove_ratio: ${edge_remove_ratio}"; exit 1 ;;
    esac

    model_name="disc_v3-${edge_remove_ratio}_multiple_${loss_type}_gnn"

    echo "============================================================"
    echo "Running: edge_remove_ratio=${edge_remove_ratio}, loss_type=${loss_type}"
    echo "model_name=${model_name}"
    echo "============================================================"

    python disc_train_multiple.py \
      -device "cuda" \
      -path "./sets_data" \
      -model_path "./sets_model" \
      -res_path "./sets_res" \
      -d_name "porto" \
      -method "seq" \
      -beta_lb 0.0001 \
      -beta_ub 100 \
      -max_T 100 \
      -gmm_comp 5 \
      -dims "[100, 20]" \
      -hidden_dim 20 \
      -n_epoch 1 \
      -bs 32 \
      -lr 0.0001 \
      -gmm_samples 100000 \
      -eval_num 2000 \
      -shortest_data_path "./porto_data" \
      -shortest_org_idx "v3-${edge_remove_ratio}_normal" \
      -shortest_new_idx "v3-${edge_remove_ratio}_except_0" \
      -save_step 1 \
      -save_iter "${save_iter}" \
      -beta_schedule front \
      -model_name "${model_name}" \
      -use_gnn \
      ${reg_flag}

  done
done