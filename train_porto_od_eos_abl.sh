# O/D conditional <eos> diffusion - ablation runs
# usage: bash train_porto_od_eos_abl.sh <eos_deg> <gpu> <suffix> [extra flags...]
eos_deg=$1
gpu=$2
suffix=$3
shift 3
export CUDA_VISIBLE_DEVICES=${gpu}

model_name="OD_EOS_porto_v3_normal_d${eos_deg}_${suffix}"

python3 main.py \
    -device "default" \
    -path "./sets_data" \
    -model_path "./sets_model" \
    -res_path "./sets_res" \
    -d_name "porto" \
    -model_name "${model_name}" \
    -method "seq" \
    -shortest_data_path "./porto_data" \
    -shortest_org_idx "v3-0.05_normal" \
    -beta_lb 0.0001 \
    -beta_ub 10 \
    -max_T 100 \
    -gmm_comp 5 \
    -dims "[100, 120, 200]" \
    -hidden_dim 32 \
    -n_epoch 1 \
    -bs 32 \
    -lr 0.0005 \
    -gmm_samples 100000 \
    -eval_num 2000 \
    -od_cond \
    -od_dropout 0.1 \
    -eos_mode \
    -eos_deg ${eos_deg} \
    -eos_canvas_len 64 \
    "$@"
