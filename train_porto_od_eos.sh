# O/D conditional diffusion with LLaDA-style <eos> length handling
# usage: bash train_porto_od_eos.sh <eos_deg> <gpu>
eos_deg=${1:-0.05}
gpu=${2:-1}
export CUDA_VISIBLE_DEVICES=${gpu}

lr="0.0005"
bs="32"
od_dropout="0.1"
canvas="64"

model_name="OD_EOS_porto_v3_normal_d${eos_deg}"

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
    -bs ${bs} \
    -lr ${lr} \
    -gmm_samples 100000 \
    -eval_num 2000 \
    -od_cond \
    -od_dropout ${od_dropout} \
    -eos_mode \
    -eos_deg ${eos_deg} \
    -eos_canvas_len ${canvas}
