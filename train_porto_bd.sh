# Block-diffusion training
# usage: bash train_porto_bd.sh <kernel: mask|graph> <block_size> <gpu> <suffix> [extra flags...]
kernel=$1
block=$2
gpu=$3
suffix=$4
shift 4
export CUDA_VISIBLE_DEVICES=${gpu}

model_name="BD_porto_v3_normal_${kernel}_blk${block}_${suffix}"

python3 main_bd.py \
    -device "default" \
    -path "./sets_data" \
    -model_path "./sets_model" \
    -res_path "./sets_res" \
    -d_name "porto" \
    -model_name "${model_name}" \
    -method "bd_train" \
    -shortest_data_path "./porto_data" \
    -shortest_org_idx "v3-0.05_normal" \
    -kernel ${kernel} \
    -block_size ${block} \
    -beta_lb 0.0001 \
    -beta_ub 10 \
    -max_T 100 \
    -od_max_len 100 \
    -drop_cond 0.1 \
    -n_epoch 1 \
    -bs 32 \
    -lr 0.0005 \
    -eval_num 1000 \
    -batch_traj_num 200 \
    -length_mode open \
    -guidance_scale 1.0 \
    "$@"
