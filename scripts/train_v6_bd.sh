# v6 데이터셋 normal 경로에 대한 masked block diffusion 학습
# usage: bash scripts/train_v6_bd.sh <block_size> <gpu> <variant: LE|LP> [extra flags...]
#   예) bash scripts/train_v6_bd.sh 4 0 LE
#   전제: bash scripts/setup_v6_data.sh && python prep_v6_node2vec.py  (각 1회)
#
# 기존 실험(porto_data / sets_model)과 완전히 분리된 경로만 사용한다.
block=$1
gpu=$2
variant=${3:-LE}
shift 3 2>/dev/null || shift $#
export CUDA_VISIBLE_DEVICES=${gpu}

fam="${variant}1.0-0.05"
model_name="BD_v6${variant}_normal_mask_blk${block}"

python3 -u main_bd.py \
    -device "default" \
    -path "./sets_data_v6" \
    -model_path "./sets_v6/${variant}/model" \
    -res_path "./sets_v6/${variant}/res" \
    -d_name "porto" \
    -model_name "${model_name}" \
    -method "bd_train" \
    -shortest_data_path "./porto_data_v6" \
    -shortest_org_idx "v6-${fam}_normal" \
    -kernel mask \
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
    -save_step 0 \
    "$@"
