#!/bin/bash
# Graph-kernel guidance for the missing block sizes 8/16/32 (§5.3 completion).
set -u
cd "$(dirname "$0")"
D=./sets_disc; L=./sets_log; M=./sets_model
mkdir -p $L

echo "=== Stage A: graph uncond pools 8/16/32 ($(date)) ==="
(CUDA_VISIBLE_DEVICES=0 python -u gen_bd_uncond_pool.py -ckpt $M/BD_porto_v3_normal_graph_blk8_v2_bd.pth  -out $D/uncond_pool_graph8.pth  -n 20000 > $L/gm_pool8.log 2>&1) &
(CUDA_VISIBLE_DEVICES=1 python -u gen_bd_uncond_pool.py -ckpt $M/BD_porto_v3_normal_graph_blk16_v2_bd.pth -out $D/uncond_pool_graph16.pth -n 20000 > $L/gm_pool16.log 2>&1) &
(CUDA_VISIBLE_DEVICES=2 python -u gen_bd_uncond_pool.py -ckpt $M/BD_porto_v3_normal_graph_blk32_v2_bd.pth -out $D/uncond_pool_graph32.pth -n 20000 > $L/gm_pool32.log 2>&1) &
wait

echo "=== Stage B: IW discs + D-CBG classifiers ($(date)) ==="
# IW model-negative discs (fast)
(CUDA_VISIBLE_DEVICES=0 python -u train_bd_disc.py -family 0.05 -frac 1 -exp e0 -neg model -pool $D/uncond_pool_graph8.pth  > $L/gm_iw8.log 2>&1) &
(CUDA_VISIBLE_DEVICES=1 python -u train_bd_disc.py -family 0.05 -frac 1 -exp e0 -neg model -pool $D/uncond_pool_graph16.pth > $L/gm_iw16.log 2>&1) &
(CUDA_VISIBLE_DEVICES=2 python -u train_bd_disc.py -family 0.05 -frac 1 -exp e0 -neg model -pool $D/uncond_pool_graph32.pth > $L/gm_iw32.log 2>&1) &
wait
# D-CBG adj classifiers (data + model), slow — one blk per GPU, two negs sequential
(CUDA_VISIBLE_DEVICES=0 python -u train_dcbg_classifier.py -kernel graph -blk 8  -adj 1 -graph_ckpt $M/BD_porto_v3_normal_graph_blk8_v2_bd.pth  > $L/gm_clf8d.log 2>&1
 CUDA_VISIBLE_DEVICES=0 python -u train_dcbg_classifier.py -kernel graph -blk 8  -adj 1 -neg model -pool $D/uncond_pool_graph8.pth  -graph_ckpt $M/BD_porto_v3_normal_graph_blk8_v2_bd.pth  > $L/gm_clf8m.log 2>&1) &
(CUDA_VISIBLE_DEVICES=1 python -u train_dcbg_classifier.py -kernel graph -blk 16 -adj 1 -graph_ckpt $M/BD_porto_v3_normal_graph_blk16_v2_bd.pth > $L/gm_clf16d.log 2>&1
 CUDA_VISIBLE_DEVICES=1 python -u train_dcbg_classifier.py -kernel graph -blk 16 -adj 1 -neg model -pool $D/uncond_pool_graph16.pth -graph_ckpt $M/BD_porto_v3_normal_graph_blk16_v2_bd.pth > $L/gm_clf16m.log 2>&1) &
(CUDA_VISIBLE_DEVICES=2 python -u train_dcbg_classifier.py -kernel graph -blk 32 -adj 1 -graph_ckpt $M/BD_porto_v3_normal_graph_blk32_v2_bd.pth > $L/gm_clf32d.log 2>&1
 CUDA_VISIBLE_DEVICES=2 python -u train_dcbg_classifier.py -kernel graph -blk 32 -adj 1 -neg model -pool $D/uncond_pool_graph32.pth -graph_ckpt $M/BD_porto_v3_normal_graph_blk32_v2_bd.pth > $L/gm_clf32m.log 2>&1) &
wait

echo "=== Stage C: evaluations ($(date)) ==="
JOBS=(
  "python -u three_way_postproc.py -ckpt $M/BD_porto_v3_normal_graph_blk8_v2_bd.pth  -disc $D/BDdisc_f0.05_p1_e0_model_graph8.pth  -tag v3g8"
  "python -u three_way_postproc.py -ckpt $M/BD_porto_v3_normal_graph_blk16_v2_bd.pth -disc $D/BDdisc_f0.05_p1_e0_model_graph16.pth -tag v3g16"
  "python -u three_way_postproc.py -ckpt $M/BD_porto_v3_normal_graph_blk32_v2_bd.pth -disc $D/BDdisc_f0.05_p1_e0_model_graph32.pth -tag v3g32"
  "python -u eval_dcbg.py -kernel graph -blk 8  -gamma 4.0 -adj 1 -approx 1 -ckpt $M/BD_porto_v3_normal_graph_blk8_v2_bd.pth  -res_suffix _gmd -clf $D/DCBGclf_graph_blk8_f0.05_p1_adj.pth"
  "python -u eval_dcbg.py -kernel graph -blk 16 -gamma 4.0 -adj 1 -approx 1 -ckpt $M/BD_porto_v3_normal_graph_blk16_v2_bd.pth -res_suffix _gmd -clf $D/DCBGclf_graph_blk16_f0.05_p1_adj.pth"
  "python -u eval_dcbg.py -kernel graph -blk 32 -gamma 4.0 -adj 1 -approx 1 -ckpt $M/BD_porto_v3_normal_graph_blk32_v2_bd.pth -res_suffix _gmd -clf $D/DCBGclf_graph_blk32_f0.05_p1_adj.pth"
  "python -u eval_dcbg.py -kernel graph -blk 8  -gamma 4.0 -adj 1 -approx 1 -ckpt $M/BD_porto_v3_normal_graph_blk8_v2_bd.pth  -res_suffix _gmm -clf $D/DCBGclf_graph_blk8_f0.05_p1_adj_modelneg.pth"
  "python -u eval_dcbg.py -kernel graph -blk 16 -gamma 4.0 -adj 1 -approx 1 -ckpt $M/BD_porto_v3_normal_graph_blk16_v2_bd.pth -res_suffix _gmm -clf $D/DCBGclf_graph_blk16_f0.05_p1_adj_modelneg.pth"
  "python -u eval_dcbg.py -kernel graph -blk 32 -gamma 4.0 -adj 1 -approx 1 -ckpt $M/BD_porto_v3_normal_graph_blk32_v2_bd.pth -res_suffix _gmm -clf $D/DCBGclf_graph_blk32_f0.05_p1_adj_modelneg.pth"
)
for g in 0 1 2 3; do
  ( for i in "${!JOBS[@]}"; do
      if (( i % 4 == g )); then
        CUDA_VISIBLE_DEVICES=$g ${JOBS[$i]} > $L/gm_job$i.log 2>&1 || echo "JOB $i FAILED"
      fi
    done ) &
done
wait
echo GM_DONE > $L/gm_done.marker
echo "ALL GRAPH-MID DONE ($(date))"
