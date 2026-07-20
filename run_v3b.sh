#!/bin/bash
# v3b round: first-order completion, placebo audit, graph-kernel Sec.5, ESS traces.
set -u
cd "$(dirname "$0")"
D=./sets_disc; L=./sets_log; M=./sets_model
mkdir -p $L
python -c "
import torch
from dcbg_plugin import PlaceboDisc
torch.save(PlaceboDisc(), '$D/placebo.pth'); print('placebo saved')"

echo "=== Stage A: graph uncond pools ($(date)) ==="
(CUDA_VISIBLE_DEVICES=0 python -u gen_bd_uncond_pool.py -ckpt $M/BD_porto_v3_normal_graph_blk64_v2_bd.pth -out $D/uncond_pool_graph64.pth -n 20000 > $L/v3b_poolg64.log 2>&1) &
(CUDA_VISIBLE_DEVICES=1 python -u gen_bd_uncond_pool.py -ckpt $M/BD_porto_v3_normal_graph_blk4_v2_bd.pth  -out $D/uncond_pool_graph4.pth  -n 20000 > $L/v3b_poolg4.log 2>&1) &
wait

echo "=== Stage B: graph discs/classifiers ($(date)) ==="
(CUDA_VISIBLE_DEVICES=0 python -u train_bd_disc.py -family 0.05 -frac 1 -exp e0 -neg model -pool $D/uncond_pool_graph64.pth > $L/v3b_iwdisc_g64.log 2>&1
 mv $D/BDdisc_f0.05_p1_e0_model_graph64.pth $D/BDdisc_f0.05_p1_e0_model_graph64.pth 2>/dev/null || true) &
(CUDA_VISIBLE_DEVICES=1 python -u train_bd_disc.py -family 0.05 -frac 1 -exp e0 -neg model -pool $D/uncond_pool_graph4.pth > $L/v3b_iwdisc_g4.log 2>&1) &
(CUDA_VISIBLE_DEVICES=2 python -u train_dcbg_classifier.py -kernel graph -blk 64 -adj 1 -graph_ckpt $M/BD_porto_v3_normal_graph_blk64_v2_bd.pth > $L/v3b_dcbgclf_g64d.log 2>&1
 CUDA_VISIBLE_DEVICES=2 python -u train_dcbg_classifier.py -kernel graph -blk 64 -adj 1 -neg model -pool $D/uncond_pool_graph64.pth -graph_ckpt $M/BD_porto_v3_normal_graph_blk64_v2_bd.pth > $L/v3b_dcbgclf_g64m.log 2>&1) &
(CUDA_VISIBLE_DEVICES=3 python -u train_dcbg_classifier.py -kernel graph -blk 4 -adj 1 -graph_ckpt $M/BD_porto_v3_normal_graph_blk4_v2_bd.pth > $L/v3b_dcbgclf_g4d.log 2>&1
 CUDA_VISIBLE_DEVICES=3 python -u train_dcbg_classifier.py -kernel graph -blk 4 -adj 1 -neg model -pool $D/uncond_pool_graph4.pth -graph_ckpt $M/BD_porto_v3_normal_graph_blk4_v2_bd.pth > $L/v3b_dcbgclf_g4m.log 2>&1) &
wait

echo "=== Stage C: evaluations ($(date)) ==="
JOBS=(
  "python -u eval_dcbg.py -kernel mask -blk 8  -gamma 4.0 -adj 1 -approx 1 -ckpt $M/BD_porto_v3_normal_mask_blk8_v2_bd.pth  -res_suffix _v3fo -clf $D/DCBGclf_mask_blk8_f0.05_p1_adj_modelneg.pth"
  "python -u eval_dcbg.py -kernel mask -blk 16 -gamma 4.0 -adj 1 -approx 1 -ckpt $M/BD_porto_v3_normal_mask_blk16_v2_bd.pth -res_suffix _v3fo -clf $D/DCBGclf_mask_blk16_f0.05_p1_adj_modelneg.pth"
  "python -u eval_dcbg.py -kernel mask -blk 32 -gamma 4.0 -adj 1 -approx 1 -ckpt $M/BD_porto_v3_normal_mask_blk32_v2_bd.pth -res_suffix _v3fo -clf $D/DCBGclf_mask_blk32_f0.05_p1_adj_modelneg.pth"
  "python -u eval_dcbg.py -kernel mask -blk 64 -gamma 4.0 -adj 1 -approx 1 -ckpt $M/BD_porto_v3_normal_mask_blk64_v2_bd.pth -res_suffix _v3fo -clf $D/DCBGclf_mask_blk64_f0.05_p1_adj_modelneg.pth"
  "python -u three_way_postproc.py -ckpt $M/BD_porto_v3_normal_mask_blk4_v2_bd.pth  -disc $D/placebo.pth -tag v3plc4"
  "python -u three_way_postproc.py -ckpt $M/BD_porto_v3_normal_mask_blk16_v2_bd.pth -disc $D/placebo.pth -tag v3plc16"
  "python -u three_way_postproc.py -ckpt $M/BD_porto_v3_normal_mask_blk64_v2_bd.pth -disc $D/placebo.pth -tag v3plc64"
  "python -u three_way_postproc.py -ckpt $M/BD_porto_v3_normal_graph_blk64_v2_bd.pth -disc $D/BDdisc_f0.05_p1_e0_model_graph64.pth -tag v3g64"
  "python -u three_way_postproc.py -ckpt $M/BD_porto_v3_normal_graph_blk4_v2_bd.pth  -disc $D/BDdisc_f0.05_p1_e0_model_graph4.pth  -tag v3g4"
  "python -u eval_dcbg.py -kernel graph -blk 64 -gamma 4.0 -adj 1 -approx 1 -ckpt $M/BD_porto_v3_normal_graph_blk64_v2_bd.pth -res_suffix _v3gd -clf $D/DCBGclf_graph_blk64_f0.05_p1_adj.pth"
  "python -u eval_dcbg.py -kernel graph -blk 64 -gamma 4.0 -adj 1 -approx 1 -ckpt $M/BD_porto_v3_normal_graph_blk64_v2_bd.pth -res_suffix _v3gm -clf $D/DCBGclf_graph_blk64_f0.05_p1_adj_modelneg.pth"
  "python -u eval_dcbg.py -kernel graph -blk 4  -gamma 4.0 -adj 1 -approx 1 -ckpt $M/BD_porto_v3_normal_graph_blk4_v2_bd.pth  -res_suffix _v3gd -clf $D/DCBGclf_graph_blk4_f0.05_p1_adj.pth"
  "python -u eval_dcbg.py -kernel graph -blk 4  -gamma 4.0 -adj 1 -approx 1 -ckpt $M/BD_porto_v3_normal_graph_blk4_v2_bd.pth  -res_suffix _v3gm -clf $D/DCBGclf_graph_blk4_f0.05_p1_adj_modelneg.pth"
  "python -u ess_trace.py"
)
for g in 0 1 2 3; do
  ( for i in "${!JOBS[@]}"; do
      if (( i % 4 == g )); then
        CUDA_VISIBLE_DEVICES=$g ${JOBS[$i]} > $L/v3b_job$i.log 2>&1 || echo "JOB $i FAILED"
      fi
    done ) &
done
wait
echo V3B_DONE > $L/v3b_done.marker
echo "ALL V3B DONE ($(date))"
