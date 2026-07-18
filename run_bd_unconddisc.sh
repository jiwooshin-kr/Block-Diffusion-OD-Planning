#!/bin/bash
# Model-distribution discriminator round (negatives = per-block unconditional
# generations = the exact Eq. 2 denominator p_theta). Run inside tmux.
# Stage A: 5 unconditional pools (one per block size)
# Stage B: 5 per-block discriminators (neg = model)
# Stage C: 10 evaluations (plain guided + adj+guided, matched disc per blk)
set -u
cd "$(dirname "$0")"
LOG=./sets_log
D=./sets_disc
mkdir -p $LOG

echo "=== Stage A: unconditional pools ($(date)) ==="
(CUDA_VISIBLE_DEVICES=0 python -u gen_bd_uncond_pool.py -blocks 4,32  2>&1 | tee $LOG/uncond_pool_a.log) &
(CUDA_VISIBLE_DEVICES=1 python -u gen_bd_uncond_pool.py -blocks 8,64 2>&1 | tee $LOG/uncond_pool_b.log) &
(CUDA_VISIBLE_DEVICES=2 python -u gen_bd_uncond_pool.py -blocks 16   2>&1 | tee $LOG/uncond_pool_c.log) &
wait

echo "=== Stage B: per-block model-negative discriminators ($(date)) ==="
(CUDA_VISIBLE_DEVICES=0 python -u train_bd_disc.py -family 0.05 -frac 1 -exp e0 -neg model -pool $D/uncond_pool_blk4.pth  2>&1 | tee $LOG/uncond_disc_blk4.log
 CUDA_VISIBLE_DEVICES=0 python -u train_bd_disc.py -family 0.05 -frac 1 -exp e0 -neg model -pool $D/uncond_pool_blk32.pth 2>&1 | tee $LOG/uncond_disc_blk32.log) &
(CUDA_VISIBLE_DEVICES=1 python -u train_bd_disc.py -family 0.05 -frac 1 -exp e0 -neg model -pool $D/uncond_pool_blk8.pth  2>&1 | tee $LOG/uncond_disc_blk8.log
 CUDA_VISIBLE_DEVICES=1 python -u train_bd_disc.py -family 0.05 -frac 1 -exp e0 -neg model -pool $D/uncond_pool_blk64.pth 2>&1 | tee $LOG/uncond_disc_blk64.log) &
(CUDA_VISIBLE_DEVICES=2 python -u train_bd_disc.py -family 0.05 -frac 1 -exp e0 -neg model -pool $D/uncond_pool_blk16.pth 2>&1 | tee $LOG/uncond_disc_blk16.log) &
wait

echo "=== Stage C: evaluations ($(date)) ==="
JOBS=(
  "python eval_bd_guidance.py -blk 4  -family 0.05 -disc $D/BDdisc_f0.05_p1_e0_model_blk4.pth  -res_name BDguid_0.05_blk4_p1_e0_modelD"
  "python eval_bd_guidance.py -blk 8  -family 0.05 -disc $D/BDdisc_f0.05_p1_e0_model_blk8.pth  -res_name BDguid_0.05_blk8_p1_e0_modelD"
  "python eval_bd_guidance.py -blk 16 -family 0.05 -disc $D/BDdisc_f0.05_p1_e0_model_blk16.pth -res_name BDguid_0.05_blk16_p1_e0_modelD"
  "python eval_bd_guidance.py -blk 32 -family 0.05 -disc $D/BDdisc_f0.05_p1_e0_model_blk32.pth -res_name BDguid_0.05_blk32_p1_e0_modelD"
  "python eval_bd_guidance.py -blk 64 -family 0.05 -disc $D/BDdisc_f0.05_p1_e0_model_blk64.pth -res_name BDguid_0.05_blk64_p1_e0_modelD"
  "python eval_bd_guidance.py -blk 4  -family 0.05 -disc $D/BDdisc_f0.05_p1_e0_model_blk4.pth  -adj_prop 1 -res_name BDguid_0.05_blk4_adj_e0_modelD"
  "python eval_bd_guidance.py -blk 8  -family 0.05 -disc $D/BDdisc_f0.05_p1_e0_model_blk8.pth  -adj_prop 1 -res_name BDguid_0.05_blk8_adj_e0_modelD"
  "python eval_bd_guidance.py -blk 16 -family 0.05 -disc $D/BDdisc_f0.05_p1_e0_model_blk16.pth -adj_prop 1 -res_name BDguid_0.05_blk16_adj_e0_modelD"
  "python eval_bd_guidance.py -blk 32 -family 0.05 -disc $D/BDdisc_f0.05_p1_e0_model_blk32.pth -adj_prop 1 -res_name BDguid_0.05_blk32_adj_e0_modelD"
  "python eval_bd_guidance.py -blk 64 -family 0.05 -disc $D/BDdisc_f0.05_p1_e0_model_blk64.pth -adj_prop 1 -res_name BDguid_0.05_blk64_adj_e0_modelD"
)
for g in 0 1 2 3; do
  (
    for i in "${!JOBS[@]}"; do
      if (( i % 4 == g )); then
        echo "[gpu$g] job $i: ${JOBS[$i]}"
        CUDA_VISIBLE_DEVICES=$g ${JOBS[$i]} > $LOG/uncondD_job$i.log 2>&1 \
          || echo "[gpu$g] JOB $i FAILED"
      fi
    done
  ) &
done
wait
echo "ALL UNCOND-DISC EXPERIMENTS DONE ($(date))"
