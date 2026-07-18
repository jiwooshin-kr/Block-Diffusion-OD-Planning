#!/bin/bash
# Full importance-weight guidance experiment pipeline (run inside tmux).
# Stage A: model-sample negative pool
# Stage B: 8 discriminators (2 per GPU)
# Stage C: 29 evaluations (7 baselines + 22 guided), round-robin over 4 GPUs
set -u
cd "$(dirname "$0")"
LOG=./sets_log
mkdir -p $LOG sets_disc sets_res

echo "=== Stage A: negative pool ($(date)) ==="
CUDA_VISIBLE_DEVICES=0 python gen_bd_neg_pool.py 2>&1 | tee $LOG/guid_pool.log

echo "=== Stage B: discriminators ($(date)) ==="
(
  CUDA_VISIBLE_DEVICES=0 python train_bd_disc.py -family 0.05 -frac 1 -exp e0 -neg data 2>&1 | tee $LOG/guid_disc_f0.05_p1_e0_data.log
  CUDA_VISIBLE_DEVICES=0 python train_bd_disc.py -family 0.05 -frac 1 -exp e0 -neg mix  2>&1 | tee $LOG/guid_disc_f0.05_p1_e0_mix.log
) &
(
  CUDA_VISIBLE_DEVICES=1 python train_bd_disc.py -family 0.1 -frac 1 -exp e0 -neg data 2>&1 | tee $LOG/guid_disc_f0.1_p1_e0_data.log
  CUDA_VISIBLE_DEVICES=1 python train_bd_disc.py -family 0.1 -frac 1 -exp e0 -neg mix  2>&1 | tee $LOG/guid_disc_f0.1_p1_e0_mix.log
) &
(
  CUDA_VISIBLE_DEVICES=2 python train_bd_disc.py -family 0.05 -frac 3 -exp e0 -neg data 2>&1 | tee $LOG/guid_disc_f0.05_p3_e0_data.log
  CUDA_VISIBLE_DEVICES=2 python train_bd_disc.py -family 0.05 -frac 3 -exp e0 -neg mix  2>&1 | tee $LOG/guid_disc_f0.05_p3_e0_mix.log
) &
(
  CUDA_VISIBLE_DEVICES=3 python train_bd_disc.py -family 0.05 -frac 1 -exp e99 -neg data 2>&1 | tee $LOG/guid_disc_f0.05_p1_e99_data.log
  CUDA_VISIBLE_DEVICES=3 python train_bd_disc.py -family 0.05 -frac 1 -exp e99 -neg mix  2>&1 | tee $LOG/guid_disc_f0.05_p1_e99_mix.log
) &
wait

echo "=== Stage C: evaluations ($(date)) ==="
D=./sets_disc
JOBS=(
  # --- baselines (7) ---
  "python eval_bd_guidance.py -blk 4  -family 0.05 -disc none -res_name BDguid_0.05_blk4_base"
  "python eval_bd_guidance.py -blk 8  -family 0.05 -disc none -res_name BDguid_0.05_blk8_base"
  "python eval_bd_guidance.py -blk 16 -family 0.05 -disc none -res_name BDguid_0.05_blk16_base"
  "python eval_bd_guidance.py -blk 32 -family 0.05 -disc none -res_name BDguid_0.05_blk32_base"
  "python eval_bd_guidance.py -blk 64 -family 0.05 -disc none -res_name BDguid_0.05_blk64_base"
  "python eval_bd_guidance.py -blk 4  -family 0.1  -disc none -res_name BDguid_0.1_blk4_base"
  "python eval_bd_guidance.py -blk 16 -family 0.1  -disc none -res_name BDguid_0.1_blk16_base"
  # --- main axis: fam 0.05, 1%, exp1 x blk{4,8,16,32,64} x neg{data,mix} (10) ---
  "python eval_bd_guidance.py -blk 4  -family 0.05 -disc $D/BDdisc_f0.05_p1_e0_data.pth -res_name BDguid_0.05_blk4_p1_e0_data"
  "python eval_bd_guidance.py -blk 8  -family 0.05 -disc $D/BDdisc_f0.05_p1_e0_data.pth -res_name BDguid_0.05_blk8_p1_e0_data"
  "python eval_bd_guidance.py -blk 16 -family 0.05 -disc $D/BDdisc_f0.05_p1_e0_data.pth -res_name BDguid_0.05_blk16_p1_e0_data"
  "python eval_bd_guidance.py -blk 32 -family 0.05 -disc $D/BDdisc_f0.05_p1_e0_data.pth -res_name BDguid_0.05_blk32_p1_e0_data"
  "python eval_bd_guidance.py -blk 64 -family 0.05 -disc $D/BDdisc_f0.05_p1_e0_data.pth -res_name BDguid_0.05_blk64_p1_e0_data"
  "python eval_bd_guidance.py -blk 4  -family 0.05 -disc $D/BDdisc_f0.05_p1_e0_mix.pth -res_name BDguid_0.05_blk4_p1_e0_mix"
  "python eval_bd_guidance.py -blk 8  -family 0.05 -disc $D/BDdisc_f0.05_p1_e0_mix.pth -res_name BDguid_0.05_blk8_p1_e0_mix"
  "python eval_bd_guidance.py -blk 16 -family 0.05 -disc $D/BDdisc_f0.05_p1_e0_mix.pth -res_name BDguid_0.05_blk16_p1_e0_mix"
  "python eval_bd_guidance.py -blk 32 -family 0.05 -disc $D/BDdisc_f0.05_p1_e0_mix.pth -res_name BDguid_0.05_blk32_p1_e0_mix"
  "python eval_bd_guidance.py -blk 64 -family 0.05 -disc $D/BDdisc_f0.05_p1_e0_mix.pth -res_name BDguid_0.05_blk64_p1_e0_mix"
  # --- pivot: fam 0.1, 1%, exp1 x blk{4,16} x neg2 (4) ---
  "python eval_bd_guidance.py -blk 4  -family 0.1 -disc $D/BDdisc_f0.1_p1_e0_data.pth -res_name BDguid_0.1_blk4_p1_e0_data"
  "python eval_bd_guidance.py -blk 16 -family 0.1 -disc $D/BDdisc_f0.1_p1_e0_data.pth -res_name BDguid_0.1_blk16_p1_e0_data"
  "python eval_bd_guidance.py -blk 4  -family 0.1 -disc $D/BDdisc_f0.1_p1_e0_mix.pth -res_name BDguid_0.1_blk4_p1_e0_mix"
  "python eval_bd_guidance.py -blk 16 -family 0.1 -disc $D/BDdisc_f0.1_p1_e0_mix.pth -res_name BDguid_0.1_blk16_p1_e0_mix"
  # --- pivot: fam 0.05, 3%, exp1 x blk{4,16} x neg2 (4) ---
  "python eval_bd_guidance.py -blk 4  -family 0.05 -disc $D/BDdisc_f0.05_p3_e0_data.pth -res_name BDguid_0.05_blk4_p3_e0_data"
  "python eval_bd_guidance.py -blk 16 -family 0.05 -disc $D/BDdisc_f0.05_p3_e0_data.pth -res_name BDguid_0.05_blk16_p3_e0_data"
  "python eval_bd_guidance.py -blk 4  -family 0.05 -disc $D/BDdisc_f0.05_p3_e0_mix.pth -res_name BDguid_0.05_blk4_p3_e0_mix"
  "python eval_bd_guidance.py -blk 16 -family 0.05 -disc $D/BDdisc_f0.05_p3_e0_mix.pth -res_name BDguid_0.05_blk16_p3_e0_mix"
  # --- pivot: fam 0.05, 1%, exp2 (disc on except 1-99) x blk{4,16} x neg2 (4) ---
  "python eval_bd_guidance.py -blk 4  -family 0.05 -disc $D/BDdisc_f0.05_p1_e99_data.pth -res_name BDguid_0.05_blk4_p1_e99_data"
  "python eval_bd_guidance.py -blk 16 -family 0.05 -disc $D/BDdisc_f0.05_p1_e99_data.pth -res_name BDguid_0.05_blk16_p1_e99_data"
  "python eval_bd_guidance.py -blk 4  -family 0.05 -disc $D/BDdisc_f0.05_p1_e99_mix.pth -res_name BDguid_0.05_blk4_p1_e99_mix"
  "python eval_bd_guidance.py -blk 16 -family 0.05 -disc $D/BDdisc_f0.05_p1_e99_mix.pth -res_name BDguid_0.05_blk16_p1_e99_mix"
  # --- proposal-entropy remedy (PDF sec.6): candidate temperature (4) ---
  "python eval_bd_guidance.py -blk 4  -family 0.05 -disc $D/BDdisc_f0.05_p1_e0_data.pth -cand_temp 1.5 -res_name BDguid_0.05_blk4_p1_e0_data_T1.5"
  "python eval_bd_guidance.py -blk 16 -family 0.05 -disc $D/BDdisc_f0.05_p1_e0_data.pth -cand_temp 1.5 -res_name BDguid_0.05_blk16_p1_e0_data_T1.5"
  "python eval_bd_guidance.py -blk 4  -family 0.05 -disc $D/BDdisc_f0.05_p1_e0_data.pth -cand_temp 2.0 -res_name BDguid_0.05_blk4_p1_e0_data_T2.0"
  "python eval_bd_guidance.py -blk 16 -family 0.05 -disc $D/BDdisc_f0.05_p1_e0_data.pth -cand_temp 2.0 -res_name BDguid_0.05_blk16_p1_e0_data_T2.0"
)
for g in 0 1 2 3; do
  (
    for i in "${!JOBS[@]}"; do
      if (( i % 4 == g )); then
        echo "[gpu$g] job $i: ${JOBS[$i]}"
        CUDA_VISIBLE_DEVICES=$g ${JOBS[$i]} > $LOG/guid_eval_job$i.log 2>&1 \
          || echo "[gpu$g] JOB $i FAILED"
      fi
    done
  ) &
done
wait
echo "ALL GUIDANCE EXPERIMENTS DONE ($(date))"
