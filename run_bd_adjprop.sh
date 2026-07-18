#!/bin/bash
# Adjacency-masked proposal experiments (Lemma 3, BD_GUIDANCE_FORMULATION.pdf §6).
# 14 evaluations: adj-only control (5) + adj+guided main axis (5) +
# adj+guided fam 0.1 (2) + adj+guided exp2 disc (2). Run inside tmux.
set -u
cd "$(dirname "$0")"
LOG=./sets_log
D=./sets_disc
mkdir -p $LOG

JOBS=(
  # --- adj-only control (no discriminator) ---
  "python eval_bd_guidance.py -blk 4  -family 0.05 -disc none -adj_prop 1 -res_name BDguid_0.05_blk4_adjonly"
  "python eval_bd_guidance.py -blk 8  -family 0.05 -disc none -adj_prop 1 -res_name BDguid_0.05_blk8_adjonly"
  "python eval_bd_guidance.py -blk 16 -family 0.05 -disc none -adj_prop 1 -res_name BDguid_0.05_blk16_adjonly"
  "python eval_bd_guidance.py -blk 32 -family 0.05 -disc none -adj_prop 1 -res_name BDguid_0.05_blk32_adjonly"
  "python eval_bd_guidance.py -blk 64 -family 0.05 -disc none -adj_prop 1 -res_name BDguid_0.05_blk64_adjonly"
  # --- adj + guided, main axis (mix disc, exp1) ---
  "python eval_bd_guidance.py -blk 4  -family 0.05 -disc $D/BDdisc_f0.05_p1_e0_mix.pth -adj_prop 1 -res_name BDguid_0.05_blk4_adj_e0_mix"
  "python eval_bd_guidance.py -blk 8  -family 0.05 -disc $D/BDdisc_f0.05_p1_e0_mix.pth -adj_prop 1 -res_name BDguid_0.05_blk8_adj_e0_mix"
  "python eval_bd_guidance.py -blk 16 -family 0.05 -disc $D/BDdisc_f0.05_p1_e0_mix.pth -adj_prop 1 -res_name BDguid_0.05_blk16_adj_e0_mix"
  "python eval_bd_guidance.py -blk 32 -family 0.05 -disc $D/BDdisc_f0.05_p1_e0_mix.pth -adj_prop 1 -res_name BDguid_0.05_blk32_adj_e0_mix"
  "python eval_bd_guidance.py -blk 64 -family 0.05 -disc $D/BDdisc_f0.05_p1_e0_mix.pth -adj_prop 1 -res_name BDguid_0.05_blk64_adj_e0_mix"
  # --- adj + guided, fam 0.1 pivot ---
  "python eval_bd_guidance.py -blk 4  -family 0.1 -disc $D/BDdisc_f0.1_p1_e0_mix.pth -adj_prop 1 -res_name BDguid_0.1_blk4_adj_e0_mix"
  "python eval_bd_guidance.py -blk 16 -family 0.1 -disc $D/BDdisc_f0.1_p1_e0_mix.pth -adj_prop 1 -res_name BDguid_0.1_blk16_adj_e0_mix"
  # --- adj + guided, exp2 (unseen-scenario disc) ---
  "python eval_bd_guidance.py -blk 4  -family 0.05 -disc $D/BDdisc_f0.05_p1_e99_mix.pth -adj_prop 1 -res_name BDguid_0.05_blk4_adj_e99_mix"
  "python eval_bd_guidance.py -blk 16 -family 0.05 -disc $D/BDdisc_f0.05_p1_e99_mix.pth -adj_prop 1 -res_name BDguid_0.05_blk16_adj_e99_mix"
)
for g in 0 1 2 3; do
  (
    for i in "${!JOBS[@]}"; do
      if (( i % 4 == g )); then
        echo "[gpu$g] job $i: ${JOBS[$i]}"
        CUDA_VISIBLE_DEVICES=$g ${JOBS[$i]} > $LOG/adjprop_job$i.log 2>&1 \
          || echo "[gpu$g] JOB $i FAILED"
      fi
    done
  ) &
done
wait
echo "ALL ADJPROP EXPERIMENTS DONE ($(date))"
