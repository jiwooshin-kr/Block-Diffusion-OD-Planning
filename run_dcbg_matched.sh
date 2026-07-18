#!/bin/bash
# Controlled 2x2 comparison round: {mechanism: D-CBG, IW} x {classifier:
# plain, adjacency-aware}, all classifiers architecture- and protocol-matched
# (data-negative, same 1% splits). Run inside tmux.
set -u
cd "$(dirname "$0")"
LOG=./sets_log
D=./sets_disc
mkdir -p $LOG

echo "=== Stage A: matched classifiers ($(date)) ==="
(CUDA_VISIBLE_DEVICES=0 python -u train_dcbg_classifier.py -kernel mask -blk 4  -adj 1 2>&1 | tee $LOG/dcbg_clf_mask4_adj.log) &
(CUDA_VISIBLE_DEVICES=1 python -u train_dcbg_classifier.py -kernel mask -blk 64 -adj 1 2>&1 | tee $LOG/dcbg_clf_mask64_adj.log) &
(CUDA_VISIBLE_DEVICES=2 python -u train_dcbg_classifier.py -kernel graph -blk 64 -adj 1 -graph_ckpt sets_model/BD_porto_v3_normal_graph_blk64_d2.0_bd.pth 2>&1 | tee $LOG/dcbg_clf_graph64_adj.log) &
(CUDA_VISIBLE_DEVICES=3 python -u train_bd_disc_plain.py 2>&1 | tee $LOG/bd_disc_plain.log) &
wait

echo "=== Stage B: evaluations ($(date)) ==="
JOBS=(
  # D-CBG + adjacency-aware classifier (gamma sweep)
  "python -u eval_dcbg.py -kernel mask  -blk 4  -gamma 1.0 -adj 1 -clf $D/DCBGclf_mask_blk4_f0.05_p1_adj.pth"
  "python -u eval_dcbg.py -kernel mask  -blk 4  -gamma 2.0 -adj 1 -clf $D/DCBGclf_mask_blk4_f0.05_p1_adj.pth"
  "python -u eval_dcbg.py -kernel mask  -blk 4  -gamma 4.0 -adj 1 -clf $D/DCBGclf_mask_blk4_f0.05_p1_adj.pth"
  "python -u eval_dcbg.py -kernel mask  -blk 64 -gamma 1.0 -adj 1 -clf $D/DCBGclf_mask_blk64_f0.05_p1_adj.pth"
  "python -u eval_dcbg.py -kernel mask  -blk 64 -gamma 2.0 -adj 1 -clf $D/DCBGclf_mask_blk64_f0.05_p1_adj.pth"
  "python -u eval_dcbg.py -kernel mask  -blk 64 -gamma 4.0 -adj 1 -clf $D/DCBGclf_mask_blk64_f0.05_p1_adj.pth"
  "python -u eval_dcbg.py -kernel graph -blk 64 -gamma 1.0 -adj 1 -clf $D/DCBGclf_graph_blk64_f0.05_p1_adj.pth"
  "python -u eval_dcbg.py -kernel graph -blk 64 -gamma 2.0 -adj 1 -clf $D/DCBGclf_graph_blk64_f0.05_p1_adj.pth"
  "python -u eval_dcbg.py -kernel graph -blk 64 -gamma 4.0 -adj 1 -clf $D/DCBGclf_graph_blk64_f0.05_p1_adj.pth"
  # IW + plain classifier (three-way gives plain-guided AND adj+plain rows)
  "python -u three_way_postproc.py -ckpt sets_model/BD_porto_v3_normal_mask_blk4_base_bd.pth  -disc $D/BDdisc_plain_f0.05_p1_e0_data.pth -tag m4plainD"
  "python -u three_way_postproc.py -ckpt sets_model/BD_porto_v3_normal_mask_blk64_base_bd.pth -disc $D/BDdisc_plain_f0.05_p1_e0_data.pth -tag m64plainD"
  "python -u three_way_postproc.py -ckpt sets_model/BD_porto_v3_normal_graph_blk64_d2.0_bd.pth -disc $D/BDdisc_plain_f0.05_p1_e0_data.pth -tag g64plainD"
  # IW + adjacency-aware DATA-negative disc on graph64 (completes the grid)
  "python -u three_way_postproc.py -ckpt sets_model/BD_porto_v3_normal_graph_blk64_d2.0_bd.pth -disc $D/BDdisc_f0.05_p1_e0_data.pth -tag g64adjdataD"
)
for g in 0 1 2 3; do
  (
    for i in "${!JOBS[@]}"; do
      if (( i % 4 == g )); then
        echo "[gpu$g] job $i: ${JOBS[$i]}"
        CUDA_VISIBLE_DEVICES=$g ${JOBS[$i]} > $LOG/matched_job$i.log 2>&1 \
          || echo "[gpu$g] JOB $i FAILED"
      fi
    done
  ) &
done
wait
echo "ALL MATCHED EXPERIMENTS DONE ($(date))"
