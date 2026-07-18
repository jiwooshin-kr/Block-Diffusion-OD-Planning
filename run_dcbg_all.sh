#!/bin/bash
# D-CBG (Schiff et al.) comparison round. Run inside tmux.
# Stage A: 3 noise-conditioned classifiers; Stage B: 9 evaluations (gamma sweep).
set -u
cd "$(dirname "$0")"
LOG=./sets_log
D=./sets_disc
mkdir -p $LOG

echo "=== Stage A: D-CBG classifiers ($(date)) ==="
(CUDA_VISIBLE_DEVICES=0 python -u train_dcbg_classifier.py -kernel mask -blk 4  2>&1 | tee $LOG/dcbg_clf_mask4.log) &
(CUDA_VISIBLE_DEVICES=1 python -u train_dcbg_classifier.py -kernel mask -blk 64 2>&1 | tee $LOG/dcbg_clf_mask64.log) &
(CUDA_VISIBLE_DEVICES=2 python -u train_dcbg_classifier.py -kernel graph -blk 64 -graph_ckpt sets_model/BD_porto_v3_normal_graph_blk64_d2.0_bd.pth 2>&1 | tee $LOG/dcbg_clf_graph64.log) &
wait

echo "=== Stage B: D-CBG evaluations ($(date)) ==="
JOBS=(
  "python -u eval_dcbg.py -kernel mask  -blk 4  -gamma 1.0 -clf $D/DCBGclf_mask_blk4_f0.05_p1.pth"
  "python -u eval_dcbg.py -kernel mask  -blk 4  -gamma 2.0 -clf $D/DCBGclf_mask_blk4_f0.05_p1.pth"
  "python -u eval_dcbg.py -kernel mask  -blk 4  -gamma 4.0 -clf $D/DCBGclf_mask_blk4_f0.05_p1.pth"
  "python -u eval_dcbg.py -kernel mask  -blk 64 -gamma 1.0 -clf $D/DCBGclf_mask_blk64_f0.05_p1.pth"
  "python -u eval_dcbg.py -kernel mask  -blk 64 -gamma 2.0 -clf $D/DCBGclf_mask_blk64_f0.05_p1.pth"
  "python -u eval_dcbg.py -kernel mask  -blk 64 -gamma 4.0 -clf $D/DCBGclf_mask_blk64_f0.05_p1.pth"
  "python -u eval_dcbg.py -kernel graph -blk 64 -gamma 1.0 -clf $D/DCBGclf_graph_blk64_f0.05_p1.pth"
  "python -u eval_dcbg.py -kernel graph -blk 64 -gamma 2.0 -clf $D/DCBGclf_graph_blk64_f0.05_p1.pth"
  "python -u eval_dcbg.py -kernel graph -blk 64 -gamma 4.0 -clf $D/DCBGclf_graph_blk64_f0.05_p1.pth"
)
for g in 0 1 2 3; do
  (
    for i in "${!JOBS[@]}"; do
      if (( i % 4 == g )); then
        echo "[gpu$g] job $i: ${JOBS[$i]}"
        CUDA_VISIBLE_DEVICES=$g ${JOBS[$i]} > $LOG/dcbg_job$i.log 2>&1 \
          || echo "[gpu$g] JOB $i FAILED"
      fi
    done
  ) &
done
wait
echo "ALL DCBG EXPERIMENTS DONE ($(date))"
