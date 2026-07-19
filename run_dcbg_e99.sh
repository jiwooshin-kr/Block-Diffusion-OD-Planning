#!/bin/bash
# D-CBG exp2 (unseen-scenario) remainder: blk64 e99 classifier + 6 exact evals.
set -u
cd "$(dirname "$0")"
if [ ! -f sets_disc/DCBGclf_mask_blk64_f0.05_p1_adj_e99.pth ]; then
  CUDA_VISIBLE_DEVICES=0 python -u train_dcbg_classifier.py -kernel mask -blk 64 -adj 1 -exp e99 2>&1 | tee sets_log/dcbg_clf_m64_adj_e99.log
fi
for g in 1.0 2.0 4.0; do
  CUDA_VISIBLE_DEVICES=0 python -u eval_dcbg.py -kernel mask -blk 4  -gamma $g -adj 1 -res_suffix _e99 -clf sets_disc/DCBGclf_mask_blk4_f0.05_p1_adj_e99.pth  > sets_log/dcbg_e99_m4_g$g.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 python -u eval_dcbg.py -kernel mask -blk 64 -gamma $g -adj 1 -res_suffix _e99 -clf sets_disc/DCBGclf_mask_blk64_f0.05_p1_adj_e99.pth > sets_log/dcbg_e99_m64_g$g.log 2>&1 &
  wait
done
echo E99_ALL_DONE > sets_log/dcbg_e99_done.marker
