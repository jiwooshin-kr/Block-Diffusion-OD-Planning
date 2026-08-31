#!/bin/bash
# v4 D-CBG classifiers — TRAINING ONLY (no sampling/eval in this script).
# D-CBG needs its own noise-conditioned classifier p(y | x_t, t): the IW discs
# (BDdisc_*_v4.pth) cannot be reused (no time conditioning, no MASK token, no
# get_log_probs). Protocol matched to the IW discs wherever it is shared:
#   -adj 1        adjacency-aware classifier (same GraphSAGE + edge channels)
#   -neg model    negatives = the block's own v4 unconditional pool (p_theta),
#                 corrupted by the same forward process as the positives
#   same 1% budget, same split seeds (e0 excludes the 1,000 reserved rows)
# 28 jobs = family {0.05, 0.1} x exp {e0 seen, e99 unseen} x blk {1..64}.
# Output: sets_disc/DCBGclf_mask_blk{B}_f{fam}_p1_adj[_e99]_modelneg_v4.pth
# e99 jobs pay a ~10 min 99-scenario data load each; e0 jobs are ~3 min total.
# Run inside tmux. Marker: sets_log/v4_dcbg_train_done.marker
set -u
cd /home/aailab/wp03052/Synthetic-Data/Block-Diffusion-OD-Planning
source /home/aailab/wp03052/venvs/od_planning_venv/bin/activate
D=sets_disc; L=sets_log
mkdir -p $L

JOBS=()
# e99 (slow: 99-scenario load) first so the round-robin spreads them evenly
for F in 0.05 0.1; do for B in 1 2 4 8 16 32 64; do
  JOBS+=("python -u tools/train/train_dcbg_classifier.py -kernel mask -blk $B -family $F -adj 1 -neg model -exp e99 -pool $D/uncond_pool_v4_blk$B.pth -outsfx _v4")
done; done
for F in 0.05 0.1; do for B in 1 2 4 8 16 32 64; do
  JOBS+=("python -u tools/train/train_dcbg_classifier.py -kernel mask -blk $B -family $F -adj 1 -neg model -exp e0 -pool $D/uncond_pool_v4_blk$B.pth -outsfx _v4")
done; done

echo "=== v4 D-CBG classifier training: ${#JOBS[@]} jobs ($(date)) ==="
for g in 0 1 2 3; do
  ( for i in "${!JOBS[@]}"; do
      if (( i % 4 == g )); then
        echo "[gpu$g] job $i: ${JOBS[$i]}"
        CUDA_VISIBLE_DEVICES=$g ${JOBS[$i]} > $L/v4_dcbgclf_job$i.log 2>&1 \
          || echo "DCBG CLF JOB $i FAILED"
      fi
    done ) &
done
wait

echo "=== trained ($(date)) ==="
ls $D/DCBGclf_mask_blk*_v4.pth 2>/dev/null | wc -l
ls $D/DCBGclf_mask_blk*_v4.pth 2>/dev/null | sed 's|.*/||'
echo V4_DCBG_TRAIN_DONE > $L/v4_dcbg_train_done.marker
echo "V4_DCBG_TRAIN_DONE ($(date))"
