#!/bin/bash
# Unattended queue for the ICLR experiment record (pdfs/iclr_iw_exp.pdf).
# Runs sequentially so each stage gets all 4 GPUs; cheapest / most-needed first,
# D-CBG exact on family 0.1 (the single most expensive stage) last.
#
#   stage 0  wait for the already-running fam 0.05 D-CBG l2r sweep (tmux l2rdcbg)
#   stage 1  fam 0.05  ablation: adj-only / IW-only / adj+IW, BOTH orders   (14 jobs)
#   stage 2  fam 0.10  IW under l2r, seen + unseen                         (14 jobs)
#   stage 3  fam 0.10  ablation, both orders                               (14 jobs)
#   stage 4  fam 0.10  D-CBG under l2r: first-order + exact, seen + unseen (28 jobs)
#
# Every stage writes its own marker in sets_log/ and its own JSON in sets_res/,
# so the report builder can be re-run at any point and fills in what exists.
# Stages are skipped if their marker already exists -> the queue is resumable.
set -u
cd /home/aailab/wp03052/Synthetic-Data/Block-Diffusion-OD-Planning
source /home/aailab/wp03052/venvs/od_planning_venv/bin/activate
L=sets_log
mkdir -p $L

stage () {                       # $1 = marker, $2.. = command
  local marker=$L/$1; shift
  if [ -f "$marker" ]; then
    echo "=== SKIP (done): $marker ==="
    return 0
  fi
  echo "=== RUN: $* ($(date)) ==="
  "$@" || echo "STAGE FAILED: $*"
  echo "=== END: $* ($(date)) ==="
}

# ---- stage 0: wait for the in-flight fam 0.05 D-CBG sweep -------------
n=0
while [ ! -f $L/l2r_dcbg_done.marker ] && [ $n -lt 400 ]; do sleep 60; n=$((n+1)); done
echo "=== stage 0: waited ${n} min for fam 0.05 D-CBG ($(date)) ==="
[ -f $L/l2r_dcbg_done.marker ] || echo "WARNING: proceeding without the fam 0.05 D-CBG marker"

# ---- stages 1-4 -------------------------------------------------------
stage abl_done.marker         env FAM=0.05 bash scripts/run_adjonly_ablation.sh
stage l2rf01_done.marker      env FAM=0.1  bash scripts/run_l2r_v4.sh
stage ablf01_done.marker      env FAM=0.1  bash scripts/run_adjonly_ablation.sh
stage l2r_dcbgf01_done.marker env FAM=0.1  bash scripts/run_l2r_dcbg.sh

echo ICLR_QUEUE_DONE > $L/iclr_queue_done.marker
echo "ICLR_QUEUE_DONE ($(date))"
