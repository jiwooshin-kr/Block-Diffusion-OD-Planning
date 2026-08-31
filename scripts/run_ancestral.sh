#!/bin/bash
# Ancestral within-block candidate proposals for the IW guidance (experiment E6).
#
# Default (mean-field): every block position is drawn independently from its own
# marginal, and the Lemma-3 mask constrains only the position about to be
# committed -- so the lookahead part of each candidate is an unconstrained
# random continuation, and candidates whose internal transitions are illegal
# carry zero target mass yet still receive a finite discriminator weight.
#
# Ancestral: positions are drawn left to right, each restricted to the legal
# successors of THAT candidate's own previous token, so every proposed block is
# a legal walk. The per-candidate proposal normaliser prod_j alpha_j does not
# cancel under self-normalisation, so the weight must be multiplied by it:
#   -anc_correct 1  ->  log w = gamma*logit + sum_j log alpha_j   (exact)
#   -anc_correct 0  ->  log w = gamma*logit                       (naive)
# Both are measured; the gap between them is how much the correction matters.
#
# fam 0.05, l2r, seen discriminator, 1,000 reserved pairs, n_is 100, RAW.
# Small blocks first so the cheap half of the table lands early; blk1 is a
# control -- with one position per block the cascade degenerates and ancestral
# must reproduce mean-field.
set -u
cd /home/aailab/wp03052/Synthetic-Data/Block-Diffusion-OD-Planning
source /home/aailab/wp03052/venvs/od_planning_venv/bin/activate
D=sets_disc; M=sets_model; R=sets_res; L=sets_log
mkdir -p $L
CK(){ echo $M/BD_porto_v3_normal_mask_blk$1_v4_bd.pth; }

echo "=== ancestral proposals: 14 jobs ($(date)) ==="
JOBS=()
for B in 1 2 4 8 16 32 64; do
  for C in 1 0; do
    T=c; [ "$C" = "0" ] && T=n     # c = corrected, n = naive
    JOBS+=("python -u tools/eval/three_way_postproc.py -family 0.05 -order l2r -cand_mode ancestral \
      -anc_correct $C -ckpt $(CK $B) -disc $D/BDdisc_f0.05_p1_e0_model_blk${B}_v4.pth \
      -tag anc${T}${B}seen")
  done
done
for g in 0 1 2 3; do
  ( for i in "${!JOBS[@]}"; do
      if (( i % 4 == g )); then
        echo "[gpu$g] job $i: ${JOBS[$i]}"
        CUDA_VISIBLE_DEVICES=$g ${JOBS[$i]} > $L/anc_job$i.log 2>&1 || echo "ANC JOB $i FAILED"
      fi
    done ) &
done
wait
echo "=== evals done ($(date)) ==="

python -u tools/collect/collect_va_table.py -family 0.05 -pairs 'corrected:ancc{B}seen,naive:ancn{B}seen' \
  -cfgs base,modelD,adj+modelD -out va_table_anc_f005.json \
  > $R/va_table_anc_f005.log 2>&1 || echo "COLLECT FAILED"
tail -50 $R/va_table_anc_f005.log

echo ANC_DONE > $L/anc_done.marker
echo "ANC_DONE ($(date))"
