#!/bin/bash
# w_gamma = 4 tempering for the IW side (see pdfs/temp_math.pdf): SNIS weights
# (D/(1-D))^4, targeting q^4 p^-3 (normalized). UNSEEN regime only (e99 discs,
# except_0 zero-shot), fam 0.05, l2r, 7 blocks; each job also re-runs base
# (w-independent) and both IW arms. Tags: wg4blk{B}uns_{cfg}.
set -u
cd /home/aailab/wp03052/Synthetic-Data/Block-Diffusion-OD-Planning
source /home/aailab/wp03052/venvs/od_planning_venv/bin/activate
D=sets_disc; M=sets_model; R=sets_res; L=sets_log
mkdir -p $L
CK(){ echo $M/BD_porto_v3_normal_mask_blk$1_v4_bd.pth; }

JOBS=()
for B in 1 2 4 8 16 32 64; do
  JOBS+=("python -u tools/eval/three_way_postproc.py -family 0.05 -order l2r -seed 7 -w_gamma 4.0 \
    -ckpt $(CK $B) -disc $D/BDdisc_f0.05_p1_e99_model_blk${B}_v4.pth -tag wg4blk${B}uns")
done

echo "=== wg4: ${#JOBS[@]} jobs ($(date)) ==="
for g in 0 1 2 3; do
  ( for i in "${!JOBS[@]}"; do
      if (( i % 4 == g )); then
        echo "[gpu$g] job $i: ${JOBS[$i]}"
        CUDA_VISIBLE_DEVICES=$g ${JOBS[$i]} > $L/wg4_job$i.log 2>&1 || echo "WG4 JOB $i FAILED"
      fi
    done ) &
done
wait

python -u tools/collect/collect_va_table.py -family 0.05 -pairs "unseen:wg4blk{B}uns" \
  -cfgs base,modelD,adj+modelD -out va_table_wg4_f005.json \
  > $R/va_table_wg4_f005.log 2>&1 || echo "COLLECT FAILED"
tail -30 $R/va_table_wg4_f005.log
grep -h "ess=" $L/wg4_job*.log | grep "_raw" | sort
echo WG4_DONE > $L/wg4_done.marker
echo "WG4_DONE ($(date))"
