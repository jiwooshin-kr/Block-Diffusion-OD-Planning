#!/bin/bash
# v4 D-CBG evaluation (exact enumeration, gamma=4) on except_0, using the
# v4 model-negative classifiers and the v4 (EOS-fixed) backbones.
# NOT launched automatically -- scope is chosen by the caller:
#
#   BLOCKS="4 16 64" FAMS="0.05" REGIMES="e0"      bash run_v4_dcbg_eval.sh   # ~40 min
#   BLOCKS="1 2 4 8 16 32 64" FAMS="0.05 0.1" REGIMES="e0 e99" bash run_v4_dcbg_eval.sh  # full
#
# Cost: exact enumeration is ~|V|=1.4k classifier calls per reveal (~2.7 s/path
# on v2 timings) => ~45 min per (blk, family, regime) config for 1,000 pairs.
# Tags: DCBG_mask{B}_g4.0_adj_v4{f01}{uns}  (raw + P1P3 rows per run)
set -u
cd /home/aailab/wp03052/Synthetic-Data/Block-Diffusion-OD-Planning
source /home/aailab/wp03052/syn_data/bin/activate
D=sets_disc; M=sets_model; L=sets_log
mkdir -p $L
BLOCKS="${BLOCKS:-4 16 64}"
FAMS="${FAMS:-0.05}"
REGIMES="${REGIMES:-e0}"
# gamma=1.0 matches the paper formula with no tempering, and matches the IW
# side (eval_bd_guidance.py -w_gamma default 1.0) -- so the two methods are
# compared at the same guidance strength. v1-era runs used gamma=4 (a tuned
# advantage for D-CBG); those numbers are in the v3 report, not here.
GAMMA="${GAMMA:-1.0}"
# VARIANTS: "exact" (enumerate all V tokens at the reveal position, ~2.9 s/path)
#           "fo"    (first-order Taylor, one fwd+bwd per reveal, ~0.08 s/path)
# Both use the SAME trained classifier -- the variant is a runtime flag
# (-approx), not a separate model. eval_dcbg tags fo runs with "_fo".
VARIANTS="${VARIANTS:-exact}"
CK(){ echo $M/BD_porto_v3_normal_mask_blk$1_v4_bd.pth; }

JOBS=()
for V in $VARIANTS; do
  AP=0; [ "$V" = "fo" ] && AP=1
  for F in $FAMS; do for R in $REGIMES; do for B in $BLOCKS; do
    E99=""; UNS=""
    if [ "$R" = "e99" ]; then E99="_e99"; UNS="uns"; fi
    F01=""; if [ "$F" = "0.1" ]; then F01="f01"; fi
    CLF=$D/DCBGclf_mask_blk${B}_f${F}_p1_adj${E99}_modelneg_v4.pth
    [ -f "$CLF" ] || { echo "MISSING CLF $CLF -- skipping"; continue; }
    JOBS+=("python -u eval_dcbg.py -kernel mask -blk $B -family $F -gamma $GAMMA -adj 1 -approx $AP -ckpt $(CK $B) -clf $CLF -res_suffix _v4${F01}${UNS}")
  done; done; done
done

echo "=== v4 D-CBG eval: ${#JOBS[@]} jobs (variants='$VARIANTS' blocks='$BLOCKS' fams='$FAMS' regimes='$REGIMES') ($(date)) ==="
[ ${#JOBS[@]} -eq 0 ] && { echo "no jobs"; exit 1; }
for g in 0 1 2 3; do
  ( for i in "${!JOBS[@]}"; do
      if (( i % 4 == g )); then
        echo "[gpu$g] job $i: ${JOBS[$i]}"
        CUDA_VISIBLE_DEVICES=$g ${JOBS[$i]} > $L/v4_dcbgeval_job$i.log 2>&1 \
          || echo "DCBG EVAL JOB $i FAILED"
      fi
    done ) &
done
wait
echo "=== results ==="
grep -h -E "^DCBG_mask.*(raw|P1P3)|sec_per_path" $L/v4_dcbgeval_job*.log 2>/dev/null | sed 's/  */ /g'
echo V4_DCBG_EVAL_DONE > $L/v4_dcbg_eval_done.marker
echo "V4_DCBG_EVAL_DONE ($(date))"
