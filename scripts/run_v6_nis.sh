#!/bin/bash
# ============================================================================
# n_is sweep: how many importance-sampling candidates does the IW guidance need?
#
#   VARIANT=LE bash scripts/run_v6_nis.sh
#   VARIANT=LP BLOCKS="2 16" NIS="10 50" GPUS="2 3" bash scripts/run_v6_nis.sh
#   # extend a finished sweep with more blocks (stage A only runs the new ones,
#   # stage B/C still collect all seven):
#   VARIANT=LE BLOCKS="4 8 32" COLLECT_BLOCKS="1 2 4 8 16 32 64" \
#     bash scripts/run_v6_nis.sh
#
# Why this axis. tools/diag/diag_proposal_diversity.py measured, on the LE
# scenario graph, that 100 mean-field candidates collapse onto 2.6-3.9 DISTINCT
# blocks at blk<=4 with adjacency masking on, because the reveal position only
# has ~2.7 scenario-legal successors. If that is the binding constraint, the
# metrics must be FLAT in n_is over this whole range, and the guidance can be
# made ~10x cheaper -- or replaced by exact enumeration -- at no cost in quality.
# A flat sweep is therefore the result, not a null finding.
#
# One three_way_postproc.py run produces four arms (base / IW-only / Adj+IW /
# Adj-only) for one (block, regime), so the job unit is (block, regime, n_is).
# base does not depend on n_is and is re-run inside every job with the same seed,
# which doubles as a consistency check: base rows must be identical across n.
#
# n_is=100 is NOT re-run -- it already exists under the gate-D tags
# (v6{V}{B}seen / v6{V}{B}uns) and is collected into the sweep tables from there.
#
# Tags: v6{V}{B}seenN{n} / v6{V}{B}unsN{n}, so nothing overwrites the gate-D
# records or another n's records.
#
# Stage A writes em_pc records; stage B collects them into
#   sets_v6/{V}/res/va_table_v6{V}_nis{n}{,_rawL,_P1P3,_P1P3L}.json
# ============================================================================
set -u
cd /home/aailab/wp03052/Synthetic-Data/Block-Diffusion-OD-Planning
source /home/aailab/wp03052/venvs/od_planning_venv/bin/activate

VARIANT="${VARIANT:-LE}"
BLOCKS="${BLOCKS:-1 2 16 64}"          # stage A: which blocks to GENERATE
# stage B/C read every block that should end up in the tables, which is NOT the
# same list when you extend a finished sweep with more blocks: running stage A on
# "4 8 32" alone would otherwise rewrite the tables with only those three blocks
# and silently drop 1/2/16/64. Missing cells are reported as MISSING, not fatal.
COLLECT_BLOCKS="${COLLECT_BLOCKS:-$BLOCKS}"
NIS="${NIS:-10 30 50 150}"        # 100 comes from the gate-D run
NIS_ALL="${NIS_ALL:-10 30 50 100 150}"   # what stage B collects
GPUS="${GPUS:-2 3}"
EVALN="${EVALN:-1000}"
STOP_AFTER="${STOP_AFTER:-}"      # A_eval | B_collect

FAM="${VARIANT}1.0-0.05"
PORTO=./porto_data_v6
S=sets_v6/$VARIANT/res
D=sets_v6/$VARIANT/disc
M=sets_v6/$VARIANT/model
L=sets_v6/$VARIANT/log
mkdir -p $L
read -r -a GPUARR <<< "$GPUS"
NG=${#GPUARR[@]}

CK() { echo "$M/BD_v6${VARIANT}_normal_mask_blk$1_bd.pth"; }

gate() { [ "$STOP_AFTER" = "$1" ] && { echo "== STOP_AFTER=$1 =="; exit 0; }; return 0; }

# ---------------------------------------------------------------- A: eval
echo "== A: n_is sweep  variant=$VARIANT  blocks=[$BLOCKS]  n_is=[$NIS]  gpus=[$GPUS] ($(date)) =="
# Each entry is "sentinel@@command". The sentinel is the LAST record file the job
# writes, so its presence means the job ran to completion -- the sweep is therefore
# resumable: relaunching (e.g. to add GPUs) re-runs only what is missing.
JOBS=()
for N in $NIS; do
  for B in $BLOCKS; do
    # seen: -adjonly 1 -> base / IW-only / Adj+IW / Adj-only in one process.
    # CFGS order puts adjonly last, so that is the sentinel.
    TG=v6${VARIANT}${B}seenN${N}
    JOBS+=("$S/em_pc/${TG}_adjonly_P1P3_em_pc_records.csv@@python -u tools/eval/three_way_postproc.py \
        -family $FAM -dver v6 -porto $PORTO -res_path $S \
        -order l2r -n_is $N -eval_num $EVALN -adjonly 1 -ckpt $(CK $B) \
        -disc $D/BDdisc_f${FAM}_p1_e0_model_blk${B}.pth -tag $TG")
    # unseen: e99 discriminator. base / Adj-only do not depend on the discriminator,
    # so this job runs base / IW-only / Adj+IW only and Adj+IW is last.
    TG=v6${VARIANT}${B}unsN${N}
    JOBS+=("$S/em_pc/${TG}_adj+modelD_P1P3_em_pc_records.csv@@python -u tools/eval/three_way_postproc.py \
        -family $FAM -dver v6 -porto $PORTO -res_path $S \
        -order l2r -n_is $N -eval_num $EVALN -ckpt $(CK $B) \
        -disc $D/BDdisc_f${FAM}_p1_e99_model_blk${B}.pth -tag $TG")
  done
done
echo "   ${#JOBS[@]} jobs over $NG gpu(s)"
for k in $(seq 0 $((NG-1))); do
  ( g=${GPUARR[$k]}
    for i in "${!JOBS[@]}"; do
      if (( i % NG == k )); then
        SENT="${JOBS[$i]%%@@*}"
        CMD="${JOBS[$i]#*@@}"
        if [ -s "$SENT" ]; then
          echo "[gpu$g] nis job $i SKIP (done)"
          continue
        fi
        echo "[gpu$g] nis job $i  ($(date +%H:%M:%S))"
        CUDA_VISIBLE_DEVICES=$g $CMD > $L/nis_job$i.log 2>&1 \
          || echo "NIS JOB $i FAILED"
        echo "[gpu$g] nis job $i done ($(date +%H:%M:%S))"
      fi
    done ) &
done
wait
echo "== A done ($(date)) =="
gate A_eval

# ---------------------------------------------------------------- B: collect
echo "== B: collect tables ($(date)) =="
BL=$(echo $COLLECT_BLOCKS | tr ' ' ',')
for N in $NIS_ALL; do
  # n_is=100 lives under the gate-D tags, everything else under the sweep tags
  if [ "$N" = "100" ]; then
    TS="v6${VARIANT}{B}seen"; TU="v6${VARIANT}{B}uns"
  else
    TS="v6${VARIANT}{B}seenN${N}"; TU="v6${VARIANT}{B}unsN${N}"
  fi
  CF="base,adjonly,modelD,adj+modelD"
  LOG=$L/nis_collect_${N}.log
  : > $LOG
  # simplification first: writes {tag}_{cfg}_{rawL|P1P3L}_em_pc_records.csv from the
  # raw / P1P3 records, so collect_va_table can then read all four variants uniformly
  for IN_OUT in "raw rawL" "P1P3 P1P3L"; do
    set -- $IN_OUT
    python -u tools/collect/postproc_loopcut.py -family $FAM -dver v6 -porto $PORTO \
      -res_path $S -blocks "$BL" -variant "$1" -out_variant "$2" -cfgs "$CF" \
      -pairs "seen:${TS},unseen:${TU}" >> $LOG 2>&1 \
      || echo "LOOPCUT n=$N $1 FAILED"
  done
  for VAR in raw rawL P1P3 P1P3L; do
    SFX=""; [ "$VAR" != "raw" ] && SFX="_$VAR"
    python -u tools/collect/collect_va_table.py -family $FAM -dver v6 -porto $PORTO \
      -res_path $S -blocks "$BL" -shape 1 -variant $VAR -cfgs "$CF" \
      -pairs "seen:${TS},unseen:${TU}" \
      -out va_table_v6${VARIANT}_nis${N}${SFX}.json >> $LOG 2>&1 \
      || echo "COLLECT n=$N $VAR FAILED"
  done
  echo "  n_is=$N collected"
done
ls -la $S/va_table_v6${VARIANT}_nis*.json
echo "== B done ($(date)) =="
gate B_collect

# ---------------------------------------------------------------- C: diversity
echo "== C: candidate-diversity diagnostic vs n_is ($(date)) =="
CUDA_VISIBLE_DEVICES=${GPUARR[0]} python -u tools/diag/diag_proposal_diversity.py \
  -variant $VARIANT -blocks "$BL" -nis_list "$(echo $NIS_ALL | tr ' ' ',')" \
  -n 100 -out $S/nis_diversity_v6${VARIANT}.json \
  > $L/nis_diversity.log 2>&1 || echo "DIVERSITY FAILED"
tail -20 $L/nis_diversity.log
echo "== ALL DONE ($(date)) =="
