#!/bin/bash
# ============================================================================
# v6 데이터 전체 파이프라인 — iclr_iw_exp.pdf §5.6.1/§5.6.2 표를 새 데이터에서 재현
#
#   usage:  VARIANT=LE bash scripts/run_v6_all.sh              (기본 LE, GPU 0-3)
#           VARIANT=LP bash scripts/run_v6_all.sh
#           VARIANT=LP GPUS="2 3" bash scripts/run_v6_all.sh
#   중단  :  VARIANT=LP GPUS="0 1" STOP_AFTER=C_disc bash scripts/run_v6_all.sh
#            (D 단계는 GPU 메모리를 거의 다 쓰므로 GPU 공유 시 여기서 끊는다)    (GPU 2,3 만 사용)
#   tmux :  tmux new -d -s v6LE 'bash scripts/run_v6_all.sh 2>&1 | tee sets_v6/LE/log/run_all.log'
#
#   A) 백본        mask blk 1..64 (7개)          normal 경로로 학습
#   B) uncond pool 블록별 20,000개                model-negative의 분모 p_theta
#   C) 판별자      e0 7개 + e99 7개               positives=except 실데이터 1%, negatives=pool
#   D) 평가        three_way x14 (order=l2r)      seen 7 + unseen 7, seen쪽은 -adjonly 1
#   E) 표 조립     collect_va_table -shape 1      valid/arrival/v&a/EM/DTW/LCS (PC 제외)
#   F) JSEV        uncond pool 20,000개 기준       (1,000쌍 표에는 표본편향이 커서 분리)
#
# 각 스테이지는 sets_v6/$VARIANT/log/*.marker 로 완료를 기록하며 재실행 시 건너뛴다(재개 가능).
# ============================================================================
set -u
cd /home/aailab/wp03052/Synthetic-Data/Block-Diffusion-OD-Planning
source /home/aailab/wp03052/venvs/od_planning_venv/bin/activate

VARIANT="${VARIANT:-LE}"
BLOCKS="${BLOCKS:-1 2 4 8 16 32 64}"
# 사용할 GPU 목록. 다른 실험과 서버를 공유할 때는 반드시 좁혀서 줄 것.
GPUS="${GPUS:-0 1 2 3}"
GPUARR=($GPUS); NG=${#GPUARR[@]}
FAM="${VARIANT}1.0-0.05"
R=sets_v6/$VARIANT
M=$R/model; D=$R/disc; S=$R/res; L=$R/log
PORTO=./porto_data_v6
N2V=./sets_data_v6/porto_node2vec.pkl
NPOOL="${NPOOL:-20000}"
STEPS="${STEPS:-4000}"
NIS="${NIS:-100}"
EVALN="${EVALN:-1000}"
mkdir -p $M $D $S $L
CK(){ echo $M/BD_v6${VARIANT}_normal_mask_blk$1_bd.pth; }

stage_done(){ [ -f "$L/$1.marker" ]; }
mark(){ echo "$1 $(date)" > "$L/$1.marker"; }
# STOP_AFTER=A_train|B_pool|C_disc|D_eval 이면 그 단계까지만 하고 멈춘다.
# GPU 를 다른 실험과 공유해 메모리가 부족할 때(특히 D 단계는 n_is x batch 만큼
# 판별자에 넣어 24GB 를 거의 다 쓴다) 안전하게 끊는 용도.
gate(){ if [ "${STOP_AFTER:-}" = "$1" ]; then
          echo "== STOP_AFTER=$1 -> 여기서 멈춘다 ($(date)) =="; exit 0
        fi; return 0; }

echo "############ v6 pipeline  VARIANT=$VARIANT  blocks=[$BLOCKS]  gpus=[$GPUS]  ($(date)) ############"

# ---------------------------------------------------------------- A: 백본
if stage_done A_train; then echo "== SKIP A (done) =="; else
echo "== A: 백본 학습 ($(date)) =="
for k in $(seq 0 $((NG-1))); do eval "lane$k=\"\""; done
i=0
for B in $BLOCKS; do
  k=$((i % NG)); i=$((i+1))
  eval "lane$k=\"\$lane$k $B\""
done
for k in $(seq 0 $((NG-1))); do
  eval "blks=\$lane$k"
  g=${GPUARR[$k]}
  echo "  gpu$g <-blk:$blks"
  ( for B in $blks; do bash scripts/train_v6_bd.sh $B $g $VARIANT > $L/train_blk$B.log 2>&1; done ) &
done
wait
n=0; for B in $BLOCKS; do [ -f "$(CK $B)" ] && n=$((n+1)) || echo "  MISSING ckpt blk$B"; done
echo "== A done: $n ckpt ($(date)) =="
[ "$n" -eq "$(echo $BLOCKS | wc -w)" ] || { echo "STAGE A INCOMPLETE - ABORT"; exit 1; }
mark A_train
fi
gate A_train

# ---------------------------------------------------------------- B: pool
if stage_done B_pool; then echo "== SKIP B (done) =="; else
echo "== B: uncond pool ($(date)) =="
i=0
for B in $BLOCKS; do
  g=${GPUARR[$((i % NG))]}; i=$((i+1))
  ( CUDA_VISIBLE_DEVICES=$g python -u tools/gen/gen_bd_uncond_pool.py \
      -ckpt $(CK $B) -out $D/uncond_pool_blk$B.pth -n $NPOOL > $L/pool_blk$B.log 2>&1 ) &
  [ $((i % NG)) -eq 0 ] && wait
done
wait
ls -la $D/uncond_pool_blk*.pth
mark B_pool
fi
gate B_pool

# ---------------------------------------------------------------- C: 판별자
if stage_done C_disc; then echo "== SKIP C (done) =="; else
echo "== C: 판별자 학습 (e0 / e99) ($(date)) =="
# e0: 블록별 개별 프로세스, 4 GPU 분산
i=0
for B in $BLOCKS; do
  g=${GPUARR[$((i % NG))]}; i=$((i+1))
  ( CUDA_VISIBLE_DEVICES=$g python -u tools/train/train_bd_disc.py \
      -family $FAM -dver v6 -porto $PORTO -n2v $N2V -out $D \
      -frac 1 -exp e0 -neg model -steps $STEPS -pool $D/uncond_pool_blk$B.pth \
      > $L/disc_e0_blk$B.log 2>&1 ) &
  [ $((i % NG)) -eq 0 ] && wait
done
wait
# e99: 99개 시나리오 positives를 한 번만 로드하고 블록별로 재사용 (1 프로세스)
JOBSPEC=""
for B in $BLOCKS; do JOBSPEC+="$B:$D/uncond_pool_blk$B.pth,"; done
CUDA_VISIBLE_DEVICES=${GPUARR[0]} python -u tools/train/train_e99_model_multi.py \
  -family $FAM -dver v6 -porto $PORTO -n2v $N2V -out $D \
  -frac 1 -steps $STEPS -jobs "${JOBSPEC%,}" > $L/disc_e99.log 2>&1 || echo "e99 FAILED"
ls -la $D/BDdisc_*.pth
mark C_disc
fi
gate C_disc

# ---------------------------------------------------------------- D: 평가
if stage_done D_eval; then echo "== SKIP D (done) =="; else
echo "== D: three_way 평가 x14 (order=l2r) ($(date)) =="
JOBS=()
for B in $BLOCKS; do
  # seen: -adjonly 1 -> base / IW-only(modelD) / Adj+IW / Adj-only 네 arm을 한 번에
  JOBS+=("python -u tools/eval/three_way_postproc.py -family $FAM -dver v6 -porto $PORTO -res_path $S \
      -order l2r -n_is $NIS -eval_num $EVALN -adjonly 1 -ckpt $(CK $B) \
      -disc $D/BDdisc_f${FAM}_p1_e0_model_blk${B}.pth -tag v6${VARIANT}${B}seen")
  # unseen: e99 판별자. Adj-only/base는 판별자와 무관하므로 seen 실행분을 재사용
  JOBS+=("python -u tools/eval/three_way_postproc.py -family $FAM -dver v6 -porto $PORTO -res_path $S \
      -order l2r -n_is $NIS -eval_num $EVALN -ckpt $(CK $B) \
      -disc $D/BDdisc_f${FAM}_p1_e99_model_blk${B}.pth -tag v6${VARIANT}${B}uns")
done
for k in $(seq 0 $((NG-1))); do
  ( g=${GPUARR[$k]}
    for i in "${!JOBS[@]}"; do
      if (( i % NG == k )); then
        echo "[gpu$g] job $i"
        CUDA_VISIBLE_DEVICES=$g ${JOBS[$i]} > $L/eval_job$i.log 2>&1 || echo "EVAL JOB $i FAILED"
      fi
    done ) &
done
wait
mark D_eval
fi
gate D_eval

# ---------------------------------------------------------------- E: 표
# unseen 그룹에는 adjonly 레코드가 없다(판별자 무관 arm이라 seen 실행분을 공유).
# 따라서 로그의 'MISSING ..._uns_adjonly_raw...' 줄은 정상이며, 보고서는 seen쪽 adjonly를 쓴다.
echo "== E: 표 조립 ($(date)) =="
BL=$(echo $BLOCKS | tr ' ' ',')
python -u tools/collect/collect_va_table.py -family $FAM -dver v6 -porto $PORTO -res_path $S \
  -blocks $BL -shape 1 -eval_num $EVALN -cfgs base,modelD,adj+modelD,adjonly \
  -pairs "seen:v6${VARIANT}{B}seen,unseen:v6${VARIANT}{B}uns" \
  -out va_table_v6${VARIANT}.json > $S/va_table_v6${VARIANT}.log 2>&1 || echo "COLLECT FAILED"
tail -40 $S/va_table_v6${VARIANT}.log

# ---------------------------------------------------------------- F: JSEV
echo "== F: JSEV (uncond pool $NPOOL개 기준) ($(date)) =="
python -u tools/eval/eval_uncond_jsev.py -blocks $BL \
  -pool_pat "$D/uncond_pool_blk{B}.pth" -porto $PORTO -norm_ver "v6-${FAM}" -n_real $NPOOL \
  -res_path $S -out jsev_v6${VARIANT}.json > $S/jsev_v6${VARIANT}.log 2>&1 || echo "JSEV FAILED"
tail -12 $S/jsev_v6${VARIANT}.log

mark ALL
echo "############ V6_${VARIANT}_ALL_DONE ($(date)) ############"
