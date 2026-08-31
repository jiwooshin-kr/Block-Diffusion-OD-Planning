#!/bin/bash
# ============================================================================
# v6 D-CBG (Schiff et al.) 비교 라운드 — IW 쪽(run_v6_all.sh D단계)과 프로토콜 일치
#
#   usage:  VARIANT=LP GPUS="2 3" bash scripts/run_v6_dcbg.sh
#           VARIANT=LP GPUS="2 3" BLOCKS="2 64" REGIMES="e0" bash scripts/run_v6_dcbg.sh
#   gamma :  VARIANT=LE GPUS="2 3" GAMMA=3 ARMS="dcbg dcbg+adjp" WAVE=_g3 bash scripts/run_v6_dcbg.sh
#            -> arm 이름이 dcbg@g3 / dcbg+adjp@g3 이 되어 gamma=1 레코드를 덮지 않는다.
#               분류기는 gamma 와 무관하므로 단계 A 는 마커로 건너뛴다(평가만 재실행).
#   tmux :  tmux new -d -s v6LPdcbg 'VARIANT=LP GPUS="2 3" bash scripts/run_v6_dcbg.sh 2>&1 | tee sets_v6/LP/log/run_dcbg.log'
#
#   전제: run_v6_all.sh 의 A(백본) + B(uncond pool) 완료.  D단계(IW 평가)까지 끝나 있으면
#         마지막 표 조립에서 IW arm 과 D-CBG arm 이 한 표로 합쳐진다.
#
#   A) 분류기  DCBGclf x (블록 x regime)   노이즈 조건부 p(y|x_t,t). IW 판별자와 프로토콜 일치:
#              -adj 1 (같은 GraphSAGE+edge 채널) / -neg model (같은 uncond pool) / -frac 1 / -steps 4000
#              IW 판별자를 재사용할 수 없다: 시간 조건 없음 + MASK 토큰 없음 + get_log_probs 없음
#   B) 평가    블록 x regime x arm.  arm 이름 = cfg 이름 = 표의 행 이름
#              dcbg          exact 열거,  마스킹 없음        <-> IW 쪽 modelD
#              dcbg+adjp     exact 열거,  Lemma-3 마스킹     <-> IW 쪽 adj+modelD
#              dcbgfo        1차 Taylor,  마스킹 없음        (같은 분류기, -approx 1)
#              dcbgfo+adjp   1차 Taylor,  Lemma-3 마스킹
#              exact/근사는 학습된 분류기가 같고 런타임 플래그만 다르다(별도 모델이 아님).
#              order=l2r, gamma=1.0(=IW 쪽 w_gamma 기본값)로 두 방법의 guidance 강도를 맞춘다
#   C) 표      collect_va_table 로 base/modelD/adj+modelD/adjonly + dcbg/dcbg+adjp 6 arm 한 표
#
#   비용(2026-08-19 v6 LE blk4 실측, 20쌍): planning 은 싸다 —
#         dcbg 0.40 s/path, dcbg+adjp 0.95 s/path (v4 시절 추정 2.9 s/path 보다 훨씬 빠름).
#         병목은 GPU 가 아니라 evaluate_em_pc + nx.shortest_path 후처리 채점(CPU, 변형 4개).
#         즉 네 arm 의 config 당 비용이 대체로 비슷하므로 "싼 arm 먼저"가 아니라
#         "exact 먼저(헤드라인), fo 나중"으로 wave 를 나눈다.
#
#   각 스테이지는 sets_v6/$VARIANT/log/*.marker 로 완료를 기록하며 재실행 시 건너뛴다.
# ============================================================================
set -u
cd /home/aailab/wp03052/Synthetic-Data/Block-Diffusion-OD-Planning
source /home/aailab/wp03052/venvs/od_planning_venv/bin/activate

VARIANT="${VARIANT:-LE}"
BLOCKS="${BLOCKS:-1 2 4 8 16 32 64}"
REGIMES="${REGIMES:-e0 e99}"          # e0 = seen 시나리오, e99 = unseen
# arm 순서 = 실행 순서. 기본값은 "싼 것 먼저" — dcbg(exact, 마스킹 없음)만 압도적으로
# 비싸므로(아래 비용 주석) 그 arm 은 별도 wave 로 돌려야 표를 빨리 볼 수 있다.
#   wave1:  ARMS="dcbg dcbg+adjp"     WAVE=_w1   exact 2 arm = 헤드라인 표
#   wave2:  ARMS="dcbgfo dcbgfo+adjp" WAVE=_w2   1차 근사 2 arm (연산예산 비교용)
# 표 조립은 그때까지 저장된 dcbg* arm 전부를 자동으로 모으므로 wave1 만 끝나도 표가 나온다.
ARMS="${ARMS:-dcbg dcbg+adjp}"
# 마커/표 이름 접미사. wave 를 나눠 돌릴 때 서로 덮어쓰지 않게 한다.
WAVE="${WAVE:-}"
# 사용할 GPU 목록. 다른 실험과 서버를 공유할 때는 반드시 좁혀서 줄 것.
GPUS="${GPUS:-0 1 2 3}"
GPUARR=($GPUS); NG=${#GPUARR[@]}

FAM="${VARIANT}1.0-0.05"
R=sets_v6/$VARIANT
M=$R/model; D=$R/disc; S=$R/res; L=$R/log
PORTO=./porto_data_v6
N2V=./sets_data_v6/porto_node2vec.pkl
STEPS="${STEPS:-4000}"
EVALN="${EVALN:-1000}"
GAMMA="${GAMMA:-1.0}"
# gamma != 1.0 이면 arm(=cfg=표의 행) 이름에 gamma 를 박는다. 이게 없으면 레코드 파일명
# {tag}_{cfg}_{variant} 이 gamma 별로 같아져서 앞서 돌린 gamma 결과를 덮어쓴다.
# '@' 는 collect_va_table 의 arm 추출 정규식 (dcbg[^_]*) 을 통과한다(밑줄이 아니므로).
GTAG=""
[ "$GAMMA" != "1.0" ] && GTAG="@g$(echo "$GAMMA" | sed 's/\.0$//')"
SFX="dcbg_v6${VARIANT}"
mkdir -p $D $S $L

CK(){   echo $M/BD_v6${VARIANT}_normal_mask_blk$1_bd.pth; }
POOL(){ echo $D/uncond_pool_blk$1.pth; }
# train_dcbg_classifier.py 의 출력 이름 규칙: DCBGclf_{kernel}_blk{B}_f{fam}_p{frac}[_adj][_e99][_modelneg]{outsfx}
CLF(){  local B=$1 RG=$2; local e99=""; [ "$RG" = "e99" ] && e99="_e99"
        echo $D/DCBGclf_mask_blk${B}_f${FAM}_p1_adj${e99}_modelneg_${SFX}.pth; }
# IW 쪽 three_way 실행과 같은 -tag 를 쓴다 -> collect_va_table 이 arm(cfg)만 추가로 읽으면 된다
TAG(){  local B=$1 RG=$2; local w="seen"; [ "$RG" = "e99" ] && w="uns"
        echo v6${VARIANT}${B}${w}; }

stage_done(){ [ -f "$L/$1.marker" ]; }
mark(){ echo "$1 $(date)" > "$L/$1.marker"; }

# 라운드로빈으로 JOBS 배열을 GPUS 에 뿌린다
run_jobs(){
  local -n JJ=$1; local pfx=$2
  echo "  ${#JJ[@]} jobs over gpus [$GPUS]"
  for k in $(seq 0 $((NG-1))); do
    ( g=${GPUARR[$k]}
      for i in "${!JJ[@]}"; do
        if (( i % NG == k )); then
          echo "[gpu$g] $pfx job $i: ${JJ[$i]}"
          CUDA_VISIBLE_DEVICES=$g ${JJ[$i]} > $L/${pfx}_job$i.log 2>&1 \
            || echo "$pfx JOB $i FAILED"
        fi
      done ) &
  done
  wait
}

echo "############ v6 D-CBG  VARIANT=$VARIANT  blocks=[$BLOCKS]  regimes=[$REGIMES]  arms=[$ARMS]  gpus=[$GPUS]  gamma=$GAMMA  ($(date)) ############"

# 전제 확인: 백본과 pool 이 있어야 한다
miss=0
for B in $BLOCKS; do
  [ -f "$(CK $B)"   ] || { echo "MISSING ckpt $(CK $B)"; miss=1; }
  [ -f "$(POOL $B)" ] || { echo "MISSING pool $(POOL $B)"; miss=1; }
done
[ "$miss" -eq 0 ] || { echo "전제(run_v6_all.sh A/B) 미완 - ABORT"; exit 1; }

# ---------------------------------------------------------------- A: 분류기
if stage_done DCBG_A_clf; then echo "== SKIP A (done) =="; else
echo "== A: D-CBG 분류기 학습 ($(date)) =="
JOBS=()
# e99 가 99 시나리오 로드로 느리므로 먼저 넣어 라운드로빈이 고르게 퍼지게 한다
for RG in $REGIMES; do
  [ "$RG" = "e99" ] || continue
  for B in $BLOCKS; do
    JOBS+=("python -u tools/train/train_dcbg_classifier.py -kernel mask -blk $B \
        -family $FAM -dver v6 -porto $PORTO -n2v $N2V -out $D \
        -adj 1 -neg model -pool $(POOL $B) -frac 1 -exp e99 -steps $STEPS -outsfx _${SFX}")
  done
done
for RG in $REGIMES; do
  [ "$RG" = "e0" ] || continue
  for B in $BLOCKS; do
    JOBS+=("python -u tools/train/train_dcbg_classifier.py -kernel mask -blk $B \
        -family $FAM -dver v6 -porto $PORTO -n2v $N2V -out $D \
        -adj 1 -neg model -pool $(POOL $B) -frac 1 -exp e0 -steps $STEPS -outsfx _${SFX}")
  done
done
run_jobs JOBS dcbgclf
n=0; for RG in $REGIMES; do for B in $BLOCKS; do
  [ -f "$(CLF $B $RG)" ] && n=$((n+1)) || echo "  MISSING clf $(CLF $B $RG)"
done; done
echo "== A done: $n clf ($(date)) =="
[ "$n" -eq "${#JOBS[@]}" ] || { echo "STAGE A INCOMPLETE - ABORT"; exit 1; }
mark DCBG_A_clf
fi

# ---------------------------------------------------------------- B: 평가
if stage_done DCBG_B_eval$WAVE; then echo "== SKIP B (done) =="; else
echo "== B: D-CBG 평가 ($(date)) =="
JOBS=()
for ARM in $ARMS; do
  # arm 이름에서 런타임 플래그를 읽는다: *+adjp -> Lemma-3 마스킹, *fo* -> 1차 근사
  case "$ARM" in *"+adjp") AP=1 ;; *) AP=0 ;; esac
  case "$ARM" in *fo*)     AX=1 ;; *) AX=0 ;; esac
  for RG in $REGIMES; do for B in $BLOCKS; do
    JOBS+=("python -u tools/eval/eval_dcbg.py -kernel mask -blk $B -gamma $GAMMA \
        -adj 1 -approx $AX -adj_prop $AP -order l2r \
        -family $FAM -dver v6 -porto $PORTO -res_path $S -eval_num $EVALN \
        -ckpt $(CK $B) -clf $(CLF $B $RG) -tag $(TAG $B $RG) -cfg ${ARM}${GTAG}")
  done; done
done
run_jobs JOBS dcbgeval$WAVE
echo "== B done ($(date)) =="
grep -h -E "sec_per_path|_raw |_P1P3 " $L/dcbgeval${WAVE}_job*.log 2>/dev/null | sed 's/  */ /g'
mark DCBG_B_eval$WAVE
fi

# ---------------------------------------------------------------- C: 표
# IW arm(base/modelD/adj+modelD/adjonly)은 run_v6_all.sh D단계 산출물을 그대로 읽는다.
# 그 단계가 아직이면 해당 줄만 MISSING 으로 빠지고 D-CBG arm 은 정상 수집된다.
echo "== C: 표 조립 (IW + D-CBG 6 arm) ($(date)) =="
BL=$(echo $BLOCKS | tr ' ' ',')
# 이번 wave 의 arm 만이 아니라 res/em_pc 에 실제로 저장된 dcbg* arm 전부를 모은다
# (wave 를 나눠 돌려도 표가 항상 "지금까지 끝난 전부"가 되도록).
DCBG_CFGS=$(ls $S/em_pc/ 2>/dev/null \
  | sed -nE "s/^v6${VARIANT}[0-9]+(seen|uns)_(dcbg[^_]*)_raw_em_pc_records\.csv$/\2/p" \
  | sort -u | paste -sd, -)
CFGS="base,modelD,adj+modelD,adjonly${DCBG_CFGS:+,$DCBG_CFGS}"
echo "  cfgs=$CFGS"
python -u tools/collect/collect_va_table.py -family $FAM -dver v6 -porto $PORTO -res_path $S \
  -blocks $BL -shape 1 -eval_num $EVALN -cfgs "$CFGS" \
  -pairs "seen:v6${VARIANT}{B}seen,unseen:v6${VARIANT}{B}uns" \
  -out va_table_v6${VARIANT}_dcbg.json > $S/va_table_v6${VARIANT}_dcbg${WAVE}.log 2>&1 || echo "COLLECT FAILED"
tail -80 $S/va_table_v6${VARIANT}_dcbg${WAVE}.log

mark DCBG_ALL$WAVE
echo "############ V6_${VARIANT}_DCBG_DONE ($(date)) ############"
