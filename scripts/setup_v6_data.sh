#!/bin/bash
# v6 데이터셋(porto_openstreetmap/path_data/v6) 실험 환경 준비 -- 서버에서 실행.
#
# 원본 파일명 : porto_{G,A,NZ,SP}-{normal|except_e}.{pkl|ts}
# 링크 파일명 : porto_shrink_{G,A,NZ,SP}_v6-{LE1.0-0.05|LP1.0-0.05}_{tag}.{pkl|ts}
#
# 이 하나의 이름 규칙이 파이프라인 전체를 커버한다:
#   main_bd.py   loader: {d_name}_shrink_G_{index}      -> -d_name porto -shortest_org_idx v6-LE1.0-0.05_normal
#   disc/eval    : porto_shrink_SP_{dver}-{family}_{tag} -> -dver v6 -family LE1.0-0.05
#   jsev         : porto_shrink_A_{norm_ver}_normal      -> -norm_ver v6-LE1.0-0.05
# 18GB는 복사하지 않고 심볼릭 링크로 이름만 맞춘다.
#
# 산출물 트리 (LE/LP가 부모를 공유):
#   sets_v6/{LE,LP}/{model,disc,res,log}
#   sets_data_v6/porto_node2vec.pkl        <- LE/LP 그래프가 동일하므로 공유
set -eu
ROOT=/home/aailab/wp03052/Synthetic-Data/Block-Diffusion-OD-Planning
SRC=/home/aailab/wp03052/Synthetic-Data/porto_openstreetmap/path_data/v6
DST=$ROOT/porto_data_v6

mkdir -p "$DST" "$ROOT/sets_data_v6"
for v in LE LP; do
  mkdir -p "$ROOT/sets_v6/$v"/{model,disc,res,log}
done

link_family () {          # $1 = 원본 하위폴더, $2 = 버전 태그(LE1.0-0.05)
  local sub=$1 fam=$2
  for tag in normal $(for e in $(seq 0 99); do echo except_$e; done); do
    ln -sfn "$SRC/$sub/porto_G-${tag}.pkl"  "$DST/porto_shrink_G_v6-${fam}_${tag}.pkl"
    ln -sfn "$SRC/$sub/porto_A-${tag}.ts"   "$DST/porto_shrink_A_v6-${fam}_${tag}.ts"
    ln -sfn "$SRC/$sub/porto_NZ-${tag}.pkl" "$DST/porto_shrink_NZ_v6-${fam}_${tag}.pkl"
    ln -sfn "$SRC/$sub/porto_SP-${tag}.pkl" "$DST/porto_shrink_SP_v6-${fam}_${tag}.pkl"
  done
  ln -sfn "$SRC/$sub/porto_tracking-A_0.05.pkl" "$DST/porto_tracking_v6-${fam}.pkl"
}

# 구 명명(porto_v6_shrink_*)으로 만든 링크가 있으면 정리
find "$DST" -maxdepth 1 -name 'porto_v6_shrink_*' -delete 2>/dev/null || true

link_family LE_1.0_0.05 "LE1.0-0.05"
link_family LP_1.0_0.05 "LP1.0-0.05"

echo "links: $(ls "$DST" | wc -l)"
for f in "$DST"/porto_shrink_SP_v6-LE1.0-0.05_normal.pkl \
         "$DST"/porto_shrink_SP_v6-LE1.0-0.05_except_0.pkl \
         "$DST"/porto_shrink_A_v6-LE1.0-0.05_except_99.ts \
         "$DST"/porto_shrink_SP_v6-LP1.0-0.05_normal.pkl; do
  [ -r "$f" ] || { echo "BROKEN: $f"; exit 1; }
done

# node2vec 캐시 이전 (구 준비 스크립트가 sets_data/porto_v6_* 로 만들어 둔 경우)
for a in node2vec path; do
  if [ -f "$ROOT/sets_data/porto_v6_${a}.pkl" ] && [ ! -f "$ROOT/sets_data_v6/porto_${a}.pkl" ]; then
    mv "$ROOT/sets_data/porto_v6_${a}.pkl" "$ROOT/sets_data_v6/porto_${a}.pkl"
    echo "moved node2vec cache: sets_data/porto_v6_${a}.pkl -> sets_data_v6/porto_${a}.pkl"
  fi
done
ls -la "$ROOT/sets_data_v6/" || true
echo "SETUP_V6_OK"
