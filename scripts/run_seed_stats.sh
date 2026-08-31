#!/bin/bash
# Three-seed replication (seeds 8 and 9 on top of the existing seed-7 record)
# for the fam 0.05 / l2r / seen headline arms:
#   base, adj-only, IW-only, adj+IW   (tools/eval/three_way_postproc.py, 4 cfgs per job)
#   Adj+D-CBG exact, gamma in {1, 4}  (tools/eval/eval_dcbg.py, adj_prop, legal-only enum)
# Same 1,000 reserved except_0 OD pairs in every run -- only the sampling seed
# varies, so the spread is the sampler's own variance on the fixed benchmark.
#
# Tags: abl{B}l2rs{S}_{cfg}  /  DCBG_mask{B}_g{G}_adj_adjp_v4l2r_s{S}
# (seed 7 keeps its original tags: abl{B}l2r_{cfg} / ..._adjp_v4l2r)
set -u
cd /home/aailab/wp03052/Synthetic-Data/Block-Diffusion-OD-Planning
source /home/aailab/wp03052/venvs/od_planning_venv/bin/activate
D=sets_disc; M=sets_model; R=sets_res; L=sets_log
mkdir -p $L
CK(){ echo $M/BD_porto_v3_normal_mask_blk$1_v4_bd.pth; }

JOBS=()
for S in 8 9; do
  for B in 1 2 4 8 16 32 64; do
    JOBS+=("python -u tools/eval/three_way_postproc.py -family 0.05 -adjonly 1 -order l2r -seed $S \
      -ckpt $(CK $B) -disc $D/BDdisc_f0.05_p1_e0_model_blk${B}_v4.pth -tag abl${B}l2rs${S}")
  done
done
for S in 8 9; do
  for G in 1.0 4.0; do
    for B in 1 2 4 8 16 32 64; do
      JOBS+=("python -u tools/eval/eval_dcbg.py -kernel mask -blk $B -family 0.05 -gamma $G -adj 1 \
        -approx 0 -adj_prop 1 -order l2r -seed $S -ckpt $(CK $B) \
        -clf $D/DCBGclf_mask_blk${B}_f0.05_p1_adj_modelneg_v4.pth -res_suffix _v4l2r_s${S}")
    done
  done
done

echo "=== seed-stats: ${#JOBS[@]} jobs ($(date)) ==="
for g in 0 1 2 3; do
  ( for i in "${!JOBS[@]}"; do
      if (( i % 4 == g )); then
        echo "[gpu$g] job $i: ${JOBS[$i]}"
        CUDA_VISIBLE_DEVICES=$g ${JOBS[$i]} > $L/seedstats_job$i.log 2>&1 \
          || echo "SEED STATS JOB $i FAILED"
      fi
    done ) &
done
wait
echo "=== evals done ($(date)) ==="

python -u tools/collect/collect_seed_stats.py > $R/seed_stats_f005.log 2>&1 || echo "COLLECT FAILED"
tail -50 $R/seed_stats_f005.log
echo SEED_STATS_DONE > $L/seed_stats_done.marker
echo "SEED_STATS_DONE ($(date))"
