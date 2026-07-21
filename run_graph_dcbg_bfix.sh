#!/bin/bash
# Retrain block-structured (bfix) graph D-CBG classifiers + re-evaluate.
# Fixes: corrupt_graph now committed-clean-prefix + current-block-only noise;
# plan_dcbg_graph feeds the full canvas to the classifier.
set -u
cd /home/aailab/wp03052/Synthetic-Data/Block-Diffusion-OD-Planning
source /home/aailab/wp03052/syn_data/bin/activate
D=sets_disc; M=sets_model; L=sets_log; R=sets_res
mkdir -p $L
CK() { echo $M/BD_porto_v3_normal_graph_blk$1_v2_bd.pth; }

echo "=== PHASE 1: train bfix classifiers ==="
# lane 0
( for B in 4 8; do
    CUDA_VISIBLE_DEVICES=0 python -u train_dcbg_classifier.py -kernel graph -blk $B -adj 1 -graph_ckpt $(CK $B) -outsfx _bfix > $L/clfbfix_g${B}d.log 2>&1
    CUDA_VISIBLE_DEVICES=0 python -u train_dcbg_classifier.py -kernel graph -blk $B -adj 1 -neg model -pool $D/uncond_pool_graph$B.pth -graph_ckpt $(CK $B) -outsfx _bfix > $L/clfbfix_g${B}m.log 2>&1
  done ) &
# lane 1
( for B in 16 32; do
    CUDA_VISIBLE_DEVICES=1 python -u train_dcbg_classifier.py -kernel graph -blk $B -adj 1 -graph_ckpt $(CK $B) -outsfx _bfix > $L/clfbfix_g${B}d.log 2>&1
    CUDA_VISIBLE_DEVICES=1 python -u train_dcbg_classifier.py -kernel graph -blk $B -adj 1 -neg model -pool $D/uncond_pool_graph$B.pth -graph_ckpt $(CK $B) -outsfx _bfix > $L/clfbfix_g${B}m.log 2>&1
  done ) &
# lane 2
( CUDA_VISIBLE_DEVICES=2 python -u train_dcbg_classifier.py -kernel graph -blk 64 -adj 1 -graph_ckpt $(CK 64) -outsfx _bfix > $L/clfbfix_g64d.log 2>&1
  CUDA_VISIBLE_DEVICES=2 python -u train_dcbg_classifier.py -kernel graph -blk 64 -adj 1 -neg model -pool $D/uncond_pool_graph64.pth -graph_ckpt $(CK 64) -outsfx _bfix > $L/clfbfix_g64m.log 2>&1 ) &
wait
echo "=== PHASE 1 done; classifiers: ==="
ls $D/DCBGclf_graph_blk*_bfix.pth

echo "=== PHASE 2: eval bfix graph D-CBG (approx, gamma 4, adj) ==="
run_eval() {  # $1=gpu $2=blk $3=negtag(_adj|_adj_modelneg) $4=ressfx
  CUDA_VISIBLE_DEVICES=$1 python -u eval_dcbg.py -kernel graph -blk $2 -gamma 4.0 -adj 1 -approx 1 \
    -ckpt $(CK $2) -res_suffix $4 -clf $D/DCBGclf_graph_blk$2_f0.05_p1$3_bfix.pth > $L/evalbfix_g$2$4.log 2>&1
}
( for B in 4 8;   do run_eval 0 $B _adj _gdbfix; run_eval 0 $B _adj_modelneg _gmbfix; done ) &
( for B in 16 32; do run_eval 1 $B _adj _gdbfix; run_eval 1 $B _adj_modelneg _gmbfix; done ) &
( run_eval 2 64 _adj _gdbfix; run_eval 2 64 _adj_modelneg _gmbfix ) &
wait
echo "=== RESULTS (bfix) ==="
grep -hE "DCBG_graph.*raw|DCBG_graph.*P1P3" $L/evalbfix_g*.log
echo "GRAPH_BFIX_ALL_DONE"
