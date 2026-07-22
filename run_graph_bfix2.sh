#!/bin/bash
# Resume block-fixed (bfix) graph D-CBG: blk4/16 classifiers already trained.
# Train remaining blk8/32/64 (data + model neg), then evaluate ALL blocks 4..64.
set -u
cd /home/aailab/wp03052/Synthetic-Data/Block-Diffusion-OD-Planning
source /home/aailab/wp03052/syn_data/bin/activate
D=sets_disc; M=sets_model; L=sets_log
mkdir -p $L
CK() { echo $M/BD_porto_v3_normal_graph_blk$1_v2_bd.pth; }

echo "=== PHASE 1: train bfix classifiers blk8/32/64 (data+model) ==="
( CUDA_VISIBLE_DEVICES=0 python -u train_dcbg_classifier.py -kernel graph -blk 8 -adj 1 -graph_ckpt $(CK 8) -outsfx _bfix > $L/clfbfix_g8d.log 2>&1
  CUDA_VISIBLE_DEVICES=0 python -u train_dcbg_classifier.py -kernel graph -blk 8 -adj 1 -neg model -pool $D/uncond_pool_graph8.pth -graph_ckpt $(CK 8) -outsfx _bfix > $L/clfbfix_g8m.log 2>&1 ) &
( CUDA_VISIBLE_DEVICES=1 python -u train_dcbg_classifier.py -kernel graph -blk 32 -adj 1 -graph_ckpt $(CK 32) -outsfx _bfix > $L/clfbfix_g32d.log 2>&1
  CUDA_VISIBLE_DEVICES=1 python -u train_dcbg_classifier.py -kernel graph -blk 32 -adj 1 -neg model -pool $D/uncond_pool_graph32.pth -graph_ckpt $(CK 32) -outsfx _bfix > $L/clfbfix_g32m.log 2>&1 ) &
( CUDA_VISIBLE_DEVICES=2 python -u train_dcbg_classifier.py -kernel graph -blk 64 -adj 1 -graph_ckpt $(CK 64) -outsfx _bfix > $L/clfbfix_g64d.log 2>&1
  CUDA_VISIBLE_DEVICES=2 python -u train_dcbg_classifier.py -kernel graph -blk 64 -adj 1 -neg model -pool $D/uncond_pool_graph64.pth -graph_ckpt $(CK 64) -outsfx _bfix > $L/clfbfix_g64m.log 2>&1 ) &
wait
echo "=== PHASE 1 done; classifiers: ==="
ls $D/DCBGclf_graph_blk*_bfix.pth

echo "=== PHASE 2: eval bfix graph D-CBG (approx, gamma 4, adj) blk4..64 ==="
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
