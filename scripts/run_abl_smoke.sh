#!/bin/bash
# 100-pair validation of the 4-arm ablation patch (checks the adjonly branch:
# plan_guided(disc=None, adj_prop=True)). Should print four *_raw lines.
set -u
cd /home/aailab/wp03052/Synthetic-Data/Block-Diffusion-OD-Planning
source /home/aailab/wp03052/venvs/od_planning_venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python -u tools/eval/three_way_postproc.py -family 0.05 -adjonly 1 -order l2r \
  -eval_num 100 -ckpt sets_model/BD_porto_v3_normal_mask_blk4_v4_bd.pth \
  -disc sets_disc/BDdisc_f0.05_p1_e0_model_blk4_v4.pth -tag ABLSMOKE
echo ABL_SMOKE_DONE
