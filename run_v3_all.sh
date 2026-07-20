#!/bin/bash
# v3 program: re-measure everything on the v2 backbones. Run inside tmux.
set -u
cd "$(dirname "$0")"
D=./sets_disc
L=./sets_log
mkdir -p $L

echo "=== Stage 0: archive v1 disc artifacts ($(date)) ==="
cd $D
for f in BDdisc_f0.05_p1_e0_model_blk*.pth DCBGclf_mask_blk*_adj_modelneg.pth uncond_pool_blk*.pth; do
  [ -f "$f" ] && mv "$f" "v1_$f"
done
cd ..

echo "=== Stage A: v2 uncond pools ($(date)) ==="
(CUDA_VISIBLE_DEVICES=0 python -u gen_bd_uncond_pool.py -ckpt sets_model/BD_porto_v3_normal_mask_blk4_v2_bd.pth  -out $D/uncond_pool_blk4.pth  > $L/v3_pool4.log 2>&1
 CUDA_VISIBLE_DEVICES=0 python -u gen_bd_uncond_pool.py -ckpt sets_model/BD_porto_v3_normal_mask_blk32_v2_bd.pth -out $D/uncond_pool_blk32.pth > $L/v3_pool32.log 2>&1) &
(CUDA_VISIBLE_DEVICES=1 python -u gen_bd_uncond_pool.py -ckpt sets_model/BD_porto_v3_normal_mask_blk8_v2_bd.pth  -out $D/uncond_pool_blk8.pth  > $L/v3_pool8.log 2>&1
 CUDA_VISIBLE_DEVICES=1 python -u gen_bd_uncond_pool.py -ckpt sets_model/BD_porto_v3_normal_mask_blk64_v2_bd.pth -out $D/uncond_pool_blk64.pth > $L/v3_pool64.log 2>&1) &
(CUDA_VISIBLE_DEVICES=2 python -u gen_bd_uncond_pool.py -ckpt sets_model/BD_porto_v3_normal_mask_blk16_v2_bd.pth -out $D/uncond_pool_blk16.pth > $L/v3_pool16.log 2>&1) &
wait

echo "=== Stage B: discriminators/classifiers ($(date)) ==="
(CUDA_VISIBLE_DEVICES=0 python -u train_bd_disc.py -family 0.05 -frac 1 -exp e0 -neg model -pool $D/uncond_pool_blk4.pth  > $L/v3_iwdisc4.log 2>&1
 CUDA_VISIBLE_DEVICES=0 python -u train_bd_disc.py -family 0.05 -frac 1 -exp e0 -neg model -pool $D/uncond_pool_blk32.pth > $L/v3_iwdisc32.log 2>&1
 CUDA_VISIBLE_DEVICES=0 python -u train_dcbg_classifier.py -kernel mask -blk 4  -adj 1 -neg model -pool $D/uncond_pool_blk4.pth  > $L/v3_dcbgclf4.log 2>&1) &
(CUDA_VISIBLE_DEVICES=1 python -u train_bd_disc.py -family 0.05 -frac 1 -exp e0 -neg model -pool $D/uncond_pool_blk8.pth  > $L/v3_iwdisc8.log 2>&1
 CUDA_VISIBLE_DEVICES=1 python -u train_bd_disc.py -family 0.05 -frac 1 -exp e0 -neg model -pool $D/uncond_pool_blk64.pth > $L/v3_iwdisc64.log 2>&1
 CUDA_VISIBLE_DEVICES=1 python -u train_dcbg_classifier.py -kernel mask -blk 8  -adj 1 -neg model -pool $D/uncond_pool_blk8.pth  > $L/v3_dcbgclf8.log 2>&1) &
(CUDA_VISIBLE_DEVICES=2 python -u train_bd_disc.py -family 0.05 -frac 1 -exp e0 -neg model -pool $D/uncond_pool_blk16.pth > $L/v3_iwdisc16.log 2>&1
 CUDA_VISIBLE_DEVICES=2 python -u train_dcbg_classifier.py -kernel mask -blk 16 -adj 1 -neg model -pool $D/uncond_pool_blk16.pth > $L/v3_dcbgclf16.log 2>&1
 CUDA_VISIBLE_DEVICES=2 python -u train_dcbg_classifier.py -kernel mask -blk 16 -adj 1 -exp e99 > $L/v3_dcbgclf16_e99.log 2>&1) &
(CUDA_VISIBLE_DEVICES=3 python -u train_dcbg_classifier.py -kernel mask -blk 32 -adj 1 -neg model -pool $D/uncond_pool_blk32.pth > $L/v3_dcbgclf32.log 2>&1
 CUDA_VISIBLE_DEVICES=3 python -u train_dcbg_classifier.py -kernel mask -blk 64 -adj 1 -neg model -pool $D/uncond_pool_blk64.pth > $L/v3_dcbgclf64.log 2>&1) &
wait
for b in 4 8 16 32 64; do
  mv $D/BDdisc_f0.05_p1_e0_model.pth $D/BDdisc_f0.05_p1_e0_model_blk$b.pth 2>/dev/null
done
# train_bd_disc names include blk from pool path; ensure names exist
ls $D/BDdisc_f0.05_p1_e0_model_blk*.pth

echo "=== Stage C: evaluations ($(date)) ==="
M=./sets_model
JOBS=(
  "python -u eval_v2_postproc.py"
  "python -u eval_v2_anatomy.py"
  "python -u three_way_postproc.py -ckpt $M/BD_porto_v3_normal_mask_blk4_v2_bd.pth  -disc $D/BDdisc_f0.05_p1_e0_model_blk4.pth  -tag v3m4"
  "python -u three_way_postproc.py -ckpt $M/BD_porto_v3_normal_mask_blk8_v2_bd.pth  -disc $D/BDdisc_f0.05_p1_e0_model_blk8.pth  -tag v3m8"
  "python -u three_way_postproc.py -ckpt $M/BD_porto_v3_normal_mask_blk16_v2_bd.pth -disc $D/BDdisc_f0.05_p1_e0_model_blk16.pth -tag v3m16"
  "python -u three_way_postproc.py -ckpt $M/BD_porto_v3_normal_mask_blk32_v2_bd.pth -disc $D/BDdisc_f0.05_p1_e0_model_blk32.pth -tag v3m32"
  "python -u three_way_postproc.py -ckpt $M/BD_porto_v3_normal_mask_blk64_v2_bd.pth -disc $D/BDdisc_f0.05_p1_e0_model_blk64.pth -tag v3m64"
  "python -u three_way_postproc.py -ckpt $M/BD_porto_v3_normal_mask_blk4_v2_bd.pth  -disc $D/BDdisc_f0.05_p1_e99_data.pth -tag v3m4e99"
  "python -u three_way_postproc.py -ckpt $M/BD_porto_v3_normal_mask_blk16_v2_bd.pth -disc $D/BDdisc_f0.05_p1_e99_data.pth -tag v3m16e99"
  "python -u eval_dcbg.py -kernel mask -blk 4  -gamma 4.0 -adj 1 -ckpt $M/BD_porto_v3_normal_mask_blk4_v2_bd.pth  -res_suffix _v3d -clf $D/DCBGclf_mask_blk4_f0.05_p1_adj.pth"
  "python -u eval_dcbg.py -kernel mask -blk 8  -gamma 4.0 -adj 1 -ckpt $M/BD_porto_v3_normal_mask_blk8_v2_bd.pth  -res_suffix _v3d -clf $D/DCBGclf_mask_blk8_f0.05_p1_adj.pth"
  "python -u eval_dcbg.py -kernel mask -blk 16 -gamma 4.0 -adj 1 -ckpt $M/BD_porto_v3_normal_mask_blk16_v2_bd.pth -res_suffix _v3d -clf $D/DCBGclf_mask_blk16_f0.05_p1_adj.pth"
  "python -u eval_dcbg.py -kernel mask -blk 32 -gamma 4.0 -adj 1 -ckpt $M/BD_porto_v3_normal_mask_blk32_v2_bd.pth -res_suffix _v3d -clf $D/DCBGclf_mask_blk32_f0.05_p1_adj.pth"
  "python -u eval_dcbg.py -kernel mask -blk 64 -gamma 4.0 -adj 1 -ckpt $M/BD_porto_v3_normal_mask_blk64_v2_bd.pth -res_suffix _v3d -clf $D/DCBGclf_mask_blk64_f0.05_p1_adj.pth"
  "python -u eval_dcbg.py -kernel mask -blk 4  -gamma 4.0 -adj 1 -ckpt $M/BD_porto_v3_normal_mask_blk4_v2_bd.pth  -res_suffix _v3m -clf $D/DCBGclf_mask_blk4_f0.05_p1_adj_modelneg.pth"
  "python -u eval_dcbg.py -kernel mask -blk 8  -gamma 4.0 -adj 1 -ckpt $M/BD_porto_v3_normal_mask_blk8_v2_bd.pth  -res_suffix _v3m -clf $D/DCBGclf_mask_blk8_f0.05_p1_adj_modelneg.pth"
  "python -u eval_dcbg.py -kernel mask -blk 16 -gamma 4.0 -adj 1 -ckpt $M/BD_porto_v3_normal_mask_blk16_v2_bd.pth -res_suffix _v3m -clf $D/DCBGclf_mask_blk16_f0.05_p1_adj_modelneg.pth"
  "python -u eval_dcbg.py -kernel mask -blk 32 -gamma 4.0 -adj 1 -ckpt $M/BD_porto_v3_normal_mask_blk32_v2_bd.pth -res_suffix _v3m -clf $D/DCBGclf_mask_blk32_f0.05_p1_adj_modelneg.pth"
  "python -u eval_dcbg.py -kernel mask -blk 64 -gamma 4.0 -adj 1 -ckpt $M/BD_porto_v3_normal_mask_blk64_v2_bd.pth -res_suffix _v3m -clf $D/DCBGclf_mask_blk64_f0.05_p1_adj_modelneg.pth"
  "python -u eval_dcbg.py -kernel mask -blk 4  -gamma 4.0 -adj 1 -approx 1 -ckpt $M/BD_porto_v3_normal_mask_blk4_v2_bd.pth -res_suffix _v3fo -clf $D/DCBGclf_mask_blk4_f0.05_p1_adj.pth"
  "python -u eval_dcbg.py -kernel mask -blk 4  -gamma 4.0 -adj 1 -ckpt $M/BD_porto_v3_normal_mask_blk4_v2_bd.pth  -res_suffix _v3e99 -clf $D/DCBGclf_mask_blk4_f0.05_p1_adj_e99.pth"
  "python -u eval_dcbg.py -kernel mask -blk 16 -gamma 4.0 -adj 1 -ckpt $M/BD_porto_v3_normal_mask_blk16_v2_bd.pth -res_suffix _v3e99 -clf $D/DCBGclf_mask_blk16_f0.05_p1_adj_e99.pth"
)
for g in 0 1 2 3; do
  ( for i in "${!JOBS[@]}"; do
      if (( i % 4 == g )); then
        CUDA_VISIBLE_DEVICES=$g ${JOBS[$i]} > $L/v3_job$i.log 2>&1 || echo "JOB $i FAILED"
      fi
    done ) &
done
wait
echo V3_DONE > $L/v3_done.marker
echo "ALL V3 DONE ($(date))"
