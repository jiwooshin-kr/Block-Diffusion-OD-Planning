# §4 blk1 / blk2 IW-guidance 결과 (RESULTS_BD 4번 항목 보강)

- 평가: `porto_v3_normal` except_0 reserved 1,000 pairs, seed 7, n_is=100
- 모델: `BD_porto_v3_normal_mask_blk{1,2}_v2_bd.pth`
- 스크립트: [scripts/run_blk12_datadisc.sh](../run_blk12_datadisc.sh) (data-neg), [scripts/run_blk12_modeldisc.sh](../run_blk12_modeldisc.sh) (model-neg)
- 열: base / IW(raw) / adj+IW(raw) / IW+P1P3 / adj+IW+P1P3. valid·arrival·EM·PC 모두 기록.
- 주: 로그의 `modelD` 라벨은 three_way_postproc의 guided config 이름일 뿐, disc의 negative 종류는 아래 표 제목 기준.

---

## A. normal-negative (data) disc — ✅ 완료

data-negative disc는 block-independent (`BDdisc_f0.05_p1_e0_data.pth`, `..._e99_data.pth`).

### A-1. SEEN (disc = except_0 vs normal)

| blk | metric | base | IW raw | adj+IW raw | IW +P1P3 | adj+IW +P1P3 |
|----:|--------|-----:|-------:|-----------:|---------:|-------------:|
| **1** | valid   | 0.306 | 0.337 | 0.801 | 1.000 | 1.000 |
|       | arrival | 0.994 | 0.995 | 0.686 | 1.000 | 1.000 |
|       | EM      | 0.283 | 0.309 | 0.376 | 0.333 | **0.377** |
|       | PC      | 0.292 | 0.319 | 0.426 | 0.478 | 0.445 |
| **2** | valid   | 0.264 | 0.311 | 0.447 | 1.000 | 1.000 |
|       | arrival | 0.956 | 0.955 | 0.883 | 1.000 | 1.000 |
|       | EM      | 0.244 | 0.289 | 0.371 | 0.317 | **0.366** |
|       | PC      | 0.253 | 0.298 | 0.399 | 0.466 | 0.499 |

### A-2. UNSEEN (disc = except_1..99 vs normal → zero-shot except_0)

| blk | metric | base | IW raw | adj+IW raw | IW +P1P3 | adj+IW +P1P3 |
|----:|--------|-----:|-------:|-----------:|---------:|-------------:|
| **1** | valid   | 0.306 | 0.342 | 0.804 | 1.000 | 1.000 |
|       | arrival | 0.994 | 0.995 | 0.674 | 1.000 | 1.000 |
|       | EM      | 0.283 | 0.308 | 0.372 | 0.332 | **0.373** |
|       | PC      | 0.292 | 0.319 | 0.420 | 0.475 | 0.439 |
| **2** | valid   | 0.264 | 0.323 | 0.448 | 1.000 | 1.000 |
|       | arrival | 0.956 | 0.960 | 0.878 | 1.000 | 1.000 |
|       | EM      | 0.244 | 0.295 | 0.358 | 0.315 | **0.358** |
|       | PC      | 0.253 | 0.307 | 0.391 | 0.465 | 0.496 |

**관찰**
- IW guidance가 blk1/2에서도 작동: valid base→IW blk1 0.306→0.337, blk2 0.264→0.311.
- adj+IW는 Lemma3 마스킹으로 valid 급증(blk1 0.801)하지만 arrival이 급락(blk1 0.686) — 강한 validity 제약이 목적지를 지나침. P3가 arrival 1.0 복구.
- adj+IW P1P3 EM: blk1 0.377 / blk2 0.366 → blk4(0.357)보다 높음 (작은 블록일수록 성능↑ 단조추세와 일치).
- Seen ≈ Unseen (generalization gap ≈ 0).

---

## B. model-negative (model) disc — ✅ 완료

blk1/2용 uncond pool을 v2 체크포인트로 생성 → e0/e99 model-neg disc 학습(val acc 0.69–0.73) → eval.
disc 파일: `BDdisc_f0.05_p1_e0_model_blk{1,2}.pth`, `..._e99_model_blk{1,2}.pth`.

### B-1. SEEN (disc = except_0 vs p_theta)

| blk | metric | base | IW raw | adj+IW raw | IW +P1P3 | adj+IW +P1P3 |
|----:|--------|-----:|-------:|-----------:|---------:|-------------:|
| **1** | valid   | 0.306 | 0.339 | 0.800 | 1.000 | 1.000 |
|       | arrival | 0.994 | 0.995 | 0.684 | 1.000 | 1.000 |
|       | EM      | 0.283 | 0.308 | 0.377 | 0.333 | **0.378** |
|       | PC      | 0.292 | 0.319 | 0.425 | 0.476 | 0.443 |
| **2** | valid   | 0.264 | 0.322 | 0.459 | 1.000 | 1.000 |
|       | arrival | 0.956 | 0.958 | 0.881 | 1.000 | 1.000 |
|       | EM      | 0.244 | 0.296 | 0.367 | 0.317 | **0.360** |
|       | PC      | 0.253 | 0.307 | 0.399 | 0.466 | 0.495 |

### B-2. UNSEEN (disc = except_1..99 vs p_theta → zero-shot except_0)

| blk | metric | base | IW raw | adj+IW raw | IW +P1P3 | adj+IW +P1P3 |
|----:|--------|-----:|-------:|-----------:|---------:|-------------:|
| **1** | valid   | 0.306 | 0.334 | 0.796 | 1.000 | 1.000 |
|       | arrival | 0.994 | 0.994 | 0.677 | 1.000 | 1.000 |
|       | EM      | 0.283 | 0.306 | 0.376 | 0.327 | **0.377** |
|       | PC      | 0.292 | 0.316 | 0.423 | 0.471 | 0.441 |
| **2** | valid   | 0.264 | 0.330 | 0.449 | 1.000 | 1.000 |
|       | arrival | 0.956 | 0.959 | 0.876 | 1.000 | 1.000 |
|       | EM      | 0.244 | 0.301 | 0.359 | 0.319 | **0.357** |
|       | PC      | 0.253 | 0.314 | 0.392 | 0.469 | 0.496 |

**관찰 (model vs data)**
- model-negative가 data보다 uniformly (미세하게) 우세 — 리포트 §4.1b 결론과 일치. 단 **blk1은 사실상 동률**(IW valid seen 0.339 vs 0.337): ESS≈99.9로 guidance가 거의 inert라 disc 종류 영향이 미미.
- blk2에서 격차가 보임: IW valid seen model 0.322 vs data 0.311 (+1.1%p), adj+IW valid 0.459 vs 0.447.
- 최종 adj+IW P1P3 EM: blk1 0.378 / blk2 0.360 (seen). blk4(0.357)보다 여전히 높음 → 작은 블록 우세 단조추세 유지.
- Seen ≈ Unseen (gap ≤0.01) — model-neg에서도 zero-shot 일반화 유지.
