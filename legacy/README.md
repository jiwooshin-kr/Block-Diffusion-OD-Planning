# legacy/ — 종료된 라운드의 스크립트

동작은 하지만(`sys.path` 부트스트랩 포함) 현재 실험 계획에는 쓰이지 않는다.
어떤 `scripts/run_*.sh`도 이 폴더를 참조하지 않는다.

## `od_pipeline/` — block diffusion 이전(EPSM_OD) 파이프라인

`main.py`(root에 유지) 시절의 평가·스모크 테스트.

- `disc_eval.py`, `disc_eval_shortest.py` — 구 판별기 평가.
  `disc_eval_shortest.py`는 root `eval_shortest.py`와 **884/893줄이 동일한 죽은 포크**(차이 29줄)로,
  참조하는 곳이 없다. 삭제해도 무방하지만 기록용으로 남겨둠.
- `eval_od_conditional.py` + `make_tiny_od_ckpt.py` — O/D 조건부 diffusion 평가 및 그 스모크용 tiny ckpt.
- `smoke_test_od.py`, `smoke_test_od_eos.py`, `smoke_test_od_abl.py` — EPSM_OD 계열 스모크 3종.
- `sweep_sample_steps.py` — respaced sampling 스텝 수 스윕.

## `v2_round/` — v2/v3 시절 평가 (v4 재학습으로 대체됨)

- `eval_bd_postproc.py`, `eval_v2_postproc.py` — 후처리 스윕. 현재는 `tools/eval/three_way_postproc.py`가 대체.
- `eval_v2_anatomy.py` — v2 실패 해부(길이 상관, miss 끝점 거리).
- `eval_l2r_normal.py` — normal 데이터에서의 reveal 순서 ablation. 현재는 각 eval의 `-order` 인자로 대체.
- `eval_uncond_diversity.py`, `eval_uncond_empc.py`, `eval_uncond_jsev.py` — 무조건 생성 지표 3종.
  현재는 `tools/eval/eval_uncond_pools.py`가 통합 담당.
