# tools/ — 실행 스크립트 (역할별)

**모든 명령은 repo root에서 실행한다.** (`sets_model/`, `porto_data/` 등 기본 경로가 cwd 상대이기 때문)

```
python tools/eval/three_way_postproc.py -ckpt ... -disc ... -tag ...
```

각 파일 상단에는 repo root를 `sys.path`에 넣는 부트스트랩 4줄이 있다. 이것 없이는
`import eval_shortest` / `dcbg_plugin` 및 체크포인트 unpickle(`models_seq.*`)이 실패한다.

| 폴더 | 역할 | 파일 |
|---|---|---|
| `train/` | 예외 상황 판별기 학습 | `train_bd_disc.py`(IW disc, 주력) · `train_e99_model_multi.py`(unseen e99 다중) · `train_dcbg_classifier.py`(D-CBG 베이스라인) · `train_bd_disc_plain.py`(adjacency 채널 제거 ablation) |
| `gen/` | 모델 샘플 pool 생성 | `gen_bd_uncond_pool.py`(무조건 생성 pool = model-negative) · `gen_bd_neg_pool.py`(mix ablation용) |
| `eval/` | guidance 평가 | `three_way_postproc.py`(주력: base/IW/adj+IW × raw/P1/P3/P1P3) · `eval_dcbg.py`(D-CBG) · `eval_bd_guidance.py`(초기 IW 평가) · `base_only.py`(무guidance 대조) · `disc_gen_eval.py`(disc 일반화) · `eval_uncond_pools.py`(무조건 생성 품질) · `eval_bd_sweep_controlled.py`(블록 스윕 조건부 평가) |
| `collect/` | 결과 표 조립 | `collect_va_table.py`(valid/arrival 10열 표) · `collect_dcbg_table.py` · `collect_postproc.py` · `collect_seed_stats.py` |
| `diag/` | 진단·계측·플롯 | `ess_trace.py` / `ess_trace_all.py`(ESS 추적) · `dbg_reveals.py` · `eval_uniq_diag.py`(후보 다양성) · `diag_dcbg_leverage.py` + `plot_dcbg_leverage.py` · `time_guidance.py`(벽시계 비교) · `smoke_test_bd.py` |

## 이동 금지 항목

체크포인트가 `torch.save(model)` 전체 pickle이라, 아래는 지금 import 경로를 유지해야 한다
(옮기면 `sets_model/*.pth`, `sets_disc/*.pth`를 못 읽는다):

- `models_seq/` (`bd_models`, `bd_disc`, `blocks`, `seq_models`)
- `dcbg_plugin.py` (top-level)
- `utils/`, `loader/`, `planner/`
- `eval_shortest.py` — 8개 스크립트가 import하는 지표/refine 라이브러리
- `main_bd.py` — 진입점이면서 `legacy/v2_round/eval_bd_postproc.py`, `tools/eval/eval_bd_sweep_controlled.py`가 import
