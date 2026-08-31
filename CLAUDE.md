# 프로젝트 워크플로우

## 구조
- 코드는 **로컬에서 수정**하고, **GPU 서버에서 실행**한다
- 서버: `wp03052@143.248.84.179`
- 서버 프로젝트 경로: `/home/aailab/wp03052/Synthetic-Data/Block-Diffusion-OD-Planning`
- 서버는 절대 직접 수정하지 않는다. 항상 로컬 수정 → 동기화 → 실행 순서

## 동기화
- 코드 수정 후 반드시 `bash sync.sh` 로 서버에 동기화한 뒤 실행할 것

## 디렉터리 구조 (2026-08 정리)
- **root**: `main_bd.py`(block diffusion 진입점) · `main.py`(구 EPSM_OD 진입점) · `eval_shortest.py`(EM/PC/refine 지표 라이브러리) · `dcbg_plugin.py`(D-CBG 베이스라인) · `prep_v6_node2vec.py` · `sync.sh`
- **이동 금지**: `models_seq/`, `dcbg_plugin.py`, `utils/`, `loader/`, `planner/`, `eval_shortest.py`, `main_bd.py`
  — 체크포인트가 `torch.save(model)` 전체 pickle이라 import 경로가 바뀌면 기존 `.pth`를 못 읽는다
- **tools/**: `train/`(판별기 학습) `gen/`(샘플 pool) `eval/`(guidance 평가) `collect/`(결과 표) `diag/`(진단·플롯) — 자세한 내용은 `tools/README.md`
- **legacy/**: `od_pipeline/`(block diffusion 이전) `v2_round/`(v4 재학습으로 대체됨) — `legacy/README.md`
- **scripts/**: 모든 `run_*.sh` / `train_*.sh` (단 `sync.sh`는 root)
- 실행은 **항상 repo root에서**: `python tools/eval/three_way_postproc.py ...`, `bash scripts/run_v6_train_all.sh`
  (기본 경로가 cwd 상대이고, 하위 폴더 스크립트는 상단 부트스트랩 4줄로 root를 `sys.path`에 넣는다)

## 서버 실행 방법
- Python 환경은 venv 사용. 모든 서버 명령 앞에 activate 필요:
  `source /home/aailab/wp03052/venvs/od_planning_venv/bin/activate`
- 실행 예시:
  ssh wp03052@143.248.84.179 "source /home/aailab/wp03052/venvs/od_planning_venv/bin/activate && cd /home/aailab/wp03052/Synthetic-Data/Block-Diffusion-OD-Planning && python main.py"

## 규칙
- 디버깅/테스트 목적의 짧은 실행만 SSH로 직접 돌릴 것
- 긴 학습(수십 분 이상)은 직접 실행하지 말 것. 검증까지만 하고 멈출 것
- 서버의 결과물 폴더(체크포인트, 로그)는 rsync --delete로부터 보호되도록 sync.sh의 exclude 목록을 유지할 것
- requirements.txt를 수정했으면 서버 venv에도 설치 필요:
  ssh wp03052@143.248.84.179 "source /home/aailab/wp03052/venvs/od_planning_venv/bin/activate && pip install -r /home/aailab/wp03052/Synthetic-Data/Block-Diffusion-OD-Planning/requirements.txt"


## Block Diffusion 구현
- 다음 github 코드를 기반으로 구현해 줘. https://github.com/kuleshov-group/bd3lms.git
- 해당 코드는 masked diffusion으로 구현이 됐지만, 나는 현재 코드에서 사용하는 uniform-state discrete diffusion으로 학습을 하고 싶어.
- masekd diffusion, uniform diffusion 모두 구현해 주고, 특히 uniform diffusion은 현재 models_seq/seq_models.py 에 구현된 loss를 따라서 구현해 줘. (pdfs/block-diffusion-loss.pdf 파일 참고)
- block diffusion에서 length 처리는 다음과 같이 해줘: 한 블록 내에서 destination 이후 토큰은 eos token으로 학습, 이후의 남은 block들은 pad 처리해서 학습 대상 (X)
- Block size는 {1, 2, 4, 8, 16, 32, 64}로 다양한 크기에서 학습하고 성능을 비교해 줘.
- 이때, 성능 평가는 porto_v3_normal로 해주고, refine() 함수를 적용하기 전/후 비교해 줘.
