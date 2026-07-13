# 프로젝트 워크플로우

## 구조
- 코드는 **로컬에서 수정**하고, **GPU 서버에서 실행**한다
- 서버: `wp03052@143.248.80.20`
- 서버 프로젝트 경로: `/home/aailab/data/wp03052/Synthetic-Data-DiffPlan`
- 서버는 절대 직접 수정하지 않는다. 항상 로컬 수정 → 동기화 → 실행 순서

## 동기화
- 코드 수정 후 반드시 `bash sync.sh` 로 서버에 동기화한 뒤 실행할 것

## 서버 실행 방법
- Python 환경은 venv 사용. 모든 서버 명령 앞에 activate 필요:
  `source /home/aailab/wp03052/env_gdp/bin/activate`
- 실행 예시:
  ssh wp03052@143.248.80.20 "source /home/aailab/wp03052/env_gdp/bin/activate && cd /home/aailab/data/wp03052/Synthetic-Data-DiffPlan && python main.py"

## 규칙
- 디버깅/테스트 목적의 짧은 실행만 SSH로 직접 돌릴 것
- 긴 학습(수십 분 이상)은 직접 실행하지 말 것. 검증까지만 하고 멈출 것
- 서버의 결과물 폴더(체크포인트, 로그)는 rsync --delete로부터 보호되도록 sync.sh의 exclude 목록을 유지할 것
- requirements.txt를 수정했으면 서버 venv에도 설치 필요:
  ssh wp03052@143.248.80.20 "source /home/aailab/wp03052/env_gdp/bin/activate && pip install -r /home/aailab/data/wp03052/Synthetic-Data-DiffPlan/requirements.txt"

## 학습 스크립트
- diffusion 학습: `train_porto_od_diffusion.sh`