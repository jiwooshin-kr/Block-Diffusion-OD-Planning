# 프로젝트 워크플로우

## 구조
- 코드는 **로컬에서 수정**하고, **GPU 서버에서 실행**한다
- 서버: `wp03052@143.248.84.179`
- 서버 프로젝트 경로: `/home/aailab/wp03052/Synthetic-Data/Block-Diffusion-OD-Planning`
- 서버는 절대 직접 수정하지 않는다. 항상 로컬 수정 → 동기화 → 실행 순서

## 동기화
- 코드 수정 후 반드시 `bash sync.sh` 로 서버에 동기화한 뒤 실행할 것

## 서버 실행 방법
- Python 환경은 venv 사용. 모든 서버 명령 앞에 activate 필요:
  `source /home/aailab/wp03052/syn_data/bin/activate`
- 실행 예시:
  ssh wp03052@143.248.84.179 "source /home/aailab/wp03052/syn_data/bin/activate && cd /home/aailab/wp03052/Synthetic-Data/Block-Diffusion-OD-Planning && python main.py"

## 규칙
- 디버깅/테스트 목적의 짧은 실행만 SSH로 직접 돌릴 것
- 긴 학습(수십 분 이상)은 직접 실행하지 말 것. 검증까지만 하고 멈출 것
- 서버의 결과물 폴더(체크포인트, 로그)는 rsync --delete로부터 보호되도록 sync.sh의 exclude 목록을 유지할 것
- requirements.txt를 수정했으면 서버 venv에도 설치 필요:
  ssh wp03052@143.248.84.179 "source /home/aailab/wp03052/syn_data/bin/activate && pip install -r /home/aailab/wp03052/Synthetic-Data/Block-Diffusion-OD-Planning/requirements.txt"


## Block Diffusion 구현
- 다음 github 코드를 기반으로 구현해 줘. https://github.com/kuleshov-group/bd3lms.git
- 해당 코드는 masked diffusion으로 구현이 됐지만, 나는 현재 코드에서 사용하는 uniform-state discrete diffusion으로 학습을 하고 싶어.
- masekd diffusion, uniform diffusion 모두 구현해 주고, 특히 uniform diffusion은 현재 models_seq/seq_models.py 에 구현된 loss를 따라서 구현해 줘. (pdfs/block-diffusion-loss.pdf 파일 참고)
- block diffusion에서 length 처리는 다음과 같이 해줘: 한 블록 내에서 destination 이후 토큰은 eos token으로 학습, 이후의 남은 block들은 pad 처리해서 학습 대상 (X)
- Block size는 {1, 2, 4, 8, 16, 32, 64}로 다양한 크기에서 학습하고 성능을 비교해 줘.
- 이때, 성능 평가는 porto_v3_normal로 해주고, refine() 함수를 적용하기 전/후 비교해 줘.
