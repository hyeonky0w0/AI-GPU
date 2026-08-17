# Baseball ML RunPod Pipeline

브랜치의 실험 코드를 하나의 RunPod Queue endpoint에서 실행하고 결과를 GitHub
Actions Artifact로 회수하는 파이프라인입니다.

## 흐름

1. `feature/**` 브랜치에서 `src/` 또는 `config/`를 push합니다.
2. Actions가 두 디렉터리를 ZIP으로 묶어 RunPod `/run`에 제출합니다.
3. `handler.py`가 작업별 임시 디렉터리에서 `src/train.py`를 실행합니다.
4. Actions가 `/status/{job-id}`를 조회하고 결과 ZIP을 복원합니다.
5. `results/`와 원본 RunPod 응답을 30일간 Artifact로 보관합니다.

## 최초 설정

1. 현재 저장소의 Dockerfile로 RunPod Serverless worker를 한 번 빌드합니다.
2. Queue endpoint가 그 빌드 이미지를 사용하도록 배포합니다.
3. GitHub repository Actions secrets에 다음 값을 저장합니다.
   - `RUNPOD_API_KEY`
   - `RUNPOD_ENDPOINT_ID`
4. Actions의 `GPU Experiment`를 수동 실행하거나 `feature/**` 브랜치에 push합니다.

Worker 이미지는 브랜치마다 다시 만들 필요가 없습니다. 단, `handler.py`, Dockerfile,
worker 의존성이 바뀐 경우에는 RunPod worker 이미지를 새로 빌드해야 합니다.

## 결과

- `metrics.json`: 브랜치, commit SHA, 모델, RMSE/MAE, 실행 시간, GPU 이름
- `predictions.csv`: 검증용 실제값과 예측값
- `train.log`: 학습 표준 출력
- `error.log`: 학습 표준 오류
- `runpod-response.json`: RunPod 작업 상태 원본

## 현재 샘플과 실제 야구 모델 연결

현재 `src/model.py`는 파이프라인 검증용 순수 Python 선형 baseline입니다. GPU 이름은
`nvidia-smi`로 기록하지만 아직 GPU 학습을 수행하지 않습니다. 실제 적용 시 브랜치에서
`src/features.py`의 샘플 데이터 로더와 `src/model.py`를 PyTorch/MLP 또는 CatBoost
구현으로 교체하고, 필요한 런타임 패키지는 worker 이미지에 미리 설치해야 합니다.

`config/experiment.yaml`은 외부 라이브러리 없이 읽을 수 있도록 JSON 문법으로 작성된
유효한 YAML 파일입니다.

## 데이터와 크기 제한

코드 번들과 반환 결과 ZIP은 각각 8 MiB로 제한합니다. 야구 원본 데이터나 대형 모델은
요청에 포함하지 말고 RunPod Network Volume 또는 S3 호환 스토리지에 저장한 뒤
`config/experiment.yaml`의 `data.uri`를 통해 참조하세요. API key나 스토리지 secret을
코드·설정·Artifact에 넣지 마세요.

## 로컬 baseline 실행

```bash
python src/train.py
```

로컬 환경에서는 GPU가 없으면 `metrics.json`의 `gpu`가 `null`인 것이 정상입니다.
