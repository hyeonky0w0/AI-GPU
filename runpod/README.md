# LG Aimers → RunPod GPU 자동화

이 디렉터리는 기존 AI-GPU RunPod 이미지의 `handler.py`와 Dockerfile을 바꾸지 않고, 현재
LG Aimers 브랜치의 작은 실험 코드만 job bundle로 전달한다. 원본 CSV, 모델, OOF는 GitHub 또는
RunPod 응답에 넣지 않는다.

## 준비

GitHub Actions secret에 다음 두 값을 등록한다.

- `RUNPOD_API_KEY`
- `RUNPOD_ENDPOINT_ID` (`cb91m1rmcgwgyk`)

RunPod Network Volume에는 다음 경로를 만든다. 파일은 GitHub에 commit하지 않는다.

```text
/runpod-volume/LG_Aimers_data/
├── train.csv
├── test.csv
├── trackman_history.csv
└── oof/
    └── 035_anchored_dynamic_moe/
```

데이터 업로드 뒤 GitHub Actions의 **Run workflow**에서 `verify_volume`을 먼저 실행한다.

## 실행 모드

- `gate_smoke`: push 시의 기본값. 035 gate 수식·범위 검사를 실행하며 데이터나 OOF를 읽지 않는다.
- `verify_volume`: 세 CSV의 존재만 확인한다. 전체 파일을 읽거나 GitHub로 전송하지 않는다.
- `gate_rolling`: Network Volume의 `oof/035_anchored_dynamic_moe`를 read-only 심볼릭 링크로 연결해 gate를 실행한다. 현재 설정은 Champion OOF 부재로 명시 승인 전 차단된다.

`gate_rolling`을 열기 전에는 2022~2024 각각에 `fold_<year>.npz`와
`features_<year>.parquet`가 있고, `p_anchor`가 890 Champion의 시간순 OOF인지 검증해야 한다.

## 범위와 제약

기존 worker 이미지에는 현재 `runpod`만 명시적으로 설치돼 있다. 실제 CatBoost, ExtraTrees,
FT-Transformer OOF 생성은 해당 Python 의존성이 이미지에 존재하는지 확인한 뒤에만 별도
이미지 갱신으로 추가한다. 이는 Dockerfile을 임의 변경하지 않는 현재 범위 밖이다.
