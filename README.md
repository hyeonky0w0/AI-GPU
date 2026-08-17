# Baseball ML GPU Test

RunPod Serverless에서 PyTorch와 CUDA 연결을 확인하기 위한 최소 프로젝트입니다.

## 실행 흐름

`handler.py`가 `src/train.py`를 실행하고 다음 결과 파일을 생성합니다.

- `results/train.log`: 표준 출력과 오류 로그
- `results/metrics.json`: CUDA 사용 가능 여부와 GPU 이름

RunPod worker의 시작 명령은 Dockerfile의 `CMD ["python", "handler.py"]`입니다.
