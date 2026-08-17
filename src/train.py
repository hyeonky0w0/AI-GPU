import json
from pathlib import Path

import torch


def main():
    print("=" * 50)
    print("GPU TEST")
    print("=" * 50)

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

        x = torch.randn(5000, 5000, device="cuda")
        _ = x @ x

        print("GPU computation finished")

    metrics = {
        "status": "success",
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
    }

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    with (results_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


if __name__ == "__main__":
    main()
