import json
import time
from pathlib import Path

import torch


def main():
    start = time.time()

    print("=" * 50)
    print("GPU TEST")
    print("=" * 50)

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    gpu_name = None

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

        print("GPU:", gpu_name)

        x = torch.randn(
            5000,
            5000,
            device="cuda",
        )

        _ = x @ x

        torch.cuda.synchronize()

        print("GPU computation finished")

    metrics = {
        "status": "success",
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": gpu_name,
        "elapsed_seconds": time.time() - start,
    }

    results_dir = Path("/app/results")
    results_dir.mkdir(exist_ok=True)

    with (results_dir / "metrics.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=2)

    print("metrics.json created")


if __name__ == "__main__":
    main()