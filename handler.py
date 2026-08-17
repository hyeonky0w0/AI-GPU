import runpod
import torch


def handler(job):
    print("===== GPU TEST =====", flush=True)

    cuda_available = torch.cuda.is_available()
    gpu_name = None

    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)

    return {
        "status": "success",
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "cuda_version": torch.version.cuda,
    }


if __name__ == "__main__":
    runpod.serverless.start({
        "handler": handler
    })
