import runpod


def handler(job):
    print("===== HANDLER START =====", flush=True)

    return {
        "status": "success",
        "message": "RunPod worker is alive",
        "input": job.get("input", {}),
    }


if __name__ == "__main__":
    runpod.serverless.start({
        "handler": handler
    })
