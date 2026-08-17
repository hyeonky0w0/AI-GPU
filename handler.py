import runpod


def handler(job):
    print("===== HANDLER STARTED =====", flush=True)

    return {
        "status": "success",
        "message": "RunPod works!",
    }


if __name__ == "__main__":
    runpod.serverless.start({
        "handler": handler
    })
