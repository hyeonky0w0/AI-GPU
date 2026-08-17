FROM pytorch/pytorch:2.12.0-cuda13.0-cudnn9-runtime

WORKDIR /app

RUN pip install --no-cache-dir runpod

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
