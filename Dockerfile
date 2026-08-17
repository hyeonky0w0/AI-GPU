FROM pytorch/pytorch:2.12.0-cuda13.0-cudnn9-runtime

WORKDIR /app

RUN pip install --no-cache-dir runpod

COPY requirements.txt /app/requirements.txt
COPY handler.py /app/handler.py
COPY src /app/src

CMD ["python", "-u", "/app/handler.py"]