FROM pytorch/pytorch:2.12.0-cuda13.0-cudnn9-runtime

WORKDIR /app

RUN python -m pip install --upgrade pip
RUN python -m pip install --no-cache-dir --index-url https://pypi.org/simple runpod==1.11.0

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
