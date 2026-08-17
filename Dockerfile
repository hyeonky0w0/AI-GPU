FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py
COPY src /app/src
COPY config /app/config

CMD ["python", "-u", "/app/handler.py"]
