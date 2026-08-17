FROM runpod/base:0.6.3-cuda11.8.0

WORKDIR /

COPY requirements.txt /requirements.txt

RUN uv pip install --system --no-cache -r /requirements.txt

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
