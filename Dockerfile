FROM runpod/base:0.6.3-cuda11.8.0

RUN ln -sf $(which python3.11) /usr/local/bin/python && \
    ln -sf $(which python3.11) /usr/local/bin/python3

COPY requirements.txt /requirements.txt

RUN uv pip install --upgrade \
    -r /requirements.txt \
    --no-cache-dir \
    --system

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
