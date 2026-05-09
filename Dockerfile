FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl dos2unix openssh-client openssh-server && \
    rm -rf /var/lib/apt/lists/*

# Install vastai CLI for self-destruct
RUN pip install --no-cache-dir vastai

# Python dependencies (torch/torchvision/timm provided by base image)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sanity check: faiss-gpu can see at least one CUDA device at runtime is
# checked inside pipeline.py — at build time we just confirm the import works.
RUN python -c "import faiss; print(f'faiss {faiss.__version__} import OK')"

# Application code
COPY pipeline.py r2_storage.py redis_queue.py region_attribution.py entrypoint.sh ./

RUN dos2unix entrypoint.sh && chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
