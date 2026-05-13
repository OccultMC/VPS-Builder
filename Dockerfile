FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl dos2unix openssh-client openssh-server && \
    rm -rf /var/lib/apt/lists/*

# faiss-gpu with cuVS support is only published via conda (no PyPI wheel).
# cuVS lifts the M cap on GpuIndexIVFPQ — without it, M>96 falls back to
# CPU which on multi-million-vector indexes is hours of swap-thrashing
# k-means. The pytorch base image ships conda so we use it directly.
#
# Channels:
#   pytorch    — faiss-gpu-cuvs built against pytorch's CUDA stack
#   nvidia     — cuVS + RAFT shared libs
#   rapidsai   — RAFT primitives faiss-gpu-cuvs links against
#   conda-forge— transitive deps
#
# libmamba solver is ~10x faster than the default for this multi-channel
# resolve (would otherwise take 5-10 min just to resolve dependencies).
RUN conda config --set solver libmamba && \
    conda install -y \
        -c pytorch -c nvidia -c rapidsai -c conda-forge \
        faiss-gpu-cuvs && \
    conda clean -ya

# Install vastai CLI for self-destruct (and other small pip-only utilities).
# NOTE: faiss is NOT in requirements.txt — installed above via conda.
RUN pip install --no-cache-dir vastai

# Python dependencies (torch/torchvision/timm provided by base image,
# faiss-gpu-cuvs provided by conda step above)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sanity check: faiss imports AND cuVS path is wired in
RUN python -c "import faiss; print(f'faiss {faiss.__version__} import OK'); \
               import faiss; cfg = faiss.GpuIndexIVFPQConfig(); \
               assert hasattr(cfg, 'use_cuvs'), 'cuVS support missing!'; \
               print('cuVS support: OK (config.use_cuvs available)')"

# Application code
COPY pipeline.py r2_storage.py redis_queue.py region_attribution.py entrypoint.sh ./

RUN dos2unix entrypoint.sh && chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
