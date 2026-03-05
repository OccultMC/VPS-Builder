FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl dos2unix openssh-client openssh-server && \
    rm -rf /var/lib/apt/lists/*

# Install vastai CLI for self-destruct
RUN pip install --no-cache-dir vastai

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY pipeline.py r2_storage.py entrypoint.sh ./

RUN dos2unix entrypoint.sh && chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
