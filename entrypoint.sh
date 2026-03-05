#!/bin/bash
set -e

echo "=== Hypervision VPS Builder ==="
echo "City: ${CITY_NAME}"
echo "Features Prefix: ${FEATURES_BUCKET_PREFIX}"
echo "Instance: ${INSTANCE_ID}"
echo "CPU: $(nproc) cores"
echo "RAM: $(free -h | awk '/^Mem:/{print $2}') total, $(free -h | awk '/^Mem:/{print $7}') available"
echo "Disk: $(df -h / | tail -1 | awk '{print $4}') free"
echo "Working Directory: $(pwd)"
echo "Files in /app:"
ls -la /app
echo "================================"

# Instance ID detection is handled by pipeline.py via R2 lookup.
if [ -z "${INSTANCE_ID}" ]; then
    echo "[INFO] INSTANCE_ID not set — pipeline.py will detect via R2"
fi

# Run the pipeline
echo "[INFO] Starting builder pipeline.py..."
python pipeline.py
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "[ERROR] Pipeline exited with code $EXIT_CODE"
    echo "[INFO] Container kept alive for debugging. SSH in to investigate."
    tail -f /dev/null
fi
