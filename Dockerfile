# CORAG in one container. Build and run (see README "Run with Docker"):
#   docker build -t corag .
#   docker run --rm -p 127.0.0.1:8080:8080 -v "$PWD/.env:/app/.env:ro" -v corag-data:/data corag
FROM python:3.13-slim

WORKDIR /app
# Chats, indexes and the downloaded local embedding model live in the /data volume
ENV PYTHONUNBUFFERED=1 CORAG_DATA_DIR=/data HF_HOME=/data/huggingface
# Listen on all interfaces inside the container; `docker run -p 127.0.0.1:8080:8080` keeps it local
ENV CORAG_HOST=0.0.0.0

# sentence-transformers needs torch; the CPU wheel avoids gigabytes of CUDA packages.
# Keep the version in step with requirements-lock.txt and CI.
RUN pip install --no-cache-dir torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt ./
# hnswlib has no prebuilt wheel: compile it, then drop the compiler in the same layer
RUN apt-get update && apt-get install -y --no-install-recommends g++ \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove g++ && rm -rf /var/lib/apt/lists/*

COPY . .
EXPOSE 8080
CMD ["python", "server.py"]
