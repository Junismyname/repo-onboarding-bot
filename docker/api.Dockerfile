FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CUDA 12.8 build of PyTorch: required for Blackwell GPUs (RTX 50xx).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

EXPOSE 8080
CMD ["uvicorn", "repobot.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
