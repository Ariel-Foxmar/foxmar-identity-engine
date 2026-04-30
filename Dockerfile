FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        build-essential \
        ca-certificates \
        curl \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install -r /app/requirements.txt

# Install optional InsightFace dependencies at build time by passing:
#   docker build --build-arg INSTALL_INSIGHTFACE=true ...
ARG INSTALL_INSIGHTFACE=false
COPY requirements-insightface.txt /app/requirements-insightface.txt
RUN if [ "$INSTALL_INSIGHTFACE" = "true" ]; then \
        python3 -m pip install -r /app/requirements-insightface.txt; \
    fi

COPY src /app/src
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh \
    && mkdir -p /models /cache /logs

ENV MODEL_DIR=/models \
    CACHE_DIR=/cache \
    LOG_DIR=/logs \
    ENGINE_MODE=stub \
    POLL_INTERVAL_SECONDS=5

ENTRYPOINT ["/app/entrypoint.sh"]
