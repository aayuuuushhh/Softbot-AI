# Uddhar API (FastAPI + PyTorch perception + agents).
#
#   docker build -t uddhar-api .                                  # CPU torch
#   docker build -t uddhar-api --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124 .
#
# Model weights are NOT baked in: mount ./models at /app/models. Without them the
# API runs the deterministic stub backends (see core/vision/inference.py).
FROM python:3.13-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgl/glib: OpenCV's runtime deps. rasterio and shapely wheels bundle GDAL/GEOS.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch + torchvision are pinned as a pair (2.6.0 <-> 0.21.0) and come from the
# index chosen above; everything else from PyPI. Installed before the code is
# copied so code edits do not re-download ~1 GB of wheels.
RUN pip install --index-url "${TORCH_INDEX}" torch==2.6.0 torchvision==0.21.0
COPY pyproject.toml ./
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))" > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt

COPY core ./core
COPY agents ./agents
COPY api ./api
COPY scripts ./scripts
COPY data/demo ./data/demo

RUN useradd --create-home --uid 1000 uddhar \
    && mkdir -p /app/models /app/data/uploads \
    && chown -R uddhar:uddhar /app/data /app/models
USER uddhar

EXPOSE 8765
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD curl -fsS http://localhost:8765/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8765"]
