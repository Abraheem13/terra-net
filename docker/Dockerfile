# Reproducible training container. Build:  docker build -t terranet -f docker/Dockerfile .
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip git curl gdal-bin libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .[dev]

# For Sionna RT generation, build the rt image instead:
#   pip install -e .[rt]
CMD ["/bin/bash"]
