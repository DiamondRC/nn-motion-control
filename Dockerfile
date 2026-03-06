# Use NVIDIA PyTorch as base for GPU+PyTorch support
FROM nvcr.io/nvidia/pytorch:25.12-py3 AS developer

# Add any system dependencies for the developer/build environment here
# RUN apt-get update -y && apt-get install -y --no-install-recommends \
#     graphviz \
#     && apt-get dist-clean

RUN apt-get update && apt-get install -y \
    python3-tk

COPY --from=ghcr.io/astral-sh/uv:0.10 /uv /uvx /bin/
