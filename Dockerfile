# Use NVIDIA PyTorch as base for GPU+PyTorch support
FROM nvcr.io/nvidia/pytorch:25.12-py3 AS developer

# Add any system dependencies for the developer/build environment here
# RUN apt-get update -y && apt-get install -y --no-install-recommends \
#     graphviz \
#     && apt-get dist-clean

COPY --from=ghcr.io/astral-sh/uv:0.10 /uv /uvx /bin/
