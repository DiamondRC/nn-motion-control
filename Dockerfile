# Use NVIDIA PyTorch as base for GPU+PyTorch support
FROM nvcr.io/nvidia/pytorch:25.12-py3 AS developer

# Add any system dependencies for the developer/build environment here
# RUN apt-get update -y && apt-get install -y --no-install-recommends \
#     graphviz \
#     && apt-get dist-clean

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
