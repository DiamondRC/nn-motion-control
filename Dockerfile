# The devcontainer should use the developer target and run as root with podman
# or docker with user namespaces.
#FROM ghcr.io/diamondlightsource/ubuntu-devcontainer:noble AS developer

# Use NVIDIA PyTorch as base for GPU+PyTorch support
FROM nvcr.io/nvidia/pytorch:25.12-py3 AS developer

# Add any system dependencies for the developer/build environment here
#RUN apt-get update -y && apt-get install -y --no-install-recommends \
#    graphviz \
#    && apt-get dist-clean

USER root
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    graphviz \
    && apt-get autoclean && rm -rf /var/lib/apt/lists/*
