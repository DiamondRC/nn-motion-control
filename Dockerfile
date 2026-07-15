# Use NVIDIA PyTorch as base for GPU+PyTorch support
FROM nvcr.io/nvidia/pytorch:25.12-py3 AS developer

# Add any system dependencies for the developer/build environment here
# RUN apt-get update -y && apt-get install -y --no-install-recommends \
#     graphviz \
#     && apt-get dist-clean

RUN apt-get update && apt-get install -y \
    python3-tk

COPY --from=ghcr.io/astral-sh/uv:0.10 /uv /uvx /bin/

# Install rtk (Rust Token Killer) — a proxy that compresses dev-command output
# before it reaches LLM contexts. https://github.com/rtk-ai/rtk
# Single static binary; downloaded per-arch and checksum-verified.
ARG RTK_VERSION=0.43.0
RUN set -eux; \
    case "$(uname -m)" in \
      aarch64|arm64) asset="rtk-aarch64-unknown-linux-gnu.tar.gz"; \
                     sha="5519f7ca12e5c143a609f0d28a0a77b97413a8dce31c2681f1a41c24519a8731" ;; \
      x86_64|amd64)  asset="rtk-x86_64-unknown-linux-musl.tar.gz"; \
                     sha="ff8a1e7766496e175291a85aeca1dc97c9ff6df33e51e5893d1fbc78fea2a609" ;; \
      *) echo "unsupported arch: $(uname -m)" >&2; exit 1 ;; \
    esac; \
    curl -fsSL -o /tmp/rtk.tar.gz \
      "https://github.com/rtk-ai/rtk/releases/download/v${RTK_VERSION}/${asset}"; \
    echo "${sha}  /tmp/rtk.tar.gz" | sha256sum -c -; \
    tar xzf /tmp/rtk.tar.gz -C /usr/local/bin rtk; \
    chmod +x /usr/local/bin/rtk; \
    rm /tmp/rtk.tar.gz
