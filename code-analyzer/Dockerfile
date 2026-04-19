# code-analyzer — Python FastAPI service driving the `claude` CLI.
#
# Base is node:22-slim because we need npm to install @anthropic-ai/claude-code.
# Python comes in on top via apt.

FROM node:22-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CODE_ANALYZER_PORT=9100 \
    CODE_ANALYZER_WORKSPACE=/workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        python3 \
        python3-pip \
        python3-venv \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Pinning to a specific minor keeps our CLI contract stable across rebuilds.
# Bump in concert with spike re-runs.
RUN npm install -g @anthropic-ai/claude-code@2.1.114

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

# Install into a venv to avoid clobbering system python.
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install -e .
ENV PATH="/opt/venv/bin:$PATH"

# Claude CLI refuses `--permission-mode bypassPermissions` under root/sudo
# as a safety check. Run as a dedicated non-root user so the CLI accepts
# the read-only Read/Grep/Glob tool allowlist we drive.
RUN useradd -m -u 10001 analyzer \
    && mkdir -p /workspace \
    && chown -R analyzer:analyzer /workspace /app /opt/venv

USER analyzer
ENV HOME=/home/analyzer

EXPOSE 9100

# uvicorn binds to CODE_ANALYZER_PORT at runtime for env-override.
CMD ["sh", "-c", "exec uvicorn code_analyzer.main:app --host 0.0.0.0 --port ${CODE_ANALYZER_PORT:-9100}"]
