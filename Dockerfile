# syntax=docker/dockerfile:1.7
# Multi-stage: build a venv with uv, build the index, ship a slim non-root runtime.
#
#   docker build -t findex .                 # also downloads the corpus and indexes it
#   docker run --rm -p 8000:8000 findex
#   docker run --rm -p 8000:8000 -v "$PWD/other.fidx:/data/index.fidx:ro" findex

FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first: this layer stays cached until uv.lock changes.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# The index is a build artifact too: fetch the public-domain corpus and index it.
FROM build AS index
ENV PATH="/app/.venv/bin:$PATH"
COPY scripts/fetch_gutenberg.py scripts/make_sections.py ./scripts/
RUN python scripts/fetch_gutenberg.py \
 && python scripts/make_sections.py \
 && findex index data/sections --out /app/index.fidx --executor processes

FROM python:3.12-slim AS runtime
RUN useradd --create-home --uid 10001 findex
COPY --from=build --chown=findex:findex /app/.venv /app/.venv
COPY --from=index --chown=findex:findex /app/index.fidx /data/index.fidx
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    INDEX_PATH=/data/index.fidx \
    HOST=0.0.0.0 \
    PORT=8000
USER findex
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=2)"
CMD ["findex", "-v", "serve"]
