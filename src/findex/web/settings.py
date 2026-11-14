"""Configuration from the environment (or a ``.env`` file) — never from code."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """``INDEX_PATH=/data/index.fidx findex serve`` — every field maps to an env var."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    index_path: Path = Path("index.fidx")
    embeddings_path: Path | None = None  # Lab 8: <index>.emb.npy, if built
    site_title: str = "findex"
    default_k: int = 10
    max_k: int = 50
    search_async: bool = False  # Lab 7 experiment: register /search as `async def`


@lru_cache
def get_settings() -> Settings:
    return Settings()
