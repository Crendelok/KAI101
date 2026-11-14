from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from findex.cli import app

runner = CliRunner()


@pytest.fixture(scope="module")
def built(corpus_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("cli") / "ix.fidx"
    result = runner.invoke(app, ["index", str(corpus_dir), "--out", str(out)])
    assert result.exit_code == 0, result.output
    return out


def test_help_lists_subcommands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("index", "search", "stats"):
        assert cmd in result.output


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0 and "findex" in result.output


def test_search_table(built: Path):
    result = runner.invoke(app, ["search", str(built), "white whale", "-k", "3"])
    assert result.exit_code == 0, result.output
    assert "The Whale" in result.output


def test_search_json_lines(built: Path):
    result = runner.invoke(app, ["search", str(built), "whale", "--json", "--scorer", "tfidf"])
    assert result.exit_code == 0
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert rows and rows[0]["rank"] == 1
    assert {"doc_id", "score", "title", "snippet", "path"} <= rows[0].keys()
    assert [r["score"] for r in rows] == sorted((r["score"] for r in rows), reverse=True)


def test_stats(built: Path):
    result = runner.invoke(app, ["stats", str(built), "--top", "5"])
    assert result.exit_code == 0
    assert "documents" in result.output and "vocabulary" in result.output


def test_missing_index_exits_nonzero(tmp_path: Path):
    result = runner.invoke(app, ["search", str(tmp_path / "nope.fidx"), "x"])
    assert result.exit_code == 1
    assert "index file not found" in result.output


def test_bad_query_exits_2(built: Path):
    result = runner.invoke(app, ["search", str(built), "(whale"])
    assert result.exit_code == 2
    assert "bad query" in result.output


def test_bad_scorer_is_a_usage_error(built: Path):
    result = runner.invoke(app, ["search", str(built), "whale", "--scorer", "magic"])
    assert result.exit_code == 2


def test_index_missing_corpus(tmp_path: Path):
    result = runner.invoke(app, ["index", str(tmp_path / "none"), "--out", str(tmp_path / "x")])
    assert result.exit_code == 1
