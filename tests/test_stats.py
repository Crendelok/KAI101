from __future__ import annotations

from pathlib import Path

import pytest

from findex.stats import eager_stats, lazy_stats, main


def test_lazy_and_eager_agree(corpus_dir: Path):
    lazy, eager = lazy_stats(corpus_dir), eager_stats(corpus_dir)
    assert lazy.documents == eager.documents == 8
    assert lazy.counts == eager.counts
    assert lazy.tokens == sum(lazy.counts.values())
    assert lazy_stats(corpus_dir, limit=2).documents == 2


def test_stats_cli_compare(corpus_dir: Path, capsys: pytest.CaptureFixture[str]):
    assert main([str(corpus_dir), "--compare"]) == 0
    out = capsys.readouterr().out
    assert "| eager (lists) | 8 |" in out and "| lazy (generators) | 8 |" in out


def test_stats_cli_report(corpus_dir: Path, capsys: pytest.CaptureFixture[str]):
    assert main([str(corpus_dir), "--top", "3", "--limit", "4"]) == 0
    out = capsys.readouterr().out
    assert "documents" in out and "top 3 terms" in out and "peak memory" in out
