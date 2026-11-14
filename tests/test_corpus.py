import gzip
import inspect
import json

from findex.corpus import iter_documents


def test_is_a_generator():
    assert inspect.isgeneratorfunction(iter_documents)


def test_txt_tree_jsonl_and_gzip(tmp_path):
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("beta", encoding="utf-8")
    (tmp_path / "ignored.md").write_text("nope", encoding="utf-8")
    lines = [json.dumps({"id": 1, "text": "gamma"}), "{not json", json.dumps({"abstract": "delta"})]
    (tmp_path / "c.jsonl").write_text("\n".join(lines), encoding="utf-8")
    with gzip.open(tmp_path / "d.jsonl.gz", "wt", encoding="utf-8") as f:
        f.write(json.dumps({"text": "epsilon"}) + "\n")

    texts = sorted(doc.text for doc in iter_documents(tmp_path))
    assert texts == ["alpha", "beta", "delta", "epsilon", "gamma"]


def test_bad_encoding_does_not_crash(tmp_path, caplog):
    (tmp_path / "bad.txt").write_bytes(b"ok \xff\xfe bytes")
    (tmp_path / "good.txt").write_text("fine", encoding="utf-8")
    texts = sorted(doc.text for doc in iter_documents(tmp_path))
    assert texts == ["fine", "ok �� bytes"]
    assert "bad UTF-8" in caplog.text


def test_lazy_first_document(tmp_path):
    for i in range(50):
        (tmp_path / f"{i}.txt").write_text(f"doc {i}", encoding="utf-8")
    gen = iter_documents(tmp_path)
    first = next(gen)  # returns after reading exactly one file
    assert first.doc_id == "0"
