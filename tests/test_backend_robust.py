"""Backend robustness: job-id validation, retryable load, disk cache, upload race."""
import json

import pytest
from fastapi.testclient import TestClient

import app as studio

client = TestClient(studio.app)


@pytest.mark.parametrize("bad", ["..%5c..%5cwindows", "..", "zzzzzzzzzzzzzzzz", "..%5csecret", "abc/def", "0123"])
def test_job_id_rejects_bad_shape(bad):
    r = client.get(f"/api/jobs/{bad}/meta")
    assert r.status_code == 404, bad


def test_traversal_cannot_read_outside_data(monkeypatch, tmp_path):
    """A job id walking out of DATA must not serve a real file outside it."""
    data = tmp_path / "data"
    data.mkdir()
    secret = tmp_path / "secret"
    secret.mkdir()
    (secret / "meta.json").write_text(json.dumps({"leak": True}))
    monkeypatch.setattr(studio, "DATA", data)
    for bad in ["..%5csecret", "..", "..%5c..%5csecret"]:
        r = client.get(f"/api/jobs/{bad}/meta")
        assert r.status_code == 404, bad
        assert "leak" not in r.text, bad


def test_job_id_accepts_valid_shape():
    # unknown but well-formed id -> 404 from the handler, proving validation passed
    r = client.get("/api/jobs/0123456789abcdef/meta")
    assert r.status_code == 404
    assert r.json()["error"] == "unknown job"


def test_disk_cache_hit_without_memory(monkeypatch, tmp_path):
    """A job already on disk must be served even when CACHE is empty (after restart)."""
    job_id = "abcdef0123456789"
    d = tmp_path / job_id
    d.mkdir()
    meta = {"id": job_id, "width": 4, "height": 4, "filename": "x.jpg"}
    (d / "meta.json").write_text(json.dumps(meta))
    (d / "photo.png").write_bytes(b"png")
    (d / "depth.f16").write_bytes(b"\x00" * 32)
    monkeypatch.setattr(studio, "DATA", tmp_path)
    monkeypatch.setattr(studio, "CACHE", {})
    monkeypatch.setattr(studio, "load_model",
                        lambda: pytest.fail("model loaded despite disk cache hit"))
    # analyze_bytes hashes the raw bytes; force the id
    monkeypatch.setattr(studio.hashlib, "sha1",
                        lambda raw: type("H", (), {"hexdigest": lambda self: job_id})())
    out = studio.analyze_bytes(b"whatever", "x.jpg")
    assert out["id"] == job_id
    assert job_id in studio.CACHE


def test_error_state_is_retryable(monkeypatch):
    """A failed load must leave status idle so the next call retries, not error."""
    monkeypatch.setitem(studio.STATE, "model", None)
    monkeypatch.setitem(studio.STATE, "status", "error")
    monkeypatch.setattr(studio, "create_model_and_transforms",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    studio.load_model()
    assert studio.STATE["status"] == "idle"
    assert "boom" in studio.STATE["message"]
