from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_match_not_found():
    # We did not ingest anything, so it should return not_found
    r = client.post("/match", json={"job_id": "nope", "cv_id": "nope"})
    assert r.status_code == 404
    assert r.json()["error"] == "not_found"


def test_rewrite_not_found():
    r = client.post("/rewrite-bullets", json={"job_id": "nope", "cv_id": "nope", "bullets": ["A", "B"]})
    assert r.status_code == 404
    assert r.json()["error"] == "not_found"
