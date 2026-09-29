import pytest
from fastapi.testclient import TestClient

from nirnaya_api.server import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "degraded")
    assert "nirnaya_solver" in body["checks"]


def test_version():
    r = client.get("/version")
    assert r.status_code == 200
    body = r.json()
    assert "api_version" in body
    assert "dependencies" in body


def test_devices():
    r = client.get("/devices")
    assert r.status_code == 200
    devices = r.json()
    assert any(d["id"] == "cpu:0" for d in devices)


def test_validate_endpoint_valid(simple_model_dict):
    r = client.post("/validate", json=simple_model_dict)
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] is True
    assert body["num_variables"] == 2


def test_validate_endpoint_invalid():
    r = client.post("/validate", json={"variables": {}, "objective": {"coefficients": {}}})
    assert r.status_code == 200
    body = r.json()
    assert body["valid"] is False
    assert body["issues"]


def test_solve_endpoint(fake_backend, simple_model_dict):
    payload = {"model": simple_model_dict, "options": {"backend": "cpu"}}
    r = client.post("/solve", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "optimal"
    assert body["observability"]["backend"] == "cpu"


def test_solve_endpoint_malformed_model_returns_structured_error(fake_backend):
    payload = {"model": {"variables": {}, "objective": {"coefficients": {}}}}
    r = client.post("/solve", json=payload)
    assert r.status_code == 422  # FastAPI/pydantic request-validation error


def test_solve_endpoint_unknown_variable_reference(fake_backend, simple_model_dict):
    simple_model_dict["constraints"]["c1"]["coefficients"]["nonexistent"] = 1.0
    payload = {"model": simple_model_dict}
    r = client.post("/solve", json=payload)
    assert r.status_code == 422


def test_solve_endpoint_explicit_gpu_without_device(fake_backend, simple_model_dict):
    payload = {"model": simple_model_dict, "options": {"backend": "gpu"}}
    r = client.post("/solve", json=payload)
    assert r.status_code == 409
    body = r.json()
    assert body["error"]["code"] == "DEVICE_UNAVAILABLE"


def test_solve_endpoint_bad_option_value(fake_backend, simple_model_dict):
    payload = {"model": simple_model_dict, "options": {"time_limit": -5}}
    r = client.post("/solve", json=payload)
    assert r.status_code in (400, 422)


def test_payload_too_large_is_rejected(fake_backend, simple_model_dict, monkeypatch):
    from nirnaya_api import server as server_mod

    monkeypatch.setattr(server_mod, "_limits", server_mod.SecurityLimits(max_payload_bytes=10))
    payload = {"model": simple_model_dict, "options": {"backend": "cpu"}}
    r = client.post("/solve", json=payload)
    assert r.status_code == 413
    body = r.json()
    assert body["error"]["code"] == "PAYLOAD_TOO_LARGE"


def test_jobs_lifecycle(fake_backend, simple_model_dict):
    payload = {"model": simple_model_dict, "options": {"backend": "cpu"}}
    r = client.post("/jobs", json=payload)
    assert r.status_code == 202
    job_id = r.json()["job_id"]

    import time

    for _ in range(50):
        r2 = client.get(f"/jobs/{job_id}")
        assert r2.status_code == 200
        body = r2.json()
        if body["status"] in ("completed", "failed"):
            break
        time.sleep(0.05)

    assert body["status"] == "completed"
    assert body["result"]["status"] == "optimal"


def test_job_not_found():
    r = client.get("/jobs/does-not-exist")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "JOB_NOT_FOUND"
