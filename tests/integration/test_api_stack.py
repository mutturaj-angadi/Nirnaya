import math

from fastapi.testclient import TestClient

from nirnaya_api import _integration
from nirnaya_api.server import app


def test_ui_schema_uses_real_solver_through_api():
    _integration.reset_backend()
    client = TestClient(app)
    model = {
        "name": "api-integration",
        "sense": "maximize",
        "variables": [
            {"name": "x", "lower": 0, "upper": 10, "type": "continuous"},
            {"name": "y", "lower": 0, "upper": 10, "type": "continuous"},
        ],
        "objective": {"offset": 5, "coefficients": {"x": 3, "y": 2}},
        "constraints": [
            {"name": "capacity", "sense": "<=", "rhs": 4, "coefficients": {"x": 1, "y": 1}}
        ],
    }
    response = client.post("/api/solve", json={
        "model": model, "backend": "cpu", "options": {"max_iterations": 200}
    })
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "optimal"
    assert result["objective"] == 17
    assert math.isclose(result["variables"]["x"], 4, abs_tol=1e-9)
    assert math.isclose(result["variables"]["y"], 0, abs_tol=1e-9)
    assert result["observability"]["backend"] == "cpu"
    assert result["presolve"]["statistics"]["original_num_variables"] == 2
    assert math.isclose(result["constraint_residuals"]["capacity"], 0, abs_tol=1e-12)
