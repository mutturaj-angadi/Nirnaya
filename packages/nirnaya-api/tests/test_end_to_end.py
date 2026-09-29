import json

from click.testing import CliRunner
from fastapi.testclient import TestClient

from nirnaya_api import Model
from nirnaya_api.cli import main
from nirnaya_api.server import app

client = TestClient(app)


def test_python_cli_rest_agree(tmp_path, fake_backend, simple_model_dict):
    # Python API
    m = Model.from_dict(simple_model_dict)
    py_result = m.solve(backend="cpu").to_dict()

    # CLI
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(simple_model_dict))
    runner = CliRunner()
    cli_out = runner.invoke(main, ["solve", str(model_path), "--backend", "cpu", "--json"])
    assert cli_out.exit_code == 0
    cli_result = json.loads(cli_out.output)

    # REST
    rest_resp = client.post(
        "/solve", json={"model": simple_model_dict, "options": {"backend": "cpu"}}
    )
    rest_result = rest_resp.json()

    for result in (py_result, cli_result, rest_result):
        assert result["status"] == "optimal"
        assert result["variable_values"] == py_result["variable_values"]
        assert result["observability"]["backend"] == "cpu"


def test_validate_then_solve_workflow(tmp_path, fake_backend, simple_model_dict):
    m = Model.from_dict(simple_model_dict)
    report = m.validate()
    assert report["valid"] is True
    result = m.solve()
    assert result.is_optimal


def test_end_to_end_infeasible_model_is_consistent(fake_backend):
    model_dict = {
        "name": "infeasible",
        "variables": {"x": {"lb": 5.0, "ub": 10.0, "kind": "continuous"}},
        "constraints": {"c1": {"coefficients": {"x": 1.0}, "sense": "<=", "rhs": 1.0}},
        "objective": {"coefficients": {"x": 1.0}, "sense": "max", "offset": 0.0},
    }
    m = Model.from_dict(model_dict)
    result = m.solve()
    assert result.status.value == "infeasible"

    rest_resp = client.post("/solve", json={"model": model_dict})
    assert rest_resp.json()["status"] == "infeasible"
