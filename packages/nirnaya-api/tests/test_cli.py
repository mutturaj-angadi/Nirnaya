import json

from click.testing import CliRunner

from nirnaya_api.cli import main


def _write(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return str(path)


def test_cli_solve_json_output(tmp_path, fake_backend, simple_model_dict):
    path = _write(tmp_path, "model.json", simple_model_dict)
    runner = CliRunner()
    result = runner.invoke(main, ["solve", path, "--backend", "cpu", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "optimal"
    assert payload["observability"]["backend"] == "cpu"


def test_cli_solve_human_output(tmp_path, fake_backend, simple_model_dict):
    path = _write(tmp_path, "model.json", simple_model_dict)
    runner = CliRunner()
    result = runner.invoke(main, ["solve", path, "--backend", "cpu"])
    assert result.exit_code == 0
    assert "status:" in result.output


def test_cli_validate_valid_model(tmp_path, fake_backend, simple_model_dict):
    path = _write(tmp_path, "model.json", simple_model_dict)
    runner = CliRunner()
    result = runner.invoke(main, ["validate", path, "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["valid"] is True


def test_cli_validate_invalid_model(tmp_path, fake_backend):
    path = _write(tmp_path, "model.json", {"variables": {}, "objective": {"coefficients": {}}})
    runner = CliRunner()
    result = runner.invoke(main, ["validate", path, "--json"])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["valid"] is False


def test_cli_devices(fake_backend):
    runner = CliRunner()
    result = runner.invoke(main, ["devices", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert any(d["id"] == "cpu:0" for d in payload["devices"])


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(main, ["version", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert "api_version" in payload
    assert "dependencies" in payload


def test_cli_benchmark(fake_backend):
    runner = CliRunner()
    result = runner.invoke(main, ["benchmark", "--sizes", "2,4", "--backend", "cpu", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert len(payload["benchmark"]) == 2


def test_cli_solve_malformed_json(tmp_path, fake_backend):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json")
    runner = CliRunner()
    result = runner.invoke(main, ["solve", str(path), "--json"])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"]["code"] == "VALIDATION_ERROR"


def test_cli_solve_nonexistent_file():
    runner = CliRunner()
    result = runner.invoke(main, ["solve", "/no/such/file.json"])
    assert result.exit_code != 0
