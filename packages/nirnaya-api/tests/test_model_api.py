import pytest

from nirnaya_api import Model, SolveStatus
from nirnaya_api.errors import ValidationError


def test_build_and_solve(fake_backend):
    m = Model("toy")
    m.add_variable("x", lb=0, ub=5)
    m.add_variable("y", lb=0, ub=5)
    m.add_constraint("c1", {"x": 1, "y": 1}, sense="<=", rhs=12)
    m.set_objective({"x": 1, "y": 2}, sense="max")

    result = m.solve(backend="cpu")

    assert result.status == SolveStatus.OPTIMAL
    assert result.backend == "cpu"
    assert result.device == "cpu:0"
    assert set(result.variable_values) == {"x", "y"}
    assert result.solve_time_s >= 0.0
    assert result.presolve_time_s >= 0.0
    assert result.iterations >= 1


def test_duplicate_variable_rejected(fake_backend):
    m = Model("toy")
    m.add_variable("x")
    with pytest.raises(ValidationError):
        m.add_variable("x")


def test_missing_objective_rejected(fake_backend):
    m = Model("toy")
    m.add_variable("x")
    with pytest.raises(ValidationError):
        m.to_dict()


def test_roundtrip_json(fake_backend):
    m = Model("toy")
    m.add_variable("x", lb=0, ub=5)
    m.set_objective({"x": 1}, sense="max")
    text = m.to_json()

    m2 = Model.from_json(text)
    assert m2.to_dict() == m.to_dict()


def test_validate_without_solving(fake_backend):
    m = Model("toy")
    m.add_variable("x", lb=0, ub=5)
    m.set_objective({"x": 1}, sense="max")
    report = m.validate()
    assert report["valid"] is True
    assert report["num_variables"] == 1


def test_infeasible_status(fake_backend):
    m = Model("toy")
    m.add_variable("x", lb=5, ub=10)
    m.add_constraint("c1", {"x": 1}, sense="<=", rhs=1)  # x>=5 but x<=1: infeasible
    m.set_objective({"x": 1}, sense="max")
    result = m.solve()
    assert result.status == SolveStatus.INFEASIBLE


def test_explicit_gpu_without_device_raises(fake_backend):
    from nirnaya_api.errors import DeviceUnavailableError

    m = Model("toy")
    m.add_variable("x", lb=0, ub=5)
    m.set_objective({"x": 1}, sense="max")
    with pytest.raises(DeviceUnavailableError):
        m.solve(backend="gpu")


def test_auto_backend_falls_back_to_cpu_with_warning(fake_backend):
    m = Model("toy")
    m.add_variable("x", lb=0, ub=5)
    m.set_objective({"x": 1}, sense="max")
    result = m.solve(backend="auto")
    assert result.backend == "cpu"
    assert any("fall" in w.lower() for w in result.warnings)


def test_auto_backend_prefers_gpu_when_available(fake_backend_with_gpu):
    m = Model("toy")
    m.add_variable("x", lb=0, ub=5)
    m.set_objective({"x": 1}, sense="max")
    result = m.solve(backend="auto")
    assert result.backend == "gpu"
    assert result.device == "gpu:0"
