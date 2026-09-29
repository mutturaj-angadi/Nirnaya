from __future__ import annotations

from collections import Counter

from scripts.noc10k_common import (
    DEFAULT_SEED, FAMILY_COUNTS, allocation, materialize_instance,
)
from scripts.analyze_noc10k import quantiles
from nirnaya_api.options import SolveOptions
from nirnaya_api.orchestrator import solve_model


def test_noc10k_family_allocation_is_exact_and_splits_holdout() -> None:
    assert sum(FAMILY_COUNTS.values()) == 10_000
    assert allocation(10_000) == FAMILY_COUNTS
    model, metadata = materialize_instance("small_sparse", DEFAULT_SEED + 100, 0,
                                            FAMILY_COUNTS["small_sparse"], DEFAULT_SEED)
    assert metadata["split"] in {"development", "validation", "holdout"}
    assert model["metadata"]["model_sha256"] == metadata["model_sha256"]
    assert len(metadata["model_sha256"]) == 64


def test_noc10k_generation_is_deterministic_and_seed_sensitive() -> None:
    args = ("industrial_style", DEFAULT_SEED + 2_000_000, 17,
            FAMILY_COUNTS["industrial_style"], DEFAULT_SEED)
    first, first_meta = materialize_instance(*args)
    second, second_meta = materialize_instance(*args)
    other, other_meta = materialize_instance("industrial_style", args[1] + 1,
        18, args[3], args[4])
    assert first == second
    assert first_meta == second_meta
    assert first_meta["model_sha256"] != other_meta["model_sha256"]
    assert first["metadata"]["instance_id"] != other["metadata"]["instance_id"]


def test_noc10k_holdout_split_is_exact_per_default_family() -> None:
    from scripts.noc10k_common import _split_map
    splits = _split_map("small_dense", FAMILY_COUNTS["small_dense"], DEFAULT_SEED)
    assert Counter(splits.values()) == {"development": 800, "validation": 100, "holdout": 100}


def test_noc10k_analyzer_uses_documented_nearest_rank_quantiles() -> None:
    assert quantiles([1, 2, 3, 4, 5]) == {
        "count": 5, "p50": 3.0, "p90": 5.0, "p95": 5.0, "p99": 5.0,
    }
    assert quantiles([None, float("nan")])["count"] == 0


def test_zero_variable_presolve_result_is_supported_by_api_adapter() -> None:
    # Development instance NOC10K-equality_heavy-00008 exposed that the API
    # tried to build an internally valid constant model as if it were user LP.
    seed = DEFAULT_SEED + 6_000_008
    model, _ = materialize_instance("equality_heavy", seed, 8,
                                    FAMILY_COUNTS["equality_heavy"], DEFAULT_SEED)
    result = solve_model(model, SolveOptions(backend="cpu", presolve=True, verbose=True))
    assert result.status.value == "optimal"
    assert len(result.variable_values) == len(model["variables"])
    assert result.presolve_info["statistics"]["reduced_num_variables"] == 0


def test_near_proportional_rows_do_not_create_false_infeasibility() -> None:
    # Development instance NOC10K-equality_heavy-00147 was feasible in
    # HiGHS and without presolve, but row merging rejected its feasible
    # interval when opposite-sense rows were reduced to the same variable.
    seed = DEFAULT_SEED + 6_000_147
    model, _ = materialize_instance("equality_heavy", seed, 147,
                                    FAMILY_COUNTS["equality_heavy"], DEFAULT_SEED)
    result = solve_model(model, SolveOptions(backend="cpu", presolve=True, verbose=True))
    assert result.status.value == "optimal"
    assert result.diagnostics.max_infeasibility <= 1e-7


def test_proportional_opposing_inequalities_keep_a_nonempty_interval() -> None:
    model = {"name": "opposing-interval", "sense": "maximize",
        "variables": [{"name": "x", "lower": 0.0, "upper": 10.0, "type": "continuous"}],
        "objective": {"offset": 0.0, "coefficients": {"x": 1.0}},
        "constraints": [
            {"name": "lower", "sense": ">=", "rhs": 2.0, "coefficients": {"x": 1.0}},
            {"name": "upper", "sense": "<=", "rhs": 8.0, "coefficients": {"x": 1.0}},
        ]}
    result = solve_model(model, SolveOptions(backend="cpu", presolve=True, verbose=True))
    assert result.status.value == "optimal"
    assert result.variable_values["x"] == 8.0
