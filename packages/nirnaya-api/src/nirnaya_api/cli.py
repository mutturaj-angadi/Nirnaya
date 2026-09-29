"""
The ``nirnaya`` command-line interface.

Commands:

    nirnaya solve model.json [options]
    nirnaya validate model.json
    nirnaya benchmark [--sizes ...]
    nirnaya devices
    nirnaya version

Every command supports ``--json`` for machine-readable output; without it,
output is a concise human-readable summary. No command ever shells out or
executes arbitrary code from the input file - files are only ever parsed
as JSON data.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Optional

import click

from . import _integration as integ
from .backends import list_devices
from .errors import NirnayaAPIError, unexpected_error
from .options import SolveOptions
from .orchestrator import solve_model, validate_model
from .security import DEFAULT_LIMITS, SecurityLimits, check_payload_size
from .version import API_SCHEMA_VERSION, JSON_MODEL_FORMAT_VERSION, __version__


def _load_json_file(path: str, limits: SecurityLimits) -> dict:
    with open(path, "rb") as f:
        raw = f.read()
    check_payload_size(raw, limits)
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        from .errors import ValidationError

        raise ValidationError(f"File {path!r} is not valid JSON: {exc}")


def _emit(data: dict, as_json: bool, human: str) -> None:
    if as_json:
        click.echo(json.dumps(data, indent=2, default=str))
    else:
        click.echo(human)


def _handle_error(exc: Exception, as_json: bool) -> None:
    err = exc if isinstance(exc, NirnayaAPIError) else unexpected_error(exc)
    if as_json:
        click.echo(json.dumps(err.to_dict(), indent=2), err=True)
    else:
        click.echo(f"error: [{err.code}] {err.message}", err=True)
    sys.exit(1)


@click.group()
@click.version_option(__version__, prog_name="nirnaya")
def main():
    """Nirnaya: solve, validate, and inspect optimization models."""


@main.command()
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--solver", default="auto", show_default=True)
@click.option("--backend", default="auto", show_default=True, help="auto | cpu | gpu")
@click.option("--device", default=None, help="Pin a specific device id, e.g. gpu:0")
@click.option("--time-limit", default=30.0, show_default=True, type=float)
@click.option("--iteration-limit", default=10_000, show_default=True, type=int)
@click.option("--feasibility-tol", default=1e-7, show_default=True, type=float)
@click.option("--optimality-tol", default=1e-7, show_default=True, type=float)
@click.option("--no-presolve", is_flag=True, help="Skip the presolve pass.")
@click.option("--verbose", is_flag=True, help="Include detailed numerical diagnostics.")
@click.option("--seed", default=None, type=int)
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def solve(
    model_path: str,
    solver: str,
    backend: str,
    device: Optional[str],
    time_limit: float,
    iteration_limit: int,
    feasibility_tol: float,
    optimality_tol: float,
    no_presolve: bool,
    verbose: bool,
    seed: Optional[int],
    as_json: bool,
):
    """Solve MODEL_PATH (a JSON file in the Nirnaya model format)."""
    try:
        model_dict = _load_json_file(model_path, DEFAULT_LIMITS)
        opts = SolveOptions(
            solver=solver,
            backend=backend,
            device=device,
            time_limit=time_limit,
            iteration_limit=iteration_limit,
            feasibility_tol=feasibility_tol,
            optimality_tol=optimality_tol,
            presolve=not no_presolve,
            verbose=verbose,
            seed=seed,
        )
        result = solve_model(model_dict, opts, DEFAULT_LIMITS)
    except Exception as exc:
        _handle_error(exc, as_json)
        return

    human_lines = [
        f"status:          {result.status.value}",
        f"objective_value: {result.objective_value}",
        f"solver:          {result.solver}",
        f"backend/device:  {result.backend}/{result.device}",
        f"solve_time_s:    {result.solve_time_s:.6f}",
        f"presolve_time_s: {result.presolve_time_s:.6f}",
        f"iterations:      {result.iterations}",
    ]
    if result.warnings:
        human_lines.append("warnings:")
        human_lines += [f"  - {w}" for w in result.warnings]
    human_lines.append("variables:")
    human_lines += [f"  {k} = {v}" for k, v in result.variable_values.items()]

    _emit(result.to_dict(), as_json, "\n".join(human_lines))
    if result.status.value not in ("optimal",):
        sys.exit(2)


@main.command()
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def validate(model_path: str, as_json: bool):
    """Validate MODEL_PATH without solving it."""
    try:
        model_dict = _load_json_file(model_path, DEFAULT_LIMITS)
        report = validate_model(model_dict, DEFAULT_LIMITS)
    except Exception as exc:
        _handle_error(exc, as_json)
        return

    human = (
        f"valid: {report['valid']}\n"
        f"variables: {report['num_variables']}\n"
        f"constraints: {report['num_constraints']}"
    )
    if report["issues"]:
        human += "\nissues:\n" + "\n".join(f"  - {i}" for i in report["issues"])
    _emit(report, as_json, human)
    sys.exit(0 if report["valid"] else 1)


@main.command()
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def devices(as_json: bool):
    """List devices/backends this installation can target."""
    try:
        devs = list_devices()
    except Exception as exc:
        _handle_error(exc, as_json)
        return
    data = {"devices": [d.__dict__ for d in devs]}
    human = "\n".join(
        f"{d.id:16s} kind={d.kind:4s} available={d.available!s:5s} name={d.name}" for d in devs
    )
    _emit(data, as_json, human)


@main.command()
@click.option(
    "--sizes",
    default="10,100,1000",
    show_default=True,
    help="Comma-separated variable counts to benchmark.",
)
@click.option("--backend", default="auto", show_default=True)
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def benchmark(sizes: str, backend: str, as_json: bool):
    """Run a small built-in benchmark suite across problem sizes.

    Generates simple synthetic knapsack-style LPs at each requested size
    and reports solve time / status, so operators can sanity-check an
    installation (and compare CPU vs GPU) without hand-writing a model.
    """
    try:
        size_list = [int(s.strip()) for s in sizes.split(",") if s.strip()]
    except ValueError as exc:
        _handle_error(_value_error_to_validation(exc), as_json)
        return

    results = []
    for n in size_list:
        model_dict = _synthetic_model(n)
        opts = SolveOptions(backend=backend, time_limit=60.0)
        t0 = time.perf_counter()
        try:
            result = solve_model(model_dict, opts, DEFAULT_LIMITS)
            wall = time.perf_counter() - t0
            results.append(
                {
                    "n_variables": n,
                    "status": result.status.value,
                    "solve_time_s": result.solve_time_s,
                    "wall_time_s": wall,
                    "backend": result.backend,
                    "device": result.device,
                }
            )
        except NirnayaAPIError as exc:
            results.append({"n_variables": n, "error": exc.to_dict()["error"]})

    human = "\n".join(
        f"n={r.get('n_variables'):<8} status={r.get('status', r.get('error', {}).get('code'))}"
        for r in results
    )
    _emit({"benchmark": results}, as_json, human)


def _synthetic_model(n: int) -> dict:
    variables = {f"x{i}": {"lb": 0.0, "ub": 1.0, "kind": "continuous"} for i in range(n)}
    coeffs = {f"x{i}": float((i % 7) + 1) for i in range(n)}
    return {
        "name": f"synthetic-{n}",
        "variables": variables,
        "constraints": {
            "cap": {"coefficients": coeffs, "sense": "<=", "rhs": max(1.0, n / 2.0)}
        },
        "objective": {"coefficients": coeffs, "sense": "max"},
    }


def _value_error_to_validation(exc: ValueError):
    from .errors import ValidationError

    return ValidationError(f"--sizes must be a comma-separated list of integers: {exc}")


@main.command()
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def version(as_json: bool):
    """Show version and integration status of Parts 1-4."""
    deps = {}
    for pkg in ("nirnaya_core", "nirnaya_presolve", "nirnaya_solver", "nirnaya_gpu"):
        try:
            mod = __import__(pkg)
            deps[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            deps[pkg] = None

    data = {
        "api_version": __version__,
        "json_model_format_version": JSON_MODEL_FORMAT_VERSION,
        "api_schema_version": API_SCHEMA_VERSION,
        "dependencies": deps,
    }
    human_lines = [f"nirnaya-api {__version__}"]
    human_lines.append(f"json_model_format_version: {JSON_MODEL_FORMAT_VERSION}")
    human_lines.append(f"api_schema_version: {API_SCHEMA_VERSION}")
    for k, v in deps.items():
        human_lines.append(f"  {k}: {v if v is not None else 'not installed'}")
    _emit(data, as_json, "\n".join(human_lines))


if __name__ == "__main__":
    main()
