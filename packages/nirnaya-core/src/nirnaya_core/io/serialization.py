"""
Deterministic JSON serialization for models and solver results.

"Deterministic" here means: given the same logical content, ``to_json``
always produces byte-identical output. This matters for reproducibility
(hashing a model to detect changes), diffing across solver runs, and
caching. It is achieved by:

* sorting dictionary keys (``sort_keys=True``),
* emitting variable/constraint collections in a fixed order (insertion
  order for :class:`~nirnaya_core.model.Model`, explicit index order for
  :class:`~nirnaya_core.model.ProblemData`),
* emitting sparse matrix entries as COO triplets sorted by ``(row, col)``
  (see :meth:`nirnaya_core.matrix.SparseMatrix.to_coo_triplets`),
* using a fixed, explicit float representation via the standard ``json``
  encoder (no locale-dependent formatting).

The on-disk/interchange format is JSON for portability, but nothing in
this module requires the *internal* numerical representation (numpy
arrays, scipy sparse matrices) to be JSON — only the boundary is JSON.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Union

from nirnaya_core.exceptions import SerializationError
from nirnaya_core.model.model import Model
from nirnaya_core.model.problem_data import ProblemData
from nirnaya_core.solution.result import SolverResult

PathLike = Union[str, Path]

#: Schema version tag embedded in every serialized document. Bump this on
#: any backward-incompatible change to the on-disk shape; readers may use
#: it to select a compatibility path.
SCHEMA_VERSION = 1

#: Sentinel strings used to represent IEEE infinities in strict JSON.
#: Strict JSON (RFC 8259) has no token for +/-Infinity, but Nirnaya models
#: legitimately use ``math.inf`` for unbounded variable bounds. Rather than
#: emit non-standard ``Infinity``/``-Infinity`` tokens (which many JSON
#: parsers reject) or silently clamp to a large finite number (which would
#: be a silent, lossy numeric conversion -- exactly what this package's
#: numerical philosophy forbids), infinities are round-tripped explicitly
#: as these sentinel strings.
_POS_INF_SENTINEL = "Infinity"
_NEG_INF_SENTINEL = "-Infinity"


def _sanitize_floats(obj: Any) -> Any:
    """Recursively replace +/-inf floats with JSON-safe sentinel strings.

    NaN is deliberately NOT sanitized here: NaN should never reach
    serialization (every numerical field in this package rejects NaN at
    construction time), so a NaN that does show up here is a bug, and
    ``json.dumps(..., allow_nan=False)`` will correctly raise on it.
    """
    if isinstance(obj, float):
        if obj == math.inf:
            return _POS_INF_SENTINEL
        if obj == -math.inf:
            return _NEG_INF_SENTINEL
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_floats(v) for v in obj]
    return obj


def _restore_floats(obj: Any) -> Any:
    """Inverse of :func:`_sanitize_floats`."""
    if isinstance(obj, str):
        if obj == _POS_INF_SENTINEL:
            return math.inf
        if obj == _NEG_INF_SENTINEL:
            return -math.inf
        return obj
    if isinstance(obj, dict):
        return {k: _restore_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_restore_floats(v) for v in obj]
    return obj


def _dumps(payload: dict[str, Any], *, indent: int | None = 2) -> str:
    try:
        sanitized = _sanitize_floats(payload)
        return json.dumps(sanitized, sort_keys=True, indent=indent, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SerializationError(f"Failed to serialize to JSON: {exc}") from exc


def _loads(text: str) -> Any:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SerializationError(f"Invalid JSON: {exc}") from exc
    return _restore_floats(raw)


def model_to_json(model: Model, *, indent: int | None = 2) -> str:
    """Serialize a :class:`~nirnaya_core.model.Model` to a deterministic JSON string."""
    payload = {"schema_version": SCHEMA_VERSION, "kind": "Model", "data": model.to_dict()}
    return _dumps(payload, indent=indent)


def model_from_json(text: str) -> Model:
    """Deserialize a :class:`~nirnaya_core.model.Model` produced by :func:`model_to_json`."""
    payload = _loads(text)
    _check_kind(payload, "Model")
    return Model.from_dict(payload["data"])


def problem_data_to_json(data: ProblemData, *, indent: int | None = 2) -> str:
    """Serialize a :class:`~nirnaya_core.model.ProblemData` to deterministic JSON."""
    payload = {"schema_version": SCHEMA_VERSION, "kind": "ProblemData", "data": data.to_dict()}
    return _dumps(payload, indent=indent)


def problem_data_from_json(text: str) -> ProblemData:
    payload = _loads(text)
    _check_kind(payload, "ProblemData")
    return ProblemData.from_dict(payload["data"])


def solver_result_to_json(result: SolverResult, *, indent: int | None = 2) -> str:
    """Serialize a :class:`~nirnaya_core.solution.SolverResult` to deterministic JSON."""
    payload = {"schema_version": SCHEMA_VERSION, "kind": "SolverResult", "data": result.to_dict()}
    return _dumps(payload, indent=indent)


def solver_result_from_json(text: str) -> SolverResult:
    payload = _loads(text)
    _check_kind(payload, "SolverResult")
    return SolverResult.from_dict(payload["data"])


def _check_kind(payload: Any, expected_kind: str) -> None:
    if not isinstance(payload, dict) or "kind" not in payload or "data" not in payload:
        raise SerializationError(
            "JSON document missing required 'kind'/'data' envelope",
            details={"expected_kind": expected_kind},
        )
    if payload["kind"] != expected_kind:
        raise SerializationError(
            f"Expected document of kind {expected_kind!r}, got {payload['kind']!r}",
            details={"expected_kind": expected_kind, "actual_kind": payload["kind"]},
        )
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise SerializationError(
            f"Unsupported schema_version {version!r}; this reader supports {SCHEMA_VERSION}",
            details={"expected_schema_version": SCHEMA_VERSION, "actual_schema_version": version},
        )


def write_json(text: str, path: PathLike) -> None:
    """Write a serialized document to ``path`` (UTF-8, trailing newline)."""
    p = Path(path)
    p.write_text(text + "\n", encoding="utf-8")


def read_json(path: PathLike) -> str:
    """Read a serialized document from ``path``."""
    p = Path(path)
    try:
        return p.read_text(encoding="utf-8")
    except OSError as exc:
        raise SerializationError(f"Failed to read {p}: {exc}") from exc


__all__ = [
    "SCHEMA_VERSION",
    "model_to_json",
    "model_from_json",
    "problem_data_to_json",
    "problem_data_from_json",
    "solver_result_to_json",
    "solver_result_from_json",
    "write_json",
    "read_json",
]
