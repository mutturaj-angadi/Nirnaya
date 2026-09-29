"""
Traceability: every transformation presolve applies is recorded as a
`TransformationRecord` in the order it was applied. This is what makes
postsolve possible (aggregations/fixes are replayed in reverse) and what
makes presolve's behavior auditable/debuggable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TransformationRecord:
    #: Sequence number in application order (0-based). Postsolve replays
    #: variable-eliminating records in *reverse* of this order.
    order: int

    #: Machine-readable transformation kind, e.g. "fixed_variable",
    #: "singleton_row", "empty_column", "doubleton_aggregation", ...
    kind: str

    #: Human-readable one-line explanation.
    description: str

    #: Names of original-space variables this transformation touched.
    variables: tuple = ()

    #: Names of original-space constraints this transformation touched.
    constraints: tuple = ()

    #: Arbitrary extra structured detail (e.g. {"value": 3.0} for a fixed
    #: variable, or {"alpha": 2.0, "beta": -1.0, "other": "x2"} for an
    #: aggregation x1 = alpha * x2 + beta). Needed by postsolve for
    #: elimination-type records; purely informational for others.
    detail: Dict[str, Any] = field(default_factory=dict)


class TraceLog:
    """Append-only ordered log of TransformationRecords."""

    def __init__(self):
        self._records: list[TransformationRecord] = []

    def add(
        self,
        kind: str,
        description: str,
        variables: tuple = (),
        constraints: tuple = (),
        detail: Optional[Dict[str, Any]] = None,
    ) -> TransformationRecord:
        rec = TransformationRecord(
            order=len(self._records),
            kind=kind,
            description=description,
            variables=tuple(variables),
            constraints=tuple(constraints),
            detail=dict(detail or {}),
        )
        self._records.append(rec)
        return rec

    @property
    def records(self) -> tuple:
        return tuple(self._records)

    def __len__(self):
        return len(self._records)

    def __iter__(self):
        return iter(self._records)
