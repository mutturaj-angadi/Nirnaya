"""
Security controls shared by the CLI and REST layers.

Everything here is defense applied *before or around* a call into the
Parts 1-4 stack - it never touches solving logic. Concretely:

  * request/payload size limits
  * model size (variable/constraint count) limits
  * wall-clock execution time limits (enforced independently of whatever
    the solver reports)
  * safe temporary file handling (no predictable paths, no world-writable
    permissions, always cleaned up)
  * no shell execution is performed anywhere in this package - the CLI and
    REST layers only ever call Python functions in-process.

All limits are conservative defaults and are configurable via
:class:`SecurityLimits` (and, for the server, environment variables - see
docs/API.md).
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import os
import stat
import tempfile
from dataclasses import dataclass
from typing import Callable, TypeVar

from .errors import PayloadTooLargeError, ResourceLimitError, SolveTimeoutError

T = TypeVar("T")


@dataclass(frozen=True)
class SecurityLimits:
    #: Maximum size, in bytes, of a raw request/file payload (JSON text)
    #: this layer will parse. Applied before JSON parsing to avoid feeding
    #: a parser arbitrarily large input.
    max_payload_bytes: int = 5 * 1024 * 1024  # 5 MiB

    #: Maximum number of variables in a model.
    max_variables: int = 200_000

    #: Maximum number of constraints in a model.
    max_constraints: int = 200_000

    #: Maximum number of (constraint, variable) coefficient entries,
    #: bounding total payload complexity beyond raw variable/constraint
    #: counts (guards against dense-matrix blowups).
    max_nonzeros: int = 5_000_000

    #: Hard ceiling on the time_limit a caller may request, regardless of
    #: what SolveOptions.time_limit says. Prevents a client from requesting
    #: an effectively unbounded solve.
    max_time_limit_s: float = 600.0

    #: Absolute wall-clock ceiling this layer enforces around every solve
    #: call, independent of the solver's own time-limit handling. Guards
    #: against a misbehaving/hanging backend.
    hard_wall_clock_s: float = 900.0


DEFAULT_LIMITS = SecurityLimits()


def check_payload_size(raw_bytes: bytes, limits: SecurityLimits = DEFAULT_LIMITS) -> None:
    if len(raw_bytes) > limits.max_payload_bytes:
        raise PayloadTooLargeError(
            f"Payload of {len(raw_bytes)} bytes exceeds the "
            f"{limits.max_payload_bytes}-byte limit."
        )


def check_model_size(model_dict: dict, limits: SecurityLimits = DEFAULT_LIMITS) -> None:
    variables = model_dict.get("variables", {}) or {}
    constraints = model_dict.get("constraints", {}) or {}

    if len(variables) > limits.max_variables:
        raise ResourceLimitError(
            f"Model has {len(variables)} variables, exceeding the limit of "
            f"{limits.max_variables}."
        )
    if len(constraints) > limits.max_constraints:
        raise ResourceLimitError(
            f"Model has {len(constraints)} constraints, exceeding the limit of "
            f"{limits.max_constraints}."
        )

    nnz = 0
    for c in constraints.values():
        nnz += len(c.get("coefficients", {}) or {})
    obj = model_dict.get("objective") or {}
    nnz += len(obj.get("coefficients", {}) or {})
    if nnz > limits.max_nonzeros:
        raise ResourceLimitError(
            f"Model has {nnz} nonzero coefficients, exceeding the limit of "
            f"{limits.max_nonzeros}."
        )


def clamp_time_limit(requested: float | None, limits: SecurityLimits = DEFAULT_LIMITS) -> float:
    """Return an enforced time limit, never exceeding the configured ceiling."""
    if requested is None:
        return limits.max_time_limit_s
    return min(requested, limits.max_time_limit_s)


def run_with_wall_clock_limit(
    fn: Callable[[], T],
    timeout_s: float,
) -> T:
    """Run ``fn`` in a worker thread and enforce a hard wall-clock ceiling.

    Raises :class:`SolveTimeoutError` if ``fn`` does not complete in time.
    Note: the underlying call is not forcibly killed (Python cannot safely
    preempt an arbitrary thread), so a well-behaved solver is still expected
    to honor its own ``time_limit``/``iteration_limit`` options; this is a
    last-resort safety net for the API layer's own promise to the caller,
    not a substitute for cooperative cancellation in nirnaya-solver.
    """
    # Deliberately not a `with ThreadPoolExecutor(...) as pool:` block: that
    # form calls shutdown(wait=True) on exit, which would block this very
    # function - and therefore the caller - until the runaway thread
    # eventually finishes, defeating the timeout entirely. Instead we shut
    # down without waiting, so a timeout returns control to the caller
    # promptly; the orphaned thread (Python cannot forcibly kill threads)
    # finishes on its own in the background.
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(fn)
    try:
        result = future.result(timeout=timeout_s)
    except concurrent.futures.TimeoutError:
        pool.shutdown(wait=False, cancel_futures=True)
        raise SolveTimeoutError(
            f"Solve did not complete within the {timeout_s}s hard wall-clock limit."
        )
    else:
        pool.shutdown(wait=False)
        return result


@contextlib.contextmanager
def safe_temp_file(suffix: str = ".json"):
    """Create a private (0600), securely-named temp file and always remove
    it, even on error.

    Yields the open file path (as ``str``); the caller is responsible for
    writing to / reading from it.
    """
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="nirnaya-")
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)  # 0600: owner read/write only
        os.close(fd)
        yield path
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.remove(path)
