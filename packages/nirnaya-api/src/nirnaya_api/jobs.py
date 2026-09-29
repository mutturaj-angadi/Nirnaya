"""
Minimal in-memory async job manager backing the optional
``POST /jobs`` / ``GET /jobs/{id}`` REST endpoints.

Design notes:
  * Single-process, in-memory only - fine for a reference deployment or a
    single-worker service; a production deployment fronting multiple
    workers would swap this for a real queue/store without changing the
    REST contract (docs/API.md documents the contract, not this
    implementation).
  * Jobs run in a bounded thread pool so a burst of submissions cannot
    exhaust resources - this is itself a resource-limit control.
  * No shell/subprocess execution occurs anywhere in this module.
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from .errors import JobNotFoundError
from .solution import Solution


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued | running | completed | failed
    result: Optional[Solution] = None
    error: Optional[dict] = None


class JobManager:
    def __init__(self, max_workers: int = 4, max_pending_jobs: int = 100):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._max_pending_jobs = max_pending_jobs

    def submit(self, fn: Callable[[], Solution]) -> Job:
        with self._lock:
            pending = sum(1 for j in self._jobs.values() if j.status in ("queued", "running"))
            if pending >= self._max_pending_jobs:
                from .errors import ResourceLimitError

                raise ResourceLimitError(
                    f"Too many pending jobs ({pending} >= {self._max_pending_jobs}); "
                    "try again later."
                )
            job = Job(id=str(uuid.uuid4()))
            self._jobs[job.id] = job

        def _run():
            with self._lock:
                job.status = "running"
            try:
                result = fn()
                with self._lock:
                    job.result = result
                    job.status = "completed"
            except Exception as exc:  # noqa: BLE001 - convert any failure into job state
                from .errors import NirnayaAPIError, unexpected_error

                err = exc if isinstance(exc, NirnayaAPIError) else unexpected_error(exc)
                with self._lock:
                    job.error = err.to_dict()["error"]
                    job.status = "failed"

        self._executor.submit(_run)
        return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(f"No job with id {job_id!r}")
        return job

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


# Process-wide default job manager used by the REST app.
default_job_manager = JobManager()
