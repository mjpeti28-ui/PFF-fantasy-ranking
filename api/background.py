"""Lightweight in-memory job manager for asynchronous tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock, Thread
from typing import Any, Callable, Dict, Optional
from uuid import uuid4


@dataclass
class JobRecord:
    job_id: str
    job_type: str
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class JobManager:
    """Manage background jobs executed in daemon threads.

    This simple manager keeps all job state in-memory. It is intended for
    single-process deployments (Render web service) and should be replaced with
    a persistent queue if horizontal scaling is required.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._jobs: Dict[str, JobRecord] = {}

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    def _set(self, record: JobRecord) -> None:
        with self._lock:
            self._jobs[record.job_id] = record

    def _update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            for key, value in fields.items():
                setattr(record, key, value)

    def create_job(
        self,
        job_type: str,
        func: Callable[..., Any],
        *,
        args: tuple[Any, ...] = (),
        kwargs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        job_id = uuid4().hex
        record = JobRecord(
            job_id=job_id,
            job_type=job_type,
            status="pending",
            created_at=self._now(),
            metadata=metadata or {},
        )
        self._set(record)

        def runner() -> None:
            self._update(job_id, status="running", started_at=self._now())
            try:
                result = func(*args, **(kwargs or {}))
            except Exception as exc:  # pragma: no cover - surfaced via API
                self._update(
                    job_id,
                    status="failed",
                    finished_at=self._now(),
                    error=str(exc),
                )
                return
            self._update(
                job_id,
                status="completed",
                finished_at=self._now(),
                result=result,
            )

        thread = Thread(target=runner, name=f"job-{job_id}", daemon=True)
        thread.start()
        return job_id

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return None
            # Return a shallow copy so callers cannot mutate internal state.
            return JobRecord(
                job_id=record.job_id,
                job_type=record.job_type,
                status=record.status,
                created_at=record.created_at,
                started_at=record.started_at,
                finished_at=record.finished_at,
                result=record.result,
                error=record.error,
                metadata=dict(record.metadata),
            )


# Global singleton used throughout the API layer.
job_manager = JobManager()

