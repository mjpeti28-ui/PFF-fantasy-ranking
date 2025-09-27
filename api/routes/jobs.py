from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from api.background import JobManager
from api.dependencies import get_job_manager, require_api_key
from api.models import JobInfo, JobStatus

router = APIRouter(prefix="/jobs", tags=["jobs"], dependencies=[Depends(require_api_key)])


def _record_to_model(job_manager: JobManager, job_id: str) -> JobInfo:
    record = job_manager.get(job_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown job")
    result_payload = record.result if isinstance(record.result, dict) else None
    try:
        status_value = JobStatus(record.status)
    except ValueError:
        status_value = JobStatus.pending
    return JobInfo(
        job_id=record.job_id,
        job_type=record.job_type,
        status=status_value,
        created_at=record.created_at,
        started_at=record.started_at,
        finished_at=record.finished_at,
        result=result_payload,
        error=record.error,
        metadata=record.metadata,
    )


@router.get("/{job_id}", response_model=JobInfo, summary="Fetch background job status")
async def get_job(job_id: str, job_manager: JobManager = Depends(get_job_manager)) -> JobInfo:
    return _record_to_model(job_manager, job_id)
