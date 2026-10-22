"""In-memory job tracking for ingestion. Swap for Redis + RQ/arq if you outgrow one process."""

import threading
import uuid
from dataclasses import dataclass
from typing import Literal

JobStatus = Literal["queued", "running", "done", "failed"]


@dataclass
class Job:
    id: str
    url: str
    status: JobStatus = "queued"
    stage: str = "queued"
    progress: float = 0.0
    repo_id: str | None = None
    error: str | None = None


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, url: str) -> Job:
        job = Job(id=uuid.uuid4().hex, url=url)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in fields.items():
                setattr(job, key, value)

    def finish(self, job_id: str, repo_id: str) -> None:
        self.update(job_id, status="done", stage="done", progress=1.0, repo_id=repo_id)

    def fail(self, job_id: str, error: str) -> None:
        self.update(job_id, status="failed", error=error)
