import time
import uuid
from dataclasses import dataclass, field
from multiprocessing import Process
from typing import Dict, List, Optional

@dataclass
class Job:
    """Represents a background job."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: str = "pending"
    pid: Optional[int] = None
    start_time: float = 0.0
    end_time: float = 0.0

class ProcessQueue:
    """Manages a queue of background processes."""
    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        self.processes: List[Process] = []

    def add_job(self, target, args) -> Job:
        """Adds a new job to the queue and starts it."""
        job = Job()
        process = Process(target=target, args=(job.id, *args))
        process.start()
        
        job.pid = process.pid
        job.start_time = time.time()
        job.status = "running"
        
        self.jobs[job.id] = job
        self.processes.append(process)
        return job

    def cleanup(self):
        """Cleans up finished processes."""
        for process in self.processes[:]:
            if not process.is_alive():
                self.processes.remove(process)

    def get_job_status(self, job_id: str) -> Optional[Job]:
        """Returns the status of a specific job."""
        return self.jobs.get(job_id)

    def update_job_status(self, job_id: str, status: str):
        """Updates the status of a job."""
        if job_id in self.jobs:
            self.jobs[job_id].status = status
            if status in ["completed", "failed"]:
                self.jobs[job_id].end_time = time.time()