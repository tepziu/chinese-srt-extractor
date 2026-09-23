"""Bounded submission; OS media-worker lease serializes actual processing."""
import os
import threading
from concurrent.futures import ThreadPoolExecutor

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='studio-worker')
_capacity = threading.BoundedSemaphore(max(2, int(os.getenv('STUDIO_MAX_PENDING_JOBS', '16'))))


def submit(target, *args):
    if not _capacity.acquire(blocking=False):
        raise TimeoutError('Hàng đợi đã đầy. Hãy chờ một công việc hoàn tất.')
    try:
        from config import jobs
        job_id = args[0] if args and isinstance(args[0], str) else None
        generation = jobs.get(job_id, {}).get('_created_at') if job_id else None
        def invoke():
            if job_id:
                job = jobs.get(job_id, {})
                if job.get('_created_at') != generation or job.get('cancel'):
                    return
            return target(*args)
        future = _executor.submit(invoke)
    except BaseException:
        _capacity.release()
        raise
    future.add_done_callback(lambda _: _capacity.release())
    return future
