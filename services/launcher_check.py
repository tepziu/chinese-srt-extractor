"""Read-only stop preflight; never terminates other software."""
from services.runtime_state import JobRegistry, TERMINAL, host_lock, process_alive
from services.pipeline_completion import reconcile_final_render
from config import OUTPUT_FOLDER


def main():
    try:
        with host_lock('media-worker', timeout=0), host_lock('douyin-scan', timeout=0):
            registry = JobRegistry()
            snapshots = registry.snapshots()
            active = []
            for jid, job in snapshots.items():
                if not job:
                    continue
                # A render may have finished before its parent recorded the
                # terminal update. Verify the stable media before blocking a
                # safe restart on a stale 85% pipeline.
                if reconcile_final_render(job, jid, OUTPUT_FOLDER):
                    continue
                status = job.get('status')
                if status in TERMINAL | {'uploaded', 'awaiting_review', 'interrupted'}:
                    continue
                if job.get('cancel'):
                    job.update(status='cancelled', message='Đã hủy')
                    continue
                owner = job.get('_owner_pid')
                if owner and not process_alive(owner):
                    job.update(status='interrupted', message='Tiến trình đã dừng')
                    continue
                active.append(jid)
            if active:
                print('Active jobs: ' + ', '.join(active))
                return 1
    except TimeoutError:
        print('Media worker or monitor scan is active')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
