from pathlib import Path

from app import app
from config import create_job, jobs


def test_stop_completed_job_does_not_destroy_success_status():
    job_id = 'stop_completed_123'
    create_job(job_id, status='done', progress=100, active_step='completed',
               artifacts={'final_video': 'kept'}, step_results={'burn': {'status': 'done'}})
    try:
        response = app.test_client().post('/api/stop/' + job_id)
        assert response.status_code == 409
        assert jobs.store.load(job_id)['status'] == 'done'
        assert jobs.store.load(job_id)['cancel'] is False
    finally:
        jobs.pop(job_id, None)


def test_stop_active_job_acknowledges_request_without_claiming_worker_exit():
    job_id = 'stop_active_123'
    create_job(job_id, status='processing', progress=32)
    try:
        response = app.test_client().post('/api/stop/' + job_id)
        assert response.status_code == 200
        assert response.json['cancel_requested'] is True
        assert response.json['worker_stopped'] is False
        assert jobs.store.load(job_id)['cancel'] is True
    finally:
        jobs.pop(job_id, None)


def test_studio_exposes_accessible_confirmation_and_status_region():
    page = app.test_client().get('/').get_data(as_text=True)
    assert 'id="stopConfirmDialog"' in page
    assert 'aria-labelledby="stopConfirmTitle"' in page
    assert 'aria-describedby="stopConfirmDescription"' in page
    assert '/static/js/stop-confirm.js' in page
    assert 'id="pipelineStopStatus"' in page
