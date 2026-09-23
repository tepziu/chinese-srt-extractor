"""Cancelable subprocesses and encoder capability checks for local media jobs."""
import os
import subprocess
import tempfile
import threading
import time
from functools import lru_cache
from pathlib import Path


def stop_process(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


@lru_cache(maxsize=1)
def encoder_args():
    from config import DEVICE, acquire_gpu_slot
    if DEVICE == 'cuda':
        try:
            with acquire_gpu_slot(timeout=30):
                probe = subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=s=64x64:d=0.04',
                    '-frames:v','1','-c:v','h264_nvenc','-f','null','-'], capture_output=True, timeout=20)
            if probe.returncode == 0:
                return ['-c:v','h264_nvenc','-preset','p4','-cq','22','-b:v','0']
        except (OSError, subprocess.SubprocessError, TimeoutError):
            pass
    return ['-c:v','libx264','-preset','fast','-crf','20']


def run_media(command, job_id=None, timeout=3600):
    from config import jobs
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=log)
        job = jobs.get(job_id)
        if job:
            job['_ffmpeg_process'] = process
        started = time.monotonic()
        try:
            while process.poll() is None:
                if job and job.get('cancel'):
                    raise RuntimeError('Đã hủy xử lý media')
                if time.monotonic()-started > timeout:
                    raise TimeoutError('Xử lý media vượt thời gian cho phép')
                time.sleep(0.1)
            log.seek(0, os.SEEK_END)
            length = log.tell()
            log.seek(max(0, length-4000))
            detail = log.read().decode('utf-8', errors='replace')
            if process.returncode:
                raise RuntimeError(f'FFmpeg lỗi: {detail[-1000:]}')
            return subprocess.CompletedProcess(command, 0, '', detail)
        finally:
            stop_process(process)
            if job:
                job.pop('_ffmpeg_process', None)


class FrameEncoder:
    """Pipe raw frames into one encoder; watchdog unblocks stalled pipe writes."""
    def __init__(self, command, job_id):
        from config import jobs
        self.job = jobs.get(job_id)
        self.log = tempfile.TemporaryFile()
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log)
        self.finished = threading.Event()
        self.touched = time.monotonic()
        self.failure = None
        if self.job:
            self.job['_ffmpeg_process'] = self.process
        self.watchdog = threading.Thread(target=self._watch, daemon=True)
        self.watchdog.start()

    def _watch(self):
        while not self.finished.wait(0.2):
            if self.job and self.job.get('cancel'):
                self.failure = 'Đã hủy render'
            elif time.monotonic() - self.touched > 300:
                self.failure = 'Encoder không tiến triển trong 5 phút'
            if self.failure:
                stop_process(self.process)
                return

    def write(self, frame):
        if self.failure:
            raise RuntimeError(self.failure)
        self.process.stdin.write(frame.tobytes())
        self.touched = time.monotonic()

    def close(self, abort=False):
        try:
            if abort:
                stop_process(self.process)
            try:
                self.process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                stop_process(self.process)
                raise TimeoutError('Encoder không thể hoàn tất file')
            if not abort and (self.failure or self.process.returncode):
                self.log.seek(0, os.SEEK_END)
                end = self.log.tell()
                self.log.seek(max(0, end-2000))
                raise RuntimeError(self.failure or self.log.read().decode('utf-8', errors='replace'))
        finally:
            self.finished.set()
            self.watchdog.join(timeout=6)
            self.log.close()
            if self.job:
                self.job.pop('_ffmpeg_process', None)
