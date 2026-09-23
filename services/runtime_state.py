"""Small local runtime: durable snapshots and OS locks shared by Web and Bot."""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

TERMINAL = {"done", "partial", "error", "failed", "cancelled"}
_local_locks: dict[str, threading.RLock] = {}
_lock_guard = threading.Lock()


def valid_job_id(value):
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{6,64}", value) is not None


def process_alive(pid):
    if not pid:
        return False
    if pid == os.getpid():
        return True
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.restype = wintypes.HANDLE
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return ctypes.get_last_error() == 5
        try:
            code = wintypes.DWORD()
            kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def runtime_dir():
    return Path(os.getenv("STUDIO_RUNTIME_DIR", str(Path(__file__).resolve().parents[1] / "runtime")))


@contextmanager
def host_lock(name, timeout=600, cancelled=None):
    """Reentrant in one thread; kernel releases the interprocess lock on crash."""
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', name):
        raise ValueError('Tên khóa không hợp lệ')
    key = str(runtime_dir() / (name + ".lock"))
    with _lock_guard:
        lock = _local_locks.setdefault(key, threading.RLock())
    deadline = time.monotonic() + max(0, timeout)
    while not lock.acquire(timeout=min(0.1, max(0, deadline-time.monotonic()))):
        if cancelled and cancelled():
            raise RuntimeError('Đã hủy tác vụ đang chờ')
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Tài nguyên đang bận: {name}")
    handles = getattr(_held_locks, "handles", None)
    if handles is None:
        handles = _held_locks.handles = {}
    handle = None
    try:
        if key in handles:
            yield
            return
        if not Path(key).parent.exists():
            Path(key).parent.mkdir(parents=True, exist_ok=True)
        handle = open(key, "a+b")
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        while True:
            if cancelled and cancelled():
                raise RuntimeError("Đã hủy (Stop)")
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Tài nguyên đang bận: {name}") from None
                time.sleep(min(0.1, max(0, deadline - time.monotonic())))
        handles[key] = handle
        try:
            yield
        finally:
            handles.pop(key, None)
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        if handle:
            handle.close()
        lock.release()


_held_locks = threading.local()


def serial_task(fn):
    """Serialize expensive work across entrypoints, allowing nested service calls."""
    from functools import wraps
    from inspect import signature
    sig = signature(fn)
    @wraps(fn)
    def wrapped(*args, **kwargs):
        bound = sig.bind(*args, **kwargs).arguments
        jid = bound.get('job_id') or (bound.get('job') or {}).get('job_id')
        generation = None
        if jid:
            from config import jobs
            generation = jobs.get(jid, {}).get('_created_at')
        def cancelled():
            if not jid:
                return False
            from config import jobs
            return jobs.get(jid, {}).get('cancel', False)
        with host_lock("media-worker", timeout=86400, cancelled=cancelled):
            if jid and jobs.get(jid, {}).get('_created_at') != generation:
                return None  # An older queued invocation was replaced by retry.
            return fn(*args, **kwargs)
    return wrapped


def public_data(value):
    if isinstance(value, dict):
        return {str(k): public_data(v) for k, v in list(value.items())
                if not str(k).startswith("_") and not any(s in str(k).lower() for s in
                ("api_key", "token", "password", "secret", "cookie"))}
    if isinstance(value, (list, tuple, set)):
        return [public_data(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value if isinstance(value, (str, int, float, bool, type(None))) else None


class JobStore:
    def __init__(self, path=None):
        self.path = Path(path) if path else runtime_dir() / "jobs.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, data TEXT NOT NULL, updated REAL NOT NULL, cancel INTEGER NOT NULL DEFAULT 0)")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        try:
            with db:
                yield db
        finally:
            db.close()

    def save(self, job, reset=False):
        data = public_data(job)
        data.update({k: job.get(k) for k in ("_created_at", "_completed_at", "_owner_pid")})
        with self.connect() as db:
            db.execute("INSERT INTO jobs VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE SET data=excluded.data, updated=excluded.updated, cancel=" +
                       ("excluded.cancel" if reset else "MAX(jobs.cancel,excluded.cancel)"),
                       (job["job_id"], json.dumps(data, ensure_ascii=False), time.time(), int(dict.get(job, "cancel", False))))

    def load(self, job_id):
        with self.connect() as db:
            row = db.execute("SELECT data,cancel FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            return None
        data = json.loads(row[0])
        data["cancel"] = bool(row[1])
        return data

    def cancel(self, job_id):
        with self.connect() as db:
            db.execute("UPDATE jobs SET cancel=1 WHERE id=?", (job_id,))

    def cancelled(self, job_id):
        with self.connect() as db:
            row = db.execute("SELECT cancel FROM jobs WHERE id=?", (job_id,)).fetchone()
        return bool(row and row[0])

    def ids(self):
        with self.connect() as db:
            return [row[0] for row in db.execute("SELECT id FROM jobs ORDER BY updated DESC")]

    def delete(self, job_id):
        with self.connect() as db:
            db.execute("DELETE FROM jobs WHERE id=?", (job_id,))


class ObservableDict(dict):
    def __init__(self, data, changed):
        self.changed = changed
        super().__init__({k: self.wrap(v) for k, v in data.items()})

    def wrap(self, value):
        return ObservableDict(value, self.changed) if isinstance(value, dict) else value

    def __setitem__(self, key, value):
        super().__setitem__(key, self.wrap(value))
        self.changed(key)

    def update(self, *args, **kwargs):
        data = dict(*args, **kwargs)
        for key, value in data.items():
            dict.__setitem__(self, key, self.wrap(value))
        self.changed("status" if "status" in data else "update")

    def setdefault(self, key, default=None):
        if key not in self:
            self[key] = default
        return self[key]

    def pop(self, key, *default):
        result = super().pop(key, *default)
        self.changed("update")
        return result


class JobRecord(ObservableDict):
    def __init__(self, data, store):
        self.store = store
        self._last_save = 0.0
        self._cancel_check = 0.0
        self._saving = False
        super().__init__(data, self.persist)

    def get(self, key, default=None):
        if key == "cancel" and not dict.get(self, "cancel") and time.monotonic() - self._cancel_check > 0.25:
            self._cancel_check = time.monotonic()
            if self.store.cancelled(dict.get(self, "job_id")):
                dict.__setitem__(self, "cancel", True)
        return super().get(key, default)

    def persist(self, key="status"):
        if self._saving:
            return
        if key == "cancel" and dict.get(self, "cancel"):
            self.store.cancel(self["job_id"])
        if key not in {"status", "pending_pipeline", "cancel", "srt_files", "review_revision"} and time.monotonic() - self._last_save < 0.25:
            return
        self._saving = True
        try:
            if dict.get(self, "status") in TERMINAL and not dict.get(self, "_completed_at"):
                dict.__setitem__(self, "_completed_at", time.time())
            self.store.save(self)
            self._last_save = time.monotonic()
        finally:
            self._saving = False


class JobRegistry(dict):
    """Keep dict compatibility for existing services; lazily restore remote jobs."""
    def __init__(self):
        super().__init__()
        self._store = None

    @property
    def store(self):
        if self._store is None or self._store.path.parent != runtime_dir():
            self._store = JobStore()
        return self._store

    def __setitem__(self, key, value):
        record = JobRecord(value, self.store)
        dict.__setitem__(self, key, record)
        self.store.save(record, reset=True)

    def get(self, key, default=None):
        value = dict.get(self, key)
        if value is not None and dict.get(value, "_owner_pid") == os.getpid():
            if value.get('status') not in TERMINAL | {'awaiting_review'}:
                return value
        if not valid_job_id(key):
            return default
        data = self.store.load(key)
        if data is None:
            return value if value is not None else default
        if (value is not None and dict.get(value, '_owner_pid') == os.getpid()
                and data.get('_owner_pid') == os.getpid()
                and value.get('status') != 'awaiting_review'
                and data.get('status') == value.get('status')):
            return value
        nested_active = [key for key, val in data.items() if isinstance(val, dict) and
                         val.get('status') in {'generating', 'processing', 'queued', 'rendering'}]
        if (nested_active or data.get('status') not in TERMINAL | {'awaiting_review', 'uploaded', 'uploading', 'interrupted'}) and not process_alive(data.get('_owner_pid')):
            for key in nested_active:
                data[key].update(status='interrupted', message='Tiến trình xử lý đã dừng; cần thử lại')
            data.update(status='interrupted', message='Tác vụ bị gián đoạn khi ứng dụng dừng. Có thể chạy lại với cache đã lưu.')
            self.store.save(data)
        record = JobRecord(data, self.store)
        if data.get('_owner_pid') == os.getpid():
            dict.__setitem__(self, key, record)
        return record

    def __getitem__(self, key):
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def __contains__(self, key):
        return self.get(key) is not None

    def pop(self, key, *default):
        self.store.delete(key)
        return dict.pop(self, key, *default)

    def snapshots(self):
        return {key: self.get(key) for key in set(self.store.ids()) | set(dict.keys(self))}


def resolve_media_path(value, *roots, must_exist=True):
    path = Path(value).resolve()
    if not any(path.is_relative_to(Path(root).resolve()) and path != Path(root).resolve() for root in roots):
        raise ValueError("Đường dẫn nằm ngoài uploads/ hoặc outputs/")
    if must_exist and not path.is_file():
        raise ValueError("File không tồn tại")
    return path
