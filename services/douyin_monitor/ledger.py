"""Durable source lifecycle ledger for the Chinese SRT Douyin monitor."""
from __future__ import annotations
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from services.runtime_state import runtime_dir, host_lock


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

class SourceLedger:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or (runtime_dir() / "douyin_source_ledger.sqlite3")).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self):
        c=sqlite3.connect(self.path, timeout=15)
        c.row_factory=sqlite3.Row
        c.execute("PRAGMA busy_timeout=15000")
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=FULL")
        return c

    def _init(self):
        with host_lock("douyin-source-ledger", timeout=15):
            with closing(self._connect()) as c, c:
                c.execute("""CREATE TABLE IF NOT EXISTS source_jobs(
                    source_platform TEXT NOT NULL,
                    channel_id TEXT NOT NULL,
                    source_work_id TEXT NOT NULL,
                    source_url TEXT,
                    source_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    artifact_path TEXT,
                    package_path TEXT,
                    last_error TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(source_platform, channel_id, source_work_id)
                )""")
                c.execute("CREATE INDEX IF NOT EXISTS idx_source_jobs_status ON source_jobs(status)")

    def claim(self, *, channel_id: str, work: dict[str, Any]) -> bool:
        work_id=str(work.get("aweme_id") or "").strip()
        if not work_id: return False
        now=_now()
        with host_lock("douyin-source-ledger", timeout=15):
            with closing(self._connect()) as c, c:
                row=c.execute("SELECT status FROM source_jobs WHERE source_platform='douyin' AND channel_id=? AND source_work_id=?",(channel_id,work_id)).fetchone()
                if row and row["status"] in {"CLAIMED","DOWNLOADED","PROCESSING","PACKAGE_READY","IMPORTED","SCHEDULED","PUBLISHED"}:
                    return False
                payload=json.dumps(work,ensure_ascii=False)
                c.execute("""INSERT INTO source_jobs(
                    source_platform,channel_id,source_work_id,source_url,source_json,status,attempts,created_at,updated_at
                ) VALUES('douyin',?,?,?,?,?,1,?,?)
                ON CONFLICT(source_platform,channel_id,source_work_id) DO UPDATE SET
                    source_json=excluded.source_json,status='CLAIMED',attempts=source_jobs.attempts+1,
                    last_error=NULL,updated_at=excluded.updated_at""",
                    (channel_id,work_id,work.get("share_url") or work.get("url"),payload,"CLAIMED",now,now))
        return True

    def mark(self, *, channel_id: str, work_id: str, status: str, artifact_path: str | None = None, package_path: str | None = None, error: str | None = None) -> None:
        with host_lock("douyin-source-ledger", timeout=15):
            with closing(self._connect()) as c, c:
                c.execute("""UPDATE source_jobs SET status=?,artifact_path=COALESCE(?,artifact_path),
                    package_path=COALESCE(?,package_path),last_error=?,updated_at=?
                    WHERE source_platform='douyin' AND channel_id=? AND source_work_id=?""",
                    (status,artifact_path,package_path,error,_now(),channel_id,str(work_id)))

    def get(self, channel_id: str, work_id: str) -> dict[str, Any] | None:
        with closing(self._connect()) as c:
            row=c.execute("SELECT * FROM source_jobs WHERE source_platform='douyin' AND channel_id=? AND source_work_id=?",(channel_id,str(work_id))).fetchone()
        return dict(row) if row else None
