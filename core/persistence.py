"""
PersistenceStore - 把用户画像和对话落到 SQLite

职责：只管落库和读库。不做业务推断，不做 session 生命周期管理。
调用方（web_demo.py/SessionStore）负责在合适的时机写进来。

设计原则：
- 只用 stdlib（sqlite3）。延续项目"低依赖"哲学。
- schema 幂等：CREATE TABLE IF NOT EXISTS，启动即建。
- 写操作尽量短事务，每次 commit，不攒批。原型够用。
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Optional

DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "onboarding.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id      TEXT PRIMARY KEY,
    nickname     TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    last_seen_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id     TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL,
    profiling_mode TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    ended_at       TEXT,
    archetype_key  TEXT,
    FOREIGN KEY(user_id) REFERENCES users(user_id)
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    turn_index  INTEGER NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    action_type TEXT,
    user_state  TEXT,
    created_at  TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS final_profiles (
    session_id            TEXT PRIMARY KEY,
    user_id               TEXT NOT NULL,
    nickname              TEXT NOT NULL,
    archetype_key         TEXT NOT NULL,
    archetype_name        TEXT NOT NULL,
    archetype_emoji       TEXT,
    description           TEXT,
    agent_promise         TEXT,
    dims_json             TEXT NOT NULL,
    macaron_promises_json TEXT,
    created_at            TEXT NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(user_id)
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class PersistenceStore:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # http.server 是多线程的，sqlite3 连接跨线程不安全，统一加锁。
        self._lock = Lock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level=None,  # 自动 commit，每条 SQL 一事务
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.executescript(SCHEMA)

    # ---------- users ----------

    def upsert_user(self, user_id: str, nickname: str):
        now = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO users(user_id, nickname, created_at, last_seen_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET "
                "nickname = excluded.nickname, last_seen_at = excluded.last_seen_at",
                (user_id, nickname, now, now),
            )

    def touch_user(self, user_id: str):
        with self._lock:
            self._conn.execute(
                "UPDATE users SET last_seen_at = ? WHERE user_id = ?",
                (_now(), user_id),
            )

    def get_user(self, user_id: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT user_id, nickname, created_at, last_seen_at FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return dict(row) if row else None

    # ---------- sessions ----------

    def register_session(self, session_id: str, user_id: str, profiling_mode: str):
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO sessions(session_id, user_id, profiling_mode, created_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, user_id, profiling_mode, _now()),
            )

    def update_session_mode(self, session_id: str, profiling_mode: str):
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET profiling_mode = ? WHERE session_id = ?",
                (profiling_mode, session_id),
            )

    def mark_session_archetype(self, session_id: str, archetype_key: str):
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET archetype_key = ? WHERE session_id = ?",
                (archetype_key, session_id),
            )

    # ---------- messages ----------

    def append_message(
        self,
        session_id: str,
        turn_index: int,
        role: str,
        content: str,
        action_type: Optional[str] = None,
        user_state: Optional[str] = None,
    ):
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages(session_id, turn_index, role, content, action_type, user_state, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, turn_index, role, content, action_type, user_state, _now()),
            )

    def get_messages(self, session_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT turn_index, role, content, action_type, user_state, created_at "
                "FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ---------- final profiles ----------

    def save_final_profile(
        self,
        session_id: str,
        user_id: str,
        nickname: str,
        archetype: dict,
        dims: dict,
        macaron_promises: Optional[list] = None,
    ):
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO final_profiles("
                "session_id, user_id, nickname, archetype_key, archetype_name, "
                "archetype_emoji, description, agent_promise, dims_json, "
                "macaron_promises_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    user_id,
                    nickname,
                    archetype.get("key", ""),
                    archetype.get("name", ""),
                    archetype.get("emoji", ""),
                    archetype.get("description", ""),
                    archetype.get("agent_promise", ""),
                    json.dumps(dims, ensure_ascii=False),
                    json.dumps(macaron_promises or [], ensure_ascii=False),
                    _now(),
                ),
            )
            self._conn.execute(
                "UPDATE sessions SET archetype_key = ? WHERE session_id = ?",
                (archetype.get("key", ""), session_id),
            )

    def get_final_profile(self, session_id: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM final_profiles WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if not row:
            return None
        data = dict(row)
        data["dims"] = json.loads(data.pop("dims_json") or "{}")
        data["macaron_promises"] = json.loads(data.pop("macaron_promises_json") or "[]")
        return data

    def list_profiles(self, limit: int = 200) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT session_id, user_id, nickname, archetype_key, archetype_name, "
                "archetype_emoji, description, dims_json, created_at "
                "FROM final_profiles ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["dims"] = json.loads(d.pop("dims_json") or "{}")
            out.append(d)
        return out

    def close(self):
        with self._lock:
            self._conn.close()
