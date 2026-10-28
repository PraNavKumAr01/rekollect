"""SQLite store. One row per *unique* command (embedded once), one row per run."""

from __future__ import annotations

import os
import re
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np

from rekollect import config as C

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
CREATE TABLE IF NOT EXISTS commands (
    id        INTEGER PRIMARY KEY,
    text      TEXT UNIQUE NOT NULL,
    first_ts  INTEGER DEFAULT 0,
    last_ts   INTEGER DEFAULT 0,
    count     INTEGER DEFAULT 0,
    fails     INTEGER DEFAULT 0,
    vec       BLOB,
    hint      TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY,
    cmd_id      INTEGER NOT NULL REFERENCES commands(id) ON DELETE CASCADE,
    ts          INTEGER DEFAULT 0,
    cwd         TEXT DEFAULT '',
    exit        INTEGER,
    duration_ms INTEGER,
    source      TEXT,
    session     TEXT DEFAULT '',
    UNIQUE (cmd_id, ts, cwd)
);
CREATE INDEX IF NOT EXISTS runs_cwd ON runs(cwd);
CREATE INDEX IF NOT EXISTS runs_cmd ON runs(cmd_id);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
-- what you chose in the picker: the local, private signal rekollect learns from
CREATE TABLE IF NOT EXISTS feedback (
    id      INTEGER PRIMARY KEY,
    ts      INTEGER NOT NULL,
    query   TEXT NOT NULL,
    qvec    BLOB,
    cmd_id  INTEGER NOT NULL REFERENCES commands(id) ON DELETE CASCADE,
    kind    TEXT NOT NULL,          -- pick | skip | irrelevant
    rank    INTEGER DEFAULT 0,      -- position it was shown at (0 = first)
    cwd     TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS feedback_cmd ON feedback(cmd_id);
-- bookmarks / annotations
CREATE TABLE IF NOT EXISTS marks (
    cmd_id  INTEGER PRIMARY KEY REFERENCES commands(id) ON DELETE CASCADE,
    name    TEXT DEFAULT '',
    tags    TEXT DEFAULT '',        -- space separated
    note    TEXT DEFAULT '',
    created INTEGER DEFAULT 0
);
"""

# columns added after the first release: (table, column, declaration)
MIGRATIONS = [
    ("runs", "session", "TEXT DEFAULT ''"),
    ("runs", "shell", "TEXT DEFAULT ''"),
    ("runs", "host", "TEXT DEFAULT ''"),
    ("runs", "repo", "TEXT DEFAULT ''"),
    ("runs", "branch", "TEXT DEFAULT ''"),
    ("commands", "hint", "TEXT DEFAULT ''"),
]


# Recorded (workflows need them) but not offered as search results.
NAVIGATION = re.compile(r"^\s*(?:cd|chdir|pushd|popd|sl|Set-Location|Push-Location|Pop-Location|z|zi)"
                        r"(?:\s+\S+)?\s*$|^\s*cd\.\.\s*$", re.IGNORECASE)


# "run in this directory or below it", for / and \ paths, with LIKE wildcards in the path escaped
_UNDER = "(r2.cwd = ? OR r2.cwd LIKE ? ESCAPE '!' OR r2.cwd LIKE ? ESCAPE '!')"


def _under_args(root: str) -> list[str]:
    base = root.rstrip("/\\")
    esc = base.replace("!", "!!").replace("%", "!%").replace("_", "!_")
    return [base, esc + "/%", esc + "\\%"]


@dataclass
class Corpus:
    """Everything the ranker needs, loaded in one go."""
    ids: np.ndarray        # (N,)
    texts: list[str]
    vecs: np.ndarray       # (N, D) float32, L2-normalised
    last_ts: np.ndarray    # (N,)
    count: np.ndarray      # (N,)
    fails: np.ndarray      # (N,)
    last_cwd: list[str]
    hints: list[str] = None  # type: ignore[assignment]
    embedded: np.ndarray = None  # type: ignore[assignment]  # False → found by words only (not embedded yet)
    marks: dict = None  # type: ignore[assignment]           # corpus index → (name, tags, note)


class Store:
    def __init__(self, path: Optional[Path] = None, readonly: bool = False):
        self.path = path or C.db_path()
        self.readonly = readonly
        if readonly:  # used by the agent server: SQLite itself refuses any write
            self.db = sqlite3.connect(f"file:{Path(self.path).as_posix()}?mode=ro", uri=True, timeout=10)
            return
        C.ensure_dirs()
        self.db = sqlite3.connect(self.path, timeout=10)
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)
        changed = False
        for table, col, decl in MIGRATIONS:
            if col not in {r[1] for r in self.db.execute(f"PRAGMA table_info({table})")}:
                self.db.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
                changed = True
        if changed:
            self.db.commit()

    def _has(self, table: str, col: str = "") -> bool:
        try:
            cols = {r[1] for r in self.db.execute(f"PRAGMA table_info({table})")}
        except sqlite3.Error:
            return False
        return bool(cols) and (not col or col in cols)

    # meta
    def get_meta(self, key: str, default: str = "") -> str:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else default

    def set_meta(self, key: str, value: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))

    # writes
    def add_run(self, text: str, ts: int, cwd: str, exit_code: Optional[int],
                duration_ms: Optional[int], source: str, session: str = "", shell: str = "",
                host: str = "", repo: str = "", branch: str = "") -> bool:
        cur = self.db.execute(
            "INSERT INTO commands(text, first_ts, last_ts) VALUES (?, ?, ?) "
            "ON CONFLICT(text) DO UPDATE SET "
            "  first_ts = CASE WHEN commands.first_ts = 0 OR (excluded.first_ts > 0 AND excluded.first_ts < commands.first_ts) "
            "                  THEN excluded.first_ts ELSE commands.first_ts END, "
            "  last_ts = MAX(commands.last_ts, excluded.last_ts) "
            "RETURNING id",
            (text, ts, ts),
        )
        cmd_id = cur.fetchone()[0]
        ins = self.db.execute(
            "INSERT OR IGNORE INTO runs(cmd_id, ts, cwd, exit, duration_ms, source, session, shell, host, "
            "repo, branch) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (cmd_id, ts, cwd, exit_code, duration_ms, source, session, shell, host, repo, branch),
        )
        if ins.rowcount:
            self.db.execute(
                "UPDATE commands SET count = count + 1, fails = fails + ? WHERE id = ?",
                (1 if (exit_code not in (None, 0)) else 0, cmd_id),
            )
            return True
        return False

    def pending(self) -> list[tuple[int, str]]:
        return self.db.execute("SELECT id, text FROM commands WHERE vec IS NULL").fetchall()

    def pending_with_marks(self) -> list[tuple[int, str, str, str, str]]:
        return self.db.execute(
            "SELECT c.id, c.text, COALESCE(m.name, ''), COALESCE(m.tags, ''), COALESCE(m.note, '') "
            "FROM commands c LEFT JOIN marks m ON m.cmd_id = c.id WHERE c.vec IS NULL").fetchall()

    # feedback (local learning)
    def add_feedback(self, query: str, qvec: Optional[np.ndarray], cmd_id: int, kind: str, rank: int,
                     cwd: str, ts: Optional[int] = None) -> None:
        blob = qvec.astype(np.float16).tobytes() if qvec is not None else None
        self.db.execute("INSERT INTO feedback(ts, query, qvec, cmd_id, kind, rank, cwd) VALUES (?,?,?,?,?,?,?)",
                        (int(ts or time.time()), query, blob, int(cmd_id), kind, int(rank), cwd))

    def feedback_rows(self, limit: int = 5000) -> list[tuple]:
        """(ts, query, qvec_blob, cmd_id, kind, rank, cwd), newest first."""
        if not self._has("feedback"):
            return []
        return self.db.execute("SELECT ts, query, qvec, cmd_id, kind, rank, cwd FROM feedback "
                               "ORDER BY ts DESC, id DESC LIMIT ?", (limit,)).fetchall()

    # bookmarks
    def set_mark(self, cmd_id: int, name: str = "", tags: str = "", note: str = "") -> None:
        self.db.execute("INSERT OR REPLACE INTO marks(cmd_id, name, tags, note, created) VALUES (?,?,?,?,?)",
                        (cmd_id, name, tags, note, int(time.time())))
        self.db.execute("UPDATE commands SET vec = NULL WHERE id = ?", (cmd_id,))  # re-embed with the note

    def remove_mark(self, cmd_id: int) -> None:
        self.db.execute("DELETE FROM marks WHERE cmd_id = ?", (cmd_id,))
        self.db.execute("UPDATE commands SET vec = NULL WHERE id = ?", (cmd_id,))

    def marks_list(self) -> list[tuple]:
        """(cmd_id, text, name, tags, note, created)"""
        if not self._has("marks"):
            return []
        return self.db.execute("SELECT m.cmd_id, c.text, m.name, m.tags, m.note, m.created FROM marks m "
                               "JOIN commands c ON c.id = m.cmd_id ORDER BY m.created DESC").fetchall()

    def find_command(self, text: str) -> Optional[int]:
        row = self.db.execute("SELECT id FROM commands WHERE text = ?", (text,)).fetchone()
        return row[0] if row else None

    def set_vectors(self, rows: Iterable[tuple[int, np.ndarray]], hints: Optional[list[str]] = None) -> None:
        rows = list(rows)
        hints = hints if hints is not None else [""] * len(rows)
        self.db.executemany(
            "UPDATE commands SET vec = ?, hint = ? WHERE id = ?",
            ((v.astype(np.float16).tobytes(), h, i) for (i, v), h in zip(rows, hints)),
        )

    def clear_vectors(self) -> None:
        self.db.execute("UPDATE commands SET vec = NULL")

    def forget(self, ids: list[int]) -> None:
        self.db.executemany("DELETE FROM commands WHERE id = ?", ((i,) for i in ids))

    def commit(self) -> None:
        self.db.commit()

    # reads
    def corpus(self, success_only: bool = False, since_ts: int = 0, cwd_prefix: str = "") -> Corpus:
        where, args = ["1"], []
        if success_only:
            where.append("c.fails < c.count")
        if since_ts:
            where.append("c.last_ts >= ?")
            args.append(since_ts)
        if cwd_prefix:
            where.append("EXISTS (SELECT 1 FROM runs r2 WHERE r2.cmd_id = c.id AND " + _UNDER + ")")
            args += _under_args(cwd_prefix)
        rows = self.db.execute(
            f"""SELECT c.id, c.text, c.last_ts, c.count, c.fails, c.vec,
                   (SELECT r.cwd FROM runs r WHERE r.cmd_id = c.id AND r.cwd != ''
                    ORDER BY r.ts DESC LIMIT 1), c.hint
                FROM commands c WHERE {' AND '.join(where)}""",
            args,
        ).fetchall()
        rows = [r for r in rows if not NAVIGATION.match(r[1])]
        if cwd_prefix and rows:
            # project scope: counts, failures, recency and place come only from runs in the project,
            # so a scoped caller learns nothing about how the command was used elsewhere
            agg = {cid: (n, f or 0, ts, cwd) for cid, n, f, ts, cwd in self.db.execute(
                "SELECT r2.cmd_id, COUNT(*), SUM(CASE WHEN r2.exit IS NOT NULL AND r2.exit != 0 THEN 1 ELSE 0 END),"
                " MAX(r2.ts), r2.cwd "  # sqlite: a bare column comes from the MAX() row
                "FROM runs r2 WHERE " + _UNDER + " GROUP BY r2.cmd_id", _under_args(cwd_prefix))}
            rows = [(r[0], r[1], agg[r[0]][2], agg[r[0]][0], agg[r[0]][1], r[5], agg[r[0]][3], r[7])
                    for r in rows if r[0] in agg]
            if success_only:
                rows = [r for r in rows if r[4] < r[3]]
        if not rows:
            return Corpus(np.zeros(0, int), [], np.zeros((0, 1), np.float32), np.zeros(0), np.zeros(0),
                          np.zeros(0), [], [], np.zeros(0, bool), {})
        # commands still waiting for the embedder are kept (zero vector): words still find them
        dim = next((len(r[5]) // 2 for r in rows if r[5]), 1)
        zero = bytes(dim * 2)
        embedded = np.array([bool(r[5]) and len(r[5]) == dim * 2 for r in rows])
        vecs = np.frombuffer(b"".join(r[5] if ok else zero for r, ok in zip(rows, embedded)),
                             dtype=np.float16).reshape(len(rows), dim)
        marks: dict = {}
        if self._has("marks"):
            by_id = {cid: (n, t, no) for cid, n, t, no in
                     self.db.execute("SELECT cmd_id, name, tags, note FROM marks")}
            marks = {i: by_id[r[0]] for i, r in enumerate(rows) if r[0] in by_id}
        return Corpus(
            ids=np.array([r[0] for r in rows]),
            texts=[r[1] for r in rows],
            vecs=vecs.astype(np.float32),
            last_ts=np.array([r[2] or 0 for r in rows], dtype=np.float64),
            count=np.array([r[3] for r in rows], dtype=np.float64),
            fails=np.array([r[4] for r in rows], dtype=np.float64),
            last_cwd=[r[6] or "" for r in rows],
            hints=[r[7] or "" for r in rows],
            embedded=embedded,
            marks=marks,
        )

    def cmd_ids_in(self, cwd: str, project: str, branch: str = "") -> tuple[set[int], set[int], set[int]]:
        """Commands run here, anywhere in this project, and on this project's current branch."""
        here = {r[0] for r in self.db.execute("SELECT DISTINCT cmd_id FROM runs WHERE cwd = ?", (cwd,))}
        proj: set[int] = set()
        on_branch: set[int] = set()
        if project:
            proj = {r[0] for r in self.db.execute(
                "SELECT DISTINCT cmd_id FROM runs r2 WHERE " + _UNDER, _under_args(project))}
            if branch and self._has("runs", "branch"):
                on_branch = {r[0] for r in self.db.execute(
                    "SELECT DISTINCT cmd_id FROM runs WHERE repo = ? AND branch = ?", (project, branch))}
        return here, proj, on_branch

    def run_details(self, cmd_id: int, under: str = "") -> Optional[tuple]:
        """Most recent run: (ts, cwd, exit, duration_ms, branch, shell, host).

        `under`: only consider runs in that directory tree (agent project scope)."""
        cols = "r2.ts, r2.cwd, r2.exit, r2.duration_ms, " + (
            "r2.branch, r2.shell, r2.host" if self._has("runs", "branch") else "'', '', ''")
        where, args = "r2.cmd_id = ?", [cmd_id]
        if under:
            where += " AND " + _UNDER
            args += _under_args(under)
        return self.db.execute(f"SELECT {cols} FROM runs r2 WHERE {where} ORDER BY r2.ts DESC LIMIT 1",
                               args).fetchone()

    def run_counts(self, cmd_id: int, under: str = "") -> tuple[int, int]:
        """(runs, failures), optionally only within a directory tree."""
        where, args = "r2.cmd_id = ?", [cmd_id]
        if under:
            where += " AND " + _UNDER
            args += _under_args(under)
        n, f = self.db.execute(f"SELECT COUNT(*), SUM(CASE WHEN r2.exit IS NOT NULL AND r2.exit != 0 "
                               f"THEN 1 ELSE 0 END) FROM runs r2 WHERE {where}", args).fetchone()
        return int(n or 0), int(f or 0)

    def timeline(self) -> list[tuple]:
        """Every timestamped run, oldest first: (ts, cmd_id, text, exit, duration_ms, cwd, session)."""
        return self.db.execute(
            "SELECT r.ts, r.cmd_id, c.text, r.exit, r.duration_ms, r.cwd, r.session "
            "FROM runs r JOIN commands c ON c.id = r.cmd_id WHERE r.ts > 0 ORDER BY r.ts, r.id").fetchall()

    def session_runs(self, session: str, limit: int) -> list[tuple]:
        rows = self.db.execute(
            "SELECT r.ts, r.cmd_id, c.text, r.exit, r.duration_ms, r.cwd, r.session "
            "FROM runs r JOIN commands c ON c.id = r.cmd_id WHERE r.session = ? "
            "ORDER BY r.ts DESC, r.id DESC LIMIT ?", (session, limit)).fetchall()
        return rows[::-1]

    def counts(self) -> tuple[int, int, int]:
        u, e = self.db.execute("SELECT COUNT(*), COUNT(vec) FROM commands").fetchone()
        r = self.db.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        return u, e, r

    def size_bytes(self) -> int:
        total = 0
        for suffix in ("", "-wal", "-shm"):
            p = Path(str(self.path) + suffix)
            if p.exists():
                total += p.stat().st_size
        return total


if os.name == "nt":
    import msvcrt

    def _try_lock(fd: int) -> bool:
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False

    def _unlock(fd: int) -> None:
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
else:
    import fcntl

    def _try_lock(fd: int) -> bool:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


@contextmanager
def index_lock(blocking: bool = False) -> Iterator[bool]:
    """Only one indexer at a time (hooks may fire from several terminals)."""
    C.ensure_dirs()
    fd = os.open(C.data_dir() / "index.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        got = _try_lock(fd)
        while blocking and not got:
            time.sleep(0.1)
            got = _try_lock(fd)
        try:
            yield got
        finally:
            if got:
                _unlock(fd)
    finally:
        os.close(fd)


def git_branch(root: str) -> str:
    """Current branch of the repo/worktree at `root`, read straight from .git (no git process)."""
    if not root:
        return ""
    try:
        dotgit = Path(root) / ".git"
        if dotgit.is_file():  # worktree / submodule: "gitdir: <path>"
            target = dotgit.read_text(encoding="utf-8").strip().removeprefix("gitdir:").strip()
            dotgit = (Path(root) / target).resolve()
        head = (dotgit / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    if head.startswith("ref: refs/heads/"):
        return head[len("ref: refs/heads/"):]
    return head[:12]  # detached HEAD


def git_root(path: str) -> str:
    """Nearest ancestor holding a .git (project context). Pure filesystem, no git call."""
    p = Path(path)
    for d in (p, *p.parents):
        try:
            if (d / ".git").exists():
                return str(d)
        except OSError:
            break
        if d == Path.home():
            break
    return ""