"""Ingest → filter → redact → dedupe → embed. Shared by `init`, `index` and search."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from rekollect import config as C
from rekollect.history import Entry, drain_spool, history_files, read_history
from rekollect.models import Embedder, spec_for
from rekollect.redact import redact
from rekollect.store import Store

# A few words of context per tool. Shell commands are terse ("k get po -A");
# appending what the tool *is* gives the embedding model something to hold on
# to when you search in plain English ("list kubernetes pods").
TOOL_HINTS = {
    "git": "git version control", "gh": "github cli", "docker": "docker containers",
    "docker-compose": "docker compose services", "podman": "containers",
    "kubectl": "kubernetes cluster", "k": "kubernetes cluster", "helm": "kubernetes helm charts",
    "terraform": "terraform infrastructure", "tf": "terraform infrastructure",
    "aws": "aws cloud", "gcloud": "google cloud", "az": "azure cloud",
    "ssh": "remote login ssh", "scp": "copy files over ssh", "rsync": "sync copy files",
    "curl": "http request download", "wget": "download file", "tar": "archive compress extract",
    "unzip": "extract zip archive", "zip": "compress zip archive", "gzip": "compress",
    "find": "find files", "fd": "find files", "grep": "search text", "rg": "search text ripgrep",
    "sed": "edit text replace", "awk": "text processing", "jq": "json processing",
    "ps": "processes", "kill": "kill process", "pkill": "kill process", "lsof": "open files ports",
    "top": "system monitor", "htop": "system monitor", "df": "disk space", "du": "disk usage size",
    "chmod": "file permissions", "chown": "file ownership", "ln": "symlink",
    "pip": "python packages", "uv": "python packages env", "poetry": "python packages",
    "python": "python script", "python3": "python script", "pytest": "python tests",
    "npm": "node packages", "pnpm": "node packages", "yarn": "node packages", "npx": "node run",
    "node": "node javascript", "cargo": "rust build", "go": "go build", "make": "build",
    "brew": "homebrew packages", "apt": "system packages", "apt-get": "system packages",
    "systemctl": "system services", "journalctl": "system logs", "crontab": "scheduled jobs",
    "ffmpeg": "video audio convert", "convert": "image convert", "magick": "image convert",
    "openssl": "certificates encryption", "psql": "postgres database", "mysql": "mysql database",
    "redis-cli": "redis", "sqlite3": "sqlite database", "nvidia-smi": "gpu status",
    "conda": "conda environment", "venv": "python virtual environment", "tmux": "terminal sessions",
    "ping": "network", "nc": "network port", "netstat": "network ports", "ss": "network ports",
    "dig": "dns lookup", "nslookup": "dns lookup", "ifconfig": "network interfaces", "ip": "network",
    # Windows / PowerShell
    "winget": "windows packages install", "choco": "windows packages install", "scoop": "windows packages install",
    "get-childitem": "list files", "gci": "list files", "dir": "list files", "get-content": "read file",
    "gc": "read file", "cat": "read file", "set-content": "write file", "remove-item": "delete files",
    "rm": "delete files", "copy-item": "copy files", "move-item": "move rename files",
    "new-item": "create file folder", "get-process": "processes", "stop-process": "kill process",
    "get-service": "system services", "restart-service": "restart system service",
    "invoke-webrequest": "http request download", "iwr": "http request download",
    "invoke-restmethod": "http api request", "irm": "http api request", "select-string": "search text",
    "sls": "search text", "test-netconnection": "network port check", "get-netipaddress": "network interfaces",
    "ipconfig": "network interfaces", "netsh": "network config", "taskkill": "kill process",
    "tasklist": "processes", "robocopy": "copy sync files", "wsl": "linux subsystem",
    "set-executionpolicy": "powershell script permissions", "expand-archive": "extract zip archive",
    "compress-archive": "compress zip archive", "get-filehash": "file checksum hash",
    "where.exe": "find executable path", "where": "find executable path", "get-command": "find command path",
    "py": "python script", "code": "open vs code editor", "explorer": "open folder",
}

_SKIP_WORDS = {"sudo", "time", "env", "nohup", "exec", "command", "builtin", "noglob", "&", "."}


# Bump when what gets embedded changes: existing vectors are then rebuilt once.
ENRICH_VERSION = 3


def hint_for(cmd: str, mark: tuple[str, str, str] = ("", "", "")) -> str:
    """Words describing what a command does: tool hints plus its step label
    ("twine upload dist/*" → "publish to the registry")."""
    from rekollect.flows import label_for
    hints: list[str] = []
    for seg in re.split(r"\s*(?:\|\||&&|\||;)\s*", cmd)[:3]:
        toks = [t for t in seg.split() if "=" not in t or t.startswith("-")]
        toks = [t for t in toks if t not in _SKIP_WORDS]
        if toks:
            head = re.split(r"[\\/]", toks[0].lstrip("&.").strip("'\""))[-1].lower()
            h = TOOL_HINTS.get(head) or TOOL_HINTS.get(head.removesuffix(".exe"))
            if h and h not in hints:
                hints.append(h)
    label = label_for(cmd)
    if label and label.lower() not in cmd.lower() and label.lower() not in hints:
        hints.insert(0, label.lower())
    from rekollect.intents import describe
    for d in reversed(describe(cmd)):  # what it does, most specific first
        if d not in hints:
            hints.insert(0, d)
    name, tags, note = mark
    for extra in (name.replace("-", " ").replace("_", " "), tags, note):  # your words describe it best
        if extra:
            hints.insert(0, extra)
    return "; ".join(hints)


def enrich(cmd: str, hint: Optional[str] = None) -> str:
    hint = hint_for(cmd) if hint is None else hint
    return f"{cmd}  ({hint})" if hint else cmd


class Filter:
    def __init__(self, cfg: C.Config):
        self.min_len = int(cfg.get_path("index.min_length", 3))
        self.redact = bool(cfg.get_path("index.redact_secrets", True))
        self.ignore = [re.compile(p) for p in cfg.get_path("index.ignore", [])]

    def __call__(self, cmd: str) -> Optional[str]:
        if cmd.startswith(" "):  # leading space = "don't remember this" (HIST_IGNORE_SPACE)
            return None
        cmd = cmd.strip()
        if len(cmd) < self.min_len or len(cmd) > 4000:
            return None
        if any(p.search(cmd) for p in self.ignore):
            return None
        return redact(cmd) if self.redact else cmd


@dataclass
class IndexReport:
    new_runs: int = 0
    embedded: int = 0
    reembedded: bool = False


class _Context:
    """repo / branch / host for runs, read from the filesystem once per directory."""

    def __init__(self, cfg: C.Config):
        import socket
        self.branch_on = bool(cfg.get_path("context.record_branch", True))
        self.host = socket.gethostname() if cfg.get_path("context.record_host", True) else ""
        self._repo: dict[str, str] = {}
        self._branch: dict[str, str] = {}

    def __call__(self, cwd: str) -> tuple[str, str]:
        from rekollect.store import git_branch, git_root
        if not cwd:
            return "", ""
        if cwd not in self._repo:
            self._repo[cwd] = git_root(cwd)
        repo = self._repo[cwd]
        if not (repo and self.branch_on):
            return repo, ""
        if repo not in self._branch:  # read at index time: close to when the command ran
            self._branch[repo] = git_branch(repo)
        return repo, self._branch[repo]


def _shell_of(session: str) -> str:
    return session.split(":", 1)[0] if ":" in session else ""


def ingest(store: Store, entries: list[Entry], cfg: C.Config) -> int:
    f = Filter(cfg)
    ctx = _Context(cfg)
    n = 0
    for e in entries:
        text = f(e.cmd)
        if text is None:
            continue
        repo, branch = ctx(e.cwd) if e.source == "hook" else ("", "")
        n += store.add_run(text, e.ts, e.cwd, e.exit, e.duration_ms, e.source, e.session,
                           _shell_of(e.session), ctx.host if e.source == "hook" else "", repo, branch)
    store.commit()
    return n


def import_histories(store: Store, cfg: C.Config) -> tuple[int, list[str]]:
    total, used = 0, []
    for kind, path in history_files():
        entries = read_history(kind, path)
        total += ingest(store, entries, cfg)
        used.append(f"{path} ({kind}, {len(entries):,} entries)")
    return total, used


def purge_ignored(store: Store, cfg: C.Config) -> int:
    """Apply the ignore list to what's already stored, whenever the list changes."""
    import hashlib
    pats = cfg.get_path("index.ignore", [])
    digest = hashlib.sha1("\n".join(pats).encode()).hexdigest()
    if store.get_meta("ignore_hash") == digest:
        return 0
    f = Filter(cfg)
    gone = [cid for cid, text in store.db.execute("SELECT id, text FROM commands")
            if any(p.search(text) for p in f.ignore)]
    store.forget(gone)
    store.set_meta("ignore_hash", digest)
    store.commit()
    return len(gone)


def ensure_model_matches(store: Store, cfg: C.Config) -> bool:
    """If the configured model changed since vectors were written, wipe them."""
    sid = f"{spec_for(cfg).id}#e{ENRICH_VERSION}"
    if store.get_meta("model_id") not in ("", sid):
        store.clear_vectors()
        store.set_meta("model_id", sid)
        store.commit()
        return True
    store.set_meta("model_id", sid)
    store.commit()
    return False


def embed_pending(store: Store, emb: Embedder,
                  progress: Optional[Callable[[int, int], None]] = None, batch: int = 256) -> int:
    todo = store.pending_with_marks()
    done = 0
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        hints = [hint_for(t, (n, tg, no)) for _, t, n, tg, no in chunk]
        vecs = emb.embed([enrich(t, h) for (_, t, *_), h in zip(chunk, hints)])
        store.set_vectors(zip((row[0] for row in chunk), vecs), hints)
        store.commit()
        done += len(chunk)
        if progress:
            progress(done, len(todo))
    return done


def sync(store: Store, cfg: C.Config, emb: Optional[Embedder] = None,
         progress: Optional[Callable[[int, int], None]] = None,
         load_emb: Optional[Callable[[], Embedder]] = None) -> IndexReport:
    """Drain hook spool and embed anything new. Cheap when there's nothing to do."""
    rep = IndexReport()
    rep.new_runs = ingest(store, drain_spool(), cfg)
    purge_ignored(store, cfg)
    rep.reembedded = ensure_model_matches(store, cfg)
    if store.pending():
        if emb is None and load_emb is not None:
            emb = load_emb()
        if emb is not None:
            rep.embedded = embed_pending(store, emb, progress)
    return rep