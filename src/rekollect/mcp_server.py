"""`rk mcp`: read-only command memory for coding assistants (Model Context Protocol, stdio).

Guarantees, in order of strength:
  * read-only: the database is opened with SQLite's mode=ro, and no tool can run,
    write or change anything. There is deliberately no "execute" tool.
  * opt-in: tools refuse to answer until the user runs `rk agents enable`.
  * scoped: by default only commands that were run inside the assistant's project
    (the folder the server was started in, or --project) are visible.
    scope="all" works only if the user allowed it (`rk agents enable --scope all`).
  * redacted: secrets were masked when recorded; output is redacted again, and the
    user's agents.exclude patterns hide whatever else they choose.
  * provenance: every result says it came from the user's shell history (not a
    suggestion), how it was recorded, how often it ran and whether it succeeded.
  * audited: every request is appended to agents.log (`rk agents log`).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from rekollect import __version__
from rekollect import config as C
from rekollect.redact import redact

PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
SERVER_INSTRUCTIONS = (
    "rekollect is the user's local shell-history memory. Use it to find commands the user has actually "
    "run before (e.g. how they run this project's tests, deploy, or start services) instead of guessing. "
    "Every result is a real command from their history, not a suggestion; prefer ones that succeeded. "
    "These tools are read-only: show commands to the user, never assume they should run without review."
)
PROVENANCE = ("source=shell-history: these commands were run by the user in their own terminal and recorded "
              "locally by rekollect. They are not AI suggestions. Secrets were redacted.")

SCOPE_PROP = {"type": "string", "enum": ["project", "all"], "default": "project",
              "description": "project: only commands run inside this project (default). "
                             "all: the user's whole history, if they allowed it."}
TOOLS = [
    {
        "name": "search_commands",
        "title": "Search command history",
        "description": "Find shell commands the user has run before, by describing what they do "
                       "(\"run the integration tests\", \"rebuild the docker image without cache\") or by keywords. "
                       "Returns real commands from history with how often they ran, whether they succeeded, "
                       "where and when.",
        "inputSchema": {"type": "object", "properties": {
            "query": {"type": "string", "description": "What the command does, in plain words or keywords."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
            "scope": SCOPE_PROP,
            "successful_only": {"type": "boolean", "default": False,
                                "description": "Only commands whose most recent runs succeeded."},
        }, "required": ["query"]},
        "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True,
                        "openWorldHint": False},
    },
    {
        "name": "recent_commands",
        "title": "Recent commands",
        "description": "The user's most recent distinct commands in this project, newest first.",
        "inputSchema": {"type": "object", "properties": {
            "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 15},
            "scope": SCOPE_PROP,
        }},
        "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True,
                        "openWorldHint": False},
    },
    {
        "name": "get_workflow",
        "title": "Find a workflow",
        "description": "The sequence of commands the user ran for a task (\"publish the package\", "
                       "\"deploy the api\"), rebuilt from one terminal session. Steps are real history.",
        "inputSchema": {"type": "object", "properties": {
            "task": {"type": "string"},
            "scope": SCOPE_PROP,
        }, "required": ["task"]},
        "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True,
                        "openWorldHint": False},
    },
    {
        "name": "list_recipes",
        "title": "Saved recipes",
        "description": "Workflows the user saved on purpose (named, step by step).",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True,
                        "openWorldHint": False},
    },
    {
        "name": "get_recipe",
        "title": "Get a saved recipe",
        "description": "Steps of one saved recipe, with its {{placeholders}} and their recorded values.",
        "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
        "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True,
                        "openWorldHint": False},
    },
]


class ToolError(Exception):
    pass


def log_path() -> Path:
    return C.data_dir() / "agents.log"


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds") if ts else ""


def _within(path: str, root: str) -> bool:
    if not path or not root:
        return False
    try:
        p, r = os.path.normcase(os.path.abspath(path)), os.path.normcase(os.path.abspath(root))
    except ValueError:
        return False
    return p == r or p.startswith(r.rstrip("\\/") + os.sep)


class Server:
    def __init__(self, cfg: C.Config, project: Optional[str] = None):
        from rekollect.store import git_root
        self.cfg = cfg
        start = os.path.abspath(project or os.getcwd())
        self.project = git_root(start) or start
        self.allowed_scope = str(cfg.get_path("agents.scope", "project"))
        self.exclude = [re.compile(p) for p in cfg.get_path("agents.exclude", [])]
        self.client = "unknown"
        self._engine: Any = None

    # ── plumbing ─────────────────────────────────────────────────────────────
    def engine(self):
        """Read-only store + ranker, built on first use (keeps `initialize` instant)."""
        if self._engine is None:
            from rekollect import models as M
            from rekollect.store import Store
            if not C.db_path().exists():
                raise ToolError("rekollect has no history yet (the user hasn't run `rk init`).")
            store = Store(readonly=True)
            emb = M.load_embedder(self.cfg) if M.is_downloaded(M.spec_for(self.cfg)) else None
            self._engine = (store, emb)
        return self._engine

    def ranker_for(self, scope: str, successful_only: bool = False):
        from rekollect.engine import make_ranker
        store, emb = self.engine()
        corpus = store.corpus(success_only=successful_only,
                              cwd_prefix=self.project if scope == "project" else "")
        return store, corpus, make_ranker(store, corpus, emb, self.cfg, cwd=self.project, learning=True)

    def check_scope(self, scope: str) -> str:
        scope = scope or "project"
        if scope not in ("project", "all"):
            raise ToolError("scope must be 'project' or 'all'")
        if scope == "all" and self.allowed_scope != "all":
            raise ToolError("The user allows project-scoped access only. They can widen it with "
                            "`rk agents enable --scope all`.")
        if scope == "project" and os.path.normcase(self.project) in {os.path.normcase(str(Path.home())),
                                                                     os.path.normcase(os.path.abspath(os.sep))}:
            raise ToolError("Not inside a project folder (started in the home or root directory), so "
                            "project scope would mean everything. Start the assistant in a project.")
        return scope

    def visible(self, text: str) -> bool:
        return not any(p.search(text) for p in self.exclude)

    def where(self, cwd: str) -> str:
        if _within(cwd, self.project):
            rel = os.path.relpath(cwd, self.project)
            return "." if rel == "." else rel
        home = str(Path.home())
        return "~" + cwd[len(home):] if cwd.startswith(home) else cwd

    def log(self, tool: str, args: dict, returned: list[str], error: str = "") -> None:
        if not self.cfg.get_path("agents.log", True):
            return
        try:
            C.ensure_dirs()
            with log_path().open("a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": int(time.time()), "client": self.client, "tool": tool,
                                    "args": args, "project": self.project, "returned": returned,
                                    "error": error}, ensure_ascii=False) + "\n")
        except OSError:
            pass

    # ── tools ────────────────────────────────────────────────────────────────
    def describe(self, store, corpus, ranker, i: int, hit=None, query: str = "", scope: str = "project") -> dict:
        """Details of one command. In project scope every detail comes from runs inside the project,
        so nothing leaks about where else (or when) it was used."""
        cid = int(corpus.ids[i])
        under = self.project if scope == "project" else ""
        d = store.run_details(cid, under) or (0, "", None, None, "", "", "")
        runs, fails = store.run_counts(cid, under)
        sources = {r[0] for r in store.db.execute("SELECT DISTINCT source FROM runs WHERE cmd_id = ?", (cid,))}
        item = {
            "command": redact(corpus.texts[i]),
            "source": "shell-history",
            "recorded_by": "rekollect shell hook" if "hook" in sources else "imported shell history",
            "times_run": runs,
            "times_failed": fails,
            "last_exit_code": d[2],
            "last_run": _iso(d[0]),
            "directory": self.where(d[1]),
        }
        if d[4]:
            item["git_branch"] = d[4]
        if d[5]:
            item["shell"] = d[5]
        if corpus.marks and i in corpus.marks:
            name, tags, note = corpus.marks[i]
            item["bookmark"] = {"name": name, "tags": tags.split(), "note": note}
        if hit is not None:
            item["why"] = ranker.explain(hit, query)
        return item

    def search_commands(self, args: dict) -> dict:
        q = str(args.get("query", "")).strip()
        if not q:
            raise ToolError("query is required")
        scope = self.check_scope(str(args.get("scope", "project")))
        limit = max(1, min(20, int(args.get("limit", 5))))
        store, corpus, ranker = self.ranker_for(scope, bool(args.get("successful_only", False)))
        hits, _ = ranker.search(q, limit * 3)
        items = [self.describe(store, corpus, ranker, h.idx, h, q, scope) for h in hits
                 if self.visible(corpus.texts[h.idx])][:limit]
        return {"scope": scope, "project": self.project, "provenance": PROVENANCE, "results": items}

    def recent_commands(self, args: dict) -> dict:
        scope = self.check_scope(str(args.get("scope", "project")))
        limit = max(1, min(50, int(args.get("limit", 15))))
        store, _ = self.engine()
        seen: set[str] = set()
        items = []
        for ts, _cid, text, ec, _dur, cwd, _sid in reversed(store.timeline()):
            if text in seen or not self.visible(text):
                continue
            if scope == "project" and not _within(cwd, self.project):
                continue
            seen.add(text)
            items.append({"command": redact(text), "source": "shell-history", "ran_at": _iso(ts),
                          "exit_code": ec, "directory": self.where(cwd)})
            if len(items) >= limit:
                break
        return {"scope": scope, "project": self.project, "provenance": PROVENANCE, "results": items}

    def get_workflow(self, args: dict) -> dict:
        from rekollect import flows as F
        task = str(args.get("task", "")).strip()
        if not task:
            raise ToolError("task is required")
        scope = self.check_scope(str(args.get("scope", "project")))
        store, corpus, ranker = self.ranker_for("all")  # flows need the whole timeline; filtered below
        found = F.find_flows(store, corpus, ranker, task, k=5)
        out = []
        for f in found:
            if scope == "project" and not all(_within(s.cwd, self.project) for s in f.steps if s.cwd):
                continue
            if scope == "project" and not any(s.cwd for s in f.steps):
                continue
            steps = [{"label": s.label, "command": redact(s.text), "exit_code": s.exit,
                      "directory": self.where(s.cwd)} for s in f.steps if self.visible(s.text)]
            out.append({"ran_at": _iso(f.start), "times_done": f.times, "failed_attempts_hidden": f.hidden_failures,
                        "steps": steps})
            if len(out) >= 2:
                break
        return {"scope": scope, "project": self.project, "provenance": PROVENANCE, "workflows": out}

    def list_recipes(self, args: dict) -> dict:
        from rekollect import flows as F
        return {"recipes": [{"name": r.name, "description": r.description, "steps": len(r.steps),
                             "values": list(r.vars)} for r in F.list_recipes()],
                "provenance": "Recipes are workflows the user saved deliberately."}

    def get_recipe(self, args: dict) -> dict:
        from rekollect import flows as F
        name = str(args.get("name", ""))
        if not F.valid_name(name) or not F.recipe_path(name).exists():
            raise ToolError(f"no recipe named {name!r}")
        r = F.load_recipe(name)
        return {"name": r.name, "description": r.description, "values": r.vars,
                "steps": [{"label": s.label, "command": redact(s.run)} for s in r.steps
                          if self.visible(s.run)],
                "provenance": "A workflow the user saved deliberately; {{name}} placeholders take the values."}

    # ── protocol ─────────────────────────────────────────────────────────────
    def call_tool(self, name: str, args: dict) -> dict:
        fn = {"search_commands": self.search_commands, "recent_commands": self.recent_commands,
              "get_workflow": self.get_workflow, "list_recipes": self.list_recipes,
              "get_recipe": self.get_recipe}.get(name)
        logged_args = {k: v for k, v in args.items() if k in ("query", "task", "scope", "limit", "name",
                                                               "successful_only")}
        if fn is None:
            return _tool_error(f"unknown tool {name!r}")
        if not self.cfg.get_path("agents.enabled", False):
            msg = ("rekollect agent access is turned off. The user can allow read-only access with "
                   "`rk agents enable` (and review requests with `rk agents log`).")
            self.log(name, logged_args, [], "disabled")
            return _tool_error(msg)
        try:
            data = fn(args)
        except ToolError as e:
            self.log(name, logged_args, [], str(e))
            return _tool_error(str(e))
        returned = [r["command"] for r in data.get("results", [])] + [
            s["command"] for w in data.get("workflows", []) for s in w["steps"]]
        self.log(name, logged_args, returned)
        return {"content": [{"type": "text", "text": _as_text(name, data)}], "structuredContent": data,
                "isError": False}

    def handle(self, msg: dict) -> Optional[dict]:
        mid = msg.get("id")
        method = msg.get("method", "")
        params = msg.get("params") or {}
        if mid is None:  # notification (initialized, cancelled, …)
            return None
        try:
            if method == "initialize":
                info = params.get("clientInfo") or {}
                self.client = f"{info.get('name', 'unknown')} {info.get('version', '')}".strip()
                asked = params.get("protocolVersion")
                version = asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[1]
                return _ok(mid, {"protocolVersion": version,
                                 "capabilities": {"tools": {"listChanged": False}},
                                 "serverInfo": {"name": "rekollect", "title": "rekollect command memory",
                                                "version": __version__},
                                 "instructions": SERVER_INSTRUCTIONS})
            if method == "ping":
                return _ok(mid, {})
            if method == "tools/list":
                return _ok(mid, {"tools": TOOLS})
            if method == "tools/call":
                return _ok(mid, self.call_tool(params.get("name", ""), params.get("arguments") or {}))
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"unknown method {method}"}}
        except Exception as e:  # never crash the client's session
            traceback.print_exc(file=sys.stderr)
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32603, "message": str(e)}}


def _ok(mid, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def _tool_error(msg: str) -> dict:
    return {"content": [{"type": "text", "text": msg}], "isError": True}


def _as_text(tool: str, data: dict) -> str:
    """Compact, readable rendering for models that only read text content."""
    lines = [PROVENANCE] if "provenance" in data else []
    if tool in ("search_commands", "recent_commands"):
        if not data["results"]:
            lines.append("No matching commands in this scope.")
        for k, r in enumerate(data["results"], 1):
            status = ("never succeeded" if r.get("times_run") and r.get("times_failed") == r.get("times_run")
                      else f"exit {r.get('last_exit_code', r.get('exit_code'))}")
            meta = [f"ran {r['times_run']}×" if "times_run" in r else "", status,
                    f"last {r.get('last_run') or r.get('ran_at')}", f"in {r['directory']}",
                    f"branch {r['git_branch']}" if r.get("git_branch") else ""]
            lines.append(f"{k}. {r['command']}\n   " + " · ".join(m for m in meta if m))
            if r.get("why"):
                lines.append("   why: " + "; ".join(r["why"]))
    elif tool == "get_workflow":
        if not data["workflows"]:
            lines.append("No matching workflow in this scope.")
        for k, w in enumerate(data["workflows"], 1):
            lines.append(f"Workflow {k} (ran {w['ran_at']}, done {w['times_done']}×):")
            lines += [f"  {i}. {s['label']}: {s['command']}" for i, s in enumerate(w["steps"], 1)]
    else:
        lines.append(json.dumps(data, indent=2, ensure_ascii=False))
    return "\n".join(lines)


def serve(cfg: C.Config, project: Optional[str] = None) -> int:
    server = Server(cfg, project)
    if not cfg.get_path("agents.enabled", False):
        print("rekollect: agent access is off; tools will refuse until `rk agents enable`", file=sys.stderr)
    stdin = sys.stdin.buffer
    out = sys.stdout.buffer
    for raw in stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            msg = json.loads(raw.decode("utf-8"))
        except ValueError:
            reply: Optional[dict] = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        else:
            if isinstance(msg, list):  # batches (older protocol versions)
                replies = [r for r in (server.handle(m) for m in msg) if r is not None]
                reply = replies or None  # type: ignore[assignment]
            else:
                reply = server.handle(msg)
        if reply is not None:
            out.write((json.dumps(reply, ensure_ascii=False) + "\n").encode("utf-8"))
            out.flush()
    return 0