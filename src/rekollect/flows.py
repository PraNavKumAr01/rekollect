"""Workflows: find the *sequence* of commands you ran for a task, and save it as a recipe.

How a workflow is found
-----------------------
1. Score every command in the index for the query (semantic + lexical, no recency).
2. Walk the timeline of runs grouped by terminal session (or by directory when an
   old run has no session) and cut it into episodes wherever you were idle for
   more than `GAP` seconds.
3. An episode that contains strongly matching commands becomes a candidate. The
   steps kept are the ones around those matches (a few before for setup, a few
   after for verification), minus failed attempts and noise like `git status`.
4. Identical sequences are merged ("done 3×"); the best few are returned.

Nothing is generated: every step is a command from your history. Step labels come
from simple rules about the command itself ("pytest" → "Run tests").

Recipes
-------
A recipe is a saved workflow in config_dir()/recipes/<name>.toml. Values that
tend to change between runs (hosts, namespaces, versions, branches, tags…)
become {{placeholders}}, shared across steps, with the recorded value as the
default. `rk recipe run` fills them once and walks you through each step.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rekollect import config as C
from rekollect.templates import VOLATILE, fill, placeholders, templatize

GAP = 20 * 60          # idle time that ends an episode
BEFORE, AFTER = 8, 3   # context kept around the matching steps
MAX_STEPS = 15
MIN_REL = 0.35

NOISE = re.compile(
    r"^\s*(git\s+(status|diff|log|show|branch)\b|cat\b|less\b|more\b|bat\b|echo\b|which\b|where(\.exe)?\b|"
    r"type\b|man\b|tldr\b|help\b|history\b|clear\b|cls\b|code\s+\.|Get-Help\b|Get-Command\b|"
    r"(rk|rekollect)\b|exit\b|pwd\b|whoami\b|date\b)", re.IGNORECASE)

LABELS: list[tuple[str, str]] = [
    (r"\b(pytest|tox|nox)\b|\b(npm|pnpm|yarn)\s+(run\s+)?test\b|\b(go|cargo|dotnet)\s+test\b|\bmake\s+test\b|"
     r"\b(jest|vitest|rspec|phpunit)\b|Invoke-Pester", "Run tests"),
    (r"\b(pip3?|uv\s+pip)\s+install\b|\buv\s+(sync|add)\b|\bpoetry\s+(install|add)\b|\b(npm|pnpm)\s+(i|install|ci)\b|"
     r"\byarn(\s+install)?\s*$|\bbundle\s+install\b|\bgo\s+mod\s+download\b|\bconda\s+install\b", "Install dependencies"),
    (r"\b(twine\s+upload|poetry\s+publish|uv\s+publish|hatch\s+publish|flit\s+publish|npm\s+publish|"
     r"cargo\s+publish|gem\s+push|dotnet\s+nuget\s+push)\b", "Publish to the registry"),
    (r"\bpython3?\s+-m\s+build\b|\b(poetry|uv|hatch|flit)\s+build\b|\bpython3?\s+setup\.py\s+(sdist|bdist)",
     "Build the package"),
    (r"\b(docker|podman)\s+(image\s+)?build\b|\bdocker\s+buildx\s+build\b", "Build the image"),
    (r"\b(docker|podman)\s+push\b", "Push the image"),
    (r"\b(docker|podman)\s+(login)\b", "Log in to the registry"),
    (r"\b(npm|pnpm|yarn)\s+(run\s+)?build\b|\b(cargo|go|dotnet)\s+build\b|\bmake(\s+(all|build))?\s*$|"
     r"\bcmake\s+--build\b|\bgradle(w)?\s+build\b|\bmvn\s+(package|install)\b", "Build"),
    (r"\b(ruff|flake8|pylint|black|isort|eslint|prettier|mypy|pyright|golangci-lint|rustfmt|cargo\s+(fmt|clippy))\b",
     "Lint / format"),
    (r"\b(bump2version|bumpversion|npm\s+version|cargo\s+release|poetry\s+version)\b", "Bump the version"),
    (r"\bgit\s+tag\b", "Tag the release"),
    (r"\bgit\s+push\b.*--tags|\bgit\s+push\s+\S+\s+v?\d", "Push the tag"),
    (r"\bgit\s+push\b", "Push to the remote"),
    (r"\bgit\s+commit\b", "Commit changes"),
    (r"\bgit\s+add\b", "Stage changes"),
    (r"\bgit\s+(pull|fetch)\b", "Get the latest changes"),
    (r"\bgit\s+(checkout|switch)\b", "Switch branch"),
    (r"\bgit\s+(merge|rebase)\b", "Integrate branches"),
    (r"\bgit\s+clone\b", "Clone the repository"),
    (r"\bgh\s+pr\s+create\b", "Open a pull request"),
    (r"\bgh\s+release\s+create\b", "Create a GitHub release"),
    (r"\bkubectl\s+apply\b|\bhelm\s+(upgrade|install)\b|\bkustomize\b", "Deploy to the cluster"),
    (r"\bkubectl\b.*\bset\s+image\b", "Roll out the new image"),
    (r"\bkubectl\s+rollout\s+(status|history)\b", "Wait for the rollout"),
    (r"\bkubectl\s+rollout\s+restart\b", "Restart the deployment"),
    (r"\bkubectl\s+(config\s+use-context|ctx)\b|\bkubectx\b", "Switch cluster context"),
    (r"\bkubectl\s+(logs|get|describe|top)\b", "Check the cluster"),
    (r"\bterraform\s+init\b", "Initialise Terraform"),
    (r"\bterraform\s+plan\b", "Plan infrastructure changes"),
    (r"\bterraform\s+apply\b", "Apply infrastructure changes"),
    (r"\b(docker\s+compose|docker-compose)\s+up\b|\bdocker\s+run\b", "Start containers"),
    (r"\b(docker\s+compose|docker-compose)\s+down\b|\bdocker\s+stop\b", "Stop containers"),
    (r"\b(python3?\s+-m\s+venv|virtualenv|uv\s+venv|conda\s+create)\b", "Create a virtual environment"),
    (r"activate(\.ps1|\.fish)?\b|\bconda\s+activate\b", "Activate the environment"),
    (r"\b(alembic\s+upgrade|manage\.py\s+migrate|prisma\s+migrate|rails\s+db:migrate)\b", "Run migrations"),
    (r"\b(npm|pnpm|yarn)\s+(run\s+)?(dev|start)\b|\bmanage\.py\s+runserver\b|\buvicorn\b|\bflask\s+run\b",
     "Start the dev server"),
    (r"^\s*(cd|Set-Location|sl|pushd)\b", "Change directory"),
    (r"\b(ssh)\b", "Connect to the server"),
    (r"\b(scp|rsync)\b", "Copy files to/from the server"),
    (r"\b(rm\s+-rf?|Remove-Item)\b.*\b(dist|build|node_modules|target|__pycache__)\b", "Clean build output"),
    (r"\b(curl|wget|Invoke-WebRequest|iwr|http)\b", "Make an HTTP request"),
]
_LABELS = [(re.compile(p, re.IGNORECASE), label) for p, label in LABELS]


def label_for(cmd: str) -> str:
    for pat, label in _LABELS:
        if pat.search(cmd):
            return label
    words = cmd.split()
    head = words[0] if words else cmd
    sub = next((w for prev, w in zip(words, words[1:])
                if not w.startswith("-") and not re.fullmatch(r"-[A-Za-z]", prev)), "")
    return f"{head} {sub}".strip() if len(sub) < 24 else head


# ── data ─────────────────────────────────────────────────────────────────────

@dataclass
class Step:
    text: str
    ts: int
    exit: Optional[int]
    cwd: str
    relevance: float = 0.0
    label: str = ""


@dataclass
class Flow:
    steps: list[Step]
    score: float
    session: str
    cwd: str
    start: int
    end: int
    times: int = 1
    hidden_failures: int = 0


@dataclass
class RecipeStep:
    label: str
    run: str
    cwd: str = ""


@dataclass
class Recipe:
    name: str
    description: str = ""
    steps: list[RecipeStep] = field(default_factory=list)
    vars: dict[str, str] = field(default_factory=dict)
    shell: str = ""
    created: str = ""
    recorded_in: str = ""


# ── finding workflows ────────────────────────────────────────────────────────

def _episodes(rows: list[tuple]) -> list[list[tuple]]:
    groups: dict[str, list[tuple]] = {}
    for r in rows:  # (ts, cmd_id, text, exit, dur, cwd, session)
        key = r[6] or (f"cwd:{r[5]}" if r[5] else "global")
        groups.setdefault(key, []).append(r)
    out = []
    for g in groups.values():
        cur = [g[0]]
        for prev, r in zip(g, g[1:]):
            if r[0] - prev[0] > GAP:
                out.append(cur)
                cur = []
            cur.append(r)
        out.append(cur)
    return out


def _clean(window: list[tuple], rel: dict[int, float]) -> tuple[list[Step], int]:
    steps: list[Step] = []
    failures = 0
    for r in window:
        ts, cmd_id, text, ec, _dur, cwd, _sid = r
        if NOISE.match(text):
            continue
        if ec not in (None, 0):
            failures += 1
            continue
        if steps and steps[-1].text == text:  # re-ran the same thing
            steps[-1] = Step(text, ts, ec, cwd, rel.get(cmd_id, 0.0), label_for(text))
            continue
        steps.append(Step(text, ts, ec, cwd, rel.get(cmd_id, 0.0), label_for(text)))
    return steps, failures


_FILLER = re.compile(
    r"\b(what|which|how|when|where|did|do|does|i|me|my|we|our|you|use|used|using|steps?|commands?|"
    r"sequence|workflow|process|the|a|an|this|that|these|those|to|for|of|in|on|with|again|last|time|"
    r"usually|normally|was|were|is|are|it|go|went|about)\b", re.IGNORECASE)


def task_query(query: str) -> str:
    """'What steps did I use to publish this package?' → 'publish package'."""
    cleaned = " ".join(_FILLER.sub(" ", re.sub(r"[?!.,]", " ", query)).split())
    return cleaned or query


def find_flows(store, corpus, ranker, query: str, k: int = 3, now: Optional[float] = None) -> list[Flow]:
    rows = store.timeline()
    if not rows or not len(corpus.texts):
        return []
    scores = ranker.relevance(task_query(query))
    rel = {int(cid): float(s) for cid, s in zip(corpus.ids, scores)}
    best_overall = max(rel.values()) if rel else 0.0
    if best_overall <= 0:
        return []
    floor = max(MIN_REL, 0.5 * best_overall)
    now = now or time.time()

    flows: dict[tuple, Flow] = {}
    for ep in _episodes(rows):
        rels = [rel.get(r[1], 0.0) for r in ep]
        best = max(rels)
        if best < floor:
            continue
        anchors = [i for i, x in enumerate(rels) if x >= max(floor, 0.6 * best)]
        lo, hi = max(0, anchors[0] - BEFORE), min(len(ep), anchors[-1] + AFTER + 1)
        steps, failures = _clean(ep[lo:hi], rel)
        if not steps:
            continue
        if len(steps) > MAX_STEPS:  # keep the part around the strongest step
            peak = max(range(len(steps)), key=lambda i: steps[i].relevance)
            a = max(0, min(peak - MAX_STEPS // 2, len(steps) - MAX_STEPS))
            steps = steps[a:a + MAX_STEPS]
        top = sorted({s.text: s.relevance for s in steps}.values(), reverse=True)[:3]
        top += [0.0] * (3 - len(top))
        age_days = (now - steps[-1].ts) / 86400
        # the best-matching step dominates; supporting matches and recency break ties
        score = 2.0 * top[0] + 0.5 * (top[1] + top[2]) + 0.1 * 0.5 ** (age_days / 30) \
            - 0.02 * max(0, len(steps) - 8)
        key = tuple(s.text for s in steps)
        f = Flow(steps, score, ep[0][6], steps[-1].cwd or ep[0][5], steps[0].ts, steps[-1].ts,
                 hidden_failures=failures)
        if key in flows:
            old = flows[key]
            f.times = old.times + 1
            f.hidden_failures += old.hidden_failures
            if old.end > f.end:
                f.steps, f.start, f.end, f.cwd = old.steps, old.start, old.end, old.cwd
            f.score = max(f.score, old.score) + 0.05
        flows[key] = f

    ranked = sorted(flows.values(), key=lambda f: -f.score)
    out: list[Flow] = []
    for f in ranked:  # skip near-duplicates of a better flow
        texts = {s.text for s in f.steps}
        if any(len(texts & {s.text for s in g.steps}) >= 0.8 * len(texts) for g in out):
            continue
        out.append(f)
        if len(out) >= k:
            break
    return out


# ── recipes: storage ─────────────────────────────────────────────────────────

NAME_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")


def recipes_dir() -> Path:
    return C.config_dir() / "recipes"


def recipe_path(name: str) -> Path:
    return recipes_dir() / f"{name}.toml"


def valid_name(name: str) -> bool:
    return bool(NAME_RE.fullmatch(name))


def _tstr(v: str) -> str:
    if "'" not in v and "\n" not in v and not re.search(r"[\x00-\x1f\x7f]", v):
        return f"'{v}'"  # literal string: backslashes (Windows paths, regexes) stay readable
    return json.dumps(v, ensure_ascii=False)


def dumps_recipe(r: Recipe) -> str:
    lines = [
        "# rekollect recipe. Edit freely: steps run in order, and {{name}} placeholders are",
        "# filled from [vars] (you're asked to confirm or change them) when you run it.",
        f"name = {_tstr(r.name)}",
        f"description = {_tstr(r.description)}",
    ]
    if r.shell:
        lines.append(f"shell = {_tstr(r.shell)}")
    if r.created:
        lines.append(f"created = {_tstr(r.created)}")
    if r.recorded_in:
        lines.append(f"recorded_in = {_tstr(r.recorded_in)}")
    lines.append("")
    if r.vars:
        lines.append("[vars]")
        for k, v in r.vars.items():
            lines.append(f"{k} = {_tstr(v)}")
        lines.append("")
    for s in r.steps:
        lines.append("[[steps]]")
        lines.append(f"label = {_tstr(s.label)}")
        lines.append(f"run = {_tstr(s.run)}")
        if s.cwd and s.cwd != r.recorded_in:  # only note directories that differ
            lines.append(f"cwd = {_tstr(s.cwd)}")
        lines.append("")
    return "\n".join(lines)


def save_recipe(r: Recipe) -> Path:
    recipes_dir().mkdir(parents=True, exist_ok=True)
    p = recipe_path(r.name)
    p.write_text(dumps_recipe(r), encoding="utf-8")
    return p


def load_recipe(name: str) -> Recipe:
    from rekollect.config import tomllib
    data = tomllib.loads(recipe_path(name).read_text(encoding="utf-8"))
    recorded = str(data.get("recorded_in", ""))
    steps = [RecipeStep(str(s.get("label", "")), str(s["run"]), str(s.get("cwd", recorded)))
             for s in data.get("steps", []) if s.get("run")]
    return Recipe(name=str(data.get("name", name)), description=str(data.get("description", "")),
                  steps=steps, vars={k: str(v) for k, v in data.get("vars", {}).items()},
                  shell=str(data.get("shell", "")), created=str(data.get("created", "")),
                  recorded_in=str(data.get("recorded_in", "")))


def list_recipes() -> list[Recipe]:
    d = recipes_dir()
    out = []
    for p in sorted(d.glob("*.toml")) if d.exists() else []:
        try:
            out.append(load_recipe(p.stem))
        except Exception:
            continue
    return out


def match_recipes(query: str, recipes: list[Recipe]) -> list[Recipe]:
    words = set(re.findall(r"[a-z0-9]{3,}", query.lower()))
    scored = []
    for r in recipes:
        hay = " ".join([r.name.replace("-", " "), r.description] + [s.label for s in r.steps]).lower()
        hits = sum(1 for w in words if w in hay)
        if hits:
            scored.append((hits, r))
    return [r for _, r in sorted(scored, key=lambda x: -x[0])]


def recipe_from_steps(name: str, steps: list[tuple[str, str, str]], description: str = "",
                      parametrize: bool = True) -> Recipe:
    """steps: (label, command, cwd). Volatile values become shared {{placeholders}}."""
    from rekollect.history import detect_shell
    registry: dict = {}
    out = []
    for label, cmd, cwd in steps:
        run = templatize(cmd, kinds=VOLATILE, registry=registry).stored() if parametrize else cmd
        out.append(RecipeStep(label or label_for(cmd), run, cwd))
    used = {p for s in out for p in placeholders(s.run)}
    variables = {slot.name: slot.default for slot in registry.values() if slot.name in used}
    cwds = [c for _, _, c in steps if c]
    return Recipe(name=name, description=description, steps=out, vars=variables, shell=detect_shell(),
                  created=time.strftime("%Y-%m-%d"), recorded_in=cwds[-1] if cwds else "")


def recipe_from_flow(name: str, flow: Flow, description: str = "", parametrize: bool = True) -> Recipe:
    return recipe_from_steps(name, [(s.label, s.text, s.cwd) for s in flow.steps], description, parametrize)


# ── recipes: running ─────────────────────────────────────────────────────────

_CD = re.compile(r"^\s*(?:cd|Set-Location|sl|pushd|chdir)(?:\s+-LiteralPath|\s+-Path)?\s+(.+?)\s*$", re.IGNORECASE)
_EXPORT = re.compile(r"^\s*(?:export\s+([A-Za-z_]\w*)=(.*)|set\s+-[gxU]+\s+([A-Za-z_]\w*)\s+(.*)|"
                     r"\$env:([A-Za-z_]\w*)\s*=\s*(.*))\s*$")
_ACTIVATE = re.compile(r"^\s*(?:source\s+|\.\s+|&\s*)?['\"]?(?P<env>.+?)[\\/](?:bin|Scripts)[\\/]"
                       r"(?:activate(?:\.fish|\.ps1|\.bat|\.csh)?|Activate\.ps1)['\"]?\s*$", re.IGNORECASE)


def _unquote(v: str) -> str:
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
        return v[1:-1]
    return v


@dataclass
class RunState:
    cwd: str
    env: dict


def builtin_step(cmd: str, state: RunState) -> Optional[str]:
    """Handle steps a child process can't do for us (cd, env vars, venv activation).

    Returns a message if handled, None if the command should be run normally.
    """
    m = _CD.match(cmd)
    if m and not re.search(r"&&|\|\||;|\|", cmd):
        target = os.path.expandvars(os.path.expanduser(_unquote(m.group(1))))
        new = Path(state.cwd, target).resolve() if not os.path.isabs(target) else Path(target)
        if not new.is_dir():
            raise FileNotFoundError(f"directory not found: {new}")
        state.cwd = str(new)
        return f"now in {new}"
    m = _EXPORT.match(cmd)
    if m:
        key = m.group(1) or m.group(3) or m.group(5)
        val = _unquote(m.group(2) or m.group(4) or m.group(6) or "")
        state.env[key] = os.path.expandvars(val)
        return f"set {key}"
    m = _ACTIVATE.match(cmd)
    if m:
        envdir = Path(state.cwd, os.path.expanduser(_unquote(m.group("env")))).resolve()
        bindir = envdir / ("Scripts" if os.name == "nt" else "bin")
        if not bindir.is_dir():
            raise FileNotFoundError(f"virtual environment not found: {envdir}")
        state.env["VIRTUAL_ENV"] = str(envdir)
        state.env["PATH"] = str(bindir) + os.pathsep + state.env.get("PATH", "")
        return f"using virtual environment {envdir}"
    return None


def shell_argv(shell: str, cmd: str) -> list[str]:
    shell = shell or "bash"
    if shell == "powershell":
        exe = shutil.which("pwsh") or shutil.which("powershell")
        if exe:
            return [exe, "-NoLogo", "-NoProfile", "-Command", cmd]
    if shell in ("zsh", "bash", "fish") and shutil.which(shell):
        return [shutil.which(shell), "-c", cmd]  # type: ignore[list-item]
    if os.name == "nt":
        exe = shutil.which("pwsh") or shutil.which("powershell")
        return [exe, "-NoLogo", "-NoProfile", "-Command", cmd] if exe else ["cmd.exe", "/c", cmd]
    return ["/bin/sh", "-c", cmd]


def run_step(cmd: str, shell: str, state: RunState) -> int:
    return subprocess.call(shell_argv(shell, cmd), cwd=state.cwd, env=state.env)


def missing_vars(recipe: Recipe) -> list[str]:
    return [p for s in recipe.steps for p in placeholders(s.run) if p not in recipe.vars]


def render_steps(recipe: Recipe, values: dict[str, str]) -> list[str]:
    return [fill(s.run, values) for s in recipe.steps]


def quote_for(shell: str, s: str) -> str:
    if shell == "powershell":
        return "'" + s.replace("'", "''") + "'"
    return shlex.quote(s)


def flow_time(f: Flow) -> str:
    mins = max(1, round((f.end - f.start) / 60)) if f.end > f.start else 0
    return f"{mins} min" if mins else "a moment"