"""Turn a concrete command into a reusable template.

    ssh -L 8888:localhost:8888 deploy@10.0.0.12
    → ssh -L {port}:localhost:{port} deploy@{host}

Detection is context-first (what the tool and the flag tell us a value is: the
value after `kubectl -n` is a namespace, the target of `ssh` is a host, the tag
of an image in `docker run` is a tag) and falls back to the shape of the value
(IP, URL, version, git sha, file path). Equal values share one placeholder, and
every placeholder remembers the original value as its default, so filling a
template with all defaults gives back the original command exactly.

Placeholders are kept as structured parts, never re-parsed from text, so commands
that already contain braces (`${f%.jpg}`, `{{.Names}}`, PowerShell script blocks)
are safe. Recipe files store them as `{{name}}`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional, Union

# kinds worth parameterising in saved recipes (files and paths stay literal there)
VOLATILE = {"host", "port", "namespace", "branch", "new_branch", "version", "tag", "image", "sha", "id", "message",
            "container", "pod", "deployment", "service", "release", "database", "user", "context",
            "region", "url", "secret", "email", "name", "statefulset", "job", "node", "local_port"}

_TOKEN = re.compile(r"""(?:"(?:\\.|[^"\\])*"?|'[^']*'?|[^\s"'])+""")
_OPS = {"&&", "||", "|", ";", "&", "|&"}

_IPV4 = re.compile(r"(?:\d{1,3}\.){3}\d{1,3}")
_URL = re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*://\S+")
_GIT_SSH = re.compile(r"[\w.-]+@[\w.-]+:[\w./~-]+")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[a-zA-Z]{2,}")
_VERSION = re.compile(r"v?\d+\.\d+(?:\.\d+)?(?:[-+.][0-9A-Za-z.]+)?")
_SHA = re.compile(r"[0-9a-f]{7,40}")
_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_PORT = re.compile(r"\d{2,5}")
_HOSTPORT = re.compile(r"(?P<host>[A-Za-z0-9.-]+):(?P<port>\d{2,5})")
_FILE_EXT = re.compile(r"[^/\\]+\.[A-Za-z0-9]{1,6}$")
_GLOBBY = re.compile(r"[*?\[\]$`{}()<>|;&]")

K8S_TYPES = {
    "deploy": "deployment", "deployment": "deployment", "deployments": "deployment",
    "svc": "service", "service": "service", "services": "service",
    "po": "pod", "pod": "pod", "pods": "pod", "sts": "statefulset", "statefulset": "statefulset",
    "job": "job", "jobs": "job", "cronjob": "job", "ns": "namespace", "namespace": "namespace",
    "no": "node", "node": "node", "nodes": "node", "cm": "name", "configmap": "name", "secret": "name",
    "secrets": "name", "ing": "name", "ingress": "name", "ds": "name", "daemonset": "name",
}
FILE_TOOLS = {"cd", "cp", "mv", "rm", "cat", "less", "more", "head", "tail", "ls", "tar", "unzip", "zip",
              "code", "vim", "vi", "nvim", "nano", "python", "python3", "py", "node", "bash", "sh", "source",
              "chmod", "chown", "ln", "mkdir", "touch", "du", "ffmpeg", "convert", "magick", "open",
              "start", "explorer", "set-location", "sl", "get-content", "gc", "remove-item", "copy-item",
              "move-item", "new-item", "scp", "rsync", "diff", "wc", "sort", "jq", "yq", "pytest"}
DOCKER_VALUE_FLAGS = {"-p", "--publish", "-v", "--volume", "-e", "--env", "--name", "-w", "--workdir",
                      "--network", "--net", "-u", "--user", "--entrypoint", "--env-file", "-l", "--label",
                      "--mount", "--platform", "--gpus", "-m", "--memory", "--cpus", "--restart",
                      "-h", "--hostname", "--add-host", "--device", "--log-driver", "--pull", "--shm-size",
                      "--tail", "-n", "--since", "--until", "--signal", "-s", "--format", "--filter",
                      "--detach-keys", "-t", "--time"}
SSH_VALUE_FLAGS = {"-i", "-L", "-R", "-D", "-p", "-l", "-o", "-F", "-J", "-W", "-b", "-c", "-E", "-e",
                   "-m", "-O", "-Q", "-S", "-w", "-P"}
# (-f is "follow" for logs but "filename" for apply, so it isn't listed)
K8S_VALUE_FLAGS = {"-n", "--namespace", "--context", "--filename", "-l", "--selector", "-o", "--output",
                   "-c", "--container", "--kubeconfig", "--cluster", "--user", "--tail", "--since",
                   "--timeout", "--set", "--values", "--version", "--kube-context"}
GIT_VALUE_FLAGS = {"-C", "-c", "-m", "--message", "-b", "-B", "--git-dir", "--work-tree", "-F", "--file"}
SCRIPT_RUNNERS = {"python", "python3", "py", "node", "deno", "bun", "ruby", "perl", "rscript", "bash", "sh",
                  "zsh", "pwsh", "powershell", "php", "lua", "julia", "tsx", "ts-node"}
DB_TOOLS = {"psql", "mysql", "mariadb", "redis-cli", "mongosh", "mongo", "pg_dump", "pg_restore", "mysqldump"}


@dataclass
class Slot:
    name: str
    default: str
    kind: str


# kinds that name the same thing; one value gets one placeholder across them
FAMILIES = ({"branch", "new_branch"}, {"port", "local_port"})

Part = Union[str, int]  # literal text, or index into slots


@dataclass
class Template:
    original: str
    parts: list[Part]
    slots: list[Slot] = field(default_factory=list)

    @property
    def has_slots(self) -> bool:
        return bool(self.slots)

    def render(self, values: Optional[dict[str, str]] = None) -> str:
        values = values or {}
        return "".join(p if isinstance(p, str) else values.get(self.slots[p].name, self.slots[p].default)
                       for p in self.parts)

    def text(self, open_: str = "{", close: str = "}") -> str:
        return "".join(p if isinstance(p, str) else f"{open_}{self.slots[p].name}{close}" for p in self.parts)

    def stored(self) -> str:
        """Form used in recipe files: {{name}}."""
        return self.text("{{", "}}")


# ── tokenising ───────────────────────────────────────────────────────────────

@dataclass
class _Tok:
    start: int
    end: int
    raw: str

    @property
    def quoted(self) -> bool:
        return len(self.raw) >= 2 and self.raw[0] == self.raw[-1] and self.raw[0] in "\"'"

    def value_span(self) -> tuple[int, int]:
        """Span of the token's value: inside quotes, after `--flag=`."""
        s, e = self.start, self.end
        if self.quoted:
            return s + 1, e - 1
        m = re.match(r"(--?[A-Za-z][\w-]*=)", self.raw)
        if m:
            s += len(m.group(1))
            if e - s >= 2 and self.raw[-1] == self.raw[len(m.group(1))] and self.raw[-1] in "\"'":
                return s + 1, e - 1
        return s, e


def _segments(cmd: str) -> list[list[_Tok]]:
    segs: list[list[_Tok]] = [[]]
    for m in _TOKEN.finditer(cmd):
        raw = m.group()
        if raw in _OPS:
            segs.append([])
        else:
            segs[-1].append(_Tok(m.start(), m.end(), raw))
    return [s for s in segs if s]


# ── detection ────────────────────────────────────────────────────────────────

Hit = tuple[int, int, str]  # absolute start, end, kind


def _is_flag(t: str) -> bool:
    return t.startswith("-") and len(t) > 1 and not _PORT.fullmatch(t[1:])


def _shape(cmd: str, s: int, e: int, head: str) -> list[Hit]:
    """Value-shape detectors for one token value cmd[s:e]."""
    v = cmd[s:e]
    if not v or v in (".", "..", "-", "--", "/", "\\", "`") or len(v) > 300:  # incl. line continuations
        return []
    if "‹redacted›" in v:
        i = v.index("‹redacted›")
        return [(s + i, s + i + len("‹redacted›"), "secret")]
    if _URL.fullmatch(v) or _GIT_SSH.fullmatch(v):
        return [(s, e, "url")]
    if _EMAIL.fullmatch(v):
        return [(s, e, "email")]
    if _UUID.fullmatch(v):
        return [(s, e, "id")]
    m = re.fullmatch(r"([\w.-]+)@([\w.-]+)", v)  # user@host
    if m:
        return [(s + m.start(2), s + m.end(2), "host")]
    m = re.fullmatch(rf"({_IPV4.pattern})(?::(\d{{2,5}}))?", v)
    if m:
        out = [(s + m.start(1), s + m.end(1), "host")]
        if m.group(2):
            out.append((s + m.start(2), s + m.end(2), "port"))
        return out
    m = _HOSTPORT.fullmatch(v)
    if m and not _VERSION.fullmatch(v) and (m.group("host") == "localhost" or "." in m.group("host")):
        return [(s + m.start("port"), s + m.end("port"), "port")]
    m = re.fullmatch(r":(\d{2,5})", v)
    if m:
        return [(s + 1, e, "port")]
    m = re.fullmatch(r"([A-Za-z0-9_.\-\[\]]+)(==|>=|@|=)(v?\d[\w.+-]*)", v)  # pkg==1.2.3, pkg@1.2.3
    if m and _VERSION.fullmatch(m.group(3)):
        return [(s + m.start(3), s + m.end(3), "version")]
    if _VERSION.fullmatch(v) and not _IPV4.fullmatch(v) and not _FILE_EXT.fullmatch(v):
        return [(s, e, "version")]
    if _SHA.fullmatch(v) and re.search(r"\d", v) and re.search(r"[a-f]", v):
        return [(s, e, "sha")]
    if _GLOBBY.search(v):
        return []
    pathish = (v.startswith(("/", "~", "./", "../", ".\\", "..\\", "\\\\")) or re.match(r"[A-Za-z]:[\\/]", v)
               or "\\" in v or (head in FILE_TOOLS and ("/" in v or _FILE_EXT.fullmatch(v)))
               or (_FILE_EXT.fullmatch(v) and not re.fullmatch(r"[\w-]+\.(com|org|net|io|dev|ai|app|co)", v)))
    if pathish:
        base = re.split(r"[\\/]", v.rstrip("/\\"))[-1]
        return [(s, e, "file" if _FILE_EXT.fullmatch(base) else "path")]
    return []


def _ports(cmd: str, s: int, e: int, kinds=("port", "port")) -> list[Hit]:
    """8080:80 / 8888:localhost:8888 → port hits."""
    v = cmd[s:e]
    out = []
    parts = list(re.finditer(r"\d{2,5}", v))
    if re.fullmatch(r"\d{2,5}(:[\w.-]+)?(:\d{2,5})?", v) and parts:
        out.append((s + parts[0].start(), s + parts[0].end(), kinds[0]))
        if len(parts) > 1 and parts[-1].end() == len(v):
            out.append((s + parts[-1].start(), s + parts[-1].end(), kinds[1]))
    return out


def _image(cmd: str, s: int, e: int) -> list[Hit]:
    v = cmd[s:e]
    m = re.fullmatch(r"(?P<name>[a-z0-9][\w./-]*?)(?::(?P<tag>[\w][\w.-]{0,127}))?(?:@sha256:[0-9a-f]+)?", v)
    if not m:
        return []
    if m.group("tag"):
        return [(s + m.start("tag"), s + m.end("tag"), "tag")]
    return [(s, e, "image")]


def _context(cmd: str, seg: list[_Tok]) -> tuple[list[Hit], set[int]]:
    """Context rules for one simple command. Returns hits and indices of tokens consumed."""
    hits: list[Hit] = []
    used: set[int] = set()
    words = [t.raw for t in seg]
    i0 = 0
    while i0 < len(words) and (words[i0] in ("sudo", "time", "env", "nohup", "exec", "command", "&", ".")
                                or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", words[i0])):
        i0 += 1
    if i0 >= len(seg):
        return hits, used
    head = re.split(r"[\\/]", words[i0].strip("'\""))[-1].lower().removesuffix(".exe")
    used.add(i0)
    args = list(range(i0 + 1, len(seg)))
    value_flags = {"kubectl": K8S_VALUE_FLAGS, "k": K8S_VALUE_FLAGS, "oc": K8S_VALUE_FLAGS, "helm": K8S_VALUE_FLAGS,
                   "git": GIT_VALUE_FLAGS, "docker": DOCKER_VALUE_FLAGS, "podman": DOCKER_VALUE_FLAGS,
                   "ssh": SSH_VALUE_FLAGS}.get(head, set())
    positionals = []
    skip_next = False
    for i in args:
        if skip_next:
            skip_next = False
            continue
        if _is_flag(words[i]):
            skip_next = words[i] in value_flags and "=" not in words[i]
        else:
            positionals.append(i)
    sub = words[positionals[0]].lower() if positionals else ""

    def val(i: int) -> tuple[int, int]:
        return seg[i].value_span()

    def take(i: int, kind: str):
        s, e = val(i)
        if e > s:
            hits.append((s, e, kind))
            used.add(i)

    def flag_value(i: int) -> Optional[int]:
        """Index of the value token for flag at i (next token), or None."""
        return i + 1 if i + 1 < len(seg) and not _is_flag(words[i + 1]) else None

    k8s = head in ("kubectl", "k", "oc", "helm")
    for i in args:
        w = words[i]
        flag, _, inline = w.partition("=")
        if not _is_flag(w):
            continue
        j: Optional[int] = None
        if inline:
            j = i  # --flag=value: value_span() handles the prefix
        else:
            j = flag_value(i)
        if j is None:
            continue
        s, e = val(j)
        v = cmd[s:e]
        kind = None
        if flag in ("-n", "--namespace") and k8s:
            kind = "namespace"
        elif flag == "--context" and k8s:
            kind = "context"
        elif flag in ("-m", "--message") and head == "git":
            kind = "message"
        elif (flag in ("-b", "-B", "-c", "-C") and head == "git" and sub in ("checkout", "switch")
              and positionals and i > positionals[0]):
            kind = "new_branch"  # git switch -c / checkout -b <name>
        elif flag in ("-t", "--tag") and head in ("docker", "podman") and "build" in words:
            hits.extend(_image(cmd, s, e)); used.add(j); continue
        elif flag == "--name" and head in ("docker", "podman"):
            kind = "container"
        elif flag in ("-p", "--publish") and head in ("docker", "podman"):
            hits.extend(_ports(cmd, s, e)[:1]); used.add(j); continue  # host side only
        elif flag in ("-L", "-R") and head == "ssh":
            hits.extend(_ports(cmd, s, e)); used.add(j); continue
        elif flag in ("-p", "--port", "-P") and _PORT.fullmatch(v) and head in DB_TOOLS | {"ssh", "scp"}:
            kind = "port"
        elif flag in ("--port",) and _PORT.fullmatch(v):
            kind = "port"
        elif flag in ("-h", "--host") and head in DB_TOOLS:
            kind = "host"
        elif flag in ("-U", "--username", "--user") and head in DB_TOOLS:
            kind = "user"
        elif flag in ("-u", "--user") and head in DB_TOOLS | {"docker", "podman"}:
            kind = "user"
        elif flag in ("-d", "--dbname", "--database") and head in DB_TOOLS:
            kind = "database"
        elif flag in ("-e", "--env") and "=" in v and head in ("docker", "podman", "kubectl"):
            k, _, _ = v.partition("=")
            if re.fullmatch(r"[A-Za-z_]\w*", k):
                hits.append((s + len(k) + 1, e, k.lower()))
                used.add(j)
            continue
        elif flag in ("--region",):
            kind = "region"
        elif flag in ("-i",) and head == "ffmpeg":
            kind = "input"
        if kind:
            take(j, kind)
        elif not inline:
            # flags that take a value we don't name: let shape detection see it, but
            # never mistake it for a positional
            used.discard(j)

    def positional_after(skip_flags: set[str], start: int) -> list[int]:
        out, k = [], start
        while k < len(seg):
            w = words[k]
            if _is_flag(w):
                if w in skip_flags and "=" not in w:
                    k += 1
            else:
                out.append(k)
            k += 1
        return out

    if head == "git" and positionals:
        rest = positionals[1:]
        if "--" in words:
            cut = words.index("--")
            rest = [i for i in rest if i < cut]
        if sub in ("checkout", "switch", "merge", "rebase") and rest:
            if words[rest[0]] not in (".", "-") and not _FILE_EXT.fullmatch(words[rest[0]]):
                take(rest[0], "branch")
        elif sub in ("push", "pull", "fetch") and len(rest) >= 2:
            for i in rest[1:]:
                take(i, "version" if _VERSION.fullmatch(words[i]) else "branch")
        elif sub == "tag" and rest and _VERSION.fullmatch(words[rest[0]]):
            take(rest[0], "version")
    elif k8s and head != "helm" and positionals:
        rest = positionals[1:]
        if sub == "rollout" and rest:
            rest = rest[1:]  # rollout status|restart|undo
        for i in rest:
            m = re.fullmatch(r"([a-z]+)/([\w.-]+)", words[i])
            if m and m.group(1) in K8S_TYPES:
                s, e = seg[i].start + m.start(2), seg[i].start + m.end(2)
                hits.append((s, e, K8S_TYPES[m.group(1)]))
                used.add(i)
        if sub in ("logs", "exec", "attach", "port-forward", "cp") and rest:
            i = rest[0]
            if "/" not in words[i] and i not in used:
                take(i, "pod")
        if sub == "port-forward":
            for i in rest[1:]:
                hits.extend(_ports(cmd, *val(i), kinds=("local_port", "port")))
                used.add(i)
        if sub == "set" and rest and words[rest[0]] == "image":
            for i in rest[1:]:
                s, e = val(i)
                eq = cmd.find("=", s, e)
                if eq > 0:  # container=image:tag
                    hits.extend(_image(cmd, eq + 1, e))
                    used.add(i)
        if sub == "config" and len(rest) >= 2 and words[rest[0]] in ("use-context", "use", "set-context"):
            take(rest[1], "context")
        if sub in ("get", "describe", "delete", "edit", "scale", "label", "annotate") and len(rest) >= 2:
            t, n = rest[0], rest[1]
            kind = K8S_TYPES.get(words[t].lower())
            if kind and "/" not in words[n]:
                take(n, kind)
    elif head == "helm" and len(positionals) >= 2 and sub in ("upgrade", "install", "uninstall", "status",
                                                               "rollback", "history", "get", "test"):
        idx = positionals[1] if sub != "get" else (positionals[2] if len(positionals) > 2 else None)
        if idx is not None:
            take(idx, "release")
        for i in positionals[2:]:
            used.add(i)  # chart refs like bitnami/redis stay literal
    elif head in ("docker", "podman") and positionals:
        rest = positionals[1:]
        if sub == "compose":
            for i in positionals[2:3]:
                used.add(i)
            if len(positionals) >= 3 and words[positionals[1]] in ("exec", "logs", "restart", "run", "stop",
                                                                  "start", "up", "build", "pull"):
                take(positionals[2], "service")
        elif sub in ("exec", "logs", "stop", "start", "restart", "rm", "attach", "inspect", "kill", "cp"):
            later = [i for i in positional_after(DOCKER_VALUE_FLAGS, positionals[0] + 1)
                     if not words[i].isdigit()]
            if later:
                take(later[0], "container")
                for i in later[1:]:
                    used.add(i)  # the command run inside the container
        elif sub in ("run", "create"):
            later = positional_after(DOCKER_VALUE_FLAGS, positionals[0] + 1)
            if later:
                hits.extend(_image(cmd, *val(later[0])))
                used.update(later)  # image + the container's command
        elif sub in ("pull", "push", "tag") and rest:
            for i in rest:
                hits.extend(_image(cmd, *val(i)))
                used.add(i)
        elif sub == "build":
            for i in rest:
                used.add(i)
    elif head == "ssh" and positionals:
        later = positional_after(SSH_VALUE_FLAGS, i0 + 1)
        if later:
            i = later[0]
            v = words[i]
            if "@" in v:
                s, e = val(i)
                at = cmd.index("@", s, e)
                hits.append((at + 1, e, "host"))
                used.add(i)
            elif not _IPV4.fullmatch(v):
                take(i, "host")
            for k in later[1:]:
                used.add(k)  # remote command
    elif head in ("scp", "rsync"):
        for i in positionals:
            s, e = val(i)
            m = re.fullmatch(r"(?:([\w.-]+)@)?([\w.-]+):(.*)", cmd[s:e])
            if m and not re.match(r"[A-Za-z]:[\\/]", cmd[s:e]):
                hits.append((s + m.start(2), s + m.end(2), "host"))
                if m.group(3):
                    hits.append((s + m.start(3), s + m.end(3), "path"))
                used.add(i)
    elif head in DB_TOOLS and positionals and head in ("psql", "mysql", "pg_dump", "mysqldump"):
        i = positionals[-1]
        if not _URL.fullmatch(words[i]) and not _FILE_EXT.fullmatch(words[i]):
            take(i, "database")
    elif head in SCRIPT_RUNNERS and positionals:
        i = positionals[0]
        if words[i] == "-m" or (i > 0 and words[i - 1] == "-m"):
            pass
        else:
            used.add(i)  # the script being run is part of the command, not a value
    if head in ("python", "python3", "py") and "-m" in words:
        j = words.index("-m")
        if j + 1 < len(words):
            used.add(j + 1)  # python -m <module>
    if head == "ffmpeg" and positionals:
        i = positionals[-1]
        if _FILE_EXT.fullmatch(words[i].strip("'\"")):
            take(i, "output")
    elif head in ("cd", "set-location", "sl", "pushd") and positionals:
        take(positionals[0], "path")
    return hits, used


def _detect(cmd: str) -> tuple[list[Hit], str]:
    hits: list[Hit] = []
    for seg in _segments(cmd):
        ctx, used = _context(cmd, seg)
        hits += ctx
        head = ""
        for k, t in enumerate(seg):
            if k in used:
                if not head:
                    head = re.split(r"[\\/]", t.raw.strip("'\""))[-1].lower().removesuffix(".exe")
                continue
            if _is_flag(t.raw) and "=" not in t.raw:
                continue
            found = _shape(cmd, *t.value_span(), head)
            # name files/paths after the long flag they belong to: --outdir ./out → {outdir}
            flag = ""
            if t.raw.startswith("--") and "=" in t.raw:
                flag = t.raw[2:t.raw.index("=")]
            elif k > 0 and seg[k - 1].raw.startswith("--") and "=" not in seg[k - 1].raw:
                flag = seg[k - 1].raw[2:]
            name = re.sub(r"[^a-z0-9]+", "_", flag.lower()).strip("_")
            if name and re.match(r"[a-z]", name) and len(name) <= 24:
                found = [(s, e, name if kind in ("file", "path") else kind) for s, e, kind in found]
            hits += found
    return hits, cmd


def templatize(cmd: str, kinds: Optional[Iterable[str]] = None, registry: Optional[dict] = None) -> Template:
    """Find variable parts of `cmd`. `kinds` limits which kinds become slots.

    `registry` (value, kind) → Slot is shared across calls so a multi-step recipe
    gets one placeholder per distinct value.
    """
    allowed = set(kinds) if kinds is not None else None
    hits, _ = _detect(cmd)
    hits = [h for h in hits if h[1] > h[0] and (allowed is None or h[2] in allowed)]
    hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))
    chosen: list[Hit] = []
    for h in hits:  # drop overlaps, earliest/longest wins
        if chosen and h[0] < chosen[-1][1]:
            continue
        chosen.append(h)

    reg = registry if registry is not None else {}
    names = {s.name for s in reg.values()}
    slots: list[Slot] = []
    slot_index: dict[str, int] = {}
    parts: list[Part] = []
    pos = 0
    for s, e, kind in chosen:
        value = cmd[s:e]
        slot = reg.get((value, kind))
        if slot is None:  # same value seen as a sibling kind (a new branch that is later pushed)
            fam = next((f for f in FAMILIES if kind in f), ())
            slot = next((sl for (v, k), sl in reg.items() if v == value and k in fam), None)
        if slot is None:
            name, n = kind, 2
            while name in names:
                name, n = f"{kind}_{n}", n + 1
            slot = Slot(name, value, kind)
            reg[(value, kind)] = slot
            names.add(name)
        if slot.name not in slot_index:
            slot_index[slot.name] = len(slots)
            slots.append(slot)
        if s > pos:
            parts.append(cmd[pos:s])
        parts.append(slot_index[slot.name])
        pos = e
    if pos < len(cmd):
        parts.append(cmd[pos:])
    return Template(cmd, parts, slots)


# ── stored form ({{name}}) ───────────────────────────────────────────────────

_STORED = re.compile(r"\{\{([a-z][a-z0-9_]*)\}\}")


def placeholders(text: str) -> list[str]:
    seen: list[str] = []
    for m in _STORED.finditer(text):
        if m.group(1) not in seen:
            seen.append(m.group(1))
    return seen


def fill(text: str, values: dict[str, str]) -> str:
    return _STORED.sub(lambda m: values.get(m.group(1), m.group(0)), text)