"""Readers for zsh / bash / fish history files and for the hook spool."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from rekollect import config as C

FS, RS = "\x1f", "\x1e"  # field / record separators used by the shell hooks


@dataclass
class Entry:
    cmd: str
    ts: int = 0  # unix seconds, 0 = unknown
    cwd: str = ""
    exit: Optional[int] = None
    duration_ms: Optional[int] = None
    source: str = "history"
    session: str = ""


# ── zsh ──────────────────────────────────────────────────────────────────────

_ZSH_EXT = re.compile(r"^: (\d+):(\d+);(.*)$", re.S)


def _unmetafy(b: bytes) -> bytes:
    """zsh stores some bytes 'metafied': 0x83 followed by (byte ^ 32)."""
    if b"\x83" not in b:
        return b
    out = bytearray()
    it = iter(b)
    for c in it:
        if c == 0x83:
            nxt = next(it, None)
            if nxt is not None:
                out.append(nxt ^ 32)
        else:
            out.append(c)
    return bytes(out)


def parse_zsh(text: str) -> Iterator[Entry]:
    cur: Optional[Entry] = None
    cont = False
    for line in text.split("\n"):
        if cont and cur is not None:
            cur.cmd = cur.cmd[:-1] + "\n" + line  # drop trailing backslash
            cont = line.endswith("\\")
            continue
        if cur is not None:
            yield cur
            cur = None
        if not line:
            continue
        m = _ZSH_EXT.match(line)
        if m:
            cur = Entry(cmd=m.group(3), ts=int(m.group(1)), duration_ms=int(m.group(2)) * 1000)
        else:
            cur = Entry(cmd=line)
        cont = line.endswith("\\")
    if cur is not None:
        yield cur


# ── bash ─────────────────────────────────────────────────────────────────────

_BASH_TS = re.compile(r"^#(\d{9,11})$")


def parse_bash(text: str) -> Iterator[Entry]:
    ts = 0
    for line in text.split("\n"):
        m = _BASH_TS.match(line)
        if m:
            ts = int(m.group(1))
            continue
        if line.strip():
            yield Entry(cmd=line, ts=ts)
            ts = 0


# ── fish ─────────────────────────────────────────────────────────────────────

def _fish_unescape(s: str) -> str:
    return s.replace("\\\\", "\x00").replace("\\n", "\n").replace("\x00", "\\")


def parse_fish(text: str) -> Iterator[Entry]:
    cur: Optional[Entry] = None
    for line in text.split("\n"):
        if line.startswith("- cmd: "):
            if cur:
                yield cur
            cur = Entry(cmd=_fish_unescape(line[7:]))
        elif line.startswith("  when: ") and cur is not None:
            try:
                cur.ts = int(line[8:].strip())
            except ValueError:
                pass
    if cur:
        yield cur


# ── PowerShell (PSReadLine) ──────────────────────────────────────────────────
# One command per line, no timestamps. Multi-line commands are stored with a
# trailing backtick on every line but the last.

def parse_pwsh(text: str) -> Iterator[Entry]:
    buf: list[str] = []
    for line in text.lstrip("\ufeff").split("\n"):
        line = line.rstrip("\r")
        if line.endswith("`"):
            buf.append(line[:-1])
            continue
        buf.append(line)
        cmd = "\n".join(buf)
        buf = []
        if cmd.strip():
            yield Entry(cmd=cmd)
    if buf and "\n".join(buf).strip():
        yield Entry(cmd="\n".join(buf))


def psreadline_dirs() -> list[Path]:
    """Where PSReadLine saves history (`(Get-PSReadLineOption).HistorySavePath`)."""
    dirs = []
    if os.name == "nt":
        appdata = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        dirs.append(Path(appdata) / "Microsoft" / "Windows" / "PowerShell" / "PSReadLine")
    else:
        xdg = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
        dirs.append(Path(xdg) / "powershell" / "PSReadLine")
    return dirs


# ── discovery ────────────────────────────────────────────────────────────────

def detect_shell() -> str:
    """Best guess at the interactive shell we're running in."""
    sh = Path(os.environ.get("SHELL", "")).name.lower()
    if sh.endswith(".exe"):
        sh = sh[:-4]
    if os.name == "nt":
        # Git Bash / MSYS set SHELL; PowerShell and cmd don't.
        return sh if sh in ("bash", "zsh", "fish") else "powershell"
    if os.environ.get("PSModulePath"):  # only set inside pwsh on macOS/Linux
        return "powershell"
    return sh or "zsh"


def history_files() -> list[tuple[str, Path]]:
    home = Path.home()
    cands: list[tuple[str, Path]] = []
    histfile = os.environ.get("HISTFILE")
    shell = Path(os.environ.get("SHELL", "")).name
    if histfile:
        cands.append(("zsh" if shell == "zsh" or "zsh" in histfile else "bash", Path(histfile)))
    cands += [
        ("zsh", home / ".zsh_history"),
        ("zsh", home / ".zhistory"),
        ("zsh", Path(os.environ.get("ZDOTDIR", home)) / ".zsh_history"),
        ("bash", home / ".bash_history"),
        ("fish", Path(os.environ.get("XDG_DATA_HOME", home / ".local/share")) / "fish" / "fish_history"),
    ]
    # ConsoleHost_history.txt, "Visual Studio Code Host_history.txt", …
    for d in psreadline_dirs():
        if d.is_dir():
            cands += [("powershell", f) for f in sorted(d.glob("*_history.txt"))]
    seen, out = set(), []
    for kind, p in cands:
        try:
            rp = p.resolve()
        except OSError:
            continue
        if rp in seen or not rp.is_file():
            continue
        seen.add(rp)
        out.append((kind, rp))
    return out


def read_history(kind: str, path: Path) -> list[Entry]:
    raw = path.read_bytes()
    if kind == "zsh":
        raw = _unmetafy(raw)
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = raw.decode("utf-16", errors="replace")
    else:
        text = raw.decode("utf-8", errors="replace")
    parser = {"zsh": parse_zsh, "bash": parse_bash, "fish": parse_fish, "powershell": parse_pwsh}[kind]
    entries = list(parser(text))
    for e in entries:
        e.source = kind
    return entries


# ── spool written by the hooks ───────────────────────────────────────────────

def drain_spool() -> list[Entry]:
    """Atomically take everything the hooks appended since the last drain."""
    sp = C.spool_path()
    if not sp.exists() or sp.stat().st_size == 0:
        return []
    work = sp.with_name(f"spool.{os.getpid()}.work")
    for attempt in range(5):
        try:
            sp.rename(work)  # shells now append to a fresh file
            break
        except FileNotFoundError:
            return []
        except OSError:  # Windows: file briefly held open by a shell's append
            if attempt == 4:
                return []
            time.sleep(0.05)
    time.sleep(0.02)  # let any in-flight append land
    data = work.read_bytes().decode("utf-8", errors="replace")
    work.unlink(missing_ok=True)
    return parse_spool(data)


def parse_spool(data: str) -> list[Entry]:
    out = []
    for rec in data.split(RS):
        if not rec.strip():
            continue
        parts = rec.split(FS, 5)
        if len(parts) == 6:      # ts, exit, duration, cwd, session, cmd
            ts, ec, dur, cwd, sid, cmd = parts
        elif len(parts) == 5:    # records from hooks before sessions existed
            ts, ec, dur, cwd, cmd = parts
            sid = ""
        else:
            continue

        def _int(x: str) -> Optional[int]:
            try:
                return int(float(x))
            except ValueError:
                return None

        out.append(Entry(cmd=cmd, ts=_int(ts) or 0, exit=_int(ec), duration_ms=_int(dur), cwd=cwd,
                         source="hook", session=sid))
    return out