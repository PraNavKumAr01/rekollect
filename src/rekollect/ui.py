"""Look & feel: theme, command syntax highlighting, result rows."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

LOGO = "◆ rekollect"


def make_theme(accent: str) -> Theme:
    return Theme({
        "accent": f"bold {accent}",
        "accent.dim": accent,
        "muted": "grey50",
        "faint": "grey35",
        "ok": "green3",
        "bad": "red3",
        "warn": "gold3",
        "cmd.head": f"bold {accent}",
        "cmd.flag": "cyan3",
        "cmd.str": "dark_sea_green3",
        "cmd.var": "gold3",
        "cmd.op": "grey50",
        "cmd.path": "sky_blue2",
        "cmd.redacted": "bold red3",
        "match": "underline",
        "sel": "on grey15",
        "bar.on": accent,
        "bar.off": "grey23",
        "slot": f"bold {accent} on grey15",
        "slot.filled": f"bold {accent} underline",
        "slot.kept": "underline",
    })


def console(accent: str = "#b48ead", stderr: bool = False, file=None) -> Console:
    return Console(theme=make_theme(accent), stderr=stderr, file=file, highlight=False,
                   soft_wrap=False)


# ── command syntax highlighting ──────────────────────────────────────────────

_LEX = re.compile(
    r"""(?P<str>"(?:\\.|[^"\\])*(?:"|$)|'[^']*(?:'|$))"""  # tolerate truncated strings
    r"""|(?P<red>‹redacted›)"""
    r"""|(?P<op>\|\||&&|[|;&<>]+)"""
    r"""|(?P<var>\$\{?[A-Za-z_][A-Za-z0-9_]*\}?)"""
    r"""|(?P<ws>\s+)"""
    r"""|(?P<word>[^\s|;&<>"']+)"""
    r"""|(?P<other>.)""",  # never drop a character
    re.S,
)


def highlight(cmd: str, query: str = "", max_len: Optional[int] = None) -> Text:
    one_line = cmd.replace("\n", " ⏎ ")
    if max_len and len(one_line) > max_len:
        one_line = one_line[: max_len - 1] + "…"
    t = Text()
    expect_head = True
    for m in _LEX.finditer(one_line):
        kind, val = m.lastgroup, m.group()
        if kind in ("ws", "other"):
            t.append(val)
            continue
        if kind == "op":
            t.append(val, "cmd.op")
            expect_head = True
            continue
        if kind == "word":
            if expect_head and "=" not in val:
                t.append(val, "cmd.head")
                expect_head = val in ("sudo", "time", "env", "nohup", "exec", "command")
                continue
            if val.startswith("-"):
                t.append(val, "cmd.flag")
            elif "/" in val or "\\" in val or val.startswith("~"):
                t.append(val, "cmd.path")
            else:
                t.append(val)
            continue
        t.append(val, {"str": "cmd.str", "var": "cmd.var", "red": "cmd.redacted"}[kind])
    # underline the bits that matched the query lexically
    for tok in {w for w in re.findall(r"[A-Za-z0-9]{2,}", query.lower())}:
        t.highlight_regex(re.compile(re.escape(tok), re.IGNORECASE), "match")
    return t


def template_text(tpl, values: Optional[dict] = None, max_len: Optional[int] = None) -> Text:
    """Template with placeholders as chips; with `values`, the filled-in command
    with the substituted values highlighted."""
    t = Text()
    for p in tpl.parts:
        if isinstance(p, str):
            t.append(p.replace("\n", " ⏎ "))
            continue
        slot = tpl.slots[p]
        if values is None:
            t.append("{" + slot.name + "}", "slot")
        else:
            v = values.get(slot.name, slot.default)
            t.append(v, "slot.filled" if v != slot.default else "slot.kept")
    if max_len and len(t) > max_len:
        t.truncate(max_len - 1)
        t.append("…")
    return t


# ── small formatters ─────────────────────────────────────────────────────────

def ago(ts: float, now: Optional[float] = None) -> str:
    if not ts:
        return "—"
    s = (now or time.time()) - ts
    for unit, secs in (("y", 31536000), ("mo", 2592000), ("w", 604800), ("d", 86400), ("h", 3600), ("m", 60)):
        if s >= secs:
            return f"{int(s // secs)}{unit}"
    return "now"


def since(ts: float, now: Optional[float] = None) -> str:
    """'3d ago' / 'just now'."""
    a = ago(ts, now)
    return "just now" if a == "now" else ("never" if a == "—" else f"{a} ago")


def short_path(p: str, width: int = 22) -> str:
    if not p:
        return ""
    home = str(Path.home())
    if p.startswith(home):
        p = "~" + p[len(home):]
    if len(p) <= width:
        return p
    tail = re.split(r"[\\/]", p)[-1]
    return ("…/" + tail)[-width:] if len(tail) < width - 2 else "…" + tail[-(width - 1):]


def bar(v: float, n: int = 5) -> Text:
    filled = max(0, min(n, round(v * n)))
    return Text("▰" * filled, "bar.on") + Text("▱" * (n - filled), "bar.off")


def human_bytes(n: float) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def status_glyph(count: float, fails: float) -> Text:
    if count and fails >= count:
        return Text("✗", "bad")
    if fails:
        return Text("◐", "warn")
    return Text("✓", "ok")


def _one_line(t: Text) -> Text:
    t.no_wrap = True
    t.overflow = "ellipsis"
    return t


def results_table(hits, corpus, query: str, width: int, why=None) -> Table:
    """`why`: per-hit list of plain-English reasons (or None)."""
    t = Table(box=None, show_header=False, pad_edge=False, padding=(0, 1), expand=False)
    t.add_column(justify="right", style="faint", no_wrap=True)       # rank
    t.add_column(no_wrap=True)                                        # relevance bar
    cmd_w = max(24, width - 46)
    t.add_column(max_width=cmd_w, overflow="fold")                   # command (reasons may wrap)
    t.add_column(no_wrap=True)                                        # status
    t.add_column(justify="right", style="muted", no_wrap=True)       # count
    t.add_column(justify="right", style="muted", no_wrap=True)       # when
    t.add_column(style="faint", no_wrap=True)                        # where
    top = hits[0].score if hits else 1.0
    for r, h in enumerate(hits, 1):
        i = h.idx
        t.add_row(
            str(r), bar(h.score / top if top else 0),
            _one_line(highlight(corpus.texts[i], query, cmd_w)),
            status_glyph(corpus.count[i], corpus.fails[i]),
            f"{int(corpus.count[i])}×", ago(corpus.last_ts[i]), short_path(corpus.last_cwd[i]),
        )
        if why:
            reasons = why[r - 1] if r - 1 < len(why) else []
            t.add_row("", "", Text("↳ " + (" · ".join(reasons) or "recent and frequent"), "accent.dim"),
                      "", "", "", "")
            nums = "  ".join(f"{k[:3]} {v:.2f}" for k, v in h.parts.items())
            t.add_row("", "", _one_line(Text("  " + nums, "faint")), "", "", "", "")
    return t


def term_width(default: int = 100) -> int:
    try:
        return os.get_terminal_size(2).columns
    except OSError:
        return default