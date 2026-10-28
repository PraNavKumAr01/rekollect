"""`rk stats` — a little dashboard of your terminal life."""

from __future__ import annotations

import re
import time
from collections import Counter

from rich.columns import Columns
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from rekollect import ui
from rekollect.store import Store, git_root

SPARK = "▁▂▃▄▅▆▇█"


def spark(values: list[float]) -> Text:
    mx = max(values) if values and max(values) > 0 else 1
    t = Text()
    for v in values:
        t.append(SPARK[min(7, int(v / mx * 7.999))] if v else " ", "bar.on" if v else "bar.off")
    return t


def hbar(label: str, v: float, mx: float, width: int = 18) -> Text:
    n = int(round(v / mx * width)) if mx else 0
    return Text.assemble((f"{label:<14.14}", ""), ("█" * n, "bar.on"), ("░" * (width - n), "bar.off"),
                         (f" {int(v):,}", "muted"))


def render(store: Store, con: Console, model_label: str) -> None:
    db = store.db
    uniq, embedded, runs = store.counts()
    if not runs:
        con.print("[muted]Nothing indexed yet. Run[/] [accent]rk init[/]")
        return
    rows = db.execute("SELECT c.text, r.ts, r.exit, r.duration_ms, r.cwd FROM runs r JOIN commands c ON c.id = r.cmd_id").fetchall()
    now = time.time()

    heads, projects, hours, weeks = Counter(), Counter(), [0] * 24, [0] * 16
    fails = known = 0
    slow = []
    for text, ts, ec, dur, cwd in rows:
        w = re.split(r"\s+", text.strip(), maxsplit=2)
        head = w[1] if w[0] in ("sudo", "time", "env", "nohup") and len(w) > 1 else w[0]
        heads[head.rsplit("/", 1)[-1]] += 1
        if cwd:
            root = git_root(cwd)
            if root:
                projects[root.rsplit("/", 1)[-1]] += 1
        if ts:
            lt = time.localtime(ts)
            hours[lt.tm_hour] += 1
            wk = int((now - ts) // 604800)
            if 0 <= wk < 16:
                weeks[15 - wk] += 1
        if ec is not None:
            known += 1
            fails += ec != 0
        if dur and dur > 0:
            slow.append((dur, text))

    first = db.execute("SELECT MIN(first_ts) FROM commands WHERE first_ts > 0").fetchone()[0]

    # headline numbers
    kpis = Table.grid(padding=(0, 4))
    for _ in range(4):
        kpis.add_column()
    succ = f"{(1 - fails / known) * 100:.0f}%" if known else "—"
    kpis.add_row(
        Text.assemble((f"{runs:,}", "accent"), ("\nruns", "muted")),
        Text.assemble((f"{uniq:,}", "accent"), ("\nunique commands", "muted")),
        Text.assemble((succ, "accent"), ("\nsuccess rate", "muted")),
        Text.assemble((ui.ago(first) if first else "—", "accent"), ("\nof history", "muted")),
    )

    top = heads.most_common(8)
    mx = top[0][1] if top else 1
    tools = Group(*[hbar(k, v, mx) for k, v in top])

    tp = projects.most_common(6)
    pmx = tp[0][1] if tp else 1
    projs = Group(*[hbar(k, v, pmx) for k, v in tp]) if tp else Text("cwd is recorded once the hook is on", "faint")

    rhythm = Group(
        Text("hour of day", "muted"), spark(hours),
        Text("0     6     12    18   23", "faint"), Text(""),
        Text("last 16 weeks", "muted"), spark(weeks),
    )

    slow.sort(reverse=True)
    longest = Table.grid(padding=(0, 2))
    longest.add_column(justify="right", style="muted")
    longest.add_column(no_wrap=True, max_width=60, overflow="ellipsis")
    for d, t in slow[:4]:
        longest.add_row(_dur(d), ui.highlight(t, max_len=60))

    con.print()
    con.print(Text.assemble((ui.LOGO, "accent"), ("  stats", "muted"), (f"   ·  {model_label}  ·  "
              f"{embedded:,}/{uniq:,} embedded  ·  {ui.human_bytes(store.size_bytes())} on disk", "faint")))
    con.print()
    con.print(kpis)
    con.print()
    box = dict(border_style="grey23", padding=(0, 1))
    con.print(Columns([Panel(tools, title="[muted]top tools", **box),
                       Panel(projs, title="[muted]busiest projects", **box),
                       Panel(rhythm, title="[muted]rhythm", **box)]))
    if slow:
        con.print(Panel(longest, title="[muted]longest running", **box))
    from rekollect.cli_learn import curve_panel
    con.print(Panel(curve_panel(store), title="[muted]getting better · where your pick was in the list", **box))


def _dur(ms: int) -> str:
    s = ms / 1000
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        return f"{s / 60:.0f}m"
    return f"{s / 3600:.1f}h"