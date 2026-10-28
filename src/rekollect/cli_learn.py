"""`rk tune`, `rk mark` / `rk marks` / `rk unmark`, and the learning-curve views."""

from __future__ import annotations

import argparse
import sys
import time

from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from rekollect import learn as L
from rekollect import ui

MIN_PICKS = 15
AUTOTUNE_EVERY = 40  # new picks between automatic re-tunes


def _con(cfg):
    from rekollect.cli import _console
    return _console(cfg)


# ── tuning ───────────────────────────────────────────────────────────────────

def _replays(store, corpus, emb, cfg, limit: int = 300):
    from rekollect.engine import make_ranker
    ranker = make_ranker(store, corpus, emb, cfg, learning=False)
    ranker.w["semantic"] = max(ranker.w.get("semantic", 0.0), 1e-6)  # always compute meaning for replay
    id_to_idx = {int(c): i for i, c in enumerate(corpus.ids)}
    fb = L.Feedback(store.feedback_rows(), id_to_idx, corpus.vecs.shape[1] if len(corpus.texts) else 0)
    picks = [k for k in range(fb.n) if fb.kind[k] == "pick"][:limit]
    return L.build_replays(ranker, fb, picks), fb


def run_tune(store, corpus, emb, cfg) -> dict:
    weights, strength, _ = L.effective_weights(cfg)
    replays, _ = _replays(store, corpus, emb, cfg)
    if len(replays) < MIN_PICKS:
        return {"n": len(replays), "too_few": True}
    return L.tune(replays, weights, strength if strength > 0 else 0.3)


def maybe_autotune(store, cfg, emb) -> None:
    """Called by the background indexer: re-tune quietly once enough new picks piled up."""
    from rekollect import config as C
    if not cfg.get_path("search.learn", True) or "weights" in C.overrides().get("search", {}):
        return
    n = store.db.execute("SELECT COUNT(*) FROM feedback WHERE kind = 'pick'").fetchone()[0] \
        if store._has("feedback") else 0
    last = int(store.get_meta("tuned_at_picks", "0") or 0)
    if n < MIN_PICKS or n - last < AUTOTUNE_EVERY:
        return
    res = run_tune(store, store.corpus(), emb, cfg)
    store.set_meta("tuned_at_picks", str(n))
    store.commit()
    if res.get("accepted"):
        L.save_learned(res)


def cmd_tune(argv: list[str], cfg) -> int:
    from rekollect.cli import _engine
    p = argparse.ArgumentParser(prog="rk tune", add_help=False)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--reset", action="store_true", help="forget learned weights")
    a = p.parse_args(argv)
    con = _con(cfg)
    if a.reset:
        L.learned_path().unlink(missing_ok=True)
        con.print("[ok]✓[/] learned weights removed; back to the defaults (your pick history is kept)")
        return 0
    store, corpus, ranker = _engine(cfg, con)
    with con.status("  [muted]Replaying your past searches…", spinner="dots"):
        t0 = time.perf_counter()
        res = run_tune(store, corpus, ranker.emb, cfg)
        dt = time.perf_counter() - t0
    if res.get("too_few"):
        con.print(f"  [muted]Not enough to learn from yet:[/] {res['n']} picks, need {MIN_PICKS}. "
                  "Every time you choose something in the picker counts.")
        return 1
    weights, strength, source = L.effective_weights(cfg)
    b, af = res["before"], res["after"]
    con.print()
    con.print(Text.assemble(("◆ rekollect tune", "accent"),
                            (f"   {res['n']} of your searches · trained on {res['train']}, checked on the "
                             f"{res['test']} most recent · {dt:.1f}s", "faint")))
    t = Table(box=None, padding=(0, 2), header_style="muted")
    t.add_column("")
    t.add_column("now", justify="right")
    t.add_column("tuned", justify="right")
    for k, label in (("hit1", "your pick ranked #1"), ("hit5", "in the top 5"), ("mrr", "MRR")):
        fmt = (lambda v: f"{v:.3f}") if k == "mrr" else (lambda v: f"{v * 100:.0f}%")
        delta = af[k] - b[k]
        style = "ok" if delta > 1e-9 else "bad" if delta < -1e-9 else "muted"
        t.add_row(label, fmt(b[k]), Text(fmt(af[k]), style))
    con.print(t)
    w = Table(box=None, padding=(0, 2), header_style="muted")
    w.add_column("signal")
    w.add_column("now", justify="right")
    w.add_column("tuned", justify="right")
    tot = sum(weights.get(s, 0) for s in L.SIGNALS) or 1
    for s in L.SIGNALS:
        w.add_row(s, f"{weights.get(s, 0) / tot:.2f}", f"{res['weights'][s]:.2f}")
    w.add_row("your history", f"{strength:.2f}", f"{res['strength']:.2f}")
    con.print(w)
    if not res["accepted"]:
        con.print("  [muted]No improvement on your recent searches, so nothing changes.[/]")
        return 0
    if a.dry_run:
        con.print("  [faint]dry run: not saved[/]")
        return 0
    L.save_learned(res)
    con.print(f"  [ok]✓[/] saved. rekollect now ranks with these weights "
              f"[faint](undo: rk tune --reset · source was: {source})[/]")
    return 0


# ── bookmarks ────────────────────────────────────────────────────────────────

def cmd_mark(argv: list[str], cfg) -> int:
    from rekollect.cli import _engine
    from rekollect.picker import Picker
    p = argparse.ArgumentParser(prog="rk mark", add_help=False)
    p.add_argument("query", nargs="*")
    p.add_argument("--name", "-n")
    p.add_argument("--tag", "-t", action="append", default=[])
    p.add_argument("--note")
    a = p.parse_args(argv)
    con = _con(cfg)
    store, corpus, ranker = _engine(cfg, con)
    if not sys.stdin.isatty():
        con.print("[bad]✗[/] rk mark needs a terminal to pick the command")
        return 2
    picker = Picker(ranker, corpus, store, cfg.get_path("ui.accent", "#b48ead"), "mark", query=" ".join(a.query))
    sel = picker.run()
    if not sel:
        return 1
    cid = store.find_command(sel)
    if cid is None:
        con.print("[bad]✗[/] pick an original command, not a filled-in template")
        return 2
    existing = {m[0]: m for m in store.marks_list()}.get(cid)
    con.print(Text("  ★ ", "accent") + ui.highlight(sel))
    name = a.name if a.name is not None else Prompt.ask("    name [faint](optional, e.g. deploy-prod)[/]",
                                                        default=existing[2] if existing else "", console=con)
    tags = " ".join(t.lstrip("#") for t in a.tag) if a.tag else Prompt.ask(
        "    tags [faint](space separated, e.g. deploy k8s)[/]", default=existing[3] if existing else "",
        console=con)
    note = a.note if a.note is not None else Prompt.ask("    note [faint](optional)[/]",
                                                        default=existing[4] if existing else "", console=con)
    store.set_mark(cid, name.strip(), " ".join(t.lstrip("#").lower() for t in tags.split()), note.strip())
    store.commit()
    con.print(f"  [ok]✓[/] bookmarked. Find it with [accent]@{name.strip() or '…'}[/] or "
              f"{' '.join('[accent]#' + t + '[/]' for t in tags.split()) or 'its tags'}; your words are searchable too.")
    return 0


def cmd_marks(argv: list[str], cfg) -> int:
    from rekollect.store import Store
    con = _con(cfg)
    want = [t.lstrip("#").lower() for t in argv]
    rows = [r for r in Store().marks_list() if all(any(x.startswith(w) for x in r[3].split()) for w in want)]
    if not rows:
        con.print("[muted]No bookmarks yet.[/] Press [accent]^B[/] in the picker, or run [accent]rk mark[/].")
        return 0
    t = Table(box=None, padding=(0, 2), header_style="muted")
    t.add_column("name", style="accent", no_wrap=True)
    t.add_column("tags", style="muted")
    t.add_column("command", overflow="fold")
    t.add_column("note", style="faint", overflow="fold")
    for _cid, text, name, tags, note, _ in rows:
        t.add_row(name or "★", " ".join("#" + x for x in tags.split()), ui.highlight(text), note)
    con.print()
    con.print(t)
    con.print("\n  [faint]search inside them with #tag or @name in the picker[/]\n")
    return 0


def cmd_unmark(argv: list[str], cfg) -> int:
    from rekollect.store import Store
    con = _con(cfg)
    if not argv:
        con.print("usage: [accent]rk unmark[/] <name or exact command>")
        return 2
    target = " ".join(argv)
    st = Store()
    hits = [r for r in st.marks_list() if r[2] == target.lstrip("@") or r[1] == target]
    if not hits:
        con.print(f"[muted]no bookmark named {target!r}[/]")
        return 1
    for r in hits:
        st.remove_mark(r[0])
    st.commit()
    con.print(f"[ok]✓[/] removed {len(hits)} bookmark{'s' if len(hits) > 1 else ''}")
    return 0


# ── learning curve ───────────────────────────────────────────────────────────

def curve_panel(store):
    """Renderable for rk stats: is rekollect getting better for you?"""
    from rich.console import Group
    rows = store.feedback_rows(20000)
    pts = L.curve(rows)
    picks = sum(p["n"] for p in pts)
    if not picks:
        return Text("Pick commands in the picker (Ctrl+G or rk) and this fills in: the position your choice\n"
                    "was at, week by week. Learning shows up as that number falling toward 1.", "faint")
    recent = pts[-4:]
    tbl = Table(box=None, padding=(0, 2), header_style="muted")
    tbl.add_column("week", style="muted")
    tbl.add_column("searches", justify="right")
    tbl.add_column("your pick was at", justify="right")
    tbl.add_column("first try", justify="right")
    for p in pts[-8:]:
        tbl.add_row(f"{p['period'] + 1}", str(p["n"]), f"#{p['avg_position']:.1f}", f"{p['first'] * 100:.0f}%")
    first = pts[0]
    last_avg = sum(p["avg_position"] * p["n"] for p in recent) / sum(p["n"] for p in recent)
    learned = L.load_learned()
    lines = [tbl, Text("")]
    trend = first["avg_position"] - last_avg
    if len(pts) > 1:
        lines.append(Text(f"  from #{first['avg_position']:.1f} in week 1 to #{last_avg:.1f} lately",
                          "ok" if trend > 0 else "muted"))
    lines.append(Text(f"  {picks} searches learned from · "
                      + (f"weights tuned {ui.since(learned['ts'])}" if learned.get("ts") else "not tuned yet (rk tune)"),
                      "faint"))
    return Group(*lines)