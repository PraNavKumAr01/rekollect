"""`rk eval`: measurable search quality."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time

from rich.table import Table
from rich.text import Text

from rekollect import evaluate as E
from rekollect import learn as L
from rekollect import models as M


def _con(cfg):
    from rekollect.cli import _console
    return _console(cfg)


def _pct(v: float) -> str:
    return f"{v * 100:.0f}%"


def _delta(cur: float, prev, pct: bool = True) -> Text:
    if prev is None:
        return Text("")
    d = cur - prev
    if abs(d) < 1e-9:
        return Text("  =", "faint")
    s = f"{d * 100:+.0f}" if pct else f"{d:+.3f}"
    return Text(f" {'▲' if d > 0 else '▼'}{s}", "ok" if d > 0 else "bad")


def cmd_eval(argv: list[str], cfg) -> int:
    p = argparse.ArgumentParser(prog="rk eval", add_help=False)
    p.add_argument("--tier", default="", help="comma list: lite,balanced,quality or 'all' (downloads if needed)")
    p.add_argument("--modes", default="semantic,lexical,hybrid")
    p.add_argument("--mine", action="store_true", help="replay your own past searches instead")
    p.add_argument("--misses", action="store_true", help="list the main-set questions that weren't answered first")
    p.add_argument("--json", action="store_true")
    p.add_argument("--fail-under", default="", metavar="hit5=0.8",
                   help="exit 1 if hybrid on the first tier scores below this (hit1, hit5 or mrr)")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--bench", default=",".join(E.BENCHES), help="devshell, devshell-holdout")
    a = p.parse_args(argv)
    if a.mine:
        return eval_mine(cfg, a.json)
    con = _con(cfg)
    benches = [E.load_bench(b.strip()) for b in a.bench.split(",") if b.strip()]
    bench = benches[0]
    modes = [m.strip() for m in a.modes.split(",") if m.strip() in E.MODES]
    need_model = any(m != "lexical" for m in modes)
    if a.tier == "all":
        tiers = list(M.REGISTRY)
    elif a.tier:
        tiers = [t.strip() for t in a.tier.split(",") if t.strip()]
    else:  # your tier, plus any other tier already on disk
        mine = str(cfg.get_path("model.tier", "balanced"))
        tiers = [mine] + [t for t, s in M.REGISTRY.items() if t != mine and M.is_downloaded(s)]
    if not need_model:
        tiers = ["none"]

    weights, strength, source = L.effective_weights(cfg)
    results: list[E.Result] = []
    if not a.json:
        sets = " + ".join(f"{b['name']} ({len(b['queries'])} questions)" for b in benches)
        con.print(Text.assemble(("◆ rekollect eval", "accent"),
                                (f"   {sets} over {len(bench['commands'])} commands · hybrid weights: {source}",
                                 "faint")))
    for tier in tiers:
        emb = None
        if tier != "none":
            if tier not in M.REGISTRY:
                con.print(f"[bad]✗[/] unknown tier {tier!r}")
                return 2
            c2 = C_with_tier(cfg, tier)
            spec = M.spec_for(c2)
            if not M.is_downloaded(spec):
                from rekollect.cli import _download_with_progress
                _download_with_progress(spec, con)
            t0 = time.perf_counter()
            emb = M.load_embedder(c2)
            load_ms = (time.perf_counter() - t0) * 1000
        else:
            load_ms = 0.0
        status = con.status(f"  [muted]{tier}: embedding the benchmark history…", spinner="dots") \
            if not a.json else _Null()
        with status:
            corpus, index_ms = E.build_corpus(bench["commands"], emb)
        for b in benches:
            for mode in modes:
                if mode != "lexical" and emb is None:
                    continue
                w = E.MODES[mode] or weights
                r = E.run(corpus, emb if mode != "lexical" else None, b["queries"], mode, w, tier, index_ms)
                r.bench = b["name"]
                r.load_ms = load_ms  # type: ignore[attr-defined]
                results.append(r)

    if a.json:
        print(json.dumps([r.row() for r in results], indent=2))
    else:
        _print(con, results)
        if a.misses:
            _print_misses(con, results)
    if not a.no_save:
        E.save(results)
    if a.fail_under:
        m = re.fullmatch(r"(hit1|hit5|mrr)=(\d*\.?\d+)", a.fail_under)
        if not m:
            con.print("[bad]✗[/] --fail-under expects e.g. hit5=0.8")
            return 2
        target = next((r for r in results if r.mode == "hybrid" and r.bench == bench["name"]),
                      results[0] if results else None)
        if target is None:
            return 2
        value = getattr(target, m.group(1))
        if value < float(m.group(2)):
            con.print(f"[bad]✗ regression:[/] {target.tier}/{target.mode} {m.group(1)} = {value:.3f} "
                      f"< {m.group(2)}")
            return 1
        if not a.json:
            con.print(f"[ok]✓[/] {target.tier}/{target.mode} {m.group(1)} = {value:.3f} ≥ {m.group(2)}")
    return 0


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def C_with_tier(cfg, tier: str):
    from rekollect.config import Config
    import copy
    c = Config(copy.deepcopy(dict(cfg)))
    c["model"]["tier"] = tier
    return c


def _print(con, results: list[E.Result]) -> None:
    t = Table(box=None, padding=(0, 1), header_style="muted")
    for col, j in (("set", "left"), ("model", "left"), ("mode", "left"), ("hit@1", "right"), ("", "left"), ("hit@5", "right"),
                   ("", "left"), ("MRR", "right"), ("", "left"), ("meaning qs", "right"), ("keyword qs", "right"),
                   ("ms/query", "right"), ("p95", "right"), ("index /1k", "right")):
        t.add_column(col, justify=j, no_wrap=True)
    last_bench = None
    for r in results:
        best = max((x.hit1 for x in results if x.bench == r.bench), default=0)
        if last_bench is not None and r.bench != last_bench:
            t.add_row(*[""] * 16)
        prev = E.previous(r.tier, r.mode, r.bench)
        para = r.by_kind.get("paraphrase", {})
        kw = r.by_kind.get("keyword", {})
        t.add_row(
            Text("" if r.bench == last_bench else r.bench.replace("devshell-", "").replace("devshell", "main"),
                 "muted"),
            Text(r.tier, "bold"), r.mode,
            Text(_pct(r.hit1), "accent" if r.hit1 == best else ""), _delta(r.hit1, prev and prev["hit1"]),
            _pct(r.hit5), _delta(r.hit5, prev and prev["hit5"]),
            f"{r.mrr:.3f}", _delta(r.mrr, prev and prev["mrr"], pct=False),
            _pct(para.get("hit1", 0)) if para else "—", _pct(kw.get("hit1", 0)) if kw else "—",
            f"{r.latency_ms:.1f}", f"{r.p95_ms:.1f}",
            f"{r.index_ms_per_1k / 1000:.1f}s" if r.mode != "lexical" else "—",
        )
        last_bench = r.bench
    con.print()
    con.print(t)
    con.print()
    con.print("  [faint]hit@1: right command first · hit@5: in the top five · meaning/keyword qs: hit@1 on "
              "questions phrased differently from the command vs. ones sharing its words · ▲▼ vs. your last run[/]")
    con.print()


def _print_misses(con, results: list[E.Result]) -> None:
    from rekollect import ui
    for r in results:
        if not r.misses or r.bench.endswith("holdout"):  # held-out misses stay unseen while tuning
            continue
        con.print(f"[muted]{r.bench} · {r.tier} · {r.mode}: {len(r.misses)} not answered first[/]")
        for q, rank, first in r.misses[:40]:
            where = f"#{rank + 1}" if rank is not None else "not in top 10"
            con.print(Text.assemble(("  ", ""), (f"{where:>14}  ", "warn" if rank is not None else "bad"),
                                    (q, "bold")))
            con.print(Text("                  got  ", "faint") + ui.highlight(first, max_len=90))
        con.print()


def eval_mine(cfg, as_json: bool) -> int:
    """Personal benchmark: your past picks are the answer key."""
    from rekollect.cli import _engine
    from rekollect.cli_learn import MIN_PICKS, _replays
    con = _con(cfg)
    store, corpus, ranker = _engine(cfg, con)
    replays, fb = _replays(store, corpus, ranker.emb, cfg)
    if len(replays) < 5:
        con.print(f"[muted]Only {len(replays)} picks recorded so far.[/] Every choice you make in the picker "
                  f"becomes a test case; {MIN_PICKS}+ make this meaningful.")
        return 1
    weights, strength, source = L.effective_weights(cfg)
    defaults = dict(C_defaults()["search"]["weights"])
    rows = [
        ("defaults, no learning", L.score_replays(replays, defaults, 0.0)),
        ("your weights, no memory of picks", L.score_replays(replays, weights, 0.0)),
        ("everything on (what you use now)", L.score_replays(replays, weights, strength)),
    ]
    if as_json:
        print(json.dumps({name: m for name, m in rows}, indent=2))
        return 0
    con.print(Text.assemble(("◆ rekollect eval --mine", "accent"),
                            (f"   your {len(replays)} past searches · each one is scored without itself", "faint")))
    t = Table(box=None, padding=(0, 2), header_style="muted")
    t.add_column("ranking")
    t.add_column("your pick #1", justify="right")
    t.add_column("top 5", justify="right")
    t.add_column("MRR", justify="right")
    for name, m in rows:
        t.add_row(name, _pct(m["hit1"]), _pct(m["hit5"]), f"{m['mrr']:.3f}")
    con.print()
    con.print(t)
    con.print(f"\n  [faint]weights: {source}[/]\n")
    from rekollect.cli_learn import curve_panel
    con.print(curve_panel(store))
    return 0


def C_defaults() -> dict:
    from rekollect.config import DEFAULTS
    return DEFAULTS


if __name__ == "__main__":  # pragma: no cover
    sys.exit(cmd_eval(sys.argv[1:], {}))