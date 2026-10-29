"""Command-line entry point. Heavy imports are lazy so `rk hook` stays instant."""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

from rekollect import __version__
from rekollect import config as C

from rekollect.hooks import SUBCOMMANDS  # noqa: E402  (cheap: hooks only imports config)

HELP_SECTIONS = [
    ("find", [
        ("Ctrl+G", "", "live picker on what you've typed so far"),
        ("rk", "[words]", "the same picker, no shortcut needed; the pick lands on your next prompt"),
        ("rk s", "<words>", "print matches  ·  -n 20  --here  --ok  --since 2w  --why  --json"),
        ("rk why", "<words>", "matches with the reasons, in plain English"),
        ("rk tpl", "'<command>'", "show a command as a reusable template (Ctrl+T in the picker)"),
        ("", "#tag  @name", "inside any search: only bookmarks with that tag / name"),
    ]),
    ("learns from you", [
        ("rk mark", "[words]", "bookmark a command with a name, #tags and a note (^B in the picker stars)"),
        ("rk marks", "[#tag]", "list bookmarks  ·  rk unmark <name>"),
        ("rk tune", "", "fit the ranking to the choices you've made  ·  --dry-run  --reset"),
        ("rk eval", "", "measure search quality on the built-in benchmark  ·  --misses  --fail-under hit5=0.9"),
        ("rk eval --mine", "", "measure it on your own past searches"),
    ]),
    ("coding assistants", [
        ("rk agents", "", "read-only access for AI assistants  ·  enable [--scope all]  disable  log  setup"),
        ("rk mcp", "", "the MCP server assistants start (stdio)"),
    ]),
    ("workflows", [
        ("rk flow", '"<task>"', "the sequence of commands you ran for a task  ·  --save <name>"),
        ("rk recipe", "", "list saved recipes"),
        ("rk recipe run", "<name>", "step through one  ·  --var k=v  --step N  --from N  --dry-run  -y"),
        ("rk recipe save", "<name>", "save what you just did in this terminal  ·  --last 5"),
        ("rk recipe", "show|edit|rm <name>", ""),
    ]),
    ("setup", [
        ("rk init", "", "download a model, index history, set up your shell"),
        ("rk vscode", "", "let the picker key through VS Code / Cursor terminals"),
        ("rk config", "", "settings  ·  set <key> <value>  unset <key>  edit"),
        ("rk models", "", "model tiers  ·  rk model <lite|balanced|quality> to switch"),
        ("rk index", "", "embed new commands now  ·  --reimport  --rebuild"),
        ("rk doctor", "", "check your setup  ·  rk keys shows what your terminal sends"),
        ("rk hook", "<shell>", "print the shell integration (zsh, bash, fish, powershell)"),
    ]),
    ("more", [
        ("rk stats", "", "your terminal habits"),
        ("rk forget", "<regex>", "delete matching commands"),
        ("rk pause / resume", "", "stop / start recording"),
    ]),
]


def _print_help(cfg: C.Config) -> None:
    from rich.table import Table
    from rich.text import Text
    con = _console(cfg)
    con.print(Text.assemble(("◆ rekollect", "accent"), (" — find that command by describing it", "muted")))
    t = Table.grid(padding=(0, 2))
    t.add_column(no_wrap=True)
    t.add_column(overflow="fold")
    for title, rows in HELP_SECTIONS:
        t.add_row("", "")
        t.add_row(Text(title, "muted"), "")
        for cmd, arg, desc in rows:
            t.add_row(Text.assemble(("  " + cmd, "accent"), (" " + arg if arg else "", "bold")), Text(desc, "faint"))
    con.print(t)
    con.print()
    con.print(Text("Commands starting with a space are never recorded; secrets are redacted before storage.",
                   "faint"))


# ── helpers ──────────────────────────────────────────────────────────────────

def _console(cfg: C.Config, stderr: bool = False):
    from rekollect import ui
    return ui.console(cfg.get_path("ui.accent", "#b48ead"), stderr=stderr)


def _label(cfg: C.Config) -> str:
    return str(cfg.get_path("model.tier", "balanced"))


def _download_with_progress(spec, con):
    from rich.progress import (BarColumn, DownloadColumn, Progress, TextColumn,
                               TransferSpeedColumn)
    from rekollect import models as M

    if M.is_downloaded(spec):
        return
    con.print(f"[muted]Downloading[/] [accent]{spec.repo}[/] [faint](~{spec.size_mb} MB, one time)[/]")
    with Progress(TextColumn("  [faint]{task.description:<32}"), BarColumn(bar_width=28,
                  complete_style="accent.dim", finished_style="ok"), DownloadColumn(), TransferSpeedColumn(),
                  console=con, transient=False) as prog:
        tasks: dict[str, int] = {}

        def cb(name: str, done: int, total: Optional[int]):
            if name not in tasks:
                tasks[name] = prog.add_task(name.rsplit("/", 1)[-1], total=total)
            prog.update(tasks[name], completed=done, total=total)

        try:
            M.download(spec, cb)
        except Exception as e:
            con.print(f"[bad]✗ download failed:[/] {e}")
            con.print("[faint]  behind a proxy or mirror? set HF_ENDPOINT; gated model? set HF_TOKEN[/]")
            raise SystemExit(1)


def _load_embedder(cfg: C.Config, con, quiet: bool = False):
    from rekollect import models as M

    spec = M.spec_for(cfg)
    if not quiet:
        _download_with_progress(spec, con)
    return M.load_embedder(cfg)


def _embed_with_progress(store, emb, con, label: str = "Embedding"):
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn
    from rekollect.indexer import embed_pending

    n = len(store.pending())
    if not n:
        return 0
    t0 = time.perf_counter()
    with Progress(TextColumn(f"  [muted]{label}[/]"), BarColumn(bar_width=28, complete_style="accent.dim",
                  finished_style="ok"), MofNCompleteColumn(), TimeRemainingColumn(), console=con) as prog:
        task = prog.add_task("", total=n)
        embed_pending(store, emb, lambda d, t: prog.update(task, completed=d))
    dt = time.perf_counter() - t0
    con.print(f"  [ok]✓[/] embedded [accent]{n:,}[/] unique commands in {dt:.1f}s "
              f"[faint]({n / max(dt, 1e-6):,.0f}/s)[/]")
    return n


def _parse_since(s: Optional[str]) -> int:
    if not s:
        return 0
    m = re.fullmatch(r"(\d+)\s*(h|d|w|mo|y)", s.strip().lower())
    if not m:
        raise SystemExit("rekollect: --since expects e.g. 3h, 7d, 2w, 6mo, 1y")
    mult = {"h": 3600, "d": 86400, "w": 604800, "mo": 2592000, "y": 31536000}[m.group(2)]
    return int(time.time() - int(m.group(1)) * mult)


def _engine(cfg: C.Config, con, *, success_only=False, since=0, here=False):
    """Open store, catch up on new commands, build the ranker."""
    from rekollect.engine import catch_up, make_ranker
    from rekollect.store import Store, git_root

    store = Store()
    if store.counts()[2] == 0:
        con.print("[muted]Nothing is indexed yet.[/] If you ran [accent]rk init[/] and it found no history, "
                  "set up the shell hook so commands get recorded, then try again. [accent]rk doctor[/] "
                  "shows what was found.")
        raise SystemExit(1)
    emb = _load_embedder(cfg, con)
    store.waiting = catch_up(store, cfg, emb)
    cwd = os.getcwd()
    project = git_root(cwd)
    corpus = store.corpus(success_only=success_only, since_ts=since, cwd_prefix=(project or cwd) if here else "")
    ranker = make_ranker(store, corpus, emb, cfg, cwd)
    return store, corpus, ranker


# ── commands ─────────────────────────────────────────────────────────────────

def cmd_search(argv: list[str], cfg: C.Config) -> int:
    p = argparse.ArgumentParser(prog="rk", add_help=False)
    p.add_argument("query", nargs="*")
    p.add_argument("-n", "--limit", type=int, default=cfg.get_path("search.limit", 8))
    p.add_argument("--here", action="store_true")
    p.add_argument("--ok", action="store_true", default=cfg.get_path("search.success_only", False))
    p.add_argument("--since")
    p.add_argument("--why", action="store_true")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    q = " ".join(a.query)
    con = _console(cfg)
    from rekollect import ui

    store, corpus, ranker = _engine(cfg, con, success_only=a.ok, since=_parse_since(a.since), here=a.here)
    hits, ms = ranker.search(q, a.limit)
    reasons = [ranker.explain(h, q) for h in hits]
    if a.json:
        import json
        print(json.dumps([{"command": corpus.texts[h.idx], "score": round(h.score, 4),
                           "runs": int(corpus.count[h.idx]), "failures": int(corpus.fails[h.idx]),
                           "last_run": int(corpus.last_ts[h.idx]), "cwd": corpus.last_cwd[h.idx],
                           "why": why_, "signals": {k: round(v, 4) for k, v in h.parts.items()}}
                          for h, why_ in zip(hits, reasons)], indent=2))
        return 0
    head = ui.Text.assemble((ui.LOGO, "accent"), ("  ", ""), (f'"{q}"' if q else "recent & frequent", "bold"),
                            (f"  ·  {len(hits)} of {len(corpus.texts):,}  ·  {ms:.0f} ms  ·  {_label(cfg)}", "faint"))
    con.print(head)
    if not hits:
        con.print("  [faint]no matches — try describing it differently[/]")
        return 1
    con.print(ui.results_table(hits, corpus, q, con.width, why=reasons if a.why else None))
    return 0


def cmd_pick(argv: list[str], cfg: C.Config) -> int:
    if argv[:1] == ["--copy-only"]:  # used by the fish `rk` function
        from rekollect.picker import copy_to_clipboard
        text = " ".join(argv[2:] if argv[1:2] == ["--"] else argv[1:])
        return 0 if copy_to_clipboard(text) else 1
    p = argparse.ArgumentParser(prog="rk pick", add_help=False)
    p.add_argument("--query", default=None)
    p.add_argument("--out", help="write the selection to this file (used by the PowerShell hook)")
    a = p.parse_args(argv)
    query = a.query if a.query is not None else os.environ.get("REKOLLECT_QUERY", "")
    try:
        open("CONIN$" if os.name == "nt" else "/dev/tty").close()
    except OSError:
        print("rekollect: pick needs a terminal", file=sys.stderr)
        return 2
    from rekollect.picker import Picker, copy_to_clipboard

    con = _console(cfg, stderr=True)
    store, corpus, ranker = _engine(cfg, con)
    try:
        picker = Picker(ranker, corpus, store, cfg.get_path("ui.accent", "#b48ead"), _label(cfg), query=query)
        sel = picker.run()
    except Exception:  # the picker often runs where a traceback would be invisible (alt screen, key handler)
        import traceback
        log = C.data_dir() / "picker.log"
        log.write_text(traceback.format_exc(), encoding="utf-8")
        print(f"rekollect: the picker crashed; details in {log}", file=sys.stderr)
        return 3
    if sel is None:
        return 1
    if a.out:
        Path(a.out).write_text(sel, encoding="utf-8")
    elif sys.stdout.isatty():  # run directly (not from the widget): show + copy
        from rekollect import ui
        con2 = _console(cfg)
        con2.print(ui.Text("❯ ", "accent") + ui.highlight(sel))
        if copy_to_clipboard(sel):
            con2.print("[faint]  copied to clipboard · press Ctrl+G at your prompt to insert directly[/]")
    else:
        sys.stdout.buffer.write(sel.encode("utf-8"))
        sys.stdout.flush()
    return 0


def cmd_index(argv: list[str], cfg: C.Config) -> int:
    p = argparse.ArgumentParser(prog="rk index", add_help=False)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--reimport", action="store_true", help="re-read shell history files")
    p.add_argument("--rebuild", action="store_true", help="re-embed everything")
    a = p.parse_args(argv)
    from rekollect.indexer import drain_spool, ensure_model_matches, import_histories, ingest
    from rekollect.store import Store, index_lock

    with index_lock(blocking=not a.quiet) as got:
        if not got:
            return 0  # another indexer is already running
        store = Store()
        if a.quiet:
            from rekollect.indexer import sync
            from rekollect import models as M
            emb_box: list = []

            def load():
                if not emb_box:
                    emb_box.append(M.load_embedder(cfg))
                return emb_box[0]
            sync(store, cfg, load_emb=load)
            try:
                _learn().maybe_autotune(store, cfg, load())
            except Exception:
                pass  # learning must never break indexing
            return 0
        con = _console(cfg)
        if a.rebuild:
            store.clear_vectors()
            store.commit()
        new = ingest(store, drain_spool(), cfg)
        if a.reimport:
            n, used = import_histories(store, cfg)
            new += n
            for u in used:
                con.print(f"  [faint]read {u}[/]")
        if ensure_model_matches(store, cfg):
            con.print("  [warn]model changed — re-embedding everything[/]")
        con.print(f"  [ok]✓[/] {new:,} new runs")
        if store.pending():
            emb = _load_embedder(cfg, con)
            _embed_with_progress(store, emb, con)
        else:
            con.print("  [ok]✓[/] index is up to date")
    return 0


def cmd_init(argv: list[str], cfg: C.Config) -> int:
    p = argparse.ArgumentParser(prog="rk init", add_help=False)
    p.add_argument("--tier", choices=["lite", "balanced", "quality"])
    p.add_argument("--yes", "-y", action="store_true")
    a = p.parse_args(argv)
    from rich.prompt import Prompt
    from rekollect import models as M
    from rekollect.indexer import ensure_model_matches, import_histories
    from rekollect.store import Store

    C.ensure_dirs()
    C.write_default_if_missing()
    cfg = C.load()
    con = _console(cfg)
    con.print()
    con.print(f"[accent]◆ rekollect[/] [muted]v{__version__} · semantic search for your shell history[/]\n")

    tier = a.tier
    if not tier and sys.stdin.isatty() and not a.yes:
        _models_table(con, cfg)
        tier = Prompt.ask("  [muted]Pick a model[/]", choices=list(M.REGISTRY), default=_label(cfg), console=con)
    tier = tier or _label(cfg)
    if tier != cfg["model"]["tier"]:
        cfg["model"]["tier"] = tier
        C.save(cfg)
    spec = M.spec_for(cfg)

    con.print()
    _download_with_progress(spec, con)
    emb = M.load_embedder(cfg)
    con.print(f"  [ok]✓[/] model ready [faint]({spec.tier}: {spec.repo}, {emb.dim or spec.dim}-d)[/]")

    store = Store()
    ensure_model_matches(store, cfg)
    with con.status("  [muted]Reading shell history…", spinner="dots"):
        n, used = import_histories(store, cfg)
    if not used:
        from rekollect.history import psreadline_dirs
        con.print("  [warn]no history files found[/] [faint](that's fine — the hook will record from now on)[/]")
        con.print(f"  [faint]  looked for ~/.zsh_history, ~/.bash_history, fish history and "
                  f"{psreadline_dirs()[0]}{os.sep}*_history.txt[/]")
    for u in used:
        con.print(f"  [ok]✓[/] read [faint]{u}[/]")
    uniq, _, runs = store.counts()
    con.print(f"  [ok]✓[/] {runs:,} runs → [accent]{uniq:,}[/] unique commands after filtering & redaction")
    _embed_with_progress(store, emb, con)
    from rekollect import ui
    con.print(f"  [ok]✓[/] index size {ui.human_bytes(store.size_bytes())}  ·  model "
              f"{ui.human_bytes(M.disk_usage(spec))}")

    # shell integration
    _install_hook(con, assume_yes=a.yes)
    _offer_vscode(con, cfg, assume_yes=a.yes)

    con.print("\n  [muted]Try it:[/]  [accent]rk[/] [bold]\"that command to free up disk space\"[/]   "
              "[muted]or press[/] [accent]Ctrl+G[/]\n")
    return 0


def _offer_vscode(con, cfg: C.Config, assume_yes: bool = False) -> None:
    from rich.prompt import Confirm
    from rekollect import vscode
    editors = [(n, p) for n, p in vscode.installed_editors() if not vscode.has_block(p)]
    if not editors:
        return
    key = cfg.get_path("ui.key", "^G")
    vk = vscode.key_label(key)
    names = ", ".join(n for n, _ in editors)
    if assume_yes or (sys.stdin.isatty() and Confirm.ask(
            f"  {names} grabs [accent]{vk}[/] in its terminal. Add a keybinding so it reaches the picker?",
            default=True, console=con)):
        cmd_vscode([], cfg)
    else:
        con.print("  [faint]later: run[/] [accent]rk vscode[/]")


def _install_hook(con, assume_yes: bool = False) -> None:
    from rich.prompt import Confirm
    from rekollect import hooks
    from rekollect.history import detect_shell

    shell = detect_shell()
    con.print()
    ask = lambda q: assume_yes or (sys.stdin.isatty() and Confirm.ask(q, default=True, console=con))  # noqa: E731
    line = hooks.RC_LINES.get(shell)

    if shell == "powershell":
        profiles = hooks.powershell_profiles()
        if not profiles:
            con.print("  [warn]couldn't find pwsh or powershell on PATH.[/] Add this to your $PROFILE:")
            con.print(f"    [accent]{line}[/]")
            return
        todo = [(ed, rc, pol) for ed, rc, pol in profiles if not hooks.has_line(rc, line)]
        for ed, rc, _ in profiles:
            if (ed, rc, _) not in todo:
                con.print(f"  [ok]✓[/] hook already in [faint]{rc}[/] ({ed})")
        if todo and ask(f"  Add live recording + [accent]Ctrl+G[/] picker to your PowerShell profile"
                        f"{'s' if len(todo) > 1 else ''}?"):
            for ed, rc, _ in todo:
                hooks.append_line(rc, line)
                con.print(f"  [ok]✓[/] added to [faint]{rc}[/] ({ed})")
            con.print("  [muted]open a new terminal to activate[/]")
        elif todo:
            con.print(f"  add this to your PowerShell profile:  [accent]{line}[/]")
        for ed, _, pol in profiles:
            if pol.lower() in ("restricted", "allsigned"):
                con.print(f"  [warn]![/] {ed} execution policy is [bold]{pol}[/], so profiles won't load. Fix with:")
                con.print("    [accent]Set-ExecutionPolicy -Scope CurrentUser RemoteSigned[/]")
        if os.name == "nt":
            con.print("  [faint]cmd.exe has no prompt hook; use PowerShell for live recording. "
                      "Searching works from any terminal.[/]")
        return

    if shell in hooks.RC_FILES:
        rc = Path(os.path.expanduser(hooks.RC_FILES[shell]))
        if hooks.has_line(rc, line):
            con.print(f"  [ok]✓[/] shell hook already in [faint]{rc}[/]")
        elif ask(f"  Add live recording + [accent]Ctrl+G[/] picker to {rc}?"):
            hooks.append_line(rc, line)
            con.print(f"  [ok]✓[/] added to [faint]{rc}[/] — open a new terminal to activate")
        else:
            con.print(f"  add this to [faint]{rc}[/]:  [accent]{line}[/]")
    else:
        con.print(f"  [warn]unknown shell {shell!r}[/] — supported: zsh, bash, fish, powershell "
                  "([accent]rk hook <shell>[/])")


def _models_table(con, cfg: C.Config) -> None:
    from rich.table import Table
    from rekollect import models as M

    speed = {"lite": "instant", "balanced": "fast", "quality": "slower"}
    t = Table(box=None, padding=(0, 1), header_style="muted")
    for col in ("", "tier", "model", "params", "dim", "disk", "speed"):
        t.add_column(col, no_wrap=True)
    active = _label(cfg)
    for name, s in M.REGISTRY.items():
        mark = "[accent]●[/]" if name == active else "[faint]○[/]"
        have = "[ok]✓ local[/]" if M.is_downloaded(s) else f"~{s.size_mb} MB"
        short = s.repo.split("/")[-1].replace("-ONNX", "").replace("-english", "")
        t.add_row(mark, f"[bold]{name}[/]", f"[faint]{short}[/]", s.params.split()[0], str(s.dim), have,
                  f"[muted]{speed.get(name, '?')}[/]")
    con.print(t)
    con.print()
    notes = Table.grid(padding=(0, 2))
    notes.add_column(no_wrap=True, style="bold")
    notes.add_column(style="faint")
    for name, s in M.REGISTRY.items():
        notes.add_row(f" {name}", s.blurb)
    con.print(notes)
    con.print()


def cmd_models(argv, cfg) -> int:
    con = _console(cfg)
    con.print(f"\n[accent]◆ rekollect[/] [muted]models · active: {_label(cfg)}[/]\n")
    _models_table(con, cfg)
    con.print("  [faint]switch with[/] [accent]rk model <tier>[/]  [faint]· or set[/] [accent]model.tier = \"custom\"[/] "
              "[faint]and point model.custom.repo at any ONNX embedding repo[/]\n")
    return 0


def cmd_model(argv, cfg) -> int:
    from rekollect import models as M
    from rekollect.indexer import ensure_model_matches
    from rekollect.store import Store, index_lock

    if not argv or argv[0] not in list(M.REGISTRY) + ["custom"]:
        print("usage: rk model <lite|balanced|quality|custom>", file=sys.stderr)
        return 2
    cfg["model"]["tier"] = argv[0]
    C.save(cfg)
    con = _console(cfg)
    spec = M.spec_for(cfg)
    _download_with_progress(spec, con)
    with index_lock(blocking=True):
        store = Store()
        ensure_model_matches(store, cfg)
        emb = M.load_embedder(cfg)
        _embed_with_progress(store, emb, con, label=f"Re-embedding with {argv[0]}")
    con.print(f"  [ok]✓[/] now using [accent]{argv[0]}[/]")
    return 0


def cmd_hook(argv, cfg) -> int:
    from rekollect import hooks
    from rekollect.history import detect_shell
    shell = argv[0] if argv else detect_shell()
    sys.stdout.write(hooks.script(shell, cfg))
    return 0


def _after_key_change(con, cfg) -> None:
    from rekollect import vscode
    for name, path, status in vscode.sync_all(cfg.get_path("ui.key", "^G")):
        con.print(f"[ok]✓[/] {name} keybinding {status} [faint]({path})[/]")
    con.print("[faint]  open a new terminal for the new key to apply[/]")
    if cfg.get_path("ui.key") == "^K":
        con.print("[warn]![/] [faint]Ctrl+K starts VS Code chords (Ctrl+K Ctrl+…); ^G is a safer choice[/]")


def cmd_config(argv, cfg) -> int:
    con = _console(cfg)
    sub = argv[0] if argv else "show"
    C.write_default_if_missing()
    if sub == "path":
        print(C.config_path())
    elif sub == "edit":
        import shlex
        import subprocess
        default = "notepad" if os.name == "nt" else "nano"
        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or default
        parts = editor.split() if os.name == "nt" else shlex.split(editor)
        return subprocess.call(parts + [str(C.config_path())])
    elif sub == "set":
        if len(argv) != 3:
            con.print("usage: [accent]rk config set[/] <dotted.key> <value>   e.g. search.weights.semantic 0.6")
            return 2
        try:
            v = C.set_path(cfg, argv[1], argv[2])
        except KeyError:
            con.print(f"[bad]✗[/] unknown key [bold]{argv[1]}[/]  [faint](see rk config)[/]")
            return 2
        except ValueError as e:
            con.print(f"[bad]✗[/] bad value for {argv[1]}: {e}")
            return 2
        C.save(cfg)
        con.print(f"[ok]✓[/] {argv[1]} = [accent]{v!r}[/]")
        if argv[1].startswith("model.tier"):
            con.print("[faint]  run[/] [accent]rk index[/] [faint]to re-embed with the new model[/]")
        if argv[1] == "ui.key":
            _after_key_change(con, cfg)
        elif argv[1].startswith("index."):
            con.print("[faint]  open a new terminal for hook changes to apply[/]")
    elif sub == "unset":
        if len(argv) != 2:
            con.print("usage: [accent]rk config unset[/] <dotted.key>   e.g. ui.key")
            return 2
        try:
            v = C.unset_path(cfg, argv[1])
        except KeyError:
            con.print(f"[bad]✗[/] unknown key [bold]{argv[1]}[/]")
            return 2
        C.save(cfg)
        con.print(f"[ok]✓[/] {argv[1]} back to default [accent]{v!r}[/]")
        if argv[1] == "ui.key":
            _after_key_change(con, cfg)
    elif sub == "reset":
        C.save(C.DEFAULTS)
        con.print("[ok]✓[/] config reset to defaults")
        _after_key_change(con, C.load())
    else:
        from rich.table import Table
        set_by_you = dict(C.flatten(C.overrides()))
        t = Table(box=None, padding=(0, 2), header_style="muted")
        t.add_column("key", no_wrap=True)
        t.add_column("value", overflow="fold")
        t.add_column("", no_wrap=True)
        for k, v in C.flatten(cfg):
            mine = k in set_by_you
            t.add_row(f"[bold]{k}[/]" if mine else f"[muted]{k}[/]",
                      f"[accent]{v!r}[/]" if mine else f"[faint]{v!r}[/]",
                      "[accent]set by you[/]" if mine else "")
        con.print(f"\n[accent]◆ rekollect config[/]  [faint]{C.config_path()}[/]\n")
        con.print(t)
        from rekollect.learn import effective_weights
        w, strength, source = effective_weights(cfg)
        con.print(f"\n  [muted]ranking in use:[/] {source}  [faint]"
                  + " ".join(f"{k} {v:.2f}" for k, v in w.items()) + f" · your history {strength:.2f}[/]")
        con.print("\n  [faint]change:[/] [accent]rk config set <key> <value>[/]   [faint]revert:[/] "
                  "[accent]rk config unset <key>[/]   [faint]all:[/] [accent]rk config reset[/]\n")
    return 0


def cmd_vscode(argv, cfg) -> int:
    from rekollect import vscode
    con = _console(cfg)
    editors = vscode.installed_editors()
    if not editors:
        con.print(f"[warn]no VS Code-family editor settings found[/] [faint](looked in {vscode.config_root()})[/]")
        return 1
    if argv and argv[0] in ("--remove", "remove"):
        for name, path in editors:
            done = vscode.remove(path)
            con.print(f"  {'[ok]✓[/] removed from' if done else '[faint]·[/] nothing in'} {name} [faint]({path})[/]")
        return 0
    key = cfg.get_path("ui.key", "^G")
    vk = vscode.key_label(key)
    rc = 0
    for name, path in editors:
        try:
            status = vscode.install(path, key)
            con.print(f"  [ok]✓[/] {name}: {vk} in the terminal now goes to your shell [faint]({status}, {path})[/]")
        except ValueError as e:
            rc = 1
            con.print(f"  [bad]✗[/] {name}: couldn't edit {path} safely ({e}). Add this inside the [ ]:")
            con.print(vscode.block(key))
    con.print("  [faint]VS Code reloads keybindings automatically; no restart needed.[/]")
    return rc


def cmd_stats(argv, cfg) -> int:
    from rekollect import stats
    from rekollect.store import Store
    stats.render(Store(), _console(cfg), _label(cfg))
    return 0


def cmd_forget(argv, cfg) -> int:
    from rich.prompt import Confirm
    from rekollect import ui
    from rekollect.store import Store

    con = _console(cfg)
    if not argv:
        con.print("usage: [accent]rk forget[/] <regex>    e.g. rk forget 'mysql -p'")
        return 2
    pat = re.compile(" ".join(argv))
    store = Store()
    rows = [(i, t) for i, t in store.db.execute("SELECT id, text FROM commands") if pat.search(t)]
    if not rows:
        con.print("[muted]nothing matches[/]")
        return 0
    for _, t in rows[:15]:
        con.print("  ", ui.highlight(t, max_len=con.width - 4))
    if len(rows) > 15:
        con.print(f"  [faint]… and {len(rows) - 15} more[/]")
    if Confirm.ask(f"Forget these {len(rows)} commands?", default=False, console=con):
        store.forget([i for i, _ in rows])
        store.commit()
        con.print(f"[ok]✓[/] forgotten {len(rows)}")
    return 0


def cmd_pause(argv, cfg, on: bool) -> int:
    C.ensure_dirs()
    con = _console(cfg)
    if on:
        C.paused_flag().touch()
        con.print("[warn]⏸[/]  recording paused (incognito). [faint]rk resume to continue[/]")
    else:
        C.paused_flag().unlink(missing_ok=True)
        con.print("[ok]▶[/]  recording resumed")
    return 0


def cmd_doctor(argv, cfg) -> int:
    from rekollect import models as M, ui
    from rekollect.history import history_files
    from rekollect.store import Store

    con = _console(cfg)
    ok = lambda b: "[ok]✓[/]" if b else "[bad]✗[/]"  # noqa: E731
    spec = M.spec_for(cfg)
    con.print(f"\n[accent]◆ rekollect doctor[/]  [faint]v{__version__} · python {sys.version.split()[0]}[/]\n")
    con.print(f"  {ok(C.config_path().exists())} config      [faint]{C.config_path()}[/]")
    store = Store()
    u, e, r = store.counts()
    con.print(f"  {ok(r > 0)} index       [faint]{C.db_path()}[/]  {r:,} runs · {u:,} unique · {e:,} embedded · "
              f"{ui.human_bytes(store.size_bytes())}")
    have = M.is_downloaded(spec)
    con.print(f"  {ok(have)} model       {spec.tier} [faint]{spec.repo}[/] {ui.human_bytes(M.disk_usage(spec)) if have else '(not downloaded)'}")
    hooked = os.environ.get("REKOLLECT_HOOKED")
    con.print(f"  {ok(bool(hooked))} shell hook  {hooked or '[faint]not active in this shell — see rk init[/]'}")
    from rekollect import vscode
    for name, path in vscode.installed_editors():
        if vscode.is_current(path):
            con.print(f"  [ok]✓[/] {name:<11} picker key passes through to the terminal")
        elif vscode.has_block(path):
            con.print(f"  [warn]![/] {name:<11} keybinding is from an older version; run [accent]rk vscode[/] "
                      "[faint](adds PowerShell and macOS Cmd keys)[/]")
        else:
            con.print(f"  {ok(False)} {name:<11} [faint]may grab the picker key in its terminal; run[/] "
                      "[accent]rk vscode[/]")
    from rekollect.learn import effective_weights
    picks = store.db.execute("SELECT COUNT(*) FROM feedback WHERE kind = 'pick'").fetchone()[0]
    con.print(f"  [ok]✓[/] learning    {picks} picks remembered · ranking: {effective_weights(cfg)[2]}")
    con.print("  [ok]✓[/] agents      " + ("on, " + str(cfg.get_path('agents.scope')) + " scope, read-only"
                                            if cfg.get_path("agents.enabled", False) else "off [faint](rk agents)[/]"))
    sp = C.spool_path()
    pend = sp.stat().st_size if sp.exists() else 0
    con.print(f"  [ok]✓[/] spool       {ui.human_bytes(pend)} waiting to be indexed")
    if C.paused_flag().exists():
        con.print("  [warn]⏸[/] recording is paused  [faint](rk resume)[/]")
    for kind, path in history_files():
        con.print(f"  [ok]✓[/] history     [faint]{path} ({kind})[/]")
    if have:
        t0 = time.perf_counter()
        emb = M.load_embedder(cfg)
        t1 = time.perf_counter()
        emb.embed(["warm up"], query=True)
        t2 = time.perf_counter()
        emb.embed(["list docker containers that exited"], query=True)
        t3 = time.perf_counter()
        con.print(f"  [ok]✓[/] latency     load {1000 * (t1 - t0):.0f} ms · first {1000 * (t2 - t1):.1f} ms · "
                  f"query {1000 * (t3 - t2):.1f} ms")
    con.print()
    return 0


# ── main ─────────────────────────────────────────────────────────────────────

def _learn():
    from rekollect import cli_learn
    return cli_learn


def _flows():
    from rekollect import cli_flows
    return cli_flows


def cmd_keys(argv, cfg) -> int:
    """Show exactly what arrives when you press keys (debugging terminal/VS Code bindings)."""
    from rekollect.picker import WINDOWS, PosixTty, WinTty
    con = _console(cfg)
    con.print("[muted]Press keys to see what reaches programs in this terminal. Press q to stop.[/]")
    con.print("[faint]Ctrl+G should show \\x07; in a VS Code PowerShell terminal with rk vscode it shows "
              "\\x1b[24~r.[/]")
    tty = WinTty() if WINDOWS else PosixTty()
    try:
        while True:
            k = tty.key()
            con.print(f"  {k!r}")
            if k == "q":
                break
    finally:
        tty.close()
    return 0


def _utf8_stdio() -> None:
    if os.name != "nt":
        return
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv: Optional[list[str]] = None) -> int:
    _utf8_stdio()
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] in ("-V", "--version"):
        print(f"rekollect {__version__}")
        return 0
    if args and args[0] == "hook":  # hot path: runs on every shell start, skip config validation noise
        return cmd_hook(args[1:], C.load())
    cfg = C.load()
    if args and args[0] in ("-h", "--help", "help"):
        _print_help(cfg)
        return 0
    try:
        if not args:
            if sys.stdin.isatty() and sys.stdout.isatty():
                return cmd_pick([], cfg)
            _print_help(cfg)
            return 0
        cmd, rest = args[0], args[1:]
        if cmd not in SUBCOMMANDS:
            return cmd_search(args, cfg)
        return {
            "init": lambda: cmd_init(rest, cfg),
            "search": lambda: cmd_search(rest, cfg),
            "s": lambda: cmd_search(rest, cfg),
            "pick": lambda: cmd_pick(rest, cfg),
            "index": lambda: cmd_index(rest, cfg),
            "models": lambda: cmd_models(rest, cfg),
            "model": lambda: cmd_model(rest, cfg),
            "config": lambda: cmd_config(rest, cfg),
            "stats": lambda: cmd_stats(rest, cfg),
            "forget": lambda: cmd_forget(rest, cfg),
            "pause": lambda: cmd_pause(rest, cfg, True),
            "resume": lambda: cmd_pause(rest, cfg, False),
            "doctor": lambda: cmd_doctor(rest, cfg),
            "vscode": lambda: cmd_vscode(rest, cfg),
            "flow": lambda: _flows().cmd_flow(rest, cfg),
            "recipe": lambda: _flows().cmd_recipe(rest, cfg),
            "recipes": lambda: _flows().cmd_recipe(["list"], cfg),
            "tpl": lambda: _flows().cmd_tpl(rest, cfg),
            "keys": lambda: cmd_keys(rest, cfg),
            "tune": lambda: _learn().cmd_tune(rest, cfg),
            "agents": lambda: __import__("rekollect.cli_agents", fromlist=["x"]).cmd_agents(rest, cfg),
            "mcp": lambda: __import__("rekollect.cli_agents", fromlist=["x"]).cmd_mcp(rest, cfg),
            "eval": lambda: __import__("rekollect.cli_eval", fromlist=["x"]).cmd_eval(rest, cfg),
            "mark": lambda: _learn().cmd_mark(rest, cfg),
            "marks": lambda: _learn().cmd_marks(rest, cfg),
            "unmark": lambda: _learn().cmd_unmark(rest, cfg),
            "why": lambda: cmd_search(rest + ["--why"], cfg),
        }[cmd]()
    except KeyboardInterrupt:
        return 130