"""`rk flow`, `rk recipe …` and `rk tpl`."""

from __future__ import annotations

import argparse
import os
import sys
import time

from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from rekollect import flows as F
from rekollect import ui


def _con(cfg):
    from rekollect.cli import _console
    return _console(cfg)


def _render_flow(con, n: int, f: F.Flow, width: int) -> None:
    where = ui.short_path(f.cwd, 40) if f.cwd else "unknown directory"
    meta = [where, ui.since(f.end) if f.end else "", f"took {F.flow_time(f)}"]
    if f.times > 1:
        meta.append(f"done {f.times}×")
    con.print(Text.assemble((f" {n} ", "bold black on #b48ead"), "  ", (" · ".join(m for m in meta if m), "muted")))
    t = Table(box=None, show_header=False, padding=(0, 1), pad_edge=False)
    t.add_column(justify="right", style="faint", no_wrap=True)
    t.add_column(no_wrap=True)
    t.add_column(no_wrap=True, max_width=max(20, width - 40), overflow="ellipsis")
    t.add_column(no_wrap=True)
    lw = max(len(s.label) for s in f.steps)
    for i, s in enumerate(f.steps, 1):
        strong = s.relevance >= 0.35
        t.add_row(f"   {i}", Text(s.label.ljust(min(lw, 28)), "accent" if strong else "muted"),
                  ui.highlight(s.text, max_len=max(20, width - 40)), Text("●" if strong else "", "accent.dim"))
    con.print(t)
    if f.hidden_failures:
        con.print(f"     [faint]· {f.hidden_failures} failed attempt{'s' if f.hidden_failures > 1 else ''} hidden[/]")
    con.print()


def cmd_flow(argv: list[str], cfg) -> int:
    from rekollect.cli import _engine
    p = argparse.ArgumentParser(prog="rk flow", add_help=False)
    p.add_argument("query", nargs="*")
    p.add_argument("-n", type=int, default=3)
    p.add_argument("--save", metavar="NAME")
    p.add_argument("--pick", type=int, default=1, help="which result --save uses")
    p.add_argument("--literal", action="store_true", help="don't turn values into {{placeholders}}")
    a = p.parse_args(argv)
    q = " ".join(a.query).strip()
    con = _con(cfg)
    if not q:
        con.print('usage: [accent]rk flow[/] [bold]"what did I do to publish this package"[/]  '
                  "[faint](-n 5, --save NAME)[/]")
        return 2
    store, corpus, ranker = _engine(cfg, con)
    t0 = time.perf_counter()
    flows = F.find_flows(store, corpus, ranker, q, k=a.n)
    ms = (time.perf_counter() - t0) * 1000
    saved = F.match_recipes(q, F.list_recipes())

    con.print(Text.assemble((ui.LOGO, "accent"), ("  flow  ", "muted"), (f'"{q}"', "bold"),
                            (f"  ·  {len(flows)} found  ·  {ms:.0f} ms", "faint")))
    con.print()
    if saved:
        con.print("[muted]saved recipes[/]")
        for r in saved[:3]:
            con.print(f"  [accent]{r.name}[/]  [faint]{len(r.steps)} steps · {r.description}[/]   "
                      f"[faint]→[/] rk recipe run {r.name}")
        con.print()
    if not flows:
        con.print("  [faint]No matching workflow yet. Workflows are rebuilt from commands recorded with "
                  "timestamps, so they appear once the shell hook has seen you do the task.[/]")
        return 1 if not saved else 0
    for n, f in enumerate(flows, 1):
        _render_flow(con, n, f, con.width)

    name = a.save
    pick = a.pick
    if not name and sys.stdin.isatty() and sys.stdout.isatty():
        ans = Prompt.ask("  [muted]Save one as a recipe?[/] [faint]e.g. '1 release-python-package', "
                         "Enter to skip[/]", default="", show_default=False, console=con).strip()
        if ans:
            parts = ans.split(None, 1)
            if len(parts) == 2 and parts[0].isdigit():
                pick, name = int(parts[0]), parts[1].strip()
            else:
                name = ans
    if not name:
        con.print("  [faint]save one with[/] [accent]rk flow \"…\" --save <name>[/] [faint](--pick N)[/]")
        return 0
    if not F.valid_name(name):
        con.print(f"[bad]✗[/] recipe names are lowercase letters, digits, . _ - (got {name!r})")
        return 2
    if not 1 <= pick <= len(flows):
        con.print(f"[bad]✗[/] --pick must be 1–{len(flows)}")
        return 2
    if F.recipe_path(name).exists() and not Confirm.ask(f"  overwrite recipe {name}?", default=False,
                                                         console=con):
        return 1
    r = F.recipe_from_flow(name, flows[pick - 1], q, parametrize=not a.literal)
    path = F.save_recipe(r)
    _print_recipe(con, r)
    con.print(f"  [ok]✓[/] saved [accent]{name}[/] [faint]({path})[/]")
    con.print(f"  [faint]run it with[/] [accent]rk recipe run {name}[/][faint], edit with[/] "
              f"[accent]rk recipe edit {name}[/]")
    return 0


def _print_recipe(con, r: F.Recipe) -> None:
    con.print()
    con.print(Text.assemble(("◆ ", "accent"), (r.name, "bold"), (f"  {r.description}" if r.description else "", "muted")))
    if r.vars:
        vt = Table(box=None, show_header=False, padding=(0, 2), pad_edge=False)
        vt.add_column(style="accent", no_wrap=True)
        vt.add_column(style="muted")
        for k, v in r.vars.items():
            vt.add_row(f"   {{{{{k}}}}}", v)
        con.print("  [faint]values (asked when you run it)[/]")
        con.print(vt)
    t = Table(box=None, show_header=False, padding=(0, 1), pad_edge=False)
    t.add_column(justify="right", style="faint")
    t.add_column(style="muted", no_wrap=True)
    t.add_column(overflow="fold")
    for i, s in enumerate(r.steps, 1):
        t.add_row(f"   {i}", s.label, _stored_text(s.run))
    con.print(t)
    con.print()


def _stored_text(run: str) -> Text:
    import re
    t = Text()
    pos = 0
    for m in re.finditer(r"\{\{([a-z][a-z0-9_]*)\}\}", run):
        t.append(ui.highlight(run[pos:m.start()]) if m.start() > pos else "")
        t.append("{" + m.group(1) + "}", "slot")
        pos = m.end()
    t.append(ui.highlight(run[pos:]))
    return t


def _edit_line(con, prompt: str, current: str) -> str:
    """Edit a command with the old text pre-filled where readline exists."""
    try:
        import readline

        def hook():
            readline.insert_text(current)
            readline.redisplay()
        readline.set_pre_input_hook(hook)
        try:
            return input(prompt).strip() or current
        finally:
            readline.set_pre_input_hook(None)
    except ImportError:  # Windows without pyreadline
        return Prompt.ask(prompt, default=current, console=con)


def _ask_values(con, r: F.Recipe, overrides: dict, yes: bool) -> dict:
    values = dict(r.vars)
    values.update(overrides)
    for name in F.missing_vars(r):
        values.setdefault(name, "")
    if not values or yes:
        return values
    con.print("  [muted]values[/] [faint](Enter keeps the recorded value)[/]")
    for k in list(values):
        if k in overrides:
            con.print(f"    [accent]{k}[/] = {values[k]}  [faint](from --var)[/]")
            continue
        values[k] = Prompt.ask(f"    [accent]{k}[/]", default=values[k], console=con)
    con.print()
    return values


def recipe_run(argv: list[str], cfg) -> int:
    p = argparse.ArgumentParser(prog="rk recipe run", add_help=False)
    p.add_argument("name")
    p.add_argument("--yes", "-y", action="store_true", help="don't ask before each step")
    p.add_argument("--step", type=int, help="run only this step")
    p.add_argument("--from", dest="start", type=int, default=1, help="start at this step")
    p.add_argument("--var", action="append", default=[], metavar="NAME=VALUE")
    p.add_argument("--dry-run", action="store_true", help="print the filled-in commands only")
    a = p.parse_args(argv)
    con = _con(cfg)
    if not F.recipe_path(a.name).exists():
        con.print(f"[bad]✗[/] no recipe [bold]{a.name}[/]  [faint](rk recipe list)[/]")
        return 2
    r = F.load_recipe(a.name)
    overrides = {}
    for kv in a.var:
        k, sep, v = kv.partition("=")
        if not sep:
            con.print(f"[bad]✗[/] --var expects NAME=VALUE, got {kv!r}")
            return 2
        overrides[k] = v
    con.print(Text.assemble(("◆ ", "accent"), (r.name, "bold"), (f"  ·  {len(r.steps)} steps", "faint"),
                            (f"  ·  {r.description}" if r.description else "", "muted")))
    con.print()
    values = _ask_values(con, r, overrides, a.yes or a.dry_run)
    commands = F.render_steps(r, values)
    if a.dry_run:
        for i, (s, c) in enumerate(zip(r.steps, commands), 1):
            con.print(f"  [faint]{i}[/]  [muted]{s.label:<24}[/] ", ui.highlight(c))
        return 0

    from rekollect.history import detect_shell
    shell = r.shell if r.shell in ("zsh", "bash", "fish", "powershell") else detect_shell()
    state = F.RunState(cwd=os.getcwd(), env=dict(os.environ))
    indices = [a.step] if a.step else list(range(a.start, len(r.steps) + 1))
    for i in indices:
        if not 1 <= i <= len(r.steps):
            con.print(f"[bad]✗[/] step {i} doesn't exist (1–{len(r.steps)})")
            return 2
    for i in indices:
        step, cmd = r.steps[i - 1], commands[i - 1]
        con.print(Text.assemble((f" {i}/{len(r.steps)} ", "bold black on #b48ead"), "  ", (step.label, "bold"),
                                (f"   {ui.short_path(state.cwd, 40)}", "faint")))
        con.print(Text("   ❯ ", "accent") + ui.highlight(cmd))
        ask = not a.yes
        while True:
            if ask:
                choice = Prompt.ask("   [faint]⏎ run · s skip · e edit · q quit[/]", default="",
                                    show_default=False, console=con).strip().lower()
                if choice == "q":
                    con.print("   [muted]stopped[/]")
                    return 1
                if choice == "s":
                    con.print("   [faint]skipped[/]\n")
                    break
                if choice == "e":
                    cmd = _edit_line(con, "   ❯ ", cmd)
                    con.print(Text("   ❯ ", "accent") + ui.highlight(cmd))
                    continue
            code = _execute(con, cmd, shell, state)
            if code == 0:
                break
            if a.yes or code == 130:
                return code
            choice = Prompt.ask("   [faint]r retry · e edit · s skip · q quit[/]", choices=["r", "e", "s", "q"],
                                default="q", show_choices=False, console=con)
            if choice == "q":
                return code
            if choice == "s":
                con.print("   [faint]skipped[/]\n")
                break
            if choice == "e":
                cmd = _edit_line(con, "   ❯ ", cmd)
                con.print(Text("   ❯ ", "accent") + ui.highlight(cmd))
            ask = False  # retry / edited command runs straight away
    con.print(f"[ok]✓[/] [bold]{r.name}[/] finished")
    return 0


def _execute(con, cmd: str, shell: str, state: F.RunState) -> int:
    try:
        msg = F.builtin_step(cmd, state)
    except FileNotFoundError as e:
        con.print(f"   [bad]✗[/] {e}")
        return 1
    if msg:
        con.print(f"   [ok]✓[/] [faint]{msg}[/]\n")
        return 0
    t0 = time.perf_counter()
    con.rule(style="grey23")
    try:
        code = F.run_step(cmd, shell, state)
    except KeyboardInterrupt:
        code = 130
    dt = time.perf_counter() - t0
    if code == 0:
        con.print(f"   [ok]✓[/] [faint]done in {dt:.1f}s[/]\n")
    else:
        con.print(f"   [bad]✗[/] exit {code} [faint]after {dt:.1f}s[/]")
    return code


def cmd_recipe(argv: list[str], cfg) -> int:
    con = _con(cfg)
    sub = argv[0] if argv else "list"
    rest = argv[1:]
    if sub in ("list", "ls"):
        rs = F.list_recipes()
        if not rs:
            con.print("[muted]No recipes yet.[/] Find a workflow with [accent]rk flow \"…\"[/] and save it, or "
                      "save what you just did with [accent]rk recipe save <name> --last 5[/].")
            return 0
        t = Table(box=None, padding=(0, 2), header_style="muted")
        t.add_column("recipe", style="accent", no_wrap=True)
        t.add_column("steps", justify="right", style="muted")
        t.add_column("values", style="faint")
        t.add_column("description", style="faint", overflow="fold")
        for r in rs:
            t.add_row(r.name, str(len(r.steps)), ", ".join(r.vars) or "—", r.description)
        con.print()
        con.print(t)
        con.print(f"\n  [faint]{F.recipes_dir()}[/]\n")
        return 0
    if sub == "run":
        return recipe_run(rest, cfg)
    if sub in ("show", "edit", "rm", "delete") and not rest:
        con.print(f"usage: [accent]rk recipe {sub}[/] <name>")
        return 2
    if sub == "show":
        if not F.recipe_path(rest[0]).exists():
            con.print(f"[bad]✗[/] no recipe [bold]{rest[0]}[/]")
            return 2
        _print_recipe(con, F.load_recipe(rest[0]))
        return 0
    if sub == "edit":
        import subprocess
        path = F.recipe_path(rest[0])
        if not path.exists():
            con.print(f"[bad]✗[/] no recipe [bold]{rest[0]}[/]")
            return 2
        default = "notepad" if os.name == "nt" else "nano"
        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or default
        return subprocess.call(editor.split() + [str(path)])
    if sub in ("rm", "delete"):
        path = F.recipe_path(rest[0])
        if not path.exists():
            con.print(f"[bad]✗[/] no recipe [bold]{rest[0]}[/]")
            return 2
        if Confirm.ask(f"Delete recipe {rest[0]}?", default=False, console=con):
            path.unlink()
            con.print(f"[ok]✓[/] deleted {rest[0]}")
        return 0
    if sub == "save":
        return recipe_save(rest, cfg)
    con.print("usage: [accent]rk recipe[/] list | show <name> | run <name> | edit <name> | rm <name> | "
              "save <name> --last N")
    return 2


def recipe_save(argv: list[str], cfg) -> int:
    """Save the last N commands of this terminal session as a recipe."""
    from rekollect.indexer import drain_spool, ingest
    from rekollect.store import Store
    p = argparse.ArgumentParser(prog="rk recipe save", add_help=False)
    p.add_argument("name")
    p.add_argument("--last", type=int, default=5)
    p.add_argument("-d", "--description", default="")
    p.add_argument("--literal", action="store_true")
    a = p.parse_args(argv)
    con = _con(cfg)
    if not F.valid_name(a.name):
        con.print(f"[bad]✗[/] recipe names are lowercase letters, digits, . _ - (got {a.name!r})")
        return 2
    store = Store()
    ingest(store, drain_spool(), cfg)  # include what was just run
    sid = os.environ.get("REKOLLECT_SESSION", "")
    rows = store.session_runs(sid, a.last + 20) if sid else store.timeline()[-(a.last + 20):]
    rows = [r for r in rows if not F.NOISE.match(r[2]) and r[3] in (None, 0)][-a.last:]
    if not rows:
        con.print("[muted]Nothing recorded in this terminal yet.[/] [faint](is the shell hook active? rk doctor)[/]")
        return 1
    r = F.recipe_from_steps(a.name, [(F.label_for(x[2]), x[2], x[5]) for x in rows], a.description,
                            parametrize=not a.literal)
    _print_recipe(con, r)
    if sys.stdin.isatty() and not Confirm.ask("  Save this recipe?", default=True, console=con):
        return 1
    path = F.save_recipe(r)
    con.print(f"  [ok]✓[/] saved [accent]{a.name}[/] [faint]({path})[/]")
    return 0


def cmd_tpl(argv: list[str], cfg) -> int:
    from rekollect.templates import templatize
    con = _con(cfg)
    if not argv:
        con.print("usage: [accent]rk tpl[/] [bold]'<command>'[/]   e.g. rk tpl 'ssh deploy@10.0.0.12'")
        return 2
    t = templatize(" ".join(argv))
    con.print(Text("  template  ", "faint") + ui.template_text(t))
    con.print(Text("  original  ", "faint") + ui.highlight(t.original))
    if t.slots:
        tb = Table(box=None, show_header=False, padding=(0, 2))
        tb.add_column(style="accent")
        tb.add_column()
        tb.add_column(style="faint")
        for s in t.slots:
            tb.add_row(f"  {{{s.name}}}", s.default, s.kind)
        con.print(tb)
    else:
        con.print("  [faint]no variable parts found[/]")
    return 0