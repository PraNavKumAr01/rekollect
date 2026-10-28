"""`rk agents …` (access control, log, setup) and `rk mcp` (the server)."""

from __future__ import annotations

import argparse
import json
import shutil

from rich.table import Table
from rich.text import Text

from rekollect import config as C
from rekollect import ui


def _con(cfg):
    from rekollect.cli import _console
    return _console(cfg)


def cmd_mcp(argv: list[str], cfg) -> int:
    from rekollect.mcp_server import serve
    p = argparse.ArgumentParser(prog="rk mcp", add_help=False)
    p.add_argument("--project", help="project folder to scope results to (default: where the server starts)")
    a = p.parse_args(argv)
    return serve(cfg, a.project)


def _requests() -> list[dict]:
    from rekollect.mcp_server import log_path
    try:
        lines = log_path().read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except ValueError:
            continue
    return out


def cmd_agents(argv: list[str], cfg) -> int:
    con = _con(cfg)
    sub = argv[0] if argv else "status"
    rest = argv[1:]
    if sub == "enable":
        p = argparse.ArgumentParser(prog="rk agents enable", add_help=False)
        p.add_argument("--scope", choices=["project", "all"], default=None)
        a = p.parse_args(rest)
        cfg["agents"]["enabled"] = True
        if a.scope:
            cfg["agents"]["scope"] = a.scope
        C.save(cfg)
        con.print(f"[ok]✓[/] coding assistants can now search your command history "
                  f"[faint](read-only · scope: {cfg['agents']['scope']})[/]")
        con.print("  [faint]connect one:[/] [accent]rk agents setup[/]   [faint]see requests:[/] "
                  "[accent]rk agents log[/]   [faint]turn off:[/] [accent]rk agents disable[/]")
        return 0
    if sub == "disable":
        cfg["agents"]["enabled"] = False
        C.save(cfg)
        con.print("[ok]✓[/] agent access off: the server stays installed but every request is refused")
        return 0
    if sub == "log":
        p = argparse.ArgumentParser(prog="rk agents log", add_help=False)
        p.add_argument("-n", type=int, default=25)
        a = p.parse_args(rest)
        reqs = _requests()[-a.n:]
        if not reqs:
            con.print("[muted]No agent requests yet.[/]")
            return 0
        t = Table(box=None, padding=(0, 2), header_style="muted")
        t.add_column("when", style="muted", no_wrap=True)
        t.add_column("client", style="faint", no_wrap=True)
        t.add_column("asked", overflow="fold")
        t.add_column("got", overflow="fold")
        for r in reqs:
            args = r.get("args", {})
            asked = f"{r['tool']}  " + " ".join(f"{k}={v}" for k, v in args.items())
            got = Text(r["error"], "bad") if r.get("error") else Text(
                f"{len(r.get('returned', []))} command(s)" + (": " + r["returned"][0][:60] if r.get("returned") else ""),
                "faint")
            t.add_row(ui.since(r["ts"]), r.get("client", ""), asked, got)
        con.print()
        con.print(t)
        con.print()
        return 0
    if sub == "setup":
        exe = shutil.which("rekollect") or "rekollect"
        con.print("\n[accent]◆ connect a coding assistant[/]  [faint](stdio MCP server · read-only)[/]\n")
        con.print("[muted]Claude Code[/]  [faint](run inside your project)[/]")
        con.print(f"  claude mcp add rekollect -- {exe} mcp\n")
        con.print("[muted]Cursor[/]  [faint].cursor/mcp.json in your project[/]")
        con.print("  " + json.dumps({"mcpServers": {"rekollect": {"command": exe, "args": ["mcp"]}}}) + "\n")
        con.print("[muted]VS Code[/]  [faint].vscode/mcp.json in your project[/]")
        con.print("  " + json.dumps({"servers": {"rekollect": {"type": "stdio", "command": exe, "args": ["mcp"]}}}) + "\n")
        con.print("[muted]Apps that don't start in a project folder[/]  [faint](e.g. desktop chat apps)[/]")
        con.print("  " + json.dumps({"mcpServers": {"rekollect": {"command": exe,
                                                                  "args": ["mcp", "--project", "/path/to/project"]}}}))
        con.print("\n  [faint]Config formats change between versions; check your tool's MCP docs if one doesn't load.[/]")
        if not cfg.get_path("agents.enabled", False):
            con.print("  [warn]![/] access is still off. Turn it on with [accent]rk agents enable[/]\n")
        return 0
    # status
    on = cfg.get_path("agents.enabled", False)
    reqs = _requests()
    con.print()
    con.print(Text.assemble(("◆ agent access  ", "accent"), ("on" if on else "off", "ok" if on else "muted")))
    con.print(f"  scope       [bold]{cfg.get_path('agents.scope', 'project')}[/]  "
              "[faint](project: only commands run inside the assistant's project · all: everything)[/]")
    con.print("  powers      [bold]search and read only[/] [faint]· no tool can run or change anything[/]")
    ex = cfg.get_path("agents.exclude", [])
    con.print(f"  hidden      {len(ex)} pattern(s) in agents.exclude [faint]+ secrets redacted[/]")
    con.print(f"  requests    {len(reqs)}" + (f" [faint]· last {ui.since(reqs[-1]['ts'])} from "
                                                f"{reqs[-1].get('client', '?')}[/]" if reqs else ""))
    con.print("\n  [accent]rk agents enable[/] [faint][--scope all][/] · [accent]disable[/] · "
              "[accent]log[/] · [accent]setup[/]\n")
    return 0