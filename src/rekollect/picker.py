"""Live, search-as-you-type picker.

POSIX: reads keys from and draws on /dev/tty, so stdout stays clean for the shell
widget (`BUFFER=$(rekollect pick)`).
Windows: reads keys with msvcrt and draws on the console (CONOUT$) in the
alternate screen buffer, so PSReadLine's prompt is untouched when we leave."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from typing import Optional

import numpy as np
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

from rekollect import ui
from rekollect.ranker import Ranker
from rekollect.store import Corpus, Store
from rekollect.templates import Template, templatize

# most important first; the footer shows as many as fit
KEYS_HELP = [("⏎", "insert (doesn't run)"), ("^Y", "copy"), ("^T", "template"), ("^E", "why"),
             ("^B", "star"), ("^X", "not this"), ("^S", "project"), ("^O", "only ✓"), ("^D", "forget"),
             ("esc", "close")]
FORM_HELP = [("⏎", "next / done"), ("↑↓", "field"), ("^U", "clear"), ("esc", "back")]


class FillForm:
    """Fill a template's placeholders. Each field starts at the value from history;
    typing replaces it, Backspace edits it."""

    def __init__(self, tpl: Template):
        self.tpl = tpl
        self.values = {s.name: s.default for s in tpl.slots}
        self.pristine = {s.name: True for s in tpl.slots}
        self.field = 0

    @property
    def name(self) -> str:
        return self.tpl.slots[self.field].name

    def handle(self, k: str) -> Optional[str]:
        """→ 'done' | 'back' | None."""
        n = self.name
        if k in ("\r", "\n"):
            if self.field == len(self.tpl.slots) - 1:
                return "done"
            self.field += 1
        elif k in ("\t", "\x1b[B", "\x1bOB", "\x0e"):
            self.field = (self.field + 1) % len(self.tpl.slots)
        elif k in ("\x1b[A", "\x1bOA", "\x10", "\x1b[Z"):
            self.field = (self.field - 1) % len(self.tpl.slots)
        elif k in ("\x1b", "\x07"):
            return "back"
        elif k == "\x03":
            return "cancel"
        elif k in ("\x7f", "\x08"):
            self.values[n] = self.values[n][:-1]
            self.pristine[n] = False
        elif k == "\x15":
            self.values[n] = ""
            self.pristine[n] = False
        elif k and not k.startswith("\x1b") and k.isprintable():
            self.values[n] = k if self.pristine[n] else self.values[n] + k
            self.pristine[n] = False
        return None

    def result(self) -> str:
        return self.tpl.render(self.values)


WINDOWS = os.name == "nt"

if WINDOWS:
    import msvcrt
else:
    import select
    import termios


class PosixTty:
    def __init__(self):
        self.fd = os.open("/dev/tty", os.O_RDWR)
        self.old = termios.tcgetattr(self.fd)
        new = termios.tcgetattr(self.fd)
        new[0] &= ~(termios.IXON | termios.ICRNL)                      # get ^S/^Q, raw CR
        new[3] &= ~(termios.ICANON | termios.ECHO | termios.ISIG | termios.IEXTEN)  # get ^C/^Z/^O
        new[6][termios.VMIN], new[6][termios.VTIME] = 1, 0
        termios.tcsetattr(self.fd, termios.TCSANOW, new)

    def close(self):
        termios.tcsetattr(self.fd, termios.TCSANOW, self.old)
        os.close(self.fd)

    def ready(self, timeout: float = 0.0) -> bool:
        return bool(select.select([self.fd], [], [], timeout)[0])

    def key(self) -> str:
        b = os.read(self.fd, 1)
        if b == b"\x1b":
            seq = b
            while self.ready(0.025):
                seq += os.read(self.fd, 1)
                if len(seq) >= 6 or (len(seq) >= 3 and seq[-1:].isalpha()) or seq.endswith(b"~"):
                    break
            return seq.decode("latin-1")
        if b and b[0] >= 0xC0:  # utf-8 multibyte
            need = 1 if b[0] < 0xE0 else 2 if b[0] < 0xF0 else 3
            b += os.read(self.fd, need)
        return b.decode("utf-8", errors="ignore")


# Windows console scan codes (after a \x00 / \xe0 prefix) → the ANSI keys handle() knows
_WIN_KEYS = {"H": "\x1b[A", "P": "\x1b[B", "K": "\x1b[D", "M": "\x1b[C", "S": "\x1b[3~",
             "G": "\x1b[H", "O": "\x1b[F"}


class WinTty:
    def close(self):
        pass

    def ready(self, timeout: float = 0.0) -> bool:
        end = time.monotonic() + timeout
        while True:
            if msvcrt.kbhit():
                return True
            if time.monotonic() >= end:
                return False
            time.sleep(0.005)

    def key(self) -> str:
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):  # function / arrow keys arrive as two codes
            return _WIN_KEYS.get(msvcrt.getwch(), "")
        if ch == "\x1b":  # terminals with VT input on send real escape sequences
            seq = ch
            while len(seq) < 6 and self.ready(0.025):
                seq += msvcrt.getwch()
                if (len(seq) >= 3 and seq[-1].isalpha()) or seq.endswith("~"):
                    break
            return seq
        if "\ud800" <= ch <= "\udbff" and msvcrt.kbhit():  # surrogate pair (emoji etc.)
            ch = (ch + msvcrt.getwch()).encode("utf-16", "surrogatepass").decode("utf-16")
        return ch


def _enable_vt(stream) -> None:
    """Make sure the Windows console interprets ANSI escapes on this handle."""
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.windll.kernel32
        h = msvcrt.get_osfhandle(stream.fileno())
        mode = wintypes.DWORD()
        if k32.GetConsoleMode(h, ctypes.byref(mode)):
            k32.SetConsoleMode(h, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        pass


def _win_console_size() -> Optional[os.terminal_size]:
    """Visible window size of the console, asked of the console itself.

    Inside a PSReadLine key handler PowerShell captures the program's stdout and
    stderr, so the usual fd-based lookups fail. CONOUT$ always names the real
    console; it must be opened with read access to query its size.
    """
    try:
        import ctypes
        from ctypes import wintypes

        class CSBI(ctypes.Structure):
            _fields_ = [("dwSize", wintypes._COORD), ("dwCursorPosition", wintypes._COORD),
                        ("wAttributes", wintypes.WORD), ("srWindow", wintypes.SMALL_RECT),
                        ("dwMaximumWindowSize", wintypes._COORD)]

        k32 = ctypes.windll.kernel32
        k32.CreateFileW.restype = wintypes.HANDLE
        h = k32.CreateFileW("CONOUT$", 0x80000000 | 0x40000000, 0x1 | 0x2, None, 3, 0, None)
        if not h or h == wintypes.HANDLE(-1).value:
            return None
        try:
            info = CSBI()
            if k32.GetConsoleScreenBufferInfo(wintypes.HANDLE(h), ctypes.byref(info)):
                w = info.srWindow.Right - info.srWindow.Left + 1
                ht = info.srWindow.Bottom - info.srWindow.Top + 1
                if w > 0 and ht > 0:
                    return os.terminal_size((w, ht))
        finally:
            k32.CloseHandle(wintypes.HANDLE(h))
    except Exception:
        pass
    return None


def terminal_size(tty_fd: Optional[int] = None) -> os.terminal_size:
    if WINDOWS:
        size = _win_console_size()
        if size:
            return size
    for fd in ([tty_fd] if tty_fd is not None else []) + [2, 1, 0]:  # stdout may be a pipe
        try:
            return os.get_terminal_size(fd)
        except (OSError, ValueError):
            continue
    hinted = os.environ.get("REKOLLECT_TERM_SIZE", "")  # set by the PowerShell hook
    m = re.fullmatch(r"(\d+)x(\d+)", hinted)
    if m and int(m.group(1)) > 0 and int(m.group(2)) > 0:
        return os.terminal_size((int(m.group(1)), int(m.group(2))))
    return shutil.get_terminal_size((100, 30))


def open_terminal(accent: str):
    """→ (key reader, output stream, rich Console) for the current platform."""
    if WINDOWS:
        out = open("CONOUT$", "w", encoding="utf-8")
        _enable_vt(out)
        size = terminal_size()
        con = Console(theme=ui.make_theme(accent), file=out, highlight=False, soft_wrap=False,
                      legacy_windows=False, force_terminal=True, color_system="truecolor",
                      width=size.columns, height=size.lines)
        return WinTty(), out, con
    tty = PosixTty()
    out = open("/dev/tty", "w")
    size = terminal_size(tty.fd)
    con = ui.console(accent, file=out)
    con.size = (size.columns, size.lines)
    return tty, out, con


def copy_to_clipboard(text: str) -> bool:
    if WINDOWS:
        try:  # clip.exe understands UTF-16 with a BOM
            subprocess.run(["clip"], input=text.encode("utf-16"), check=True, timeout=3,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            return True
        except Exception:
            return False
    for cmd in (["pbcopy"], ["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "-ib"]):
        if shutil.which(cmd[0]):
            try:
                subprocess.run(cmd, input=text.encode(), check=True, timeout=2)
                return True
            except Exception:
                continue
    return False


class Picker:
    def __init__(self, ranker: Ranker, corpus: Corpus, store: Store, accent: str, model_label: str,
                 query: str = "", rows: int = 10):
        self.r, self.c, self.store = ranker, corpus, store
        self.query = query
        self.sel = 0
        self.only_project = False
        self.only_ok = False
        self.removed = np.zeros(len(corpus.texts), dtype=bool)
        self.toast = ""
        self.armed_forget = False
        self.model_label = model_label
        self.tty, self.tty_out, self.con = open_terminal(accent)
        self.pref_rows = rows
        self.rows = rows
        self.fit()
        self.hits = []
        self.ms = 0.0
        self.ok_mask = corpus.fails < corpus.count
        self.proj_mask = ranker.context > 0
        self.tpl_view = False
        self.explain = False
        self.form: Optional[FillForm] = None
        self._tpl: dict[int, Template] = {}
        self._details: dict[int, Optional[tuple]] = {}
        self.cwd = os.getcwd()
        self.waiting = int(getattr(store, "waiting", 0) or 0)

    def template(self, i: int) -> Template:
        if i not in self._tpl:
            self._tpl[i] = templatize(self.c.texts[i])
        return self._tpl[i]

    def fit(self) -> None:
        """Match the real console size (re-read on every redraw, so resizing works)."""
        size = terminal_size(getattr(self.tty, "fd", None))
        self.con.size = (size.columns, size.lines)
        # header, prompt, rule, preview, rule, footer, spare line; the preview shrinks on short screens
        self.preview_lines = 3 if size.lines >= 16 else 1
        chrome = 6 + self.preview_lines
        self.rows = max(1, min(self.pref_rows, size.lines - chrome))

    # ── state ────────────────────────────────────────────────────────────────
    def refresh(self):
        mask = ~self.removed
        if self.only_ok:
            mask &= self.ok_mask
        if self.only_project:
            mask &= self.proj_mask
        self.hits, self.ms = self.r.search(self.query, self.rows, mask=mask)
        self.sel = min(self.sel, max(0, len(self.hits) - 1))

    def current(self) -> Optional[int]:
        return self.hits[self.sel].idx if self.hits else None

    # ── learning ─────────────────────────────────────────────────────────────
    def record(self, kind: str, items: list[tuple[int, int]]) -> None:
        """Log feedback for the current query; this is what rekollect learns from."""
        if not self.query.strip() or getattr(self.store, "readonly", False):
            return
        try:
            for i, rank in items:
                self.store.add_feedback(self.query.strip(), self.r.last_qv, int(self.c.ids[i]), kind, rank,
                                        self.cwd)
            self.store.commit()
        except Exception:
            pass  # never lose a selection over bookkeeping

    def record_pick(self, i: int) -> None:
        shown_above = [(h.idx, r) for r, h in enumerate(self.hits[:self.sel]) if h.idx != i]
        self.record("pick", [(i, self.sel)])
        self.record("skip", shown_above)

    def toggle_star(self) -> None:
        i = self.current()
        if i is None:
            return
        cid = int(self.c.ids[i])
        marks = self.c.marks if self.c.marks is not None else {}
        if i in marks:
            self.store.remove_mark(cid)
            del marks[i]
            self.r.marked[i] = 0.0
            self.toast = "unstarred"
        else:
            self.store.set_mark(cid)
            marks[i] = ("", "", "")
            self.r.marked[i] = 1.0
            self.toast = "★ starred — rk mark adds a name, #tags and a note"
        self.c.marks = marks
        self.store.commit()

    # ── drawing ──────────────────────────────────────────────────────────────
    def render(self):
        old_rows = self.rows
        self.fit()
        if self.rows > old_rows and self.form is None:
            self.refresh()
        if self.form is not None:
            return self.render_form()
        w = self.con.width
        head = Text.assemble((ui.LOGO, "accent"), ("  ", ""))
        chips = [self.model_label, f"{len(self.c.texts):,} cmds", f"{self.ms:.0f} ms"]
        if self.waiting:
            chips.append(f"{self.waiting:,} still indexing · found by words meanwhile")
        if self.only_project:
            chips.append("project")
        if self.only_ok:
            chips.append("only ✓")
        if self.tpl_view:
            chips.append("templates")
        for ch in chips:
            head.append(f" {ch} ", "muted on grey11")
            head.append(" ")
        prompt = Text.assemble(("❯ ", "accent"), (self.query, "bold"), ("▏", "accent.dim"))
        if not self.query:
            prompt.append("describe the command you're looking for…  (#tag, @bookmark)", "faint")

        grid = Table.grid(expand=True, padding=(0, 1))
        grid.add_column(width=1, no_wrap=True)
        grid.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
        show_path = w >= 80
        grid.add_column(no_wrap=True, width=27 if show_path else 12)
        marks = self.c.marks or {}
        for r in range(self.rows):
            if r < len(self.hits):
                i = self.hits[r].idx
                meta = Text.assemble(ui.status_glyph(self.c.count[i], self.c.fails[i]),
                                     (f"{int(self.c.count[i])}×".rjust(5) + ui.ago(self.c.last_ts[i]).rjust(5), "muted"))
                if show_path:
                    meta.append("  " + ui.short_path(self.c.last_cwd[i], 14), "faint")
                if self.tpl_view:
                    cmd = ui.template_text(self.template(i), max_len=w)
                else:
                    cmd = ui.highlight(self.c.texts[i], self.query, max_len=w)
                if i in marks:
                    star = Text("★ ", "accent")
                    if marks[i][0]:
                        star.append(marks[i][0] + " ", "bold")
                    cmd = star + cmd
                if r == self.sel:
                    grid.add_row(Text("▌", "accent"), cmd, meta, style="sel")
                else:
                    grid.add_row(Text(" "), cmd, meta)
            elif r == 0:
                grid.add_row(" ", Text("no matches — try fewer or different words", "faint"), "")
            else:
                grid.add_row(" ", "", "")

        rule = Text("─" * min(w, 200), "faint")
        return Group(head, prompt, rule, grid, rule, self.render_preview(w), rule, self.render_footer(w))

    def details(self, i: int) -> Optional[tuple]:
        if i not in self._details:
            try:
                self._details[i] = self.store.run_details(int(self.c.ids[i]))
            except Exception:
                self._details[i] = None
        return self._details[i]

    def render_preview(self, w: int) -> Group:
        """The selected command in full, where/when it ran, and its template or reasons."""
        cur = self.current()
        lines: list[Text] = []
        if cur is None:
            return Group(*[Text("") for _ in range(self.preview_lines)])
        tpl = self.template(cur)
        if self.tpl_view:
            badge = Text(" TEMPLATE ", "bold black on #b48ead") + Text(" ")
            body = ui.template_text(tpl, max_len=(w - 12) * 2)
            alt = Text(" ORIGINAL ", "muted on grey15") + Text(" ") + ui.highlight(self.c.texts[cur], max_len=w - 12)
        else:
            badge = Text(" ORIGINAL ", "muted on grey15") + Text(" ")
            body = ui.highlight(self.c.texts[cur], self.query, max_len=(w - 12) * 2)
            alt = (Text(" ^T ", "muted on grey15") + Text(" ") + ui.template_text(tpl, max_len=w - 8)
                   if tpl.has_slots else None)
        first = badge + body
        first.no_wrap = self.preview_lines == 1
        first.overflow = "ellipsis"
        lines.append(first)
        if self.preview_lines == 1:
            return Group(*lines)
        # where & when
        d = self.details(cur)
        meta = Text("  ", "faint")
        bits = []
        if d:
            ts, cwd, ec, dur, branch, shell, host = d
            if cwd:
                bits.append(ui.short_path(cwd, 30))
            if branch:
                bits.append(f"⎇ {branch}")
            if shell:
                bits.append(shell)
        count, fails = int(self.c.count[cur]), int(self.c.fails[cur])
        bits.append(f"{count} run{'s' if count != 1 else ''}" + (f", {fails} failed" if fails else ", all ok" if count > 1 else ""))
        if self.c.last_ts[cur]:
            bits.append(f"last {ui.since(self.c.last_ts[cur])}")
        if d and d[3]:
            bits.append(f"took {d[3] / 1000:.1f}s")
        marks = self.c.marks or {}
        if cur in marks:
            name, tags, note = marks[cur]
            bits.append("★ " + " ".join(x for x in (name, " ".join("#" + t for t in tags.split()), note) if x))
        meta.append(" · ".join(bits), "faint")
        lines.append(meta)
        if self.explain:
            hit = self.hits[self.sel]
            reasons = self.r.explain(hit, self.query) or ["recent and frequent"]
            lines.append(Text("  why  ", "muted") + Text(" · ".join(reasons), "accent.dim"))
        elif self.tpl_view:
            lines.append(Text("  ") + alt + (Text("   ⏎ fill in", "muted") if tpl.has_slots else Text("")))
        elif alt is not None:
            lines.append(alt)
        else:
            lines.append(Text(""))
        for ln in lines:
            ln.no_wrap = True
            ln.overflow = "ellipsis"
        return Group(*lines)

    def render_footer(self, w: int) -> Text:
        if self.toast:
            return Text(self.toast, "accent.dim")
        foot = Text()
        for k, d in KEYS_HELP:
            piece = len(k) + len(d) + 4
            if len(foot) + piece > w - 1:
                break
            foot.append(k, "bold" if k == "⏎" else "muted")
            foot.append(f" {d}   ", "faint")
        return foot

    def render_form(self):
        f = self.form
        w = self.con.width
        head = Text.assemble((ui.LOGO, "accent"), ("  ", ""), (" fill in template ", "muted on grey11"))
        info = Table.grid(padding=(0, 2))
        info.add_column(style="faint", no_wrap=True)
        info.add_column(no_wrap=True, overflow="ellipsis")
        info.add_row("template", ui.template_text(f.tpl, max_len=w - 12))
        info.add_row("was", ui.highlight(f.tpl.original, max_len=w - 12))
        fields = Table.grid(expand=True, padding=(0, 1))
        fields.add_column(width=1)
        fields.add_column(no_wrap=True)
        fields.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
        fields.add_column(no_wrap=True, style="faint")
        width = max(len(s.name) for s in f.tpl.slots)
        room = max(1, self.con.height - 9)  # header, 2 info, 3 rules, result, footer, spare
        first = min(max(0, f.field - room // 2), max(0, len(f.tpl.slots) - room))
        for k, s in list(enumerate(f.tpl.slots))[first:first + room]:
            v = f.values[s.name]
            active = k == f.field
            if active:
                val = Text(v, "faint" if f.pristine[s.name] else "bold") + Text("▏", "accent")
            else:
                val = Text(v or "(empty)", "faint" if f.pristine[s.name] or not v else "")
            note = "from history" if f.pristine[s.name] else f"was {s.default}"
            fields.add_row(Text("▌", "accent") if active else Text(" "),
                           Text(s.name.ljust(width), "accent" if active else "muted"), val, note,
                           style="sel" if active else None)
        result = Text("  ❯ ", "accent") + ui.template_text(f.tpl, f.values, max_len=w - 6)
        foot = Text()
        for k, d in FORM_HELP:
            foot.append(k, "muted")
            foot.append(f" {d}   ", "faint")
        if w >= 110:
            foot.append("goes to your prompt to check first; nothing runs yet", "faint")
        rule = Text("─" * min(w, 200), "faint")
        return Group(head, info, rule, fields, rule, result, rule, foot)

    # ── loop ─────────────────────────────────────────────────────────────────
    def run(self) -> Optional[str]:
        tty = self.tty
        result: Optional[str] = None
        try:
            self.refresh()
            if not WINDOWS:
                self.tty_out.write("\r\n")  # start below the prompt line…
                self.tty_out.flush()
            with Live(self.render(), console=self.con, auto_refresh=False, transient=True,
                      screen=WINDOWS, redirect_stdout=False, redirect_stderr=False) as live:
                while True:
                    k = tty.key()
                    if self.form is not None:
                        r = self.form.handle(k)
                        if r == "done":
                            result = self.form.result()
                            self.record_pick(self.current())
                            break
                        if r == "cancel":
                            break
                        if r == "back":
                            self.form = None
                        live.update(self.render(), refresh=True)
                        continue
                    action = self.handle(k)
                    # swallow buffered keystrokes before re-ranking (fast typists)
                    while action is None and tty.ready(0):
                        action = self.handle(tty.key())
                    if action == "accept":
                        i = self.current()
                        if i is not None and self.tpl_view and self.template(i).has_slots:
                            self.form = FillForm(self.template(i))
                            live.update(self.render(), refresh=True)
                            continue
                        result = self.c.texts[i] if i is not None else None
                        if i is not None:
                            self.record_pick(i)
                        break
                    if action == "cancel":
                        break
                    if action != "noop":
                        self.refresh()
                    live.update(self.render(), refresh=True)
        except KeyboardInterrupt:
            result = None
        finally:
            if not WINDOWS:
                self.tty_out.write("\x1b[1A\r\x1b[2K")  # …and hand the line back for the shell to redraw
                self.tty_out.flush()
            tty.close()
            self.tty_out.close()
        return result

    def handle(self, k: str) -> Optional[str]:
        """Returns 'accept' | 'cancel' | 'noop' | None (state changed → re-rank)."""
        self.toast = "" if k not in ("\x04", "\x18", "\x02") else self.toast
        if k != "\x04":
            self.armed_forget = False
        if k in ("\r", "\n", "\t"):
            return "accept"
        if k in ("\x1b", "\x03", "\x07"):
            return "cancel"
        if k in ("\x1b[A", "\x1bOA", "\x10"):
            self.sel = max(0, self.sel - 1)
            return "noop"
        if k in ("\x1b[B", "\x1bOB", "\x0e"):
            self.sel = min(len(self.hits) - 1, self.sel + 1) if self.hits else 0
            return "noop"
        if k in ("\x7f", "\x08"):
            self.query = self.query[:-1]
        elif k == "\x15":
            self.query = ""
        elif k == "\x17":
            self.query = self.query.rstrip()
            self.query = self.query[: self.query.rfind(" ") + 1] if " " in self.query else ""
        elif k == "\x14":
            self.tpl_view = not self.tpl_view
            return "noop"
        elif k == "\x05":
            self.explain = not self.explain
            return "noop"
        elif k == "\x02":
            self.toggle_star()
            return "noop"
        elif k == "\x18":
            i = self.current()
            if i is None:
                return "noop"
            self.record("irrelevant", [(i, self.sel)])
            self.removed[i] = True
            self.toast = "got it — it'll rank lower for searches like this"
        elif k == "\x13":
            self.only_project = not self.only_project
        elif k == "\x0f":
            self.only_ok = not self.only_ok
        elif k == "\x19":
            i = self.current()
            if i is not None:
                self.toast = "copied to clipboard ✓" if copy_to_clipboard(self.c.texts[i]) else "no clipboard tool found"
            return "noop"
        elif k == "\x04":
            i = self.current()
            if i is None:
                return "noop"
            if not self.armed_forget:
                self.armed_forget = True
                self.toast = "press ^D again to permanently forget this command"
                return "noop"
            self.store.forget([int(self.c.ids[i])])
            self.store.commit()
            self.removed[i] = True
            self.armed_forget = False
            self.toast = "forgotten ✓"
        elif len(k) == 1 and k.isprintable():
            self.query += k
            self.sel = 0
        elif len(k) > 1 and not k.startswith("\x1b") and k.isprintable():
            self.query += k
            self.sel = 0
        else:
            return "noop"
        return None