"""Make the picker key reach the shell inside VS Code-family terminals.

VS Code resolves keybindings before the integrated terminal sees a key. Ctrl+G in
a focused terminal runs "Go to Recent Directory", so the shell never gets it.
We add user keybindings (user keybindings beat the defaults):

  * unbind Go to Recent Directory from the key
  * PowerShell terminals: send ESC[24~r ("F12, r"). On Windows a raw Ctrl+letter
    byte doesn't reliably reach PSReadLine as a key press through ConPTY, so VS
    Code's own shell integration maps Ctrl+Space, Shift+Enter, ... to "F12,<letter>"
    sequences the same way. The PowerShell hook binds F12,r to the picker.
  * every other shell: send the raw control byte (Ctrl+G = 0x07), which is what
    the shell would have received without VS Code in the way.
  * macOS: VS Code puts terminal commands on Cmd (Go to Recent Directory is Cmd+G),
    so Cmd+<key> is bound the same way as Ctrl+<key>; both open the picker.

keybindings.json is JSONC (comments, trailing commas), so we never re-serialise
it: we splice our lines in as text and verify the result still parses.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

from rekollect.hooks import PS_ALT_SEQUENCE

EDITORS = ("Code", "Code - Insiders", "VSCodium", "Cursor", "Windsurf")
MARK_BEGIN = "// >>> rekollect: send the picker key to the shell (rk vscode --remove to undo)"
MARK_END = "// <<< rekollect"

# Built-in terminal bindings that would otherwise grab the key
_CONFLICTS = {
    "ctrl+g": ["workbench.action.terminal.goToRecentDirectory"],
    "ctrl+r": ["workbench.action.terminal.runRecentCommand"],
    "ctrl+k": [],  # chord prefix; our binding still wins for a single press, but we warn
}


def config_root() -> Path:
    if os.name == "nt":
        return Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support"
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")


def installed_editors() -> list[tuple[str, Path]]:
    """(editor name, keybindings.json path) for every VS Code-family editor found."""
    root = config_root()
    return [(e, root / e / "User" / "keybindings.json") for e in EDITORS if (root / e / "User").is_dir()]


def in_vscode_terminal() -> bool:
    return os.environ.get("TERM_PROGRAM") == "vscode"


IS_MAC = sys.platform == "darwin"


def vscode_key(key: str) -> tuple[str, str]:
    """'^G' → ('ctrl+g', '\\x07')."""
    ch = key[1].lower()
    return f"ctrl+{ch}", chr(ord(ch) & 0x1F)


def vscode_keys(key: str, mac: bool = IS_MAC) -> list[str]:
    """Every VS Code key that should open the picker: ctrl+g, plus cmd+g on macOS."""
    vk, _ = vscode_key(key)
    return [vk, "cmd+" + vk.split("+", 1)[1]] if mac else [vk]


def key_label(key: str, mac: bool = IS_MAC) -> str:
    return " / ".join(vscode_keys(key, mac))


def block(key: str, indent: str = "  ", mac: bool = IS_MAC) -> str:
    _, seq = vscode_key(key)
    rules = []
    for vk in vscode_keys(key, mac):
        conflicts = _CONFLICTS.get("ctrl+" + vk.split("+", 1)[1], [])
        rules += [{"key": vk, "command": "-" + cmd} for cmd in conflicts]
        rules.append({"key": vk, "command": "workbench.action.terminal.sendSequence",
                      "args": {"text": PS_ALT_SEQUENCE}, "when": "terminalFocus && terminalShellType == 'pwsh'"})
        rules.append({"key": vk, "command": "workbench.action.terminal.sendSequence",
                      "args": {"text": seq}, "when": "terminalFocus && terminalShellType != 'pwsh'"})
    lines = [MARK_BEGIN] + [json.dumps(r) + ("," if i < len(rules) - 1 else "") for i, r in enumerate(rules)]
    lines.append(MARK_END)
    return "\n".join(indent + ln for ln in lines)


# ── JSONC helpers ────────────────────────────────────────────────────────────

def _mask(text: str, strings: bool = True) -> str:
    """Blank out comments (and strings if asked), keeping offsets identical."""
    out = list(text)
    i, n = 0, len(text)

    def blank(a: int, b: int) -> None:
        for k in range(a, b):
            if out[k] != "\n":
                out[k] = " "

    while i < n:
        c = text[i]
        if c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            if strings:
                blank(i, min(j + 1, n))
            i = j + 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            blank(i, j)
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            blank(i, j)
            i = j
        else:
            i += 1
    return "".join(out)


def parse_jsonc(text: str):
    code = _mask(text, strings=False)
    code = re.sub(r",(\s*[\]}])", r"\1", code)  # trailing commas
    return json.loads(code) if code.strip() else []


def _remove_block(text: str) -> str:
    pat = re.compile(r"\n?[ \t]*" + re.escape(MARK_BEGIN) + r".*?" + re.escape(MARK_END) + r"[^\n]*", re.S)
    return pat.sub("", text)


def splice(text: str, key: Optional[str]) -> str:
    """Return keybindings.json text with our block replaced/added (or removed when key is None)."""
    text = _remove_block(text)
    if key is None:
        return text
    masked = _mask(text)
    end = masked.rfind("]")
    if end < 0:
        if masked.strip():
            raise ValueError("keybindings.json doesn't look like a JSON array")
        return "[\n" + block(key) + "\n]\n"
    before = masked[:end].rstrip()
    if not before.endswith("["):
        if before.count("[") == 0:
            raise ValueError("keybindings.json doesn't look like a JSON array")
    needs_comma = not (before.endswith("[") or before.endswith(","))
    cut = len(before)
    middle = text[cut:end].rstrip()  # comments between the last entry and ']'
    return (text[:cut] + ("," if needs_comma else "") + middle + "\n" + block(key) + "\n" + text[end:])


def install(path: Path, key: str) -> str:
    """→ 'added' | 'updated' | 'unchanged'. Raises ValueError if the file can't be edited safely."""
    old = path.read_text(encoding="utf-8-sig") if path.exists() else ""
    new = splice(old, key)
    parse_jsonc(new)  # never write something VS Code can't read
    if new == old:
        return "unchanged"
    if path.exists():
        path.with_name(path.name + ".rekollect.bak").write_text(old, encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new, encoding="utf-8")
    return "updated" if MARK_BEGIN in old else "added"


def remove(path: Path) -> bool:
    if not path.exists():
        return False
    old = path.read_text(encoding="utf-8-sig")
    new = splice(old, None)
    if new == old:
        return False
    parse_jsonc(new)
    path.write_text(new, encoding="utf-8")
    return True


def is_current(path: Path) -> bool:
    """Block present and written by this version (older blocks lacked the PowerShell rule)."""
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8-sig", errors="ignore")
    return MARK_BEGIN in text and "\\u001b[24~r" in text and (not IS_MAC or '"cmd+' in text)


def has_block(path: Path) -> bool:
    return path.exists() and MARK_BEGIN in path.read_text(encoding="utf-8-sig", errors="ignore")


def sync_all(key: str) -> list[tuple[str, Path, str]]:
    """Re-point existing rekollect blocks at a new key (called after `rk config set ui.key`)."""
    out = []
    for name, path in installed_editors():
        if has_block(path):
            try:
                out.append((name, path, install(path, key)))
            except ValueError as e:
                out.append((name, path, f"error: {e}"))
    return out