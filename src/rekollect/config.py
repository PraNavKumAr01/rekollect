"""Paths and user configuration (TOML, XDG-friendly)."""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import Any, Optional

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


# ── paths ────────────────────────────────────────────────────────────────────

def _xdg(var: str, fallback: str) -> Path:
    v = os.environ.get(var)
    return Path(v) if v else Path.home() / fallback


def home() -> Optional[Path]:
    """Override everything with REKOLLECT_HOME (handy for tests / portable installs)."""
    v = os.environ.get("REKOLLECT_HOME")
    return Path(v) if v else None


def config_dir() -> Path:
    h = home()
    return h if h else _xdg("XDG_CONFIG_HOME", ".config") / "rekollect"


def data_dir() -> Path:
    h = home()
    return h if h else _xdg("XDG_DATA_HOME", ".local/share") / "rekollect"


def config_path() -> Path:
    return config_dir() / "config.toml"


def db_path() -> Path:
    return data_dir() / "index.db"


def spool_path() -> Path:
    return data_dir() / "spool"


def models_dir() -> Path:
    return data_dir() / "models"


def paused_flag() -> Path:
    return data_dir() / "paused"


def ensure_dirs() -> None:
    for d in (config_dir(), data_dir(), models_dir()):
        d.mkdir(parents=True, exist_ok=True)


# ── defaults ─────────────────────────────────────────────────────────────────

DEFAULTS: dict[str, Any] = {
    "model": {
        # lite | balanced | quality | custom   (see `rk models`)
        "tier": "balanced",
        # CPU threads for ONNX inference. Low on purpose: we embed tiny strings.
        "threads": 2,
        # only used when tier = "custom": any ONNX sentence-embedding repo on HF
        "custom": {
            "repo": "",
            "onnx_file": "onnx/model_quantized.onnx",
            "pooling": "mean",  # mean | cls
            "query_prefix": "",
            "doc_prefix": "",
        },
    },
    "index": {
        # embed new commands in the background every N commands
        "auto": True,
        "every": 10,
        "min_length": 3,
        # replace API keys / tokens / passwords with ‹redacted› before storing
        "redact_secrets": True,
        # commands matching any of these regexes are never indexed
        "ignore": [
            # (cd is kept: workflows need it. Search hides navigation commands.)
            r"^(?i:ls|ll|la|l|pwd|clear|cls|dir|gci|exit|history|h|fg|bg|jobs)(\s+\S{0,24})?$",
            r"^(rk|rekollect)(\s|$)",
            # editors' shell-integration bootstrap lines (VS Code types these into new terminals)
            r"shellIntegration[\w.-]*\.(ps1|sh|bash|zsh|fish)\b",
        ],
    },
    "search": {
        "limit": 8,
        "success_only": False,
        # how much each signal matters in the final ranking (they're normalized)
        "weights": {
            "semantic": 0.50,
            "lexical": 0.25,
            "recency": 0.10,
            "frequency": 0.08,
            "context": 0.07,
        },
        # recency half-life in days
        "half_life_days": 21,
        # learn from what you pick in the picker (stays on this machine)
        "learn": True,
        # how strongly your past choices move results (rk tune adjusts this for you)
        "learning": 0.3,
    },
    "context": {
        # extra context stored with each command run (used for ranking and filters)
        "record_branch": True,
        "record_host": True,
    },
    "agents": {
        # read-only access for coding assistants via `rk mcp` (off until you enable it)
        "enabled": False,
        # project: only commands run inside the assistant's project · all: everything
        "scope": "project",
        # extra regexes for commands agents must never see
        "exclude": [],
        # keep a log of every request (rk agents log)
        "log": True,
    },
    "ui": {
        "accent": "#b48ead",
        "key": "^G",  # picker key binding inserted by `rk hook`
    },
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


class Config(dict):
    """dict with dotted get: cfg.get_path('search.weights.semantic')."""

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


FORMAT_MARKER = "# format: overrides-only"

# Defaults that shipped in earlier versions. v0.1 copied every default into the
# user's file, so a value equal to one of these was never really chosen by the user.
_PAST_DEFAULTS: dict[str, list[Any]] = {
    "index.ignore": [[r"^(ls|ll|la|l|cd|pwd|clear|exit|history|fg|bg|jobs)(\s+\S{0,24})?$",
                      r"^(rk|rekollect)(\s|$)"],
                     [r"^(?i:ls|ll|la|l|cd|cd\.\.|pwd|clear|cls|dir|gci|exit|history|h|fg|bg|jobs)(\s+\S{0,24})?$",
                      r"^(rk|rekollect)(\s|$)"]],
}


def _read_user() -> dict:
    p = config_path()
    if not p.exists():
        return {}
    try:
        text = p.read_text(encoding="utf-8")
        user = tomllib.loads(text)
    except Exception as e:  # never crash the shell over a typo
        print(f"rekollect: ignoring invalid config ({e})", file=sys.stderr)
        return {}
    if FORMAT_MARKER not in text:  # old full copy → keep only real overrides, rewrite once
        user = _drop_defaults(user)
        try:
            p.write_text(dumps(user), encoding="utf-8")
        except OSError:
            pass
    return user


def _drop_defaults(user: dict, prefix: str = "") -> dict:
    out: dict = {}
    for k, v in user.items():
        key = f"{prefix}{k}"
        default = Config(DEFAULTS).get_path(key, _MISSING)
        if isinstance(v, dict) and isinstance(default, dict):
            sub_ = _drop_defaults(v, key + ".")
            if sub_:
                out[k] = sub_
        elif v != default and v not in _PAST_DEFAULTS.get(key, []):
            out[k] = v
    return out


_MISSING = object()


def overrides() -> dict:
    """Only what the user explicitly set (the contents of config.toml)."""
    return _read_user()


def load() -> Config:
    return Config(_merge(DEFAULTS, _read_user()))


# ── tiny TOML writer (enough for our own config shape) ───────────────────────

def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        if "\\" in v and "'" not in v:
            return f"'{v}'"  # literal string, no escaping needed (regexes!)
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, list):
        if not v:
            return "[]"
        inner = ",\n".join("  " + _toml_value(x) for x in v)
        return "[\n" + inner + ",\n]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{k} = {_toml_value(x)}" for k, x in v.items()) + " }"
    raise TypeError(f"can't serialise {type(v)}")


HEADER = f"""\
# rekollect configuration
# Only the settings you change live here; everything else uses the built-in
# defaults (listed, commented out, at the bottom). Change things with
#   rk config set <key> <value>     e.g. rk config set ui.key ^G
#   rk config unset <key>           go back to the default
{FORMAT_MARKER}
"""


def _sections(cfg: dict) -> list[str]:
    lines: list[str] = []
    for section, body in cfg.items():
        lines.append(f"[{section}]")
        nested = []
        for k, v in body.items():
            if isinstance(v, dict) and (section, k) in (("model", "custom"), ("search", "weights")):
                nested.append((k, v))
            else:
                lines.append(f"{k} = {_toml_value(v)}")
        for k, v in nested:
            lines.append(f"\n[{section}.{k}]")
            for kk, vv in v.items():
                lines.append(f"{kk} = {_toml_value(vv)}")
        lines.append("")
    return lines


def dumps(user: dict) -> str:
    """Overrides as live TOML, then every default as a commented reference."""
    lines = [HEADER]
    lines += _sections(user) if user else ["# (no overrides yet)", ""]
    lines += ["", "# ── defaults " + "─" * 60]
    lines += ["# " + ln if ln else "#" for ln in "\n".join(_sections(DEFAULTS)).split("\n")]
    return "\n".join(lines) + "\n"


def diff(cfg: dict, base: dict = DEFAULTS) -> dict:
    out: dict = {}
    for k, v in cfg.items():
        b = base.get(k, _MISSING) if isinstance(base, dict) else _MISSING
        if isinstance(v, dict) and isinstance(b, dict):
            d = diff(v, b)
            if d:
                out[k] = d
        elif v != b:
            out[k] = v
    return out


def save(cfg: dict) -> None:
    """Persist only the values that differ from DEFAULTS."""
    ensure_dirs()
    config_path().write_text(dumps(diff(cfg)), encoding="utf-8")


def write_default_if_missing() -> bool:
    if config_path().exists():
        return False
    save(DEFAULTS)
    return True


def flatten(cfg: dict, prefix: str = "") -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    for k, v in cfg.items():
        if isinstance(v, dict):
            out += flatten(v, f"{prefix}{k}.")
        else:
            out.append((f"{prefix}{k}", v))
    return out


def coerce(raw: str, current: Any) -> Any:
    """Turn CLI text into the type of the existing value."""
    if isinstance(current, bool):
        if raw.lower() in ("1", "true", "yes", "on"):
            return True
        if raw.lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"expected true/false, got {raw!r}")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        return [x.strip() for x in raw.split(",") if x.strip()]
    return raw


def set_path(cfg: dict, dotted: str, raw: str) -> Any:
    parts = dotted.split(".")
    node = cfg
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            raise KeyError(dotted)
        node = node[p]
    if parts[-1] not in node:
        raise KeyError(dotted)
    value = coerce(raw, node[parts[-1]])
    if dotted == "ui.key":
        value = normalize_key(value)
    node[parts[-1]] = value
    return value


def unset_path(cfg: dict, dotted: str) -> Any:
    default = Config(DEFAULTS).get_path(dotted, _MISSING)
    if default is _MISSING:
        raise KeyError(dotted)
    parts = dotted.split(".")
    node = cfg
    for p in parts[:-1]:
        node = node[p]
    node[parts[-1]] = copy.deepcopy(default)
    return default


def normalize_key(raw: str) -> str:
    """Accept ^G, C-g, ctrl+g, Ctrl-G → canonical ^G."""
    import re
    m = re.fullmatch(r"\s*(?:\^|c-|ctrl[+-])([a-z])\s*", raw, re.IGNORECASE)
    if not m:
        raise ValueError(f"expected a Ctrl+letter key like ^G or ctrl+g, got {raw!r}")
    return "^" + m.group(1).upper()