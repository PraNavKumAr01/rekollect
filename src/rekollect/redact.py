"""Keep secrets out of the index. Values are replaced, the command shape stays."""

from __future__ import annotations

import re

MASK = "‹redacted›"

_PATTERNS: list[tuple[re.Pattern, str]] = [
    # FOO_API_KEY=xxx / export GITHUB_TOKEN=xxx / DB_PASSWORD="xxx"
    (re.compile(r"\b([A-Za-z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|PASS|CREDENTIALS?|AUTH)[A-Za-z0-9_]*)=(\"[^\"]*\"|'[^']*'|\S+)",
                re.IGNORECASE), r"\1=" + MASK),
    # --password xxx / --token=xxx / --api-key xxx
    (re.compile(r"(--?(?:password|passwd|pass|token|api[-_]?key|secret|auth|access[-_]?key)(?:=|\s+))(\"[^\"]*\"|'[^']*'|\S+)",
                re.IGNORECASE), r"\1" + MASK),
    # Authorization: Bearer xxx  /  -u user:pass
    (re.compile(r"(Bearer\s+)[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE), r"\1" + MASK),
    (re.compile(r"(\s-u\s+[^\s:]+:)(\S+)"), r"\1" + MASK),
    # scheme://user:pass@host
    (re.compile(r"(\b[a-z][a-z0-9+.-]*://[^\s:/@]+:)([^\s@]+)(@)", re.IGNORECASE), r"\1" + MASK + r"\3"),
    # well-known token shapes
    (re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}"), MASK),
    (re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}"), MASK),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"), MASK),
    (re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}"), MASK),
    (re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"), MASK),
    (re.compile(r"\bhf_[A-Za-z0-9]{30,}"), MASK),
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), MASK),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), MASK),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"), MASK),  # JWT
]


def redact(cmd: str) -> str:
    for pat, repl in _PATTERNS:
        cmd = pat.sub(repl, cmd)
    return cmd