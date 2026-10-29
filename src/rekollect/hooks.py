"""Shell integration. The hooks are pure shell: recording a command costs one
`printf >> file` (no Python startup on your prompt). Every N commands a
background, disowned `rekollect index` embeds what's new.

Each hook also defines:
  * a key binding (ui.key, default Ctrl+G) that opens the picker on the current line
  * an `rk` shell function: `rk` or `rk <words>` opens the picker without any key
    binding and puts the chosen command on your *next* prompt, ready to edit.
    (`rk <subcommand>` and anything with flags goes straight to the CLI.)
  * a session id per terminal, so workflows can be rebuilt from what ran together.
"""

from __future__ import annotations

from pathlib import Path

from rekollect import config as C

# CLI subcommands the `rk` shell function must pass straight through
SUBCOMMANDS = ("init", "search", "s", "pick", "index", "hook", "models", "model", "config", "stats",
               "forget", "pause", "resume", "doctor", "help", "vscode", "flow", "recipe", "recipes",
               "tpl", "keys", "mark", "marks", "unmark", "tune", "eval", "agents", "mcp", "why")

# VS Code on Windows can't deliver every Ctrl+key to PSReadLine through ConPTY, so (like VS Code's
# own shell integration does for Ctrl+Space etc.) it sends F12 followed by a letter instead.
PS_ALT_CHORD = "F12,r"
PS_ALT_SEQUENCE = "\x1b[24~r"

ZSH = r"""
# ── rekollect (zsh) ──────────────────────────────────────────────
[[ -o interactive ]] || return 0
zmodload zsh/datetime 2>/dev/null
typeset -g __rk_dir=@DIR@ __rk_spool=@SPOOL@ __rk_every=@EVERY@ __rk_auto=@AUTO@
typeset -g __rk_cmd="" __rk_start="" ; typeset -gi __rk_n=0
typeset -g __rk_sid="zsh:$$.${EPOCHSECONDS:-$(date +%s)}"
export REKOLLECT_HOOKED=zsh REKOLLECT_SESSION=$__rk_sid

__rk_preexec() { __rk_cmd="$1"; __rk_start=$EPOCHREALTIME }
__rk_precmd() {
  local ec=$?
  [[ -z "$__rk_cmd" ]] && return
  local cmd="$__rk_cmd"; __rk_cmd=""
  [[ "$cmd" == " "* || -e "$__rk_dir/paused" ]] && return
  local -i dur=$(( (EPOCHREALTIME - __rk_start) * 1000 ))
  printf '%s\x1f%s\x1f%s\x1f%s\x1f%s\x1f%s\x1e' "${EPOCHREALTIME%.*}" "$ec" "$dur" "$PWD" "$__rk_sid" "$cmd" >>| "$__rk_spool"
  if (( __rk_auto && ++__rk_n >= __rk_every )); then
    __rk_n=0
    command rekollect index --quiet &>/dev/null &!
  fi
}
autoload -Uz add-zsh-hook
add-zsh-hook preexec __rk_preexec
add-zsh-hook precmd __rk_precmd

__rk_widget() {
  local sel
  sel=$(command rekollect pick --query "$LBUFFER" </dev/tty)
  local ret=$?
  if [[ -n "$sel" ]]; then BUFFER="$sel"; CURSOR=${#BUFFER}; fi
  zle reset-prompt
  return $ret
}
zle -N __rk_widget
# Bind in every keymap (emacs and vi users alike), and once more at the first prompt:
# oh-my-zsh, prezto and zsh-vi-mode can rebuild keymaps after this file has run.
__rk_bind() { local m; for m in emacs viins vicmd; do bindkey -M $m '@KEY@' __rk_widget; done }
__rk_bind
__rk_bind_once() { __rk_bind; add-zsh-hook -d precmd __rk_bind_once }
add-zsh-hook precmd __rk_bind_once
typeset -ga zvm_after_init_commands; zvm_after_init_commands+=(__rk_bind)

rk() {
  if (( $# )) && [[ " @SUBS@ " == *" $1 "* || " $* " == *" -"* ]]; then command rekollect "$@"; return; fi
  local sel
  sel=$(command rekollect pick --query "$*" </dev/tty) || return
  [[ -n "$sel" ]] && print -z -- "$sel"
}
""".lstrip()

BASH = r"""
# ── rekollect (bash) ─────────────────────────────────────────────
[[ $- == *i* ]] || return 0
__rk_dir=@DIR@; __rk_spool=@SPOOL@; __rk_every=@EVERY@; __rk_auto=@AUTO@
__rk_n=0; __rk_last=""; __rk_armed=""; __rk_start=""
__rk_sid="bash:$$.$(date +%s)"
export REKOLLECT_HOOKED=bash REKOLLECT_SESSION=$__rk_sid

__rk_now() { if [[ -n "$EPOCHREALTIME" ]]; then echo "${EPOCHREALTIME/[.,]/}"; else echo "$(date +%s)000000"; fi; }
__rk_debug() { [[ -n "$COMP_LINE" || -z "$__rk_armed" ]] && return; __rk_armed=""; __rk_start=$(__rk_now); }
__rk_precmd() {
  local ec=$? entry num cmd
  entry=$(HISTTIMEFORMAT= builtin history 1)
  if [[ $entry =~ ^[[:space:]]*([0-9]+)[*]?[[:space:]][[:space:]](.*)$ ]]; then
    num=${BASH_REMATCH[1]}; cmd=${BASH_REMATCH[2]}
    if [[ "$num" != "$__rk_last" && -n "$__rk_last" && -n "$__rk_start" && ! -e "$__rk_dir/paused" && "$cmd" != " "* ]]; then
      local dur=$(( ($(__rk_now) - __rk_start) / 1000 ))
      printf '%s\x1f%s\x1f%s\x1f%s\x1f%s\x1f%s\x1e' "$(date +%s)" "$ec" "$dur" "$PWD" "$__rk_sid" "$cmd" >> "$__rk_spool"
      if (( __rk_auto && ++__rk_n >= __rk_every )); then
        __rk_n=0
        ( command rekollect index --quiet >/dev/null 2>&1 & )
      fi
    fi
    __rk_last=$num
  fi
  __rk_start=""
}
__rk_arm() { __rk_armed=1; }
trap '__rk_debug' DEBUG
PROMPT_COMMAND="__rk_precmd${PROMPT_COMMAND:+;$PROMPT_COMMAND};__rk_arm"

__rk_widget() {
  local sel
  sel=$(command rekollect pick --query "$READLINE_LINE" </dev/tty) || return
  READLINE_LINE="$sel"; READLINE_POINT=${#sel}
}
bind -x '"@BKEY@": __rk_widget'

rk() {
  if (( $# )) && [[ " @SUBS@ " == *" $1 "* || " $* " == *" -"* ]]; then command rekollect "$@"; return; fi
  local sel
  sel=$(command rekollect pick --query "$*" </dev/tty) || return
  [[ -z "$sel" ]] && return
  history -s -- "$sel"
  printf '\e[2m  ↑ to edit and run:\e[0m %s\n' "$sel" >&2
}
""".lstrip()

FISH = r"""
# ── rekollect (fish) ─────────────────────────────────────────────
status is-interactive; or exit 0
set -g __rk_dir @DIR@
set -g __rk_spool @SPOOL@
set -g __rk_every @EVERY@
set -g __rk_auto @AUTO@
set -g __rk_n 0
set -g __rk_sid fish:$fish_pid.(date +%s)
set -gx REKOLLECT_HOOKED fish
set -gx REKOLLECT_SESSION $__rk_sid

function __rk_postexec --on-event fish_postexec
    set -l ec $status
    set -l cmd $argv[1]
    test -z "$cmd"; and return
    string match -q -- ' *' $cmd; and return
    test -e $__rk_dir/paused; and return
    printf '%s\x1f%s\x1f%s\x1f%s\x1f%s\x1f%s\x1e' (date +%s) $ec $CMD_DURATION $PWD $__rk_sid "$cmd" >> $__rk_spool
    set -g __rk_n (math $__rk_n + 1)
    if test $__rk_auto -eq 1 -a $__rk_n -ge $__rk_every
        set -g __rk_n 0
        command rekollect index --quiet >/dev/null 2>&1 &
        disown 2>/dev/null
    end
end

function __rk_widget
    set -l sel (command rekollect pick --query (commandline) </dev/tty | string collect)
    if test -n "$sel"
        commandline -r -- $sel
    end
    commandline -f repaint
end
bind @FKEY@ __rk_widget

function rk
    if test (count $argv) -gt 0
        if contains -- $argv[1] @SUBS@; or string match -q -- '-*' $argv
            command rekollect $argv
            return
        end
    end
    set -l sel (command rekollect pick --query "$argv" </dev/tty | string collect)
    test -n "$sel"; or return
    # fish can't pre-fill the next prompt from a function, so show the pick and copy it;
    # Ctrl+G (the key binding) edits in place.
    if command rekollect pick --copy-only -- $sel 2>/dev/null
        printf '\e[2m  copied to clipboard (Ctrl+G edits in place in fish):\e[0m %s\n' $sel >&2
    else
        printf '\e[2m  picked (Ctrl+G edits in place in fish):\e[0m %s\n' $sel >&2
    end
end
""".lstrip()


POWERSHELL = r"""
# -- rekollect (PowerShell) ------------------------------------------
if (-not $global:__rk_loaded) {
    $global:__rk_loaded = $true
    $global:__rk_dir   = @DIR@
    $global:__rk_spool = @SPOOL@
    $global:__rk_every = @EVERY@
    $global:__rk_auto  = @AUTO@
    $global:__rk_n     = 0
    $global:__rk_win   = ($PSVersionTable.PSVersion.Major -le 5) -or $IsWindows
    $global:__rk_utf8  = New-Object System.Text.UTF8Encoding $false
    $global:__rk_sid   = "powershell:$PID.$([DateTimeOffset]::Now.ToUnixTimeSeconds())"
    $global:__rk_subs  = @(@SUBS_PS@)
    $global:__rk_exe   = (Get-Command rekollect -CommandType Application -ErrorAction SilentlyContinue |
                          Select-Object -First 1).Source
    if (-not $global:__rk_exe) { $global:__rk_exe = 'rekollect' }
    $__rk_h = Get-History -Count 1
    $global:__rk_last = if ($__rk_h) { $__rk_h.Id } else { 0 }
    $env:REKOLLECT_HOOKED = 'powershell'
    $env:REKOLLECT_SESSION = $global:__rk_sid

    $global:__rk_orig_prompt = $function:prompt
    function global:prompt {
        $ok = $global:?
        $code = $global:LASTEXITCODE
        try {
            $h = Get-History -Count 1
            if ($h -and $h.Id -ne $global:__rk_last) {
                $global:__rk_last = $h.Id
                $cmd = $h.CommandLine
                if ($cmd -and -not $cmd.StartsWith(' ') -and
                    -not (Test-Path -LiteralPath (Join-Path $global:__rk_dir 'paused'))) {
                    $ec = if ($ok) { 0 } elseif ($code) { $code } else { 1 }
                    $ts = [DateTimeOffset]::new($h.StartExecutionTime).ToUnixTimeSeconds()
                    $ms = [int64]($h.EndExecutionTime - $h.StartExecutionTime).TotalMilliseconds
                    $FS = [char]0x1f; $RS = [char]0x1e
                    $rec = "$ts$FS$ec$FS$ms$FS$((Get-Location).ProviderPath)$FS$($global:__rk_sid)$FS$cmd$RS"
                    [System.IO.File]::AppendAllText($global:__rk_spool, $rec, $global:__rk_utf8)
                    if ($global:__rk_auto -and (++$global:__rk_n -ge $global:__rk_every)) {
                        $global:__rk_n = 0
                        if ($global:__rk_win) {
                            Start-Process -FilePath $global:__rk_exe -ArgumentList 'index', '--quiet' -WindowStyle Hidden
                        } else {
                            Start-Process -FilePath $global:__rk_exe -ArgumentList 'index', '--quiet' `
                                -RedirectStandardOutput '/dev/null' -RedirectStandardError '/dev/null'
                        }
                    }
                }
            }
        } catch { }
        $global:LASTEXITCODE = $code
        if ($global:__rk_orig_prompt) { & $global:__rk_orig_prompt } else { "PS $($executionContext.SessionState.Path.CurrentLocation)> " }
    }

    # Runs the picker; returns the chosen command or $null.
    function global:__rk_pick([string]$query) {
        $out = [System.IO.Path]::GetTempFileName()
        $env:REKOLLECT_QUERY = $query
        try { $sz = $Host.UI.RawUI.WindowSize; $env:REKOLLECT_TERM_SIZE = "$($sz.Width)x$($sz.Height)" } catch { }
        try {
            & $global:__rk_exe pick --out $out
            if ($LASTEXITCODE -eq 0) { return [System.IO.File]::ReadAllText($out, $global:__rk_utf8) }
            if ($LASTEXITCODE -ge 2) { Write-Host "rekollect: picker failed (rk doctor for details)" -ForegroundColor Yellow }
            return $null
        } finally {
            Remove-Item Env:REKOLLECT_QUERY -ErrorAction SilentlyContinue
            Remove-Item Env:REKOLLECT_TERM_SIZE -ErrorAction SilentlyContinue
            Remove-Item -LiteralPath $out -ErrorAction SilentlyContinue
        }
    }

    # `rk` / `rk <words>`: picker without a key binding; the choice lands on your next prompt.
    function global:rk {
        if ($args.Count -gt 0 -and (($global:__rk_subs -contains "$($args[0])") -or
                                    ($args | Where-Object { "$_".StartsWith('-') }))) {
            & $global:__rk_exe @args
            return
        }
        $sel = __rk_pick ($args -join ' ')
        if (-not $sel) { return }
        if (Get-Module PSReadLine) {
            $global:__rk_pending = $sel
            $null = Register-EngineEvent -SourceIdentifier PowerShell.OnIdle -MaxTriggerCount 1 -Action {
                if ($global:__rk_pending) {
                    [Microsoft.PowerShell.PSConsoleReadLine]::Insert($global:__rk_pending)
                    $global:__rk_pending = $null
                }
            }
        } else {
            Write-Host $sel
        }
    }

    if (Get-Module PSReadLine) {
        $__rk_handler = {
            $line = $null; $cursor = $null
            [Microsoft.PowerShell.PSConsoleReadLine]::GetBufferState([ref]$line, [ref]$cursor)
            $sel = __rk_pick $line
            if ($sel) { [Microsoft.PowerShell.PSConsoleReadLine]::Replace(0, $line.Length, $sel) }
            [Microsoft.PowerShell.PSConsoleReadLine]::InvokePrompt()
        }
        foreach ($__rk_chord in @('@PSKEY@', '@PSALT@')) {
            Set-PSReadLineKeyHandler -Chord $__rk_chord -BriefDescription 'rekollect' `
                -Description 'Semantic search of shell history' -ScriptBlock $__rk_handler
        }
    }
}
""".lstrip()


def _q(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def _psq(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def _key_variants(key: str) -> tuple[str, str, str, str]:
    """'^G' → zsh '^G', bash '\\C-g', fish '\\cg', PowerShell 'Ctrl+g'."""
    if key.startswith("^") and len(key) == 2:
        ch = key[1].lower()
        return key, f"\\C-{ch}", f"\\c{ch}", f"Ctrl+{ch}"
    return key, key, key, key


def script(shell: str, cfg: C.Config) -> str:
    shell = ALIASES.get(shell, shell)
    tpl = {"zsh": ZSH, "bash": BASH, "fish": FISH, "powershell": POWERSHELL}.get(shell)
    if tpl is None:
        raise SystemExit(f"rekollect: unsupported shell {shell!r} (zsh, bash, fish, powershell)")
    zkey, bkey, fkey, pkey = _key_variants(str(cfg.get_path("ui.key", "^G")))
    q = _psq if shell == "powershell" else _q
    if shell == "powershell":
        tpl = tpl.replace("@AUTO@", "$true" if cfg.get_path("index.auto", True) else "$false")
        tpl = tpl.replace("@SUBS_PS@", ", ".join(_psq(x) for x in SUBCOMMANDS))
        tpl = tpl.replace("@PSALT@", PS_ALT_CHORD)
    return (tpl.replace("@DIR@", q(str(C.data_dir())))
               .replace("@SUBS@", " ".join(SUBCOMMANDS))
               .replace("@SPOOL@", q(str(C.spool_path())))
               .replace("@PSKEY@", pkey)
               .replace("@EVERY@", str(max(1, int(cfg.get_path("index.every", 10)))))
               .replace("@AUTO@", "1" if cfg.get_path("index.auto", True) else "0")
               .replace("@KEY@", zkey).replace("@BKEY@", bkey).replace("@FKEY@", fkey))


ALIASES = {"pwsh": "powershell", "ps": "powershell", "posh": "powershell", "powershell.exe": "powershell"}

RC_FILES = {"zsh": "~/.zshrc", "bash": "~/.bashrc", "fish": "~/.config/fish/config.fish"}
RC_LINES = {
    "zsh": 'eval "$(rekollect hook zsh)"',
    "bash": 'eval "$(rekollect hook bash)"',
    "fish": "rekollect hook fish | source",
    "powershell": "Invoke-Expression (& rekollect hook powershell | Out-String)",
}


def _run_ps(exe: str, command: str) -> str:
    import subprocess
    try:
        r = subprocess.run([exe, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command],
                           capture_output=True, text=True, timeout=20)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def powershell_profiles() -> list[tuple[str, Path, str]]:
    """(edition, profile path, execution policy) for each PowerShell on PATH.

    Uses $PROFILE.CurrentUserAllHosts (profile.ps1) so the hook loads in every host:
    Windows Terminal, the VS Code terminal and the VS Code PowerShell extension.
    """
    import shutil
    out = []
    for exe, edition in (("pwsh", "PowerShell 7"), ("powershell", "Windows PowerShell")):
        path = shutil.which(exe)
        if not path:
            continue
        res = _run_ps(path, "$PROFILE.CurrentUserAllHosts; Get-ExecutionPolicy")
        lines = [ln.strip() for ln in res.splitlines() if ln.strip()]
        if lines:
            out.append((edition, Path(lines[0]), lines[1] if len(lines) > 1 else ""))
    return out


def append_line(rc: Path, line: str) -> None:
    """Append without breaking the file's encoding (Windows profiles are often UTF-16)."""
    rc.parent.mkdir(parents=True, exist_ok=True)
    block = f"\n# rekollect: semantic shell history\n{line}\n"
    head = rc.read_bytes()[:2] if rc.exists() else b""
    if head in (b"\xff\xfe", b"\xfe\xff"):
        enc = "utf-16-le" if head == b"\xff\xfe" else "utf-16-be"
        with rc.open("ab") as f:
            f.write(block.replace("\n", "\r\n").encode(enc))
    else:
        nl = "\r\n" if rc.suffix == ".ps1" else "\n"
        with rc.open("a", encoding="utf-8", newline="") as f:
            f.write(block.replace("\n", nl))


def has_line(rc: Path, line: str) -> bool:
    if not rc.exists():
        return False
    raw = rc.read_bytes()
    text = raw.decode("utf-16", errors="ignore") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") \
        else raw.decode("utf-8", errors="ignore")
    return line in text