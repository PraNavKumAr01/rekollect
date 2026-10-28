import os
import json
import struct

import numpy as np
import pytest

from rekollect.history import parse_bash, parse_fish, parse_spool, parse_zsh, _unmetafy
from rekollect.indexer import Filter, enrich
from rekollect.models import read_safetensors
from rekollect.ranker import Ranker, tokens
from rekollect.redact import MASK, redact
from rekollect.store import Corpus
from rekollect import config as C


# ── history parsers ──────────────────────────────────────────────────────────

def test_zsh_extended_and_multiline():
    text = ": 1700000000:3;git status\n: 1700000100:0;for f in *.png; do\\\n  echo $f\\\ndone\n: 1700000200:0;ls\n"
    es = list(parse_zsh(text))
    assert [e.cmd for e in es] == ["git status", "for f in *.png; do\n  echo $f\ndone", "ls"]
    assert es[0].ts == 1700000000 and es[0].duration_ms == 3000


def test_zsh_plain():
    assert [e.cmd for e in parse_zsh("ls -la\ngit pull\n")] == ["ls -la", "git pull"]


def test_zsh_unmetafy():
    assert _unmetafy(b"a\x83\xa2b") == b"a\x82b"


def test_bash_with_timestamps():
    es = list(parse_bash("#1700000000\nmake build\nnpm test\n"))
    assert es[0].cmd == "make build" and es[0].ts == 1700000000
    assert es[1].cmd == "npm test" and es[1].ts == 0


def test_fish():
    es = list(parse_fish("- cmd: echo hi\\nthere\n  when: 1700000000\n- cmd: ls\n  when: 1700000001\n  paths:\n    - x\n"))
    assert es[0].cmd == "echo hi\nthere" and es[0].ts == 1700000000
    assert es[1].cmd == "ls"


def test_spool():
    data = "1700000000\x1f0\x1f12\x1f/home/a\x1fgit push\x1e1700000001\x1f1\x1f5\x1f/tmp\x1fmake\x1e"
    es = parse_spool(data)
    assert es[0].cmd == "git push" and es[0].cwd == "/home/a" and es[0].exit == 0
    assert es[1].exit == 1 and es[1].duration_ms == 5


# ── privacy ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cmd,leak", [
    ("export OPENAI_API_KEY=sk-abc123abc123abc123abc123", "sk-abc"),
    ("curl -H 'Authorization: Bearer abcdefghijklmnop' x", "abcdefghijklmnop"),
    ("git clone https://bob:hunter2@github.com/x/y", "hunter2"),
    ("mysql --password=s3cret -h db", "s3cret"),
    ("gh auth login --with-token ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "ghp_"),
    ("aws configure set aws_access_key_id AKIAABCDEFGHIJKLMNOP", "AKIAABCDEFGHIJKLMNOP"),
    ("HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz123456 python x.py", "hf_abc"),
])
def test_redact(cmd, leak):
    out = redact(cmd)
    assert leak not in out and MASK in out


def test_redact_leaves_normal_commands():
    for c in ["docker run -p 8080:80 nginx", "git commit -m 'fix token parsing'", "ls -la ~/keys"]:
        assert redact(c) == c


def test_filter():
    f = Filter(C.Config(C.DEFAULTS))
    assert f(" secret command") is None  # leading space
    assert f("ls") is None and f("rk docker") is None
    assert f("cd ..") == "cd .."  # kept for workflows…
    from rekollect.store import NAVIGATION
    assert NAVIGATION.match("cd ..") and NAVIGATION.match("Set-Location C:\\x") and not NAVIGATION.match("cd x && make")
    assert f("docker compose up -d") == "docker compose up -d"


def test_enrich():
    assert "kubernetes" in enrich("sudo kubectl get pods -A")
    assert enrich("weird-tool --x") == "weird-tool --x"


# ── safetensors ──────────────────────────────────────────────────────────────

def test_safetensors_roundtrip(tmp_path):
    arr = np.arange(12, dtype=np.float16).reshape(3, 4)
    header = {"embeddings": {"dtype": "F16", "shape": [3, 4], "data_offsets": [0, arr.nbytes]}}
    h = json.dumps(header).encode()
    p = tmp_path / "m.safetensors"
    p.write_bytes(struct.pack("<Q", len(h)) + h + arr.tobytes())
    out = read_safetensors(p)
    assert np.array_equal(out["embeddings"], arr)


# ── ranking ──────────────────────────────────────────────────────────────────

class FakeEmb:
    """Bag-of-words hashing embedder: deterministic, no downloads."""
    dim = 64

    def embed(self, texts, query=False):
        out = np.zeros((len(texts), self.dim), np.float32)
        for i, t in enumerate(texts):
            for tok in tokens(t):
                out[i, hash(tok) % self.dim] += 1
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(n, 1e-9)


def _corpus(texts, ts=None, counts=None, fails=None):
    emb = FakeEmb()
    n = len(texts)
    return Corpus(ids=np.arange(n), texts=texts, vecs=emb.embed(texts),
                  last_ts=np.array(ts or [1.7e9] * n, float), count=np.array(counts or [1] * n, float),
                  fails=np.array(fails or [0] * n, float), last_cwd=[""] * n)


W = C.DEFAULTS["search"]["weights"]


def test_lexical_exact_beats_noise():
    c = _corpus(["git push origin main", "docker ps -a", "kubectl get pods -n kube-system"])
    hits, _ = Ranker(c, FakeEmb(), W, now=1.7e9).search("kubectl pods")
    assert c.texts[hits[0].idx].startswith("kubectl")


def test_prefix_search_as_you_type():
    c = _corpus(["terraform apply -auto-approve", "tar -xzf data.tgz"])
    hits, _ = Ranker(c, FakeEmb(), W, now=1.7e9).search("terraf")
    assert c.texts[hits[0].idx].startswith("terraform")


def test_failed_commands_are_demoted():
    # same bag of words → same semantic & lexical score; the never-succeeded one must lose the tie
    c = _corpus(["make deploy staging", "make staging deploy"], counts=[3, 3], fails=[3, 0])
    hits, _ = Ranker(c, FakeEmb(), W, now=1.7e9).search("staging make")
    assert c.texts[hits[0].idx] == "make staging deploy"


def test_mask_and_empty_query():
    c = _corpus(["a b c", "d e f"], ts=[1.7e9, 1.6e9])
    r = Ranker(c, FakeEmb(), W, now=1.7e9)
    hits, _ = r.search("", 5)
    assert hits[0].idx == 0
    hits, _ = r.search("", 5, mask=np.array([False, True]))
    assert [h.idx for h in hits] == [1]


# ── config ───────────────────────────────────────────────────────────────────

def test_config_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("REKOLLECT_HOME", str(tmp_path))
    C.save(C.DEFAULTS)
    cfg = C.load()
    assert cfg == C.DEFAULTS
    C.set_path(cfg, "search.weights.semantic", "0.7")
    C.set_path(cfg, "index.auto", "off")
    C.save(cfg)
    cfg2 = C.load()
    assert cfg2.get_path("search.weights.semantic") == 0.7 and cfg2.get_path("index.auto") is False
    assert cfg2.get_path("index.ignore") == C.DEFAULTS["index"]["ignore"]


def test_highlight_never_drops_characters():
    from rekollect.ui import highlight
    for s in ['psql -c "SELECT x(\'a\')" && echo $HOME', "echo 'unterminated", 'a "b', "x | y; z & w > f",
              "for f in *; do\n echo $f\ndone", "curl -H 'Authorization: Bearer ‹redacted›' http://x"]:
        assert highlight(s).plain == s.replace("\n", " ⏎ ")
        assert highlight(s, max_len=12).plain.rstrip("…") in s.replace("\n", " ⏎ ")


def test_partial_word_leans_lexical():
    c = _corpus(["terraform apply tfplan", "tar -xzf data.tgz", "git status"])
    hits, _ = Ranker(c, FakeEmb(), W, now=1.7e9).search("terraf")
    assert c.texts[hits[0].idx].startswith("terraform")


# ── Windows / PowerShell ─────────────────────────────────────────────────────

def test_parse_pwsh_history():
    from rekollect.history import parse_pwsh
    text = "﻿git status\r\npytest -x tests\r\nGet-ChildItem `\r\n  -Recurse `\r\n  -Filter *.py\r\n\r\nls\r\n"
    assert [e.cmd for e in parse_pwsh(text)] == [
        "git status", "pytest -x tests", "Get-ChildItem \n  -Recurse \n  -Filter *.py", "ls"]


def test_psreadline_history_is_discovered(tmp_path, monkeypatch):
    from rekollect import history
    d = tmp_path / "PSReadLine"
    d.mkdir()
    (d / "ConsoleHost_history.txt").write_text("pytest\n", encoding="utf-8")
    (d / "Visual Studio Code Host_history.txt").write_text("npm test\n", encoding="utf-8")
    monkeypatch.setattr(history, "psreadline_dirs", lambda: [d])
    monkeypatch.setattr(history.Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HISTFILE", raising=False)
    found = [(k, p.name) for k, p in history.history_files()]
    assert ("powershell", "ConsoleHost_history.txt") in found
    assert ("powershell", "Visual Studio Code Host_history.txt") in found
    cmds = [e.cmd for k, p in history.history_files() for e in history.read_history(k, p)]
    assert sorted(cmds) == ["npm test", "pytest"]


def test_utf16_history_file(tmp_path):
    from rekollect.history import read_history
    p = tmp_path / "h.txt"
    p.write_bytes("docker ps\r\necho ünïcode\r\n".encode("utf-16"))
    assert [e.cmd for e in read_history("powershell", p)] == ["docker ps", "echo ünïcode"]


def test_detect_shell(monkeypatch):
    from rekollect import history
    monkeypatch.setattr(history.os, "name", "nt")
    monkeypatch.delenv("SHELL", raising=False)
    assert history.detect_shell() == "powershell"
    monkeypatch.setenv("SHELL", "/usr/bin/bash")  # Git Bash
    assert history.detect_shell() == "bash"


def test_powershell_hook_script():
    from rekollect import hooks
    cfg = C.Config(C.DEFAULTS)
    s = hooks.script("pwsh", cfg)
    assert "@" + "DIR@" not in s and "@PSKEY@" not in s
    assert "Ctrl+g" in s and "$global:__rk_every = 10" in s and "$global:__rk_auto  = $true" in s
    assert s.strip().endswith("}")


def test_append_line_keeps_utf16_profile(tmp_path):
    from rekollect import hooks
    rc = tmp_path / "profile.ps1"
    rc.write_bytes("Set-Alias g git\r\n".encode("utf-16"))
    line = hooks.RC_LINES["powershell"]
    hooks.append_line(rc, line)
    text = rc.read_bytes().decode("utf-16")
    assert text.startswith("Set-Alias g git") and line in text and hooks.has_line(rc, line)


def test_index_lock_is_exclusive(tmp_path, monkeypatch):
    monkeypatch.setenv("REKOLLECT_HOME", str(tmp_path))
    from rekollect.store import index_lock
    import multiprocessing as mp
    with index_lock() as got:
        assert got
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(target=_try_lock_in_child, args=(str(tmp_path), q))
        p.start(); p.join(20)
        assert q.get(timeout=5) is False
    with index_lock() as got:
        assert got


def _try_lock_in_child(home, q):
    import os
    os.environ["REKOLLECT_HOME"] = home
    from rekollect.store import index_lock
    with index_lock() as got:
        q.put(got)


def test_windows_key_decoding(monkeypatch):
    from rekollect import picker

    class FakeMsvcrt:
        def __init__(self, chars):
            self.chars = list(chars)

        def getwch(self):
            return self.chars.pop(0)

        def kbhit(self):
            return bool(self.chars)

    monkeypatch.setattr(picker, "msvcrt", FakeMsvcrt(["\xe0", "H", "\x00", "P", "a", "\r", "\x1b"]), raising=False)
    t = picker.WinTty()
    assert [t.key() for _ in range(5)] == ["\x1b[A", "\x1b[B", "a", "\r", "\x1b"]


# ── config: overrides only ───────────────────────────────────────────────────

def test_config_file_holds_only_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("REKOLLECT_HOME", str(tmp_path))
    cfg = C.load()
    C.set_path(cfg, "ui.key", "ctrl+k")
    C.save(cfg)
    assert C.overrides() == {"ui": {"key": "^K"}}
    # a later change to a built-in default must reach users who never touched that setting
    monkeypatch.setitem(C.DEFAULTS["search"], "limit", 15)
    cfg = C.load()
    assert cfg.get_path("search.limit") == 15
    C.unset_path(cfg, "ui.key")
    C.save(cfg)
    assert C.overrides() == {}


def test_old_full_config_is_migrated(tmp_path, monkeypatch):
    import copy
    monkeypatch.setenv("REKOLLECT_HOME", str(tmp_path))
    old = copy.deepcopy(C.DEFAULTS)
    old["ui"]["key"] = "^K"
    old["search"]["limit"] = 12
    old["index"]["ignore"] = C._PAST_DEFAULTS["index.ignore"][0]
    C.config_path().write_text("\n".join(["# old"] + C._sections(old)), encoding="utf-8")
    assert C.overrides() == {"ui": {"key": "^K"}, "search": {"limit": 12}}
    assert C.FORMAT_MARKER in C.config_path().read_text(encoding="utf-8")
    assert C.load().get_path("index.ignore") == C.DEFAULTS["index"]["ignore"]


@pytest.mark.parametrize("raw,want", [("^g", "^G"), ("ctrl+g", "^G"), ("Ctrl-K", "^K"), ("C-r", "^R")])
def test_normalize_key(raw, want):
    assert C.normalize_key(raw) == want


def test_normalize_key_rejects():
    with pytest.raises(ValueError):
        C.normalize_key("F5")


# ── VS Code keybindings ──────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "",
    "[]",
    "// Place your key bindings in this file to override the defaults\n[\n]\n",
    '// mine\n[\n  {\n    "key": "ctrl+t", // c ] \n    "command": "x"\n  }\n  // last ]\n]\n',
    '[\n  { "key": "ctrl+e", "command": "x" },\n]',
    '[ { "key": "ctrl+e", "command": "type", "args": {"text": "a]b//c"} } ]',
])
def test_vscode_splice(text):
    from rekollect import vscode as V
    out = V.splice(text, "^G")
    data = V.parse_jsonc(out)
    assert data[-1] == {"key": "ctrl+g", "command": "workbench.action.terminal.sendSequence",
                        "args": {"text": "\x07"}, "when": "terminalFocus && terminalShellType != 'pwsh'"}
    assert data[-2] == {"key": "ctrl+g", "command": "workbench.action.terminal.sendSequence",
                        "args": {"text": "\x1b[24~r"}, "when": "terminalFocus && terminalShellType == 'pwsh'"}
    assert {"key": "ctrl+g", "command": "-workbench.action.terminal.goToRecentDirectory"} in data
    assert V.splice(out, "^G") == out                       # idempotent
    swapped = V.splice(out, "^K")
    assert swapped.count(V.MARK_BEGIN) == 1 and V.parse_jsonc(swapped)[-1]["args"]["text"] == "\x0b"
    if text.strip():
        assert V.parse_jsonc(V.splice(out, None)) == V.parse_jsonc(text)   # clean removal


def test_vscode_install_backs_up_and_refuses_garbage(tmp_path):
    from rekollect import vscode as V
    p = tmp_path / "keybindings.json"
    p.write_text('[{"key": "ctrl+e", "command": "x"}]', encoding="utf-8")
    assert V.install(p, "^G") == "added"
    assert (tmp_path / "keybindings.json.rekollect.bak").exists()
    assert V.install(p, "^G") == "unchanged"
    p.write_text("{ not an array", encoding="utf-8")
    with pytest.raises(ValueError):
        V.install(p, "^G")
    assert p.read_text(encoding="utf-8") == "{ not an array"


# ── templates ────────────────────────────────────────────────────────────────

TEMPLATE_CASES = [
    ("ssh deploy@10.0.0.12", "ssh deploy@{host}"),
    ("ssh -L 8888:localhost:8888 gpu-box", "ssh -L {port}:localhost:{port} {host}"),
    ("ssh -p 2222 ubuntu@ec2-3-90-1-1.compute.amazonaws.com", "ssh -p {port} ubuntu@{host}"),
    ("kubectl logs -f deploy/api -n prod", "kubectl logs -f deploy/{deployment} -n {namespace}"),
    ("kubectl -n prod set image deploy/api api=ghcr.io/acme/api:2.4.1",
     "kubectl -n {namespace} set image deploy/{deployment} api=ghcr.io/acme/api:{tag}"),
    ("kubectl config use-context prod-eu", "kubectl config use-context {context}"),
    ("kubectl port-forward svc/grafana 3000:80 -n monitoring",
     "kubectl port-forward svc/{service} {local_port}:{port} -n {namespace}"),
    ("helm upgrade --install redis bitnami/redis -n cache", "helm upgrade --install {release} bitnami/redis -n {namespace}"),
    ("docker run -d --name web -p 8080:80 nginx:1.25.3", "docker run -d --name {container} -p {port}:80 nginx:{tag}"),
    ("docker logs -f --tail 200 worker", "docker logs -f --tail 200 {container}"),
    ("docker build -t ghcr.io/acme/api:2.4.0 .", "docker build -t ghcr.io/acme/api:{tag} ."),
    ("git push --force-with-lease origin feature/auth", "git push --force-with-lease origin {branch}"),
    ("git -C ~/src/app checkout -b feat/x", "git -C {path} checkout -b {new_branch}"),
    ("git tag v1.4.0 && git push origin v1.4.0", "git tag {version} && git push origin {version}"),
    ('git commit -m "fix token parsing"', 'git commit -m "{message}"'),
    ("pip install torch==2.4.1", "pip install torch=={version}"),
    ("psql -h db.internal.acme.io -p 5433 -U analytics warehouse", "psql -h {host} -p {port} -U {user} {database}"),
    ("curl -H 'Authorization: Bearer ‹redacted›' https://api.acme.io/v1/users",
     "curl -H 'Authorization: Bearer {secret}' {url}"),
    ("lsof -i :8000", "lsof -i :{port}"),
    ("for f in *.jpg; do convert \"$f\" \"${f%.jpg}.webp\"; done", "for f in *.jpg; do convert \"$f\" \"${f%.jpg}.webp\"; done"),
    ("docker ps --format '{{.Names}}'", "docker ps --format '{{.Names}}'"),
    ("git status", "git status"),
]


@pytest.mark.parametrize("cmd,want", TEMPLATE_CASES)
def test_templatize(cmd, want):
    from rekollect.templates import templatize
    t = templatize(cmd)
    assert t.text() == want
    assert t.render() == cmd  # defaults give back the original exactly


def test_template_fill_and_shared_registry():
    from rekollect.templates import VOLATILE, fill, placeholders, templatize
    t = templatize("ssh -L 8888:localhost:8888 gpu-box")
    assert t.render({"port": "9999", "host": "gpu-2"}) == "ssh -L 9999:localhost:9999 gpu-2"
    reg: dict = {}
    a = templatize("docker build -t app:2.4.0 .", VOLATILE, reg).stored()
    b = templatize("docker push app:2.4.0", VOLATILE, reg).stored()
    assert a == "docker build -t app:{{tag}} ." and b == "docker push app:{{tag}}"
    assert placeholders(a + b) == ["tag"] and fill(b, {"tag": "3.0"}) == "docker push app:3.0"


def test_picker_fill_form():
    from rekollect.picker import FillForm
    from rekollect.templates import templatize
    f = FillForm(templatize("ssh -L 8888:localhost:8888 gpu-box"))
    for k in "9999":
        f.handle(k)          # typing replaces the remembered value
    assert f.handle("\r") is None
    f.handle("\x7f")         # backspace edits instead
    for k in "2":
        f.handle(k)
    assert f.handle("\r") == "done"
    assert f.result() == "ssh -L 9999:localhost:9999 gpu-bo2"


# ── workflows & recipes ──────────────────────────────────────────────────────

def _flow_store(tmp_path, monkeypatch):
    monkeypatch.setenv("REKOLLECT_HOME", str(tmp_path))
    from rekollect.store import Store
    st = Store()
    t = 1_700_000_000
    def sess(sid, start, cwd, cmds):
        for k, c in enumerate(cmds):
            ec = 0
            if isinstance(c, tuple):
                c, ec = c
            st.add_run(c, start + 60 * k, cwd, ec, 1000, "hook", sid)
    sess("a", t, "/p", ["git pull", ("pytest -q", 1), "git status", "pytest -q", "python -m build",
                        "twine upload dist/*", "git tag v1.0.0"])
    sess("a", t + 3 * 3600, "/p", ["ffmpeg -i a.mov b.mp4"])               # same session, after a long gap
    sess("b", t + 600, "/q", ["kubectl apply -f k8s/", "kubectl rollout status deploy/web"])
    st.commit()
    emb = FakeEmb()
    rows = st.db.execute("SELECT id, text FROM commands").fetchall()
    st.set_vectors([(i, v) for (i, _), v in zip(rows, emb.embed([t for _, t in rows]))])
    st.commit()
    return st, emb


def test_find_flows_groups_by_session_and_gap(tmp_path, monkeypatch):
    from rekollect import flows as F
    st, emb = _flow_store(tmp_path, monkeypatch)
    corpus = st.corpus()
    r = Ranker(corpus, emb, W)
    found = F.find_flows(st, corpus, r, "twine upload", now=1_700_000_000)
    top = found[0]
    texts = [s.text for s in top.steps]
    assert "twine upload dist/*" in texts and "python -m build" in texts
    assert "ffmpeg -i a.mov b.mp4" not in texts          # idle gap ends the episode
    assert "kubectl apply -f k8s/" not in texts          # other terminal session
    assert "git status" not in texts                     # noise
    assert texts.count("pytest -q") == 1 and top.hidden_failures == 1
    assert [s.label for s in top.steps][-3:] == ["Build the package", "Publish to the registry", "Tag the release"]


def test_task_query_strips_question_words():
    from rekollect.flows import task_query
    assert task_query("What steps did I use to publish this package?") == "publish package"
    assert task_query("how") == "how"


def test_recipe_roundtrip_and_placeholders(tmp_path, monkeypatch):
    monkeypatch.setenv("REKOLLECT_HOME", str(tmp_path))
    from rekollect import flows as F
    r = F.recipe_from_steps("ship-it", [("", "docker build -t app:2.4.0 .", "/w"), ("", "docker push app:2.4.0", "/w"),
                                        ("", r"cd C:\Users\me\it's here", "/w"), ("", "echo 'multi\nline'", "/w")],
                            "ship")
    assert r.vars == {"tag": "2.4.0"}
    F.save_recipe(r)
    back = F.load_recipe("ship-it")
    assert [s.run for s in back.steps] == [s.run for s in r.steps] and back.vars == r.vars
    assert [s.label for s in back.steps][:2] == ["Build the image", "Push the image"]
    assert F.render_steps(back, {"tag": "3.0"})[1] == "docker push app:3.0"
    assert F.list_recipes()[0].name == "ship-it"


def test_runner_builtin_steps(tmp_path):
    from rekollect import flows as F
    (tmp_path / "proj" / ".venv" / ("Scripts" if os.name == "nt" else "bin")).mkdir(parents=True)
    st = F.RunState(cwd=str(tmp_path), env={"PATH": "/usr/bin"})
    assert F.builtin_step("cd proj", st) and st.cwd == str(tmp_path / "proj")
    assert F.builtin_step("export STAGE=prod", st) and st.env["STAGE"] == "prod"
    assert F.builtin_step("$env:REGION = 'eu'", st) and st.env["REGION"] == "eu"
    assert F.builtin_step("source .venv/bin/activate", st)
    assert st.env["VIRTUAL_ENV"].endswith(".venv") and st.env["PATH"].split(os.pathsep)[0].endswith(("bin", "Scripts"))
    assert F.builtin_step("cd proj && make", st) is None   # compound commands run in the shell
    with pytest.raises(FileNotFoundError):
        F.builtin_step("cd nowhere", st)


# ── benchmark ────────────────────────────────────────────────────────────────

def test_benchmark_files_are_consistent():
    from rekollect import evaluate as E
    for name in E.BENCHES:
        b = E.load_bench(name)
        cmds = set(b["commands"])
        assert len(cmds) == len(b["commands"])
        for q in b["queries"]:
            assert q["expect"] and all(e in cmds for e in q["expect"]), q


def test_benchmark_lexical_regression_floor():
    """Keyword search needs no model, so it gates every change to hints, intents and BM25."""
    from rekollect import evaluate as E
    for name, floor in (("devshell", 0.78), ("devshell-holdout", 0.6)):
        b = E.load_bench(name)
        corpus, _ = E.build_corpus(b["commands"], None)
        r = E.run(corpus, None, b["queries"], "lexical", {"lexical": 1.0}, "none", 0.0)
        assert r.hit1 >= floor, (name, r.hit1)
        assert r.by_kind["keyword"]["hit1"] == 1.0 if "keyword" in r.by_kind else True


def test_identifier_queries_prefer_exact_matches():
    c = _corpus(["ping -c 4 10.0.0.1", "ssh deploy@10.0.0.12", "git status"])
    hits, _ = Ranker(c, FakeEmb(), W, now=1.7e9).search("10.0.0.12")
    assert c.texts[hits[0].idx] == "ssh deploy@10.0.0.12"


def test_intents_describe_commands():
    from rekollect.intents import describe
    assert "undo commit, keep changes staged" in describe("git reset --soft HEAD~1")
    assert any("without cache" in d for d in describe("docker build --no-cache -t api ."))
    assert describe("some-unknown-tool --flag") == []


def test_project_scope_is_path_aware(tmp_path, monkeypatch):
    monkeypatch.setenv("REKOLLECT_HOME", str(tmp_path))
    from rekollect.store import Store
    st = Store()
    st.add_run("make in", 1, "/w/proj", 0, 1, "hook")
    st.add_run("make sub", 2, "/w/proj/sub", 0, 1, "hook")
    st.add_run("make sibling", 3, "/w/proj-other", 0, 1, "hook")
    st.add_run("make win", 4, "C:\\w\\proj\\src", 0, 1, "hook")
    st.add_run("make under_score", 5, "/w/pXoj", 0, 1, "hook")
    st.commit()
    assert sorted(st.corpus(cwd_prefix="/w/proj").texts) == ["make in", "make sub"]
    assert st.corpus(cwd_prefix="C:\\w\\proj").texts == ["make win"]
    assert st.corpus(cwd_prefix="/w/p_oj").texts == []  # _ is literal, not a wildcard
    # scoped stats only count runs inside the project
    st.add_run("make in", 10, "/elsewhere", 2, 1, "hook")
    st.add_run("make in", 11, "/elsewhere", 2, 1, "hook")
    st.commit()
    c = st.corpus(cwd_prefix="/w/proj")
    i = c.texts.index("make in")
    assert (c.count[i], c.fails[i], c.last_ts[i], c.last_cwd[i]) == (1, 0, 1, "/w/proj")
    g = st.corpus()
    assert g.count[g.texts.index("make in")] == 3


# ── agent server (MCP) ───────────────────────────────────────────────────────

def _mcp_setup(tmp_path, monkeypatch, enabled=True, scope="project"):
    monkeypatch.setenv("REKOLLECT_HOME", str(tmp_path / "rk"))
    from rekollect.store import Store
    proj = tmp_path / "proj"
    (proj / ".git").mkdir(parents=True)
    st = Store()
    st.add_run("pytest tests/integration -m integration", 100, str(proj), 0, 1, "hook", "zsh:1.1")
    st.add_run("export TOKEN=‹redacted›", 101, str(proj), 0, 1, "hook", "zsh:1.1")
    st.add_run("ssh prod-db-1", 102, str(tmp_path / "elsewhere"), 0, 1, "hook", "zsh:1.1")
    st.add_run("cat ~/.secret-notes", 103, str(proj), 0, 1, "hook", "zsh:1.1")
    st.commit()
    cfg = C.Config(C._merge(C.DEFAULTS, {"agents": {"enabled": enabled, "scope": scope,
                                                     "exclude": [r"secret-notes"]},
                                          "model": {"tier": "lite"}}))
    from rekollect.mcp_server import Server
    return Server(cfg, str(proj)), st


def _call(server, name, args):
    return server.handle({"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                          "params": {"name": name, "arguments": args}})["result"]


def test_mcp_refuses_when_disabled(tmp_path, monkeypatch):
    server, _ = _mcp_setup(tmp_path, monkeypatch, enabled=False)
    init = server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                          "params": {"protocolVersion": "2025-06-18", "clientInfo": {"name": "t"}}})
    assert init["result"]["protocolVersion"] == "2025-06-18"
    res = _call(server, "search_commands", {"query": "integration tests"})
    assert res["isError"] and "rk agents enable" in res["content"][0]["text"]


def test_mcp_project_scope_exclusions_and_provenance(tmp_path, monkeypatch):
    server, _ = _mcp_setup(tmp_path, monkeypatch)
    tools = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]
    assert all(t["annotations"]["readOnlyHint"] for t in tools)
    assert not any("run" in t["name"] or "exec" in t["name"] for t in tools)  # no execution powers
    res = _call(server, "search_commands", {"query": "integration tests", "limit": 10})
    assert not res["isError"]
    cmds = [r["command"] for r in res["structuredContent"]["results"]]
    assert "pytest tests/integration -m integration" in cmds
    assert "ssh prod-db-1" not in cmds                          # outside the project
    assert not any("secret-notes" in c for c in cmds)            # user exclusion
    r0 = res["structuredContent"]["results"][0]
    assert r0["source"] == "shell-history" and r0["recorded_by"] == "rekollect shell hook"
    assert "not AI suggestions" in res["content"][0]["text"]
    assert _call(server, "search_commands", {"query": "x", "scope": "all"})["isError"]  # not allowed


def test_mcp_scope_all_when_allowed_and_store_is_readonly(tmp_path, monkeypatch):
    import sqlite3
    server, _ = _mcp_setup(tmp_path, monkeypatch, scope="all")
    res = _call(server, "recent_commands", {"scope": "all", "limit": 10})
    assert "ssh prod-db-1" in [r["command"] for r in res["structuredContent"]["results"]]
    store, _ = server.engine()
    with pytest.raises(sqlite3.OperationalError):
        store.db.execute("DELETE FROM commands")


def test_new_commands_are_searchable_before_embedding(tmp_path, monkeypatch):
    monkeypatch.setenv("REKOLLECT_HOME", str(tmp_path))
    from rekollect.store import Store
    st = Store()
    st.add_run("helm upgrade --install api ./chart", 1, "/w", 0, 1, "hook")
    st.commit()
    rows = st.db.execute("SELECT id, text FROM commands").fetchall()
    st.set_vectors([(rows[0][0], FakeEmb().embed([rows[0][1]])[0])])
    st.add_run("kubectl drain node-7 --ignore-daemonsets", 2, "/w", 0, 1, "hook")  # not embedded yet
    st.commit()
    c = st.corpus()
    assert len(c.texts) == 2 and list(c.embedded) == [True, False]
    hits, _ = Ranker(c, FakeEmb(), W, now=10).search("drain node")
    assert c.texts[hits[0].idx].startswith("kubectl drain")
    assert "still being indexed" in " ".join(Ranker(c, FakeEmb(), W, now=10).explain(hits[0], "drain node"))


def test_feedback_moves_similar_searches(tmp_path, monkeypatch):
    from rekollect.learn import Feedback
    c = _corpus(["make deploy staging", "make deploy production", "make test"])
    emb = FakeEmb()
    q = "deploy please"
    qv = emb.embed([q])[0]
    rows = [(1_700_000_000, q, qv.astype(np.float16).tobytes(), 1, "pick", 1, "")]
    fb = Feedback(rows, {0: 0, 1: 1, 2: 2}, emb.dim, now=1_700_000_000)
    plain = Ranker(c, emb, W, now=1.7e9)
    learned = Ranker(c, emb, W, now=1.7e9, feedback=fb, learning=0.3)
    assert plain.search("deploy please", 1)[0][0].idx == 0
    top = learned.search("deploy please", 1)[0][0]
    assert top.idx == 1 and "you chose it for a similar search" in learned.explain(top, "deploy please")


def test_tune_improves_replayed_choices():
    from rekollect.learn import Replay, score_replays, tune
    rng = np.random.default_rng(0)
    replays = []
    for _ in range(40):  # the user always wants the most *frequent* candidate; semantic is noise
        n = 20
        parts = np.vstack([rng.random(n), rng.random(n), rng.random(n), rng.random(n), np.zeros(n), np.zeros(n)])
        target = int(np.argmax(parts[3]))
        replays.append(Replay(target, np.arange(n), parts, np.ones(n)))
    w = dict(C.DEFAULTS["search"]["weights"])
    before = score_replays(replays, w, 0.3)["mrr"]
    res = tune(replays, w, 0.3)
    assert res["accepted"] and res["after"]["mrr"] > before and res["weights"]["frequency"] > 0.08
