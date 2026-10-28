<div align="center">

# ◆ rekollect

**Find that command by describing it.**

Semantic search for your shell history. Runs fully offline on a tiny embedding model.

`zsh` · `bash` · `fish` · `PowerShell` · macOS · Linux · Windows

<img src="docs/demo.gif" width="900" alt="rekollect in a real zsh session: search by meaning, the Ctrl+G picker, filling a template and finding a whole release workflow">

<!-- Full 5-minute walkthrough: in GitHub's web editor, drag docs/rekollect-demo.mp4 onto this line.
     GitHub uploads it and pastes a https://github.com/user-attachments/assets/… link that plays inline. -->

</div>

```text
$ rk "shrink that video"
◆ rekollect  "shrink that video"  ·  8 of 12,408  ·  9 ms  ·  balanced
1  ▰▰▰▰▰  ffmpeg -i input.mov -vcodec libx264 -crf 28 output.mp4      ✓   8×  2w  ~/talks
2  ▰▰▰▱▱  ffmpeg -i talk.mp4 -vn -acodec mp3 talk.mp3                  ✓   2×  1mo ~/talks
3  ▰▰▱▱▱  convert screenshot.png -resize 50% small.png                 ✓   5×  3d  ~/blog
```

You remember *what* you did, not the exact flags. `Ctrl+R` only matches exact substrings, so
"undo last commit but keep changes" never finds `git reset --soft HEAD~1`. rekollect does. It also
turns past commands into reusable templates, and past sessions into workflows you can replay.

## Features

- **Search by meaning.** A local embedding model plus BM25, so "free up docker space" finds
  `docker system prune -af --volumes`, and exact tokens like hostnames still match.
- **Live picker on `Ctrl+G`, or just type `rk`.** Search as you type, then press Enter to put
  the command on your prompt. It's ready to edit, never auto-run.
- **Templates.** `ssh deploy@10.0.0.12` becomes `ssh deploy@{host}`; fill in a new host and go.
  Hosts, ports, namespaces, image tags, branches, versions, paths and more are detected.
- **Workflows and recipes.** Ask "what did I do to publish this package?" and get the actual
  sequence from your history, then save it as a recipe you can replay step by step.
- **Gets better as you use it.** Every pick teaches it what *you* mean. Similar searches later
  put that command first, `^X` says "not this", and `rk tune` refits the ranking to your habits.
  All local, with no telemetry.
- **Measured, not guessed.** `rk eval` scores search quality on a built-in benchmark, plus a
  held-out set, so changes to scoring or models can't quietly regress. `rk eval --mine` scores it
  on your own past searches.
- **Memory for your coding assistant.** `rk mcp` lets Claude Code, Cursor, VS Code and other MCP
  clients look up commands you've actually run ("how do I run the integration tests here?")
  instead of inventing them. It's read-only, project-scoped, opt-in and logged.
- **Knows where you are.** It boosts commands you ran in the current directory, git project or
  branch, and demotes ones that never succeeded.
- **Zero-cost recording.** The hook is one `printf >> file` per command, with no Python on your
  prompt. Embedding happens in the background every *N* commands.
- **Private by default.** Everything stays on your machine. API keys, tokens, passwords and
  `user:pass@` URLs are redacted *before* storage. Prefix a command with a space to keep it out
  entirely, or use `rk pause` for incognito.
- **Three model tiers.** From 34 MB of static embeddings to EmbeddingGemma. Switch with one
  command and it re-embeds for you.
- **Extras:** bookmarks with names, `#tags` and notes; plain-English "why this result";
  `rk stats` dashboard; `--json` for scripting; `rk forget` to delete commands.

## Install

```bash
pipx install rekollect      # or: uv tool install rekollect
rk init
```

`rk init` lets you pick a model, downloads it once, indexes your existing history, and offers to
add one line to your shell rc file:

```bash
eval "$(rekollect hook zsh)"     # zsh  (~/.zshrc)
eval "$(rekollect hook bash)"    # bash (~/.bashrc)
rekollect hook fish | source     # fish (~/.config/fish/config.fish)
```

```powershell
# PowerShell 7 and Windows PowerShell 5.1 ($PROFILE.CurrentUserAllHosts, so it also
# loads in the VS Code terminal and the PowerShell extension)
Invoke-Expression (& rekollect hook powershell | Out-String)
```

### VS Code, Cursor and other VS Code-based editors

VS Code handles keys before its terminal does, and in a focused terminal Ctrl+G runs *Go to Recent
Directory*. Typing `rk` always works. To get Ctrl+G back, `rk init` offers to fix it, or you can run
it any time:

```bash
rk vscode            # add the keybinding to every VS Code-family editor found
rk vscode --remove   # take it out again
```

It adds a clearly marked block to your `keybindings.json` (comments and existing entries are kept,
and a backup is written next to it):

```jsonc
// >>> rekollect: send the picker key to the shell (rk vscode --remove to undo)
{"key": "ctrl+g", "command": "-workbench.action.terminal.goToRecentDirectory"},
{"key": "ctrl+g", "command": "workbench.action.terminal.sendSequence", "args": {"text": "\u001b[24~r"}, "when": "terminalFocus && terminalShellType == 'pwsh'"},
{"key": "ctrl+g", "command": "workbench.action.terminal.sendSequence", "args": {"text": "\u0007"}, "when": "terminalFocus && terminalShellType != 'pwsh'"}
// <<< rekollect
```

For most shells VS Code types the raw Ctrl+G byte, which is what the shell would have got without
VS Code in the way. On Windows that byte doesn't reliably reach PSReadLine as a key press, so for
PowerShell it sends `F12, r` instead. VS Code's own shell integration uses the same trick for
Ctrl+Space and Shift+Enter. The PowerShell hook binds both. Changing `ui.key` updates this block
automatically, and `rk keys` shows exactly what your terminal sends.

### Windows notes

- `rk init` reads your existing PSReadLine history (`%APPDATA%\Microsoft\Windows\PowerShell\PSReadLine\*_history.txt`,
  which includes the separate VS Code host history) and adds the hook to every PowerShell it finds.
- If Windows PowerShell says *running scripts is disabled*, your profile can't load. Run
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once. `rk init` warns about this.
- PSReadLine history has no timestamps, so imported commands start without recency information.
  Everything recorded by the hook afterwards has timestamps, exit codes, durations and directories.
- `cmd.exe` has no prompt hook, so it can't record. Searching (`rk …`) works from any terminal.

## Usage

### Find a command

Three ways to open the picker, all with live results that change as you type:

| | |
|---|---|
| `Ctrl+G` | at your prompt; whatever you've typed so far becomes the search |
| `rk` | with the shell hook, opens the picker with no shortcut needed |
| `rk docker clean` + Enter | same, pre-filled with your words |

With `rk`, the command you pick is waiting on your **next prompt** (zsh, PowerShell). In bash it's
one `↑` away. `rk` is a shell function, so it also works where a terminal or editor grabs Ctrl+G.

```bash
rk s "that ssh tunnel to the gpu box"    # print matches instead of opening the picker
rk s -n 20 --here --ok --since 2w kube    # this project, only successful, last 2 weeks
rk s "docker cleanup" --why              # show how each signal contributed
```

| key in picker | action |
|---|---|
| type | search as you type |
| `↑` `↓` | move |
| `⏎` / `Tab` | put the command on your prompt |
| `^T` | template view: see each command with its variable parts as `{placeholders}` |
| `^E` | why: the reasons this result is here, in plain English |
| `^B` | star (bookmark) it; `rk mark` adds a name, tags and a note |
| `^X` | not this: hide it, and rank it lower for searches like this one |
| `^S` | only this project |
| `^O` | only commands that succeeded |
| `^Y` | copy to clipboard |
| `^D` `^D` | forget this command forever |
| `Esc` | close |

The panel under the list previews the selected command in full: where it ran, on which branch,
how often it ran and whether it succeeded, plus its template (`^T`) or its reasons (`^E`). The
footer spells out that Enter *inserts* the command; nothing runs until you press Enter again at
your prompt.

Type `#deploy` or `@prod-api` anywhere in a search to narrow it to your bookmarks.

### Reuse a command as a template

Press `^T` in the picker. Each result shows its template, with the original underneath:

```text
▌ ssh -L {port}:localhost:{port} {host}                          ✓   8×  1mo
  was  ssh -L 8888:localhost:8888 gpu-box   ⏎ fill in
```

Enter opens a small form. Every field starts with the value from history: typing replaces it,
Backspace edits it, and Enter moves on. Values used more than once (the two ports above) are one
field. The filled command goes to your prompt for a last look; nothing runs until you press Enter.

What gets detected, mostly from context rather than guesswork:

| | example | becomes |
|---|---|---|
| hosts, users@hosts, IPs | `ssh deploy@10.0.0.12` | `ssh deploy@{host}` |
| ports | `lsof -i :8000`, `-p 8080:80`, `-L 8888:…` | `:{port}` |
| Kubernetes | `kubectl logs deploy/api -n prod` | `deploy/{deployment} -n {namespace}` |
| | `kubectl config use-context prod-eu` | `{context}` |
| Helm releases | `helm upgrade --install redis …` | `{release}` |
| containers, images, tags | `docker run --name web nginx:1.25.3` | `--name {container} nginx:{tag}` |
| git | `git push origin feature/auth`, `git tag v1.4.0` | `{branch}`, `{version}` |
| versions, shas, ids | `pip install torch==2.4.1`, `git revert 3f9c2ab` | `{version}`, `{sha}` |
| databases | `psql -h db.acme.io -U analytics warehouse` | `-h {host} -U {user} {database}` |
| files, paths, URLs, secrets | `tar -xzvf data.tar.gz -C ~/data` | `{file}`, `{path}`, `{url}`, `{secret}` |

`rk tpl '<command>'` shows the template for any command.

### Find a workflow

```bash
rk flow "what steps did I use to publish this package"
```

```text
◆ rekollect  flow  "what steps did I use to publish this package"  ·  2 found  ·  8 ms

 1   ~/projects/rekollect · 5d ago · took 6 min
   1  Get the latest changes   git pull
   2  Install dependencies     pip install -e ".[dev]"  ●
   3  Lint / format            ruff check .
   4  Run tests                pytest -q
   5  Clean build output       rm -rf dist
   6  Build the package        python -m build          ●
   7  Publish to the registry  twine upload dist/*      ●
   8  Tag the release          git tag v0.2.0
   9  Push the tag             git push origin v0.2.0
```

Every step is a command you actually ran; nothing is generated. rekollect finds the commands that
match your task (● marks them), then takes the steps around them from the same terminal session,
until you'd been idle for 20 minutes. Failed attempts and noise like `git status` are dropped,
and repeats of the same sequence are merged ("done 3×"). Step labels come from rules about the
command itself (`twine upload` → "Publish to the registry").

Workflows are rebuilt from commands recorded by the shell hook (which stores a session id per
terminal). Imported history without timestamps can't be grouped this way.

### Save and replay recipes

```bash
rk flow "deploy the api" --save deploy-api   # save a workflow (or answer the prompt after rk flow)
rk recipe save release --last 6              # save the last 6 commands from this terminal
rk recipe                                    # list
rk recipe run deploy-api                     # step through it
rk recipe run deploy-api --var tag=2.5.0 --var namespace=staging --dry-run
```

When saving, values that change between runs become placeholders shared by every step, with the
recorded value as the default:

```text
◆ deploy-api  deploy the api
  values (asked when you run it)
   {{context}}       prod-eu
   {{tag}}           2.4.1
   {{namespace}}     prod
   {{deployment}}    api
   1  Switch cluster context  kubectl config use-context {context}
   2  Build the image         docker build -t ghcr.io/acme/api:{tag} .
   3  Push the image          docker push ghcr.io/acme/api:{tag}
   4  Roll out the new image  kubectl -n {namespace} set image deploy/{deployment} api=ghcr.io/acme/api:{tag}
   5  Wait for the rollout    kubectl rollout status deploy/{deployment} -n {namespace}
```

`rk recipe run` asks for the values once, then shows each step and waits: Enter runs it, `s` skips,
`e` edits it first, `q` stops. If a step fails you can retry, edit, skip or stop. `--step 3` runs
one step and `--from 3` resumes. `cd`, environment variables (`export`, `$env:`) and
virtualenv activation carry over to later steps, as they would in your shell.

Recipes are plain TOML in `~/.config/rekollect/recipes/`. Edit them with `rk recipe edit <name>`,
or keep them in git and share them with your team.

## Gets better as you use it

Your choices are the training data, and they never leave your machine.

1. **It remembers what you chose.** Each pick in the picker is logged: the search, what you
   picked, the position it was shown at, and what you passed over. When a new search means the
   same as an old one (similar embedding, or mostly the same words), the old choice is boosted.
   Results you skipped count slightly against; `^X` counts strongly against.
2. **It picks up habits.** Commands you choose often get a small boost on every search, and
   bookmarks a bigger one.
3. **It retunes itself.** `rk tune` replays your logged searches and adjusts how much meaning,
   words, recency, frequency, place and your own history count, to put your actual choices
   first. It trains on older searches and keeps the change only if it also wins on your most
   recent ones. It runs automatically in the background every 40 new picks; `rk tune --reset`
   undoes it.

You can watch it happen. `rk stats` shows, week by week, the position your choice was at when you
picked it, and that number falls toward #1. In a simulated six weeks of use (one person with
consistent habits the defaults don't know about), first-try picks went from 79% in week 1 to 95%
in week 6 from memory alone, and `rk tune` then took the held-out recent searches from 97% to 100%.

```bash
rk eval --mine     # how the current ranking does on your own past searches
rk tune --dry-run  # what tuning would change, without saving it
```

## Measuring search quality

```bash
rk eval                         # every downloaded model × semantic / keyword / hybrid
rk eval --tier all              # download and compare all three tiers
rk eval --misses                # which questions weren't answered first, and what came first instead
rk eval --fail-under hit5=0.9   # exit 1 below a threshold: a regression gate for CI
```

The built-in benchmark is a realistic developer history (233 commands across git, Docker,
Kubernetes, Python, Node, files, networking, databases, cloud and PowerShell) and 93 questions
with known answers, mostly phrased nothing like the command ("undo my last commit but keep the
changes"). It reports how often the right command came **first** (hit@1), how often it was in the
**top five** (hit@5), MRR, and latency. Every run is saved, and the next one shows ▲▼ deltas.

A second **held-out** set of 52 questions was written before any tuning, and its misses are never
listed while improving things, so it shows whether improvements generalise rather than just fit
the questions. Both sets were written by the same author, so they're a regression gate, not an
independent study. Your own picks (`rk eval --mine`) are the real test.

These numbers came from building this release, using the `lite` architecture (a static embedding
model) as a stand-in. The machine it was developed on couldn't download the shipped models, so run
`rk eval` yourself for your models:

| | main hit@1 | main hit@5 | held-out hit@1 | held-out hit@5 |
|---|---|---|---|---|
| hybrid, before | 51% | 75% | 46% | 83% |
| + built-in descriptions of what commands do, exact-match boost | **85%** | **96%** | **77%** | **96%** |
| keyword only (BM25), after | 82% | 96% | 65% | 90% |
| meaning only, after | 74% | 91% | 75% | 94% |

The big lever wasn't the model. Commands rarely say what they're for (`git reset --soft HEAD~1`
never mentions "undo"), so rekollect now indexes a short plain-English description of common
tools, subcommands and flags with every command, a small built-in tldr. CI fails if keyword-only
search drops below its floor.

## Coding assistants (MCP)

```bash
rk agents enable          # off by default
rk agents setup           # config snippets for Claude Code, Cursor, VS Code, …
rk agents log             # every request an assistant made, and what it got
```

`rk mcp` is a stdio [Model Context Protocol](https://modelcontextprotocol.io) server with five
tools: `search_commands`, `recent_commands`, `get_workflow`, `list_recipes` and `get_recipe`.
An assistant asked to "run the integration tests like I did last time" can retrieve
`pytest tests/integration -m integration` from your history instead of guessing.

| guarantee | how |
|---|---|
| **read-only** | the database is opened in SQLite read-only mode, and there is no tool that runs, writes or changes anything |
| **opt-in** | every tool refuses until you run `rk agents enable`; `rk agents disable` turns it off again |
| **project-scoped** | by default an assistant only sees commands run inside its project folder; seeing everything takes `rk agents enable --scope all` |
| **redacted** | secrets were masked when recorded and are masked again on the way out; `agents.exclude` regexes hide anything else |
| **provenance** | every result says it is `source: shell-history`, recorded by the hook or imported, with run count, exit status, directory and branch, and why it matched |
| **audited** | each request is logged with the client, the question and what was returned (`rk agents log`) |

Commands that were just run become visible to assistants after the next background index
(every *N* commands), because the server never writes.

## Models

All models run on CPU via ONNX Runtime, except `lite`, which doesn't need a neural net at all.

| tier | model | params | dim | download | when to pick it |
|---|---|---|---|---|---|
| `lite` | [potion-code-16M-v2](https://huggingface.co/minishlab/potion-code-16M-v2) | 16M static | 256 | ~34 MB | Old laptops, huge histories, or when you want instant results |
| `balanced` ★ | [granite-embedding-small-english-r2](https://huggingface.co/onnx-community/granite-embedding-small-english-r2-ONNX) int8 | 47M | 384 | ~55 MB | The default: strong retrieval quality for its size, Apache-2.0 |
| `quality` | [embeddinggemma-300m](https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX) int8 | 308M | 768 | ~300 MB | Vague natural-language queries; Gemma terms of use |
| `custom` | any ONNX sentence-embedding repo | | | | Set `model.custom.repo`, `pooling` and prefixes |

```bash
rk models            # compare, see what's downloaded
rk model lite        # switch tier + re-embed
```

## How it works

```text
your shell ──printf──▶ spool ──every N cmds──▶ rk index (background)
                                                │  filter → redact → dedupe
                                                ▼
                                      SQLite: unique commands + runs
                                      (vectors stored as float16)
                                                │
query ──embed──▶ cosine ─┐                      │
      ──BM25───▶ lexical ├─▶ weighted blend ◀───┘ recency · frequency · project
                         ┘        │
                                  ▼
                        ranked results / picker
```

1. **Recording.** Hooks append `ts, exit, duration, cwd, session, cmd` to a spool file using
   ASCII unit/record separators, so multi-line commands survive intact. Recording costs
   microseconds.
2. **Indexing.** Each *unique* command is embedded once. If you ran `git status` 4,000 times,
   that's still one vector. Before embedding, each command gets a few words about what it does:
   the tool (`kubectl` → "kubernetes cluster") and its intent (`twine upload` → "publish to the
   registry"). Shell commands are terse, and this gives both the model and keyword search
   something to match plain-English queries against.
3. **Hybrid ranking.** Five signals are each normalized to [0, 1] per query, then mixed with
   weights you can tune:
   - **semantic:** cosine similarity, rescaled against the median so models with different
     cosine ranges behave the same
   - **lexical:** BM25 plus an exact-substring bonus; the last word you're typing is treated as
     a prefix
   - **recency:** exponential decay with a 21-day half-life
   - **frequency:** log-scaled run count
   - **context:** 1.0 if run in this directory, 0.6 if run in this git project

   Commands that never once succeeded get a 0.85× penalty. While you're mid-word ("terraf"),
   weight shifts from semantic to lexical, because half a word embeds as noise.
4. **Static vs. transformer.** The `lite` tier uses [Model2Vec](https://github.com/MinishLab/model2vec):
   every token's vector is precomputed, so embedding is just lookup, average and normalize. It
   uses no ONNX, is memory-mapped, and pairs especially well with BM25. The other tiers run a
   real int8 transformer.

## Configuration

`~/.config/rekollect/config.toml`. The file holds only the settings you changed, so new
defaults from upgrades still reach you. `rk config` shows every effective value and marks the ones
you set:

```bash
rk config set index.every 25                 # embed every 25 commands
rk config set search.weights.semantic 0.6     # lean more on meaning
rk config set search.half_life_days 60        # care less about recency
rk config set ui.key '^R'                     # bind the picker to Ctrl+R instead
rk config unset ui.key                        # back to the default
rk config set ui.accent '#7aa2f7'             # tokyo-night-ish
rk config edit                                # open in $EDITOR
```

| key | default | |
|---|---|---|
| `model.tier` | `balanced` | `lite` · `balanced` · `quality` · `custom` |
| `model.threads` | `2` | CPU threads for ONNX |
| `index.auto` / `index.every` | `true` / `10` | Background embedding cadence |
| `index.redact_secrets` | `true` | Mask keys/tokens/passwords before storage |
| `index.ignore` | trivial `ls`/`pwd`/… | Regexes never indexed (`cd` is kept for workflows but hidden from search) |
| `search.limit` | `8` | |
| `search.weights.*` | `.50 .25 .10 .08 .07` | semantic, lexical, recency, frequency, context |
| `ui.accent` / `ui.key` | `#b48ead` / `^G` | `^G` becomes `Ctrl+g` in PowerShell |
| `search.learn` / `search.learning` | `true` / `0.3` | learn from your picks / how strongly (rk tune adjusts it) |
| `context.record_branch` / `record_host` | `true` / `true` | store git branch and hostname with each run |
| `agents.enabled` / `agents.scope` | `false` / `project` | coding-assistant access (rk agents) |
| `agents.exclude` | `[]` | regexes for commands assistants must never see |

Data lives in `~/.local/share/rekollect`, or `$REKOLLECT_HOME` if set.

## Privacy

- Nothing leaves your machine. The only network call is the one-time model download from Hugging Face.
  Set `HF_ENDPOINT` to use a mirror.
- Secrets are redacted with pattern rules: `*_KEY=`, `--password`, `Bearer …`, `user:pass@`,
  `sk-…`, `ghp_…`, `AKIA…`, JWTs and more. This is best effort, so use a leading space for
  anything sensitive.
- `rk forget '<regex>'` or `^D ^D` in the picker deletes commands. `rk pause` stops recording.
- Context stored with each run: directory, exit code, duration, a per-terminal session id, the
  shell, and (switchable off) the git branch (`context.record_branch`) and hostname
  (`context.record_host`). The branch is read from `.git/HEAD`; no git process runs.
- What you pick in the picker is stored locally to improve ranking (`search.learn = false` turns
  it off). Coding assistants see nothing unless you run `rk agents enable`.

## Development

```bash
git clone https://github.com/YOUR_USERNAME/rekollect && cd rekollect
pip install -e ".[dev]"
pytest
```

## License

MIT
