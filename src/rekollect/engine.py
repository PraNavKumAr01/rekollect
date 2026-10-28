"""Build a ranker the same way everywhere (CLI, picker, agent server, tuner)."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Optional

from rekollect import config as C
from rekollect.learn import Feedback, effective_weights
from rekollect.ranker import Ranker
from rekollect.store import Corpus, Store, git_branch, git_root

SYNC_EMBED_LIMIT = 64  # more pending than this (with a transformer model) → finish in the background


def catch_up(store: Store, cfg: C.Config, emb) -> int:
    """Index new commands; embed a few right now, leave a big backlog to a background indexer.

    Anything not embedded yet is still found by its words, so search never waits.
    Returns how many commands are still waiting.
    """
    from rekollect.indexer import drain_spool, embed_pending, ensure_model_matches, ingest, purge_ignored
    ingest(store, drain_spool(), cfg)
    purge_ignored(store, cfg)
    ensure_model_matches(store, cfg)
    pending = len(store.pending())
    if not pending:
        return 0
    if emb.spec.kind == "static" or pending <= SYNC_EMBED_LIMIT:
        embed_pending(store, emb)
        return 0
    try:  # detached background indexer
        kw: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL, "stdin": subprocess.DEVNULL}
        if os.name == "nt":
            kw["creationflags"] = 0x00000008 | 0x08000000  # DETACHED_PROCESS | CREATE_NO_WINDOW
        else:
            kw["start_new_session"] = True
        subprocess.Popen([sys.executable, "-m", "rekollect", "index", "--quiet"], **kw)
    except OSError:
        pass
    return pending


def make_ranker(store: Store, corpus: Corpus, emb, cfg: C.Config, cwd: Optional[str] = None,
                learning: bool = True) -> Ranker:
    cwd = cwd or os.getcwd()
    project = git_root(cwd)
    branch = git_branch(project) if project and cfg.get_path("context.record_branch", True) else ""
    here_ids, proj_ids, branch_ids = store.cmd_ids_in(cwd, project, branch)
    weights, strength, _ = effective_weights(cfg)
    feedback = None
    if learning and strength > 0:
        id_to_idx = {int(c): i for i, c in enumerate(corpus.ids)}
        dim = corpus.vecs.shape[1] if len(corpus.texts) else 0
        rows = store.feedback_rows()
        if rows:
            feedback = Feedback(rows, id_to_idx, dim)
    return Ranker(corpus, emb, weights, here_ids, proj_ids,
                  half_life_days=float(cfg.get_path("search.half_life_days", 21)),
                  feedback=feedback, learning=strength if learning else 0.0, branch_ids=branch_ids)