"""Measure search quality: `rk eval`.

Standard benchmark: a realistic developer history and questions with known answers
(bench/devshell.json). Each configuration (model tier × retrieval mode) ranks the
history for every question; we record where the first acceptable answer landed.

  hit@1   the right command was first
  hit@5   it was in the top five (what the picker shows without scrolling)
  MRR@10  mean of 1/position, counting misses beyond 10 as 0
  latency per query (embedding the query + ranking), mean and p95

All runs are saved to data_dir/eval-history.jsonl so every change to scoring,
models or normalisation can be compared with the last run (`--fail-under` turns
it into a regression gate for CI).
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from importlib import resources
from typing import Optional

import numpy as np

from rekollect import __version__
from rekollect import config as C
from rekollect.indexer import enrich, hint_for
from rekollect.ranker import Ranker
from rekollect.store import Corpus

MODES = {
    "semantic": {"semantic": 1.0},
    "lexical": {"lexical": 1.0},
    "hybrid": None,  # your effective weights
}


def load_bench(name: str = "devshell") -> dict:
    return json.loads(resources.files("rekollect").joinpath("bench", f"{name}.json").read_text(encoding="utf-8"))


@dataclass
class Result:
    tier: str
    mode: str
    hit1: float
    hit5: float
    mrr: float
    latency_ms: float
    p95_ms: float
    index_ms_per_1k: float = 0.0
    by_kind: dict = field(default_factory=dict)
    misses: list = field(default_factory=list)  # (query, best rank or None, what came first)
    bench: str = ""

    def row(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "misses"}


def build_corpus(commands: list[str], emb) -> tuple[Corpus, float]:
    hints = [hint_for(c) for c in commands]
    t0 = time.perf_counter()
    if emb is not None:
        vecs = emb.embed([enrich(c, h) for c, h in zip(commands, hints)])
    else:
        vecs = np.zeros((len(commands), 1), dtype=np.float32)
    ms = (time.perf_counter() - t0) * 1000
    n = len(commands)
    corpus = Corpus(ids=np.arange(n), texts=list(commands), vecs=vecs.astype(np.float32),
                    last_ts=np.full(n, time.time()), count=np.ones(n), fails=np.zeros(n),
                    last_cwd=[""] * n, hints=hints, embedded=np.full(n, emb is not None), marks={})
    return corpus, ms


def run(corpus: Corpus, emb, queries: list[dict], mode: str, weights: dict, tier: str,
        index_ms: float) -> Result:
    ranker = Ranker(corpus, emb, weights, adaptive=(mode == "hybrid"))
    pos_of = {t: i for i, t in enumerate(corpus.texts)}
    ranks, lat = [], []
    kinds: dict[str, list[Optional[int]]] = {}
    misses = []
    if emb is not None:
        emb.embed(["warm up"], query=True)
    for q in queries:
        t0 = time.perf_counter()
        hits, _ = ranker.search(q["q"], 10)
        lat.append((time.perf_counter() - t0) * 1000)
        want = {pos_of[e] for e in q["expect"]}
        rank = next((r for r, h in enumerate(hits) if h.idx in want), None)
        ranks.append(rank)
        kinds.setdefault(q.get("kind", "paraphrase"), []).append(rank)
        if rank != 0:
            misses.append((q["q"], rank, corpus.texts[hits[0].idx] if hits else ""))

    def metrics(rs: list[Optional[int]]) -> tuple[float, float, float]:
        n = max(1, len(rs))
        return (sum(r == 0 for r in rs) / n, sum(r is not None and r < 5 for r in rs) / n,
                sum(1 / (r + 1) for r in rs if r is not None) / n)

    h1, h5, mrr = metrics(ranks)
    by_kind = {k: dict(zip(("hit1", "hit5", "mrr"), metrics(v)), n=len(v)) for k, v in kinds.items()}
    lat_sorted = sorted(lat)
    return Result(tier, mode, h1, h5, mrr, statistics.mean(lat), lat_sorted[int(0.95 * (len(lat) - 1))],
                  index_ms * 1000 / max(1, len(corpus.texts)), by_kind, misses)


def history_path():
    return C.data_dir() / "eval-history.jsonl"


def previous(tier: str, mode: str, bench: str) -> Optional[dict]:
    try:
        lines = history_path().read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("tier") == tier and r.get("mode") == mode and r.get("bench") == bench:
            return r
    return None


BENCHES = ("devshell", "devshell-holdout")


def save(results: list[Result]) -> None:
    C.ensure_dirs()
    with history_path().open("a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({**r.row(), "version": __version__, "ts": int(time.time())}) + "\n")