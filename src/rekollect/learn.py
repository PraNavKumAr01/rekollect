"""Getting better with use: everything here is computed from your own picks, locally.

Three mechanisms, from immediate to slow:

1. Memory of choices (immediate). Each pick in the picker is logged with the
   query, the query's embedding and the position the command was shown at.
   For a new search, earlier searches that meant the same thing (similar
   embedding, or most of the same words) vote for what you chose then. Results
   you passed over count a little against; ^X ("not this") counts a lot.

2. Habits (gradual). Commands you pick often get a small, query-independent
   boost; bookmarks (^B) a slightly larger one.

3. Tuned weights (periodic). `rk tune` replays logged searches and adjusts how
   much meaning, words, recency, frequency, place and your own history count,
   to put your actual choices first. It only keeps the change if it also wins
   on the most recent searches it didn't train on.

The learning curve is the position your choice was at when you picked it: if
rekollect is learning, that drifts toward #1.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from rekollect import config as C

KIND_WEIGHT = {"pick": 1.0, "skip": -0.15, "irrelevant": -1.5}
HALF_LIFE_DAYS = 120.0
_TOK = re.compile(r"[a-z0-9]+")


def _toks(s: str) -> set[str]:
    return set(_TOK.findall(s.lower()))


@dataclass
class Evidence:
    picks: int = 0
    rejections: int = 0
    similar_queries: int = 0


class Feedback:
    """Past choices, ready to score a new query against."""

    def __init__(self, rows: list[tuple], id_to_idx: dict[int, int], dim: int, now: Optional[float] = None):
        now = now or time.time()
        keep = [r for r in rows if r[3] in id_to_idx]
        self.n = len(keep)
        self.idx = np.array([id_to_idx[r[3]] for r in keep], dtype=np.int64)
        self.kind = [r[4] for r in keep]
        self.weight = np.array([KIND_WEIGHT.get(r[4], 0.0) for r in keep])
        age = np.array([(now - r[0]) / 86400.0 for r in keep]) if keep else np.zeros(0)
        self.decay = np.power(0.5, np.clip(age, 0, None) / HALF_LIFE_DAYS)
        self.queries = [r[1] for r in keep]
        self.qtoks = [_toks(r[1]) for r in keep]
        self.ts = np.array([r[0] for r in keep])
        vecs = np.zeros((self.n, max(dim, 1)), dtype=np.float32)
        self.has_vec = np.zeros(self.n, dtype=bool)
        for k, r in enumerate(keep):
            if r[2] and len(r[2]) == dim * 2:  # same model as now (dims match)
                v = np.frombuffer(r[2], dtype=np.float16).astype(np.float32)
                nv = np.linalg.norm(v)
                if nv > 0:
                    vecs[k] = v / nv
                    self.has_vec[k] = True
        self.vecs = vecs
        size = (int(self.idx.max()) + 1) if self.n else 0
        picks = np.bincount(self.idx[np.array(self.kind) == "pick"], minlength=size) if self.n else np.zeros(0)
        self.pick_counts = picks

    def similarity(self, query: str, qv: Optional[np.ndarray]) -> np.ndarray:
        """How alike each logged query is to this one, in [0, 1]."""
        if not self.n:
            return np.zeros(0)
        qt = _toks(query)
        jac = np.array([len(qt & t) / len(qt | t) if (qt or t) else 0.0 for t in self.qtoks])
        sem = np.zeros(self.n)
        if qv is not None and self.has_vec.any() and len(qv) == self.vecs.shape[1]:
            nq = np.linalg.norm(qv)
            if nq > 0:
                cos = self.vecs @ (qv / nq)
                sem = np.where(self.has_vec, np.clip((cos - 0.6) / 0.35, 0, 1), 0)
        exact = np.array([q.strip().lower() == query.strip().lower() for q in self.queries], dtype=float)
        return np.maximum.reduce([jac, sem, exact])

    def signal(self, query: str, qv: Optional[np.ndarray], n: int, exclude: Optional[np.ndarray] = None,
               marked: Optional[np.ndarray] = None) -> tuple[np.ndarray, dict[int, Evidence]]:
        """Personal score per corpus command in [-1, 1], plus the evidence behind it."""
        out = np.zeros(n)
        ev: dict[int, Evidence] = {}
        if self.n:
            sim = self.similarity(query, qv) if query.strip() else np.zeros(self.n)
            related = sim >= 0.5
            if exclude is not None:
                related &= ~exclude
            contrib = self.weight * sim * self.decay * related
            np.add.at(out, self.idx, contrib)
            for k in np.nonzero(related)[0]:
                e = ev.setdefault(int(self.idx[k]), Evidence())
                e.similar_queries += 1
                if self.kind[k] == "pick":
                    e.picks += 1
                elif self.kind[k] == "irrelevant":
                    e.rejections += 1
            # habits: what you pick often, whatever you searched
            pc = self.pick_counts
            if exclude is not None and exclude.any():
                pc = pc - np.bincount(self.idx[exclude & (np.array(self.kind) == "pick")], minlength=len(pc))
            if len(pc):
                habit = np.log1p(np.clip(pc, 0, None)) / math.log1p(max(1, pc.max()))
                out[:len(habit)] += 0.15 * habit[:n]
        if marked is not None:
            out += 0.35 * marked
        return np.tanh(out), ev


# ── learned weights ──────────────────────────────────────────────────────────

SIGNALS = ("semantic", "lexical", "recency", "frequency", "context")


def learned_path():
    return C.data_dir() / "learned.json"


def load_learned() -> dict:
    try:
        return json.loads(learned_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def effective_weights(cfg: C.Config) -> tuple[dict, float, str]:
    """(weights, learning strength, where they came from). Explicit config wins over learned."""
    weights = dict(cfg.get_path("search.weights", {}))
    strength = float(cfg.get_path("search.learning", 0.3))
    user_set = C.overrides().get("search", {})
    if not cfg.get_path("search.learn", True):
        return weights, 0.0, "config (learning off)"
    learned = load_learned()
    if learned.get("weights") and "weights" not in user_set:
        weights = {**weights, **learned["weights"]}
        if "learning" not in user_set and "strength" in learned:
            strength = float(learned["strength"])
        return weights, strength, f"tuned on {learned.get('n', 0)} of your searches"
    return weights, strength, "defaults" if not user_set else "your config"


@dataclass
class Replay:
    """One logged pick, with every signal precomputed for its candidates."""
    target: int            # corpus index of what was picked
    cand: np.ndarray       # candidate corpus indices
    parts: np.ndarray      # (len(SIGNALS) + 1, len(cand)): signals + personal
    health: np.ndarray


def build_replays(ranker, feedback: Feedback, picks: list[int], max_cand: int = 200) -> list[Replay]:
    """picks: positions in `feedback` of 'pick' events to replay (leave-one-out for the personal part)."""
    out = []
    for k in picks:
        query = feedback.queries[k]
        target = int(feedback.idx[k])
        sem, lex = ranker._signals(query)
        qv = ranker.last_qv
        same_moment = np.abs(feedback.ts - feedback.ts[k]) <= 1  # this search's own events
        personal, _ = feedback.signal(query, qv, ranker.n, exclude=same_moment)
        base = 0.5 * sem + 0.5 * lex
        cand = np.argsort(-base)[:max_cand]
        if target not in cand:
            cand = np.append(cand, target)
        parts = np.vstack([sem[cand], lex[cand], ranker.recency[cand], ranker.frequency[cand],
                           ranker.context[cand], personal[cand]])
        out.append(Replay(target, cand, parts, ranker.health[cand]))
    return out


def score_replays(replays: list[Replay], weights: dict, strength: float) -> dict:
    tot = sum(max(0.0, weights.get(s, 0.0)) for s in SIGNALS) or 1.0
    w = np.array([max(0.0, weights.get(s, 0.0)) / tot for s in SIGNALS] + [strength])
    hit1 = hit5 = 0
    rr = 0.0
    for r in replays:
        base = w[:-1] @ r.parts[:-1]
        score = base * r.health + w[-1] * r.parts[-1]
        t = int(np.nonzero(r.cand == r.target)[0][0])
        rank = int((score > score[t]).sum())
        hit1 += rank == 0
        hit5 += rank < 5
        rr += 1.0 / (rank + 1)
    n = max(1, len(replays))
    return {"hit1": hit1 / n, "hit5": hit5 / n, "mrr": rr / n, "n": len(replays)}


def tune(replays: list[Replay], weights: dict, strength: float, holdout: float = 0.25) -> dict:
    """Coordinate search over weights on older searches, checked on the newest ones."""
    # replays are newest-first (feedback order): hold out the newest searches
    cut = max(1, int(len(replays) * holdout))
    test, train = replays[:cut], replays[cut:]
    if len(train) < 10:
        train = replays
    cur_w, cur_s = dict(weights), float(strength)
    best = score_replays(train, cur_w, cur_s)["mrr"]
    factors = (0.5, 0.75, 1.25, 1.6, 2.5)
    # trust region: tuning nudges each signal, it never switches one off. A few hundred
    # picks can't justify dropping (say) keyword matching that every other query relies on.
    lo = {k: 0.4 * v for k, v in weights.items()}
    hi = {k: 2.5 * v if v else 0.1 for k, v in weights.items()}
    for _ in range(3):
        improved = False
        for key in SIGNALS + ("strength",):
            for f in factors:
                w2, s2 = dict(cur_w), cur_s
                if key == "strength":
                    s2 = min(1.5, max(0.0, cur_s * f if cur_s else 0.1 * f))
                else:
                    v = cur_w.get(key, 0.0) * f if cur_w.get(key, 0.0) else 0.05 * f
                    w2[key] = min(hi.get(key, 0.1), max(lo.get(key, 0.0), v))
                if w2 == cur_w and s2 == cur_s:
                    continue
                if sum(w2.get(s, 0) for s in ("semantic", "lexical")) <= 0:
                    continue
                m = score_replays(train, w2, s2)["mrr"]
                if m > best + 1e-6:
                    best, cur_w, cur_s, improved = m, w2, s2, True
        if not improved:
            break
    before = score_replays(test, weights, strength)
    after = score_replays(test, cur_w, cur_s)
    tot = sum(cur_w.get(s, 0) for s in SIGNALS) or 1.0
    return {
        "weights": {s: round(cur_w.get(s, 0.0) / tot, 4) for s in SIGNALS},
        "strength": round(cur_s, 4),
        "before": before, "after": after,
        "accepted": after["mrr"] >= before["mrr"] - 1e-9 and (after["mrr"] > before["mrr"] or
                                                              after["hit1"] > before["hit1"]),
        "n": len(replays), "train": len(train), "test": len(test),
    }


def save_learned(result: dict) -> None:
    learned_path().write_text(json.dumps({
        "weights": result["weights"], "strength": result["strength"], "n": result["n"],
        "before": result["before"], "after": result["after"], "ts": int(time.time()),
    }, indent=2), encoding="utf-8")


# ── learning curve ───────────────────────────────────────────────────────────

def curve(rows: list[tuple], bucket_days: int = 7) -> list[dict]:
    """Where your pick was shown, per period: avg position and share picked at #1."""
    picks = sorted((r[0], r[5]) for r in rows if r[4] == "pick")
    if not picks:
        return []
    start = picks[0][0]
    buckets: dict[int, list[int]] = {}
    for ts, rank in picks:
        buckets.setdefault(int((ts - start) // (bucket_days * 86400)), []).append(int(rank))
    return [{"period": b, "n": len(v), "avg_position": 1 + sum(v) / len(v),
             "first": sum(1 for x in v if x == 0) / len(v)} for b, v in sorted(buckets.items())]