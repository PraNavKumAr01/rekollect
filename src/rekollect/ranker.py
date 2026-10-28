"""Hybrid ranking: dense semantic + BM25 lexical + recency + frequency + context.

Why hybrid?  Embeddings are great at "how did I shrink that video" → `ffmpeg
-crf 28`, but terrible at exact tokens like a hostname or a branch name. BM25 is
the reverse. Each signal is normalised to [0, 1] per query, then mixed with
user-tunable weights, so the blend behaves the same across embedding models
whose raw cosine ranges differ a lot.
"""

from __future__ import annotations

import bisect
import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from rekollect.models import Embedder
from rekollect.store import Corpus

_TOK = re.compile(r"[a-z0-9]+")
_FILTER = re.compile(r"(?:^|\s)([#@])([\w.-]+)")
STOPWORDS = {"the", "a", "an", "to", "of", "in", "on", "for", "and", "or", "my", "i", "it", "is", "with", "that",
             "this", "from", "by", "at", "be", "do", "did", "how", "what", "which", "me", "all", "up"}


def tokens(s: str) -> list[str]:
    return _TOK.findall(s.lower())


def parse_query(query: str) -> tuple[str, list[str], list[str]]:
    """'#deploy @prod-api restart pods' → ('restart pods', ['deploy'], ['prod-api'])."""
    tags = [m.group(2).lower() for m in _FILTER.finditer(query) if m.group(1) == "#"]
    names = [m.group(2).lower() for m in _FILTER.finditer(query) if m.group(1) == "@"]
    text = _FILTER.sub(" ", query)
    if query.endswith(" ") and not text.endswith(" "):
        text += " "
    return text if text.strip() else text.strip(), tags, names


@dataclass
class Hit:
    idx: int
    score: float
    parts: dict[str, float] = field(default_factory=dict)


class Ranker:
    K1, B = 1.2, 0.75

    def __init__(self, corpus: Corpus, emb: Optional[Embedder], weights: dict[str, float],
                 here_ids: set[int] = frozenset(), proj_ids: set[int] = frozenset(),
                 half_life_days: float = 21.0, now: Optional[float] = None,
                 feedback=None, learning: float = 0.0, branch_ids: set[int] = frozenset(),
                 adaptive: bool = True):
        self.c = corpus
        self.emb = emb
        self.feedback = feedback      # learn.Feedback or None
        self.adaptive = adaptive      # False: use the weights exactly as given (benchmarks)
        self.learning = float(learning)
        self.last_qv: Optional[np.ndarray] = None
        self.last_evidence: dict = {}
        tot = sum(max(0.0, float(v)) for v in weights.values()) or 1.0
        self.w = {k: max(0.0, float(v)) / tot for k, v in weights.items()}
        n = len(corpus.texts)
        self.n = n

        # BM25 postings
        post: dict[str, list[tuple[int, int]]] = defaultdict(list)
        dl = np.zeros(n, dtype=np.float32)
        hints = corpus.hints or [""] * len(corpus.texts)
        for i, (t, h) in enumerate(zip(corpus.texts, hints)):
            toks = tokens(t) + tokens(h)  # what it does counts as words too
            dl[i] = len(toks)
            tf: dict[str, int] = defaultdict(int)
            for tok in toks:
                tf[tok] += 1
            for tok, f in tf.items():
                post[tok].append((i, f))
        self.post = {k: (np.array([a for a, _ in v]), np.array([b for _, b in v], dtype=np.float32))
                     for k, v in post.items()}
        self.vocab = sorted(self.post)
        self.dl = dl
        self.avgdl = float(dl.mean()) if n else 1.0
        self.lower = [t.lower() for t in corpus.texts]

        # query-independent priors
        now = now or time.time()
        age_days = np.where(corpus.last_ts > 0, (now - corpus.last_ts) / 86400.0, 3650.0)
        self.recency = np.power(0.5, np.clip(age_days, 0, None) / max(half_life_days, 0.1))
        mx = math.log1p(corpus.count.max()) if n else 1.0
        self.frequency = np.log1p(corpus.count) / (mx or 1.0)
        ctx = np.zeros(n, dtype=np.float64)
        self.place = [""] * n  # for explanations
        if n and (here_ids or proj_ids):
            id_list = corpus.ids.tolist()
            for i, cid in enumerate(id_list):
                if cid in here_ids:
                    ctx[i], self.place[i] = 1.0, "dir"
                elif cid in proj_ids:
                    ctx[i], self.place[i] = 0.6, "repo"
                if cid in branch_ids and ctx[i] < 1.0:
                    ctx[i] = min(1.0, ctx[i] + 0.25)
                    self.place[i] = (self.place[i] + "+branch") if self.place[i] else "branch"
        self.context = ctx
        marks = corpus.marks or {}
        self.marked = np.zeros(n)
        for i in marks:
            self.marked[i] = 1.0
        self.tags = [set(marks[i][1].lower().split()) if i in marks else set() for i in range(n)]
        self.names = [marks[i][0].lower() if i in marks else "" for i in range(n)]
        # commands that never once succeeded are probably not what you want
        self.health = np.where((corpus.count > 0) & (corpus.fails >= corpus.count), 0.85, 1.0)

    # ── lexical ──────────────────────────────────────────────────────────────
    def _idf(self, df: int) -> float:
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def _bm25(self, q: str) -> np.ndarray:
        scores = np.zeros(self.n, dtype=np.float32)
        qt = tokens(q)
        if not qt:
            return scores
        prefix = None if q.endswith(" ") else qt[-1]
        terms: list[tuple[str, float]] = [(t, 1.0) for t in qt]
        if prefix and len(prefix) >= 2:  # search-as-you-type: last word is a prefix
            lo = bisect.bisect_left(self.vocab, prefix)
            hi = bisect.bisect_left(self.vocab, prefix + "￿")
            terms += [(t, 0.7) for t in self.vocab[lo:min(hi, lo + 64)] if t != prefix]
        norm = self.K1 * (1 - self.B + self.B * self.dl / max(self.avgdl, 1e-6))
        for t, boost in terms:
            p = self.post.get(t)
            if p is None:
                continue
            idx, tf = p
            s = self._idf(len(idx)) * tf * (self.K1 + 1) / (tf + norm[idx])
            scores[idx] += boost * s
        return scores

    def _signals(self, query: str) -> tuple[np.ndarray, np.ndarray]:
        """Per-command semantic and lexical match in [0, 1]."""
        q = query.strip()
        sem = np.zeros(self.n)
        self.last_qv = None
        if self.emb is not None and self.w.get("semantic", 0) > 0 and self.n and q:
            qv = self.emb.embed([q], query=True)[0]
            self.last_qv = qv
            sims = self.c.vecs @ qv
            ref = float(np.median(sims))
            top = float(sims.max())
            sem = np.clip((sims - ref) / max(top - ref, 1e-6), 0, 1)
        bm = self._bm25(query)  # lexical: BM25 + exact substring bonus
        lex = bm / bm.max() if self.n and bm.max() > 0 else bm
        ql = q.lower()
        self.last_exact = np.zeros(self.n)
        if len(ql) >= 3:
            sub = np.fromiter((ql in t for t in self.lower), dtype=bool, count=self.n)
            self.last_exact = sub * 1.0
            lex = np.maximum(lex, sub * 1.0)
        return sem, lex

    _IDENT = re.compile(r"[A-Za-z0-9]*[\d_./:@\\-][A-Za-z0-9_./:@\\-]*")

    def _identifier_like(self, q: str) -> bool:
        """A short query containing a token that looks like an identifier and appears verbatim."""
        words = q.split()
        if len(words) > 3:
            return False
        for w_ in words:
            if len(w_) >= 3 and self._IDENT.fullmatch(w_) and re.search(r"[\d_./:@\\]", w_):
                lw = w_.lower()
                if any(lw in t for t in self.lower):
                    return True
        return False

    def relevance(self, query: str) -> np.ndarray:
        """How well each command matches the query, ignoring recency/frequency/context."""
        sem, lex = self._signals(query)
        ws, wl = self.w.get("semantic", 0), self.w.get("lexical", 0)
        return (ws * sem + wl * lex) / ((ws + wl) or 1.0)

    # ── main entry ───────────────────────────────────────────────────────────
    def personal(self, query: str, qv: Optional[np.ndarray]) -> np.ndarray:
        if self.feedback is None and not self.marked.any():
            self.last_evidence = {}
            return np.zeros(self.n)
        if self.feedback is None:
            self.last_evidence = {}
            return np.tanh(0.35 * self.marked)
        sig, ev = self.feedback.signal(query, qv, self.n, marked=self.marked)
        self.last_evidence = ev
        return sig

    def search(self, query: str, limit: int = 8, mask: Optional[np.ndarray] = None) -> tuple[list[Hit], float]:
        t0 = time.perf_counter()
        if self.n == 0:
            return [], 0.0
        keep = np.ones(self.n) if mask is None else mask.astype(np.float64)
        query, tags, names = parse_query(query)
        for tag in tags:   # #tag: only bookmarks carrying a matching tag
            keep = keep * np.array([any(t.startswith(tag) for t in ts) for ts in self.tags], dtype=float)
        for name in names:  # @name: bookmarks by name
            keep = keep * np.array([nm.startswith(name) for nm in self.names], dtype=float)
        q = query.strip()
        if not q:  # empty query → "frecency" view, nudged by what you tend to pick
            pers = self.personal("", None)
            base = (0.55 * self.recency + 0.30 * self.frequency + 0.15 * self.context + 0.2 * pers) * keep
            order = [i for i in np.argsort(-base)[:limit] if base[i] > 0]
            hits = [Hit(int(i), float(base[i]), {"recency": float(self.recency[i]),
                                                  "frequency": float(self.frequency[i]),
                                                  "context": float(self.context[i]),
                                                  "personal": float(pers[i])}) for i in order]
            return hits, (time.perf_counter() - t0) * 1000

        sem, lex = self._signals(query)
        pers = self.personal(query, self.last_qv) if self.learning > 0 or self.marked.any() else np.zeros(self.n)
        w = dict(self.w)
        qt = tokens(q)
        if self.adaptive and len(qt) == 1 and not query.endswith(" ") and qt[0] not in self.post:
            # a single half-typed word ("terraf") embeds as noise; lean on lexical until it's complete
            shift = 0.65 * w.get("semantic", 0)
            w["semantic"] = w.get("semantic", 0) - shift
            w["lexical"] = w.get("lexical", 0) + shift
        elif self.adaptive and self._identifier_like(q):
            # "10.0.0.12", "pg_dump", "exp42": exact tokens matter more than meaning,
            # and containing the exact string beats sharing pieces of it ("10.0.0.1")
            shift = 0.5 * w.get("semantic", 0)
            w["semantic"] = w.get("semantic", 0) - shift
            w["lexical"] = w.get("lexical", 0) + shift
            if self.last_exact.any():
                lex = 0.4 * lex + 0.6 * self.last_exact
        score = (w.get("semantic", 0) * sem + w.get("lexical", 0) * lex
                 + w.get("recency", 0) * self.recency + w.get("frequency", 0) * self.frequency
                 + w.get("context", 0) * self.context) * self.health * keep
        score = score + max(self.learning, 0.1 if self.marked.any() else 0.0) * pers * keep

        k = min(limit, self.n)
        order = np.argpartition(-score, k - 1)[:k]
        order = order[np.argsort(-score[order])]
        hits = [Hit(int(i), float(score[i]), {
            "semantic": float(sem[i]), "lexical": float(lex[i]), "recency": float(self.recency[i]),
            "frequency": float(self.frequency[i]), "context": float(self.context[i]),
            "personal": float(pers[i]),
        }) for i in order if score[i] > 0]
        return hits, (time.perf_counter() - t0) * 1000

    # ── plain-English reasons ────────────────────────────────────────────────
    def explain(self, hit: Hit, query: str, now: Optional[float] = None) -> list[str]:
        from rekollect.ui import since
        i, p = hit.idx, hit.parts
        out: list[str] = []
        q, _, _ = parse_query(query)
        sem, lex = p.get("semantic", 0.0), p.get("lexical", 0.0)
        if q.strip():
            if self.c.embedded is not None and not self.c.embedded[i]:
                out.append("found by its words (meaning is still being indexed)")
            elif sem >= 0.75:
                out.append("means what you described")
            elif sem >= 0.4:
                out.append("close to what you described")
            words = sorted({t for t in tokens(q) if len(t) > 1 and t not in STOPWORDS and t in tokens(
                self.c.texts[i] + " " +
                                                                            (self.c.hints[i] if self.c.hints else ""))})
            if words and lex >= 0.2:
                out.append("contains " + ", ".join(f"“{w}”" for w in words[:4]))
        ev = self.last_evidence.get(i)
        if ev and ev.picks:
            out.append(f"you chose it for {'a similar search' if ev.picks == 1 else f'{ev.picks} similar searches'}")
        if ev and ev.rejections:
            out.append("you marked it “not this” for a similar search")
        if self.c.marks and i in self.c.marks:
            name, tags, note = self.c.marks[i]
            label = f"bookmarked{' as “' + name + '”' if name else ''}"
            if tags:
                label += " " + " ".join("#" + t for t in tags.split())
            out.append(label)
        place = self.place[i]
        if "dir" in place:
            out.append("run in this directory")
        elif "repo" in place:
            out.append("run in this repository" + (" on this branch" if "branch" in place else ""))
        elif "branch" in place:
            out.append("run on this branch")
        count, fails = int(self.c.count[i]), int(self.c.fails[i])
        if count and fails >= count:
            out.append("has never succeeded")
        elif fails:
            out.append(f"succeeded {count - fails} of {count} times")
        elif count > 1:
            out.append(f"always succeeded ({count} runs)")
        if self.c.last_ts[i]:
            out.append(f"last run {since(self.c.last_ts[i], now)}")
        return out