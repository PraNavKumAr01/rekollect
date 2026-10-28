"""Embedding model registry, downloader and the two inference backends.

Two very different kinds of embedder live here:

* StaticEmbedder  – Model2Vec "static" embeddings. Every token already has a
  precomputed vector, so embedding a command is: tokenize → look up rows →
  weighted average → L2 normalise. No neural net runs at query time, which is
  why it is ~1000x faster than a transformer and needs no ONNX Runtime at all.

* OnnxEmbedder    – a real (small) transformer exported to ONNX and
  int8-quantised, run with onnxruntime on CPU. Contextual, so better at
  understanding paraphrases, but needs a forward pass per batch.
"""

from __future__ import annotations

import json
import os
import struct
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from rekollect import config as C


# ── registry ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelSpec:
    tier: str
    repo: str
    kind: str  # "static" | "onnx"
    files: tuple[str, ...]
    dim: int
    params: str
    size_mb: int
    blurb: str
    onnx_file: str = ""
    optional_files: tuple[str, ...] = ()
    pooling: str = "mean"  # mean | cls (ignored if graph outputs sentence_embedding)
    query_prefix: str = ""
    doc_prefix: str = ""
    max_tokens: int = 128  # shell commands are short; keeps compute tiny
    license: str = ""

    @property
    def id(self) -> str:
        """Identity stored in the DB; if it changes we re-embed everything."""
        return f"{self.repo}:{self.onnx_file or 'static'}"


REGISTRY: dict[str, ModelSpec] = {
    "lite": ModelSpec(
        tier="lite",
        repo="minishlab/potion-code-16M-v2",
        kind="static",
        files=("config.json", "tokenizer.json", "model.safetensors"),
        dim=256,
        params="16M (static)",
        size_mb=34,
        blurb="Model2Vec static embeddings distilled from a code retriever. "
        "No transformer at runtime: a lookup + average. Pairs very well with BM25.",
        license="MIT",
    ),
    "balanced": ModelSpec(
        tier="balanced",
        repo="onnx-community/granite-embedding-small-english-r2-ONNX",
        kind="onnx",
        files=("tokenizer.json", "onnx/model_quantized.onnx", "onnx/model_quantized.onnx_data"),
        onnx_file="onnx/model_quantized.onnx",
        dim=384,
        params="47M",
        size_mb=55,
        pooling="cls",
        blurb="IBM Granite Embedding small R2 (ModernBERT), int8. Strong retrieval "
        "and code quality for its size. Recommended.",
        license="Apache-2.0",
    ),
    "quality": ModelSpec(
        tier="quality",
        repo="onnx-community/embeddinggemma-300m-ONNX",
        kind="onnx",
        files=("tokenizer.json", "onnx/model_quantized.onnx"),
        optional_files=("onnx/model_quantized.onnx_data",),
        onnx_file="onnx/model_quantized.onnx",
        dim=768,
        params="308M",
        size_mb=310,
        pooling="mean",
        query_prefix="task: search result | query: ",
        doc_prefix="title: none | text: ",
        blurb="Google EmbeddingGemma-300M, int8. Best understanding of vague, "
        "natural-language queries. Heavier; uses the Gemma terms of use.",
        license="Gemma",
    ),
}


def spec_for(cfg: C.Config) -> ModelSpec:
    tier = cfg.get_path("model.tier", "balanced")
    if tier == "custom":
        c = cfg.get_path("model.custom", {})
        if not c.get("repo"):
            raise SystemExit("rekollect: model.tier = 'custom' but model.custom.repo is empty")
        f = c.get("onnx_file", "onnx/model_quantized.onnx")
        return ModelSpec(
            tier="custom", repo=c["repo"], kind="onnx",
            files=("tokenizer.json", f), optional_files=(f + "_data",), onnx_file=f,
            dim=0, params="?", size_mb=0, blurb="User-supplied ONNX model",
            pooling=c.get("pooling", "mean"),
            query_prefix=c.get("query_prefix", ""), doc_prefix=c.get("doc_prefix", ""),
        )
    if tier not in REGISTRY:
        raise SystemExit(f"rekollect: unknown model tier {tier!r} (choose: {', '.join(REGISTRY)}, custom)")
    return REGISTRY[tier]


def model_path(spec: ModelSpec) -> Path:
    return C.models_dir() / spec.repo.replace("/", "--")


def is_downloaded(spec: ModelSpec) -> bool:
    root = model_path(spec)
    return all((root / f).exists() for f in spec.files)


def disk_usage(spec: ModelSpec) -> int:
    root = model_path(spec)
    if not root.exists():
        return 0
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


# ── downloader (plain HTTPS, no huggingface_hub dependency) ──────────────────

ProgressCb = Callable[[str, int, Optional[int]], None]  # (filename, done, total)


def _hf_url(repo: str, path: str) -> str:
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    return f"{endpoint}/{repo}/resolve/main/{path}"


def _fetch(url: str, dest: Path, name: str, progress: Optional[ProgressCb]) -> None:
    headers = {"User-Agent": "rekollect/0.1"}
    if tok := os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {tok}"
    req = urllib.request.Request(url, headers=headers)
    tmp = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = r.headers.get("Content-Length")
        total_i = int(total) if total else None
        done = 0
        while chunk := r.read(1 << 16):
            f.write(chunk)
            done += len(chunk)
            if progress:
                progress(name, done, total_i)
    tmp.replace(dest)


def download(spec: ModelSpec, progress: Optional[ProgressCb] = None) -> Path:
    root = model_path(spec)
    for f in spec.files:
        dest = root / f
        if not dest.exists():
            _fetch(_hf_url(spec.repo, f), dest, f, progress)
    for f in spec.optional_files:
        dest = root / f
        if dest.exists():
            continue
        try:
            _fetch(_hf_url(spec.repo, f), dest, f, progress)
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
    return root


# ── safetensors reader (numpy only, memory-mapped) ───────────────────────────

_ST_DTYPES = {"F16": np.float16, "F32": np.float32, "BF16": None, "I8": np.int8,
              "I32": np.int32, "I64": np.int64, "U8": np.uint8, "F64": np.float64}


def read_safetensors(path: Path) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    base = 8 + n
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    out: dict[str, np.ndarray] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dt = _ST_DTYPES.get(meta["dtype"])
        start, end = meta["data_offsets"]
        raw = mm[base + start: base + end]
        if meta["dtype"] == "BF16":  # widen bf16 → f32
            arr = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif dt is None:
            raise ValueError(f"unsupported safetensors dtype {meta['dtype']}")
        else:
            arr = raw.view(dt)
        out[name] = arr.reshape(meta["shape"])
    return out


# ── embedders ────────────────────────────────────────────────────────────────

def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


class Embedder:
    spec: ModelSpec
    dim: int

    def embed(self, texts: Sequence[str], *, query: bool = False) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class StaticEmbedder(Embedder):
    def __init__(self, spec: ModelSpec, root: Path):
        from tokenizers import Tokenizer

        self.spec = spec
        self.tok = Tokenizer.from_file(str(root / "tokenizer.json"))
        self.tok.no_padding()
        self.tok.no_truncation()
        cfg = json.loads((root / "config.json").read_text(encoding="utf-8")) if (root / "config.json").exists() else {}
        self.normalize = bool(cfg.get("normalize", True))
        t = read_safetensors(root / "model.safetensors")
        key = "embeddings" if "embeddings" in t else next(iter(t))
        self.table = t[key]  # stays memory-mapped (float16): ~32 MB on disk, ~0 RSS until touched
        self.weights = t.get("weights")
        self.mapping = t.get("mapping")
        self.dim = int(self.table.shape[1])
        unk = None
        for cand in ("[UNK]", "<unk>", "<UNK>"):
            unk = self.tok.token_to_id(cand)
            if unk is not None:
                break
        self.unk = unk

    def embed(self, texts: Sequence[str], *, query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        encs = self.tok.encode_batch(list(texts), add_special_tokens=False)
        ids_list = []
        for e in encs:
            ids = np.asarray(e.ids, dtype=np.int64)
            if self.unk is not None and ids.size:
                ids = ids[ids != self.unk]
            ids_list.append(ids)
        lengths = np.array([len(i) for i in ids_list])
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        if lengths.sum() == 0:
            return out
        flat = np.concatenate([i for i in ids_list if len(i)])
        rows = self.mapping[flat] if self.mapping is not None else flat
        vecs = self.table[rows].astype(np.float32)
        if self.weights is not None:
            vecs *= self.weights[flat].astype(np.float32)[:, None]
        nz = np.nonzero(lengths)[0]
        starts = np.concatenate([[0], np.cumsum(lengths[nz])[:-1]])
        sums = np.add.reduceat(vecs, starts, axis=0)
        out[nz] = sums / lengths[nz, None]
        return _l2(out) if self.normalize else out


class OnnxEmbedder(Embedder):
    def __init__(self, spec: ModelSpec, root: Path, threads: int = 2):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.spec = spec
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, int(threads))
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(root / spec.onnx_file), so, providers=["CPUExecutionProvider"])
        self.in_names = {i.name for i in self.sess.get_inputs()}
        self.out_names = [o.name for o in self.sess.get_outputs()]

        self.tok = Tokenizer.from_file(str(root / "tokenizer.json"))
        self.tok.enable_truncation(max_length=spec.max_tokens)
        pad_id, pad_tok = 0, "[PAD]"
        for cand in ("[PAD]", "<pad>", "<PAD>"):
            tid = self.tok.token_to_id(cand)
            if tid is not None:
                pad_id, pad_tok = tid, cand
                break
        self.tok.enable_padding(pad_id=pad_id, pad_token=pad_tok)
        self.dim = spec.dim or 0

    def _run(self, texts: list[str]) -> np.ndarray:
        encs = self.tok.encode_batch(texts)
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        feed = {}
        if "input_ids" in self.in_names:
            feed["input_ids"] = ids
        if "attention_mask" in self.in_names:
            feed["attention_mask"] = mask
        if "token_type_ids" in self.in_names:
            feed["token_type_ids"] = np.zeros_like(ids)
        if "sentence_embedding" in self.out_names:
            (emb,) = self.sess.run(["sentence_embedding"], feed)
            return emb.astype(np.float32)
        hidden = self.sess.run([self.out_names[0]], feed)[0].astype(np.float32)
        if self.spec.pooling == "cls":
            return hidden[:, 0]
        m = mask[..., None].astype(np.float32)
        return (hidden * m).sum(1) / np.maximum(m.sum(1), 1e-9)

    def embed(self, texts: Sequence[str], *, query: bool = False, batch: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim or 1), dtype=np.float32)
        prefix = self.spec.query_prefix if query else self.spec.doc_prefix
        items = [prefix + t for t in texts]
        order = np.argsort([len(t) for t in items])  # length-bucketing = less padding
        out: Optional[np.ndarray] = None
        for i in range(0, len(items), batch):
            idx = order[i:i + batch]
            vec = self._run([items[j] for j in idx])
            if out is None:
                out = np.zeros((len(items), vec.shape[1]), dtype=np.float32)
                self.dim = vec.shape[1]
            out[idx] = vec
        return _l2(out)  # type: ignore[arg-type]


def load_embedder(cfg: C.Config, progress: Optional[ProgressCb] = None) -> Embedder:
    spec = spec_for(cfg)
    root = model_path(spec)
    if not is_downloaded(spec):
        download(spec, progress)
    if spec.kind == "static":
        return StaticEmbedder(spec, root)
    return OnnxEmbedder(spec, root, threads=cfg.get_path("model.threads", 2))


def timed(fn, *a, **kw):
    t = time.perf_counter()
    r = fn(*a, **kw)
    return r, (time.perf_counter() - t) * 1000