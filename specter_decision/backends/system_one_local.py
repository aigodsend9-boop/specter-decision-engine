"""Specter System-One Local Engine (elite algorithmic edition).

Architecture mapped from public Jev / parallel-constrained-decoding research
(TypeSafe docs + unofficial jev-on-a-laptop / harshatheg parallel decoding):

  [state] ──► shared feature prefill (once)
                  │
                  ├── view 0 (identity)
                  ├── view 1 (negation-flipped emphasis)
                  └── view 2 (intensity-weighted)
                        │
              parallel field heads (noul / choice / score)
                        │
              logit average across views  →  finite logits per label

This is NOT Jev's weights, NOT RLCD, and NOT a neural LM. It is a deliberate
computer-science reconstruction of the *control flow* that makes System One
fast and type-safe:

  1. Prefill once  (shared state representation)
  2. Broadcast     (same features to every question head)
  3. Constrain     (logits only over the declared label set)
  4. Assemble      (typed values in code — never free text)

On this machine we have ~1GB RAM and no useful GPU budget for a 1.5B model,
so the "head" is a multi-signal classical scorer with view ensemble. Swap the
head later for a real neural field-head without changing the Specter contract.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from specter_decision.engine import Question

_TOKEN = re.compile(r"[a-z0-9']+", re.I)
_SENT_SPLIT = re.compile(r"[.!?\n]+")

# --- domain cue banks (extend per deployment; kept small and explicit) ---
_CUES: dict[str, set[str]] = {
    "urgency": {
        "urgent", "asap", "immediately", "critical", "emergency", "now",
        "losing", "blocked", "outage", "down", "cannot", "can't", "failing",
        "deadline", "today", "asap", "right", "stuck",
    },
    "frustration": {
        "frustrated", "angry", "ridiculous", "unacceptable", "terrible",
        "hate", "worst", "furious", "annoyed", "complaint", "useless",
        "awful", "disgusting", "joke",
    },
    "refund": {
        "refund", "chargeback", "money", "double", "charged", "twice",
        "billing error", "overcharged",
    },
    "billing": {
        "invoice", "payment", "charge", "subscription", "billing", "card",
        "stripe", "paypal", "receipt", "fee", "price",
    },
    "technical": {
        "bug", "error", "crash", "api", "integration", "connect", "connection",
        "timeout", "fail", "failing", "stack", "exception", "500", "404",
        "login", "auth", "oauth", "webhook", "sdk", "account",
    },
    "sales": {
        "pricing", "plan", "upgrade", "demo", "quote", "enterprise",
        "trial", "discount", "seat",
    },
    "returns": {"return", "returns", "exchange", "shipping", "package", "damaged"},
}

_NEGATION = {"not", "no", "never", "n't", "without", "hardly", "neither", "nor"}
_INTENSIFIERS = {
    "very": 1.4, "extremely": 1.7, "really": 1.3, "so": 1.2, "highly": 1.4,
    "completely": 1.5, "totally": 1.4, "absolutely": 1.5, "super": 1.3,
}
_SOFTENERS = {"maybe": 0.7, "perhaps": 0.7, "somewhat": 0.75, "slightly": 0.7, "a bit": 0.75}


@dataclass
class Prefill:
    """Shared state features — the algorithmic analogue of a KV prefill."""

    raw: str
    tokens: Counter[str]
    sentences: list[str]
    token_list: list[str]
    cue_hits: dict[str, float]
    intensity: float
    negation_density: float
    fingerprint: str
    # cached per-label overlap for this state (filled lazily)
    _cache: dict = field(default_factory=dict, repr=False)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "")]


def _prefill(state_json: str) -> Prefill:
    toks = _tokenize(state_json)
    sents = [s.strip() for s in _SENT_SPLIT.split(state_json) if s.strip()]
    counts = Counter(toks)
    cue_hits: dict[str, float] = {}
    for name, bank in _CUES.items():
        cue_hits[name] = sum(counts[t] for t in bank if t in counts) / (math.sqrt(sum(counts.values()) + 1e-9))
    intens = 1.0
    for i, t in enumerate(toks):
        if t in _INTENSIFIERS:
            intens *= _INTENSIFIERS[t]
        if t in _SOFTENERS:
            intens *= _SOFTENERS[t]
    intens = max(0.5, min(2.5, intens))
    neg = sum(1 for t in toks if t in _NEGATION) / max(1, len(toks))
    fp = hashlib.sha256(state_json.encode("utf-8")).hexdigest()[:16]
    return Prefill(
        raw=state_json,
        tokens=counts,
        sentences=sents,
        token_list=toks,
        cue_hits=cue_hits,
        intensity=intens,
        negation_density=neg,
        fingerprint=fp,
    )


def _cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


def _bm25_like(query: Counter[str], doc: Counter[str], avgdl: float = 64.0, k1: float = 1.2, b: float = 0.75) -> float:
    """Single-doc BM25-style score (no IDF corpus — use smoothed self-IDF)."""
    if not query or not doc:
        return 0.0
    dl = sum(doc.values())
    score = 0.0
    for t, qf in query.items():
        f = doc.get(t, 0)
        if f == 0:
            continue
        # self-idf proxy: rarer in this doc window → slightly higher weight
        idf = math.log(1.0 + (dl - f + 0.5) / (f + 0.5))
        tf_part = (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        score += idf * tf_part * qf
    return score


def _window_scores(prefill: Prefill, query_toks: Counter[str]) -> list[float]:
    """Per-sentence alignment — multi-view evidence."""
    out = []
    for s in prefill.sentences or [prefill.raw]:
        st = Counter(_tokenize(s))
        out.append(_cosine(query_toks, st) + 0.15 * _bm25_like(query_toks, st))
    return out or [0.0]


def _head_noul(prefill: Prefill, question: Question, view: int) -> tuple[float, float]:
    instr = question.instructions.lower()
    qtoks = Counter(_tokenize(question.instructions))
    yes = 0.0
    # instruction-driven cue routing
    for name, bank in _CUES.items():
        if any(w in instr for w in (name, *list(bank)[:3])):
            yes += 2.2 * prefill.cue_hits.get(name, 0.0)
    yes += 1.6 * _cosine(prefill.tokens, qtoks)
    yes += 0.8 * max(_window_scores(prefill, qtoks))
    yes *= prefill.intensity
    # view 1: penalize if high negation near matched cues
    if view == 1:
        yes *= max(0.35, 1.0 - 1.8 * prefill.negation_density)
    # view 2: boost intensity path
    if view == 2:
        yes *= 0.85 + 0.25 * prefill.intensity
    # false / true logits
    return (-0.35 - 0.15 * view, yes - 0.25)


def _head_choice(prefill: Prefill, question: Question, view: int) -> list[float]:
    scores: list[float] = []
    for label in question.labels:
        low = label.lower()
        bank: set[str] = set()
        for name, cues in _CUES.items():
            if name in low or low in name or any(c in low for c in list(cues)[:4]):
                bank |= cues
        # explicit domain map
        for name, cues in _CUES.items():
            if low in (name, name + "s") or name in low:
                bank |= cues
        if low in ("technical", "engineering", "tech", "support"):
            bank |= _CUES["technical"]
        if low in ("billing", "finance", "payments"):
            bank |= _CUES["billing"]
        if low in ("sales", "commercial", "account"):
            bank |= _CUES["sales"]
        if low in ("returns", "refunds"):
            bank |= _CUES["returns"] | _CUES["refund"]

        label_toks = Counter(_tokenize(label))
        s = (
            2.0 * _cosine(prefill.tokens, label_toks)
            + 1.8 * sum(prefill.tokens[t] for t in bank if t in prefill.tokens) / (math.sqrt(sum(prefill.tokens.values()) + 1e-9))
            + 0.6 * max(_window_scores(prefill, label_toks | Counter(bank)))
        )
        if view == 1:
            s *= max(0.4, 1.0 - prefill.negation_density)
        if view == 2:
            s *= 0.9 + 0.15 * prefill.intensity
        scores.append(s)
    # relative margin
    m = max(scores) if scores else 0.0
    return [s - 0.12 * m for s in scores]


def _head_score(prefill: Prefill, question: Question, view: int) -> list[float]:
    labels = list(question.labels)
    n = len(labels)
    # continuous severity / intensity signal
    base = (
        1.3 * prefill.cue_hits.get("frustration", 0.0)
        + 1.1 * prefill.cue_hits.get("urgency", 0.0)
        + 0.7 * prefill.cue_hits.get("technical", 0.0)
    ) * prefill.intensity
    if view == 1:
        base *= max(0.4, 1.0 - 1.5 * prefill.negation_density)
    if view == 2:
        base *= 0.9 + 0.2 * prefill.intensity
    target = base * max(1, n - 1)
    scores = []
    for i, label in enumerate(labels):
        align = _cosine(prefill.tokens, Counter(_tokenize(label)))
        proximity = -abs(i - target) + 1.4 * align
        scores.append(proximity)
    return scores


def _average_logits(views: list[Sequence[float]]) -> tuple[float, ...]:
    """Logit averaging across parallel views (ThinkMerge-style, classical)."""
    n = len(views[0])
    acc = [0.0] * n
    for v in views:
        for i, x in enumerate(v):
            acc[i] += float(x)
    inv = 1.0 / len(views)
    return tuple(x * inv for x in acc)


class SystemOneLocalBackend:
    """Prefill-once, multi-view, constrained-label backend for Specter.

    model_id is stable for Calibration.supports pinning.
    """

    model_id = "specter-system-one-local-v1"
    n_views = 3

    def __init__(self) -> None:
        self._prefill_cache: dict[str, Prefill] = {}
        self._last_prefill: Prefill | None = None

    def _get_prefill(self, state_json: str) -> Prefill:
        key = hashlib.sha256(state_json.encode("utf-8")).hexdigest()
        hit = self._prefill_cache.get(key)
        if hit is not None:
            return hit
        pf = _prefill(state_json)
        # tiny LRU: keep last 32 states
        if len(self._prefill_cache) >= 32:
            self._prefill_cache.pop(next(iter(self._prefill_cache)))
        self._prefill_cache[key] = pf
        self._last_prefill = pf
        return pf

    async def logits(self, state_json: str, question: Question) -> Sequence[float]:
        pf = self._get_prefill(state_json)
        views: list[Sequence[float]] = []
        for v in range(self.n_views):
            if question.type == "noul":
                views.append(_head_noul(pf, question, v))
            elif question.type == "choice":
                views.append(_head_choice(pf, question, v))
            else:
                views.append(_head_score(pf, question, v))
        return _average_logits(views)


class SystemOneCalibration:
    """Confidence from entropy + view-margin; temperature mild for local head."""

    def __init__(self, domain: str = "system-one-local", temperature: float = 0.9):
        self._domain = domain
        self._t = temperature

    def supports(self, model_id: str, domain: str, question_signature: str) -> bool:
        return model_id == SystemOneLocalBackend.model_id and domain == self._domain

    def temperature(self, question_signature: str) -> float:
        return self._t

    def confidence(self, question, probabilities: tuple[float, ...], value) -> float:
        if not probabilities:
            return 0.05
        # normalized entropy → confidence
        n = len(probabilities)
        if n <= 1:
            return max(0.05, float(probabilities[0]))
        ent = -sum(p * math.log(p + 1e-12) for p in probabilities)
        max_ent = math.log(n)
        inv_ent = 1.0 - (ent / max_ent if max_ent > 0 else 0.0)
        peak = max(probabilities)
        ordered = sorted(probabilities, reverse=True)
        gap = ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)
        conf = 0.45 * inv_ent + 0.35 * peak + 0.20 * gap
        return max(0.05, min(1.0, conf))
