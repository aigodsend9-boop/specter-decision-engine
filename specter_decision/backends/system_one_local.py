"""Specter System-One Local — lean core.

Control flow (public System-One idea, no vendor weights):
  state → prefill once → constrained field logits → engine softmax

TempleOS-minded constraints applied here:
  - fixed tables, no dynamic registries
  - one pass over tokens
  - slots, no incidental allocations on the hot path
  - views are affine transforms of a single base score, not three full rescores
"""
from __future__ import annotations

import math
import re
from typing import Sequence

from specter_decision.engine import Question

_TOKEN = re.compile(r"[a-z0-9']+")

# Fixed cue tables (order is ABI for Prefill.hits indices).
_CUE_NAMES = ("urgency", "frustration", "refund", "billing", "technical", "sales", "returns")
_CUE_SETS: tuple[frozenset[str], ...] = (
    frozenset("urgent asap immediately critical emergency now losing blocked outage down cannot can't failing deadline today stuck".split()),
    frozenset("frustrated angry ridiculous unacceptable terrible hate worst furious annoyed complaint useless awful joke".split()),
    frozenset("refund chargeback money double charged twice overcharged".split()),
    frozenset("invoice payment charge subscription billing card stripe paypal receipt fee price".split()),
    frozenset("bug error crash api integration connect connection timeout fail failing stack exception login auth oauth webhook sdk account".split()),
    frozenset("pricing plan upgrade demo quote enterprise trial discount seat".split()),
    frozenset("return returns exchange shipping package damaged".split()),
)
_N_CUES = len(_CUE_SETS)

_NEG = frozenset("not no never n't without hardly neither nor".split())
_INT = {"very": 1.35, "extremely": 1.6, "really": 1.25, "so": 1.15, "highly": 1.35,
        "completely": 1.45, "totally": 1.35, "absolutely": 1.45}
_SOFT = {"maybe": 0.72, "perhaps": 0.72, "somewhat": 0.78, "slightly": 0.72}

# label-name → cue index for choice heads (O(1))
_LABEL_CUE: dict[str, int] = {
    "urgent": 0, "urgency": 0,
    "frustration": 1, "frustrated": 1, "angry": 1,
    "refund": 2, "refunds": 2,
    "billing": 3, "finance": 3, "payments": 3, "payment": 3,
    "technical": 4, "engineering": 4, "tech": 4, "support": 4,
    "sales": 5, "commercial": 5, "account": 5,
    "returns": 6, "return": 6, "shipping": 6,
}


class Prefill:
    __slots__ = ("tf", "vocab", "hits", "intensity", "neg_density", "n_tok", "key")

    def __init__(self, tf: dict[str, int], vocab: frozenset[str], hits: list[float],
                 intensity: float, neg_density: float, n_tok: int, key: int):
        self.tf = tf
        self.vocab = vocab
        self.hits = hits
        self.intensity = intensity
        self.neg_density = neg_density
        self.n_tok = n_tok
        self.key = key

    @property
    def fingerprint(self) -> str:
        return format(self.key & 0xFFFFFFFFFFFFFFFF, "016x")


def _prefill(state_json: str) -> Prefill:
    # Single pass: lowercase tokens, counts, intensity, negation.
    raw = state_json.lower()
    tf: dict[str, int] = {}
    n_tok = 0
    intens = 1.0
    neg = 0
    for m in _TOKEN.finditer(raw):
        t = m.group(0)
        tf[t] = tf.get(t, 0) + 1
        n_tok += 1
        if t in _NEG:
            neg += 1
        f = _INT.get(t)
        if f is not None:
            intens *= f
        else:
            f = _SOFT.get(t)
            if f is not None:
                intens *= f
    if intens < 0.5:
        intens = 0.5
    elif intens > 2.4:
        intens = 2.4
    scale = 1.0 / math.sqrt(n_tok + 1e-9)
    hits = [0.0] * _N_CUES
    for i, bank in enumerate(_CUE_SETS):
        s = 0
        for w in bank:
            c = tf.get(w)
            if c:
                s += c
        hits[i] = s * scale
    key = hash(state_json)
    return Prefill(tf, frozenset(tf), hits, intens, neg / max(1, n_tok), n_tok, key)


def _overlap(tf: dict[str, int], words: frozenset[str], scale: float) -> float:
    s = 0
    for w in words:
        c = tf.get(w)
        if c:
            s += c
    return s * scale


def _base_noul(pf: Prefill, instr: str) -> float:
    yes = 0.0
    # instruction routes which cues matter
    if "urgent" in instr or "time" in instr or "asap" in instr:
        yes += 2.4 * pf.hits[0]
    if "refund" in instr or "chargeback" in instr:
        yes += 2.4 * pf.hits[2]
    if "frustrat" in instr or "angry" in instr:
        yes += 2.4 * pf.hits[1]
    # generic: shared tokens between instruction and state
    scale = 1.0 / math.sqrt(pf.n_tok + 1e-9)
    shared = 0
    for m in _TOKEN.finditer(instr):
        t = m.group(0)
        c = pf.tf.get(t)
        if c:
            shared += c
    yes += 1.5 * shared * scale
    return yes * pf.intensity


def _base_choice(pf: Prefill, labels: tuple[str, ...]) -> list[float]:
    scale = 1.0 / math.sqrt(pf.n_tok + 1e-9)
    out: list[float] = []
    for label in labels:
        low = label.lower()
        s = 0.0
        idx = _LABEL_CUE.get(low)
        if idx is not None:
            s += 2.0 * pf.hits[idx]
        # token overlap with label words
        for m in _TOKEN.finditer(low):
            t = m.group(0)
            c = pf.tf.get(t)
            if c:
                s += c * scale * 1.8
            idx2 = _LABEL_CUE.get(t)
            if idx2 is not None:
                s += 1.2 * pf.hits[idx2]
        out.append(s * pf.intensity)
    return out


def _base_score(pf: Prefill, n_levels: int) -> list[float]:
    base = (1.3 * pf.hits[1] + 1.1 * pf.hits[0] + 0.7 * pf.hits[4]) * pf.intensity
    target = base * max(1, n_levels - 1)
    return [-abs(i - target) for i in range(n_levels)]


def _view_pair(base_no: float, base_yes: float, pf: Prefill) -> tuple[float, float, float, float, float, float]:
    """Three (no, yes) pairs as flat tuple — no list-of-lists."""
    n1 = max(0.35, 1.0 - 1.8 * pf.neg_density)
    n2 = 0.85 + 0.25 * pf.intensity
    return (
        base_no, base_yes,
        base_no, base_yes * n1,
        base_no, base_yes * n2,
    )


def _avg3(a: float, b: float, c: float) -> float:
    return (a + b + c) * (1.0 / 3.0)


class SystemOneLocalBackend:
    """Prefill-once constrained local backend. model_id stable for calibration pin."""

    model_id = "specter-system-one-local-v1"
    __slots__ = ("_cache", "_last", "_order")

    def __init__(self) -> None:
        self._cache: dict[int, Prefill] = {}
        self._order: list[int] = []
        self._last: Prefill | None = None

    def _get(self, state_json: str) -> Prefill:
        k = hash(state_json)
        hit = self._cache.get(k)
        if hit is not None:
            self._last = hit
            return hit
        pf = _prefill(state_json)
        if len(self._order) >= 32:
            old = self._order.pop(0)
            self._cache.pop(old, None)
        self._cache[k] = pf
        self._order.append(k)
        self._last = pf
        return pf

    @property
    def _last_prefill(self) -> Prefill | None:
        return self._last

    async def logits(self, state_json: str, question: Question) -> Sequence[float]:
        pf = self._get(state_json)
        t = question.type
        if t == "noul":
            yes = _base_noul(pf, question.instructions.lower())
            # views: identity, negation damp, intensity boost — average yes/no
            n1 = max(0.35, 1.0 - 1.8 * pf.neg_density)
            n2 = 0.85 + 0.25 * pf.intensity
            y = _avg3(yes, yes * n1, yes * n2)
            return (-0.40, y - 0.25)
        if t == "choice":
            labels = question.labels
            base = _base_choice(pf, labels)
            n1 = max(0.40, 1.0 - pf.neg_density)
            n2 = 0.90 + 0.15 * pf.intensity
            m = 0.0
            out: list[float] = []
            for s in base:
                v = _avg3(s, s * n1, s * n2)
                if v > m:
                    m = v
                out.append(v)
            # relative margin (stable argmax, non-degenerate softmax)
            return tuple(v - 0.12 * m for v in out)
        # score
        labels = question.labels
        base = _base_score(pf, len(labels))
        n1 = max(0.40, 1.0 - 1.5 * pf.neg_density)
        n2 = 0.90 + 0.20 * pf.intensity
        return tuple(_avg3(s, s * n1, s * n2) for s in base)


class SystemOneCalibration:
    """Entropy + peak + gap confidence. Domain-pinned."""

    __slots__ = ("_domain", "_t")

    def __init__(self, domain: str = "system-one-local", temperature: float = 0.9):
        self._domain = domain
        self._t = float(temperature)

    def supports(self, model_id: str, domain: str, question_signature: str) -> bool:
        return model_id == SystemOneLocalBackend.model_id and domain == self._domain

    def temperature(self, question_signature: str) -> float:
        return self._t

    def confidence(self, question, probabilities: tuple[float, ...], value) -> float:
        n = len(probabilities)
        if n == 0:
            return 0.05
        if n == 1:
            return max(0.05, float(probabilities[0]))
        peak = probabilities[0]
        second = 0.0
        for p in probabilities:
            if p > peak:
                second = peak
                peak = p
            elif p > second:
                second = p
        ent = 0.0
        for p in probabilities:
            if p > 1e-15:
                ent -= p * math.log(p)
        inv_ent = 1.0 - ent / math.log(n)
        conf = 0.45 * inv_ent + 0.35 * peak + 0.20 * (peak - second)
        if conf < 0.05:
            return 0.05
        if conf > 1.0:
            return 1.0
        return conf
