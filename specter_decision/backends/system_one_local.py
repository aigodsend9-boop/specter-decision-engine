"""Specter System-One Local v3 — bit-mask lexicon + amortized prefill cache.

Cache theory
------------
Let C_prefill be cost of building Prefill(state), C_head cost of one field head.
Without cache, Q questions cost Q * (C_prefill + C_head).
With content-addressed cache keyed by hash(state):

    cost = C_prefill + Q * C_head

Amortized per question: C_prefill/Q + C_head.
At Q=28, prefill is ~3.6% of the no-cache prefill bill.

Bit-level lexicon
-----------------
Each known word maps to:
  bank_mask : uint7 bitfield — membership in cue banks B_0..B_6
  flags     : neg / intens / soft
  factor    : intensity multiplier

One open-dict probe per token updates all bank counts via bit tests.
Unknown tokens only increment n_tok (O(1) reject).

Optional C core (libprefill.so) accelerates prefill when present.
"""
from __future__ import annotations

import math
import os
import re
from typing import Sequence

from specter_decision.engine import Question

_TOKEN = re.compile(r"[a-z0-9']+")

# bank index ABI
_N_CUES = 7
# word -> (bank_mask:int, neg:bool, factor:float)
_LEX: dict[str, tuple[int, bool, float]] = {}


def _lex_add(words: str, bit: int) -> None:
    for w in words.split():
        mask, neg, fac = _LEX.get(w, (0, False, 1.0))
        _LEX[w] = (mask | (1 << bit), neg, fac)


def _lex_flag(words: str, *, neg: bool = False, factor: float = 1.0) -> None:
    for w in words.split():
        mask, n, f = _LEX.get(w, (0, False, 1.0))
        _LEX[w] = (mask, n or neg, factor if factor != 1.0 else f)


_lex_add("urgent asap immediately critical emergency now losing blocked outage down cannot failing deadline today stuck", 0)
_lex_add("frustrated angry ridiculous unacceptable terrible hate worst furious annoyed complaint useless awful joke", 1)
_lex_add("refund chargeback money double charged twice overcharged", 2)
_lex_add("invoice payment charge subscription billing card stripe paypal receipt fee price", 3)
_lex_add("bug error crash api integration connect connection timeout fail failing stack exception login auth oauth webhook sdk account", 4)
_lex_add("pricing plan upgrade demo quote enterprise trial discount seat", 5)
_lex_add("return returns exchange shipping package damaged", 6)
_lex_flag("not no never without hardly neither nor", neg=True)
_lex_flag("very", factor=1.35)
_lex_flag("extremely", factor=1.60)
_lex_flag("really", factor=1.25)
_lex_flag("so", factor=1.15)
_lex_flag("highly", factor=1.35)
_lex_flag("completely", factor=1.45)
_lex_flag("totally", factor=1.35)
_lex_flag("absolutely", factor=1.45)
_lex_flag("maybe", factor=0.72)
_lex_flag("perhaps", factor=0.72)
_lex_flag("somewhat", factor=0.78)
_lex_flag("slightly", factor=0.72)

_LABEL_CUE: dict[str, int] = {
    "urgent": 0, "urgency": 0, "frustration": 1, "frustrated": 1, "angry": 1,
    "refund": 2, "refunds": 2, "billing": 3, "finance": 3, "payments": 3, "payment": 3,
    "technical": 4, "engineering": 4, "tech": 4, "support": 4,
    "sales": 5, "commercial": 5, "account": 5, "returns": 6, "return": 6, "shipping": 6,
}

# optional native
_C_PREFILL = None
try:
    import ctypes
    from ctypes import c_char_p, c_size_t, c_uint32, c_int, c_float, Structure, POINTER

    class _PrefillOut(Structure):
        _fields_ = [
            ("key", c_uint32),
            ("n_tok", c_int),
            ("intensity", c_float),
            ("neg_density", c_float),
            ("hits", c_float * _N_CUES),
        ]

    for path in (
        os.environ.get("SPECTER_PREFILL_SO"),
        os.path.join(os.path.dirname(__file__), "libprefill.so"),
        "/tmp/libprefill.so",
        "/home/workdir/artifacts/native/libprefill.so",
    ):
        if path and os.path.isfile(path):
            _lib = ctypes.CDLL(path)
            _lib.specter_prefill.argtypes = [c_char_p, c_size_t, POINTER(_PrefillOut)]
            _lib.specter_prefill.restype = c_uint32
            _C_PREFILL = _lib
            break
except Exception:
    _C_PREFILL = None


class Prefill:
    __slots__ = ("hits", "intensity", "neg_density", "n_tok", "key", "tf")

    def __init__(self, hits: list[float], intensity: float, neg_density: float,
                 n_tok: int, key: int, tf: dict[str, int] | None = None):
        self.hits = hits
        self.intensity = intensity
        self.neg_density = neg_density
        self.n_tok = n_tok
        self.key = key
        self.tf = tf or {}

    @property
    def fingerprint(self) -> str:
        return format(self.key & 0xFFFFFFFFFFFFFFFF, "016x")


def _prefill_py(state_json: str) -> Prefill:
    raw = state_json.lower()
    counts = [0] * _N_CUES
    n_tok = 0
    neg = 0
    intens = 1.0
    tf: dict[str, int] = {}
    for m in _TOKEN.finditer(raw):
        t = m.group(0)
        n_tok += 1
        tf[t] = tf.get(t, 0) + 1
        meta = _LEX.get(t)
        if meta is None:
            continue
        mask, is_neg, fac = meta
        if mask:
            # branchless-ish bit walk for 7 banks
            m = mask
            b = 0
            while m:
                if m & 1:
                    counts[b] += 1
                m >>= 1
                b += 1
        if is_neg:
            neg += 1
        if fac != 1.0:
            intens *= fac
    if intens < 0.5:
        intens = 0.5
    elif intens > 2.4:
        intens = 2.4
    scale = 1.0 / math.sqrt(n_tok + 1e-9)
    hits = [c * scale for c in counts]
    return Prefill(hits, intens, neg / max(1, n_tok), n_tok, hash(state_json), tf)


def _prefill_c(state_json: str) -> Prefill:
    out = _PrefillOut()
    data = state_json.encode("utf-8", errors="ignore")
    _C_PREFILL.specter_prefill(data, len(data), ctypes.byref(out))
    hits = [float(out.hits[i]) for i in range(_N_CUES)]
    return Prefill(hits, float(out.intensity), float(out.neg_density),
                   int(out.n_tok), int(out.key), None)


def _prefill(state_json: str) -> Prefill:
    if _C_PREFILL is not None:
        try:
            return _prefill_c(state_json)
        except Exception:
            pass
    return _prefill_py(state_json)


def _base_noul(pf: Prefill, instr: str) -> float:
    yes = 0.0
    if "urgent" in instr or "time" in instr or "asap" in instr:
        yes += 2.4 * pf.hits[0]
    if "refund" in instr or "chargeback" in instr:
        yes += 2.4 * pf.hits[2]
    if "frustrat" in instr or "angry" in instr:
        yes += 2.4 * pf.hits[1]
    if pf.tf:
        scale = 1.0 / math.sqrt(pf.n_tok + 1e-9)
        shared = 0
        for m in _TOKEN.finditer(instr):
            c = pf.tf.get(m.group(0))
            if c:
                shared += c
        yes += 1.5 * shared * scale
    return yes * pf.intensity


def _base_choice(pf: Prefill, labels: tuple[str, ...]) -> list[float]:
    scale = 1.0 / math.sqrt(pf.n_tok + 1e-9)
    out: list[float] = []
    tf = pf.tf
    for label in labels:
        low = label.lower()
        s = 0.0
        idx = _LABEL_CUE.get(low)
        if idx is not None:
            s += 2.0 * pf.hits[idx]
        if tf:
            for m in _TOKEN.finditer(low):
                t = m.group(0)
                c = tf.get(t)
                if c:
                    s += c * scale * 1.8
                idx2 = _LABEL_CUE.get(t)
                if idx2 is not None:
                    s += 1.2 * pf.hits[idx2]
        else:
            # C prefill path without tf: rely on label cue index only
            pass
        out.append(s * pf.intensity)
    return out


def _base_score(pf: Prefill, n_levels: int) -> list[float]:
    base = (1.3 * pf.hits[1] + 1.1 * pf.hits[0] + 0.7 * pf.hits[4]) * pf.intensity
    target = base * max(1, n_levels - 1)
    return [-abs(i - target) for i in range(n_levels)]


def _avg3(a: float, b: float, c: float) -> float:
    return (a + b + c) * (1.0 / 3.0)


class SystemOneLocalBackend:
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
        if len(self._order) >= 64:
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
            n1 = max(0.35, 1.0 - 1.8 * pf.neg_density)
            n2 = 0.85 + 0.25 * pf.intensity
            y = _avg3(yes, yes * n1, yes * n2)
            return (-0.40, y - 0.25)
        if t == "choice":
            base = _base_choice(pf, question.labels)
            n1 = max(0.40, 1.0 - pf.neg_density)
            n2 = 0.90 + 0.15 * pf.intensity
            m = 0.0
            out: list[float] = []
            for s in base:
                v = _avg3(s, s * n1, s * n2)
                if v > m:
                    m = v
                out.append(v)
            return tuple(v - 0.12 * m for v in out)
        base = _base_score(pf, len(question.labels))
        n1 = max(0.40, 1.0 - 1.5 * pf.neg_density)
        n2 = 0.90 + 0.20 * pf.intensity
        return tuple(_avg3(s, s * n1, s * n2) for s in base)


class SystemOneCalibration:
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
        peak = second = 0.0
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
        return 0.05 if conf < 0.05 else (1.0 if conf > 1.0 else conf)
