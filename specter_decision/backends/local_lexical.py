"""Local lexical backend — functional stand-in, not a trained System One model.

Design goals aligned with the public Jev/TypeSafe *contract* (not their weights):
- One state snapshot, many independent questions
- Logits over the exact label set (noul/choice/score)
- No free-form text generation
- Deterministic given the same state + question

This uses token overlap, cue lexicons, and simple rubric alignment so the
Specter engine can run end-to-end without a GPU or API key. It is honest about
limits: it is not calibrated intelligence and must not be sold as such.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

from specter_decision.engine import Question

_TOKEN = re.compile(r"[a-z0-9_]+", re.I)

# Lightweight cue banks (English). Extend per domain; not a model.
_URGENCY = {
    "urgent", "asap", "immediately", "critical", "emergency", "right now",
    "losing", "blocked", "outage", "down", "cannot", "can't", "failing",
}
_FRUSTRATION = {
    "frustrated", "angry", "ridiculous", "unacceptable", "terrible",
    "hate", "worst", "furious", "annoyed", "complaint",
}
_REFUND = {"refund", "chargeback", "money back", "double charged", "charged twice"}
_BILLING = {"invoice", "payment", "charge", "subscription", "billing", "card", "stripe"}
_TECH = {"bug", "error", "crash", "api", "integration", "connect", "timeout", "fail"}
_SALES = {"pricing", "plan", "upgrade", "demo", "quote", "enterprise"}


def _tokens(text: str) -> Counter[str]:
    return Counter(t.lower() for t in _TOKEN.findall(text or ""))


def _blob(state_json: str, question: Question) -> str:
    # Instructions participate; routing id does not (engine already strips id).
    return f"{state_json}\n{question.instructions}\n{' '.join(question.criteria)}"


def _overlap(a: Counter[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    hits = sum(a[t] for t in b if t in a)
    return hits / (sum(a.values()) ** 0.5 + 1e-6)


def _cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class LocalLexicalBackend:
    """Deterministic lexical scorer that emits finite logits per label.

    model_id is stable so Calibration.supports can pin a domain artifact.
    """

    model_id = "specter-local-lexical-v1"

    async def logits(self, state_json: str, question: Question) -> Sequence[float]:
        text = _blob(state_json, question)
        toks = _tokens(text)
        labels = list(question.labels)
        n = len(labels)
        if n == 0:
            return ()

        if question.type == "noul":
            # labels: false, true — score "yes" cues from instructions + state
            instr = question.instructions.lower()
            yes = 0.0
            if any(w in instr for w in ("urgent", "urgency", "time-sensitive")):
                yes += 2.0 * _overlap(toks, _URGENCY)
            if any(w in instr for w in ("refund", "chargeback")):
                yes += 2.0 * _overlap(toks, _REFUND)
            if any(w in instr for w in ("frustrat", "angry")):
                yes += 2.0 * _overlap(toks, _FRUSTRATION)
            # Generic: lexical overlap between instruction content words and state
            instr_toks = _tokens(question.instructions)
            yes += 1.5 * _cosine(toks, instr_toks)
            # Bias slightly toward no when no signal
            return (-0.4, yes - 0.2)

        if question.type == "choice":
            scores: list[float] = []
            for label in labels:
                desc = label
                # criteria may only be the option name in Specter; boost known domains
                bank = set()
                low = label.lower()
                if low in ("billing", "payments", "finance"):
                    bank |= _BILLING
                elif low in ("technical", "engineering", "support", "tech"):
                    bank |= _TECH | {"connect", "connection", "stripe", "account", "failing", "fails"}
                elif low in ("sales", "account", "commercial"):
                    bank |= _SALES
                elif low in ("returns", "refunds"):
                    bank |= _REFUND
                label_toks = _tokens(f"{label} {desc}")
                s = 2.0 * _cosine(toks, label_toks) + 1.5 * _overlap(toks, bank)
                scores.append(s)
            # Soft margin so argmax is stable but distribution is non-degenerate
            m = max(scores) if scores else 0.0
            return tuple(s - m * 0.15 for s in scores)

        # score: ordered rubric levels 0..n-1
        # Higher levels align with stronger negative/urgency language by default
        base = (
            1.2 * _overlap(toks, _FRUSTRATION)
            + 1.0 * _overlap(toks, _URGENCY)
            + 0.8 * _overlap(toks, _TECH)
        )
        # Map continuous signal onto ordered levels via peaked logits
        scores = []
        for i, label in enumerate(labels):
            label_toks = _tokens(label)
            align = _cosine(toks, label_toks)
            # Prefer higher index when base is high
            target = base * (n - 1)
            proximity = -abs(i - target) + 1.5 * align
            scores.append(proximity)
        return tuple(scores)
