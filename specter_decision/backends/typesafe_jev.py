"""Adapter: call TypeSafe Jev API and map answers back to Specter logits.

Requires TYPESAFE_API_KEY. This is not Jev's weights — it is a thin client that
lets Specter's engine consume the real System One service when you have access.

Mapping strategy:
- Jev returns probabilities (and for choice/score, the full distribution).
- Specter Backend must return *logits*. We invert softmax approximately via
  log(p + eps) so the engine's temperature+softmax recovers a close distribution.
"""
from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from typing import Sequence

from specter_decision.engine import Question

_DEFAULT_URL = "https://api.typesafe.ai/v1/systemone"
_EPS = 1e-8


def _probs_to_logits(probs: list[float]) -> tuple[float, ...]:
    clamped = [max(_EPS, min(1.0 - _EPS, float(p))) for p in probs]
    logs = [math.log(p) for p in clamped]
    # Center for numerical stability (engine does its own peak subtraction)
    mean = sum(logs) / len(logs)
    return tuple(x - mean for x in logs)


class TypeSafeJevBackend:
    """HTTP backend that forwards one question at a time to Jev.

    Specter evaluates questions independently; this adapter honors that by
    sending a single-question request per logits() call. Parallelism stays in
    Specter's asyncio gather + semaphore.
    """

    model_id = "jev-latest"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = _DEFAULT_URL,
        timeout: float = 30.0,
    ):
        key = api_key or os.environ.get("TYPESAFE_API_KEY", "")
        if not key or not isinstance(key, str):
            raise ValueError("TYPESAFE_API_KEY required for TypeSafeJevBackend")
        self._key = key
        self._url = base_url.rstrip("/")
        self._timeout = timeout

    async def logits(self, state_json: str, question: Question) -> Sequence[float]:
        # Reconstruct state object if possible
        try:
            state = json.loads(state_json)
        except json.JSONDecodeError:
            state = state_json

        q_body: dict = {
            "type": question.type,
            "instructions": question.instructions,
        }
        if question.type == "choice":
            # TypeSafe criteria is a map name -> description
            q_body["criteria"] = {c: c for c in question.criteria}
        elif question.type == "score":
            # TypeSafe score criteria is an ordered list of level descriptions
            q_body["criteria"] = list(question.criteria)
        elif question.type == "noul":
            pass

        payload = {
            "state": state,
            "model": self.model_id,
            "questions": {"_q": q_body},
        }
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self._url,
            data=raw,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"TypeSafe HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"TypeSafe transport error: {exc}") from exc

        answers = body.get("answers") or {}
        ans = answers.get("_q")
        if not isinstance(ans, dict):
            raise RuntimeError("TypeSafe response missing answers._q")

        labels = list(question.labels)
        if question.type == "noul":
            # Jev: {"type":"noul","noul": 0.99} — sometimes without full dist
            p_yes = float(ans.get("noul", ans.get("value", 0.5)))
            p_yes = max(0.0, min(1.0, p_yes))
            return _probs_to_logits([1.0 - p_yes, p_yes])

        probs_map = ans.get("probabilities") or {}
        if question.type == "choice":
            ordered = [float(probs_map.get(c, 0.0)) for c in question.criteria]
        else:
            # score: keys may be "0","1",... or level text
            ordered = []
            for i, label in enumerate(labels):
                if str(i) in probs_map:
                    ordered.append(float(probs_map[str(i)]))
                elif label in probs_map:
                    ordered.append(float(probs_map[label]))
                else:
                    ordered.append(0.0)
        s = sum(ordered)
        if s <= 0:
            ordered = [1.0 / len(labels)] * len(labels)
        else:
            ordered = [p / s for p in ordered]
        return _probs_to_logits(ordered)
