#!/usr/bin/env python3
"""System-One local demo — same ticket shape as TypeSafe quickstart."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from specter_decision import Question, SpecterDecisionEngine
from specter_decision.backends.system_one_local import SystemOneLocalBackend, SystemOneCalibration

STATE = {
    "ticket": (
        "Hi, I've been trying to connect my Stripe account for 3 days and it "
        "keeps failing. I'm losing sales. Please help ASAP."
    )
}

QUESTIONS = [
    Question("department", "choice", "Which team should handle this",
             ("billing", "technical", "sales")),
    Question("frustration", "score", "How frustrated the customer appears",
             ("Calm, just stating facts", "Frustrated but civil", "Very angry, strong language")),
    Question("is_urgent", "noul", "The message conveys urgency or time-sensitivity"),
    Question("wants_refund", "noul", "The customer is requesting a refund"),
    Question("channel", "choice", "Best first response channel",
             ("email", "phone", "chat")),
]


async def run() -> dict:
    backend = SystemOneLocalBackend()
    cal = SystemOneCalibration(domain="system-one-local")
    engine = SpecterDecisionEngine(
        backend=backend, calibration=cal, domain="system-one-local",
        concurrency=16, timeout=2.0,
    )
    t0 = time.perf_counter()
    answers = await engine.decide(STATE, QUESTIONS)
    ms = (time.perf_counter() - t0) * 1000
    return {
        "backend": backend.model_id,
        "latency_ms": round(ms, 2),
        "prefill_fingerprint": backend._last_prefill.fingerprint if backend._last_prefill else None,
        "decisions": answers,
    }


if __name__ == "__main__":
    out = asyncio.run(run())
    print(json.dumps(out, indent=2, ensure_ascii=True))
