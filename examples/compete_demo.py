#!/usr/bin/env python3
"""End-to-end Specter demo with a functional local backend.

Mirrors the public TypeSafe quickstart ticket shape so you can compare contracts.
Run from repo root after installing the wheel:

  python examples/compete_demo.py

Optional: set TYPESAFE_API_KEY and pass --jev to route through real Jev.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

# Allow running against a checkout without install
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from specter_decision import Question, SpecterDecisionEngine
from specter_decision.backends.local_lexical import LocalLexicalBackend
from specter_decision.calibration.simple import SimpleCalibration


STATE = {
    "ticket": (
        "Hi, I've been trying to connect my Stripe account for 3 days and it "
        "keeps failing. I'm losing sales. Please help ASAP."
    )
}

QUESTIONS = [
    Question(
        "department",
        "choice",
        "Which team should handle this",
        ("billing", "technical", "sales"),
    ),
    Question(
        "frustration",
        "score",
        "How frustrated the customer appears",
        ("Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"),
    ),
    Question(
        "is_urgent",
        "noul",
        "The message conveys urgency or time-sensitivity",
    ),
]


async def main(use_jev: bool) -> int:
    if use_jev:
        from specter_decision.backends.typesafe_jev import TypeSafeJevBackend

        backend = TypeSafeJevBackend()
        domain = "typesafe-proxy"
    else:
        backend = LocalLexicalBackend()
        domain = "support-demo-v1"

    calibration = SimpleCalibration(backend.model_id, domain, temperature=0.85)
    engine = SpecterDecisionEngine(
        backend=backend,
        calibration=calibration,
        domain=domain,
        concurrency=8,
        timeout=30.0 if use_jev else 2.0,
    )
    answers = await engine.decide(STATE, QUESTIONS)
    print(json.dumps({"backend": backend.model_id, "domain": domain, "decisions": answers}, indent=2))
    ok = all(d.get("status") == "ok" for d in answers)
    return 0 if ok else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--jev", action="store_true", help="Use TypeSafe Jev API (needs TYPESAFE_API_KEY)")
    args = p.parse_args()
    raise SystemExit(asyncio.run(main(args.jev)))
