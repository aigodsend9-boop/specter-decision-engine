#!/usr/bin/env python3
"""Cross-language prefill microbench + cache theory numbers."""
from __future__ import annotations

import math
import sys
import time
import hashlib

STATE = (
    "Hi, I've been trying to connect my Stripe account for 3 days and it keeps "
    "failing. I'm losing sales. Please help ASAP."
)

def py_bench():
    from specter_decision.backends.system_one_local import _prefill, SystemOneLocalBackend
    from specter_decision.engine import Question
    import asyncio

    N = 20_000
    t0 = time.perf_counter()
    for _ in range(N):
        _prefill(STATE)
    ms = (time.perf_counter() - t0) * 1000
    print(f"Python _prefill {N}: {ms:.2f} ms ({1000*ms/N:.3f} us/op)")

    b = SystemOneLocalBackend()
    q = Question("u", "noul", "Is this urgent?")

    async def run():
        await b.logits(STATE, q)  # warm cache
        t0 = time.perf_counter()
        for _ in range(N):
            await b.logits(STATE, q)
        return (time.perf_counter() - t0) * 1000

    ms = asyncio.run(run())
    print(f"Python logits cached {N}: {ms:.2f} ms ({1000*ms/N:.3f} us/op)")

    # cache hit rate model: 1 prefill + (Q-1) hits for Q questions
    for Q in (1, 5, 28, 128):
        cost = 1.0 + 0.0 * (Q - 1)  # amortized prefill units
        print(f"  amortized prefill units for Q={Q}: {cost/Q:.4f} (ideal cache)")

if __name__ == "__main__":
    py_bench()
