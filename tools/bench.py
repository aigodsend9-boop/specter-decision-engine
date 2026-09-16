"""Banco de medição reprodutível do kernel.

Mede quatro coisas distintas, porque misturá-las é a forma mais comum de
publicar número sem significado:

1. **Custo do kernel** — backend de custo zero. Isola o overhead do
   snapshot, validação, softmax e projeção.
2. **Lote vs fan-out** — mesmo backend, mesma carga, com e sem o
   protocolo `logits_batch`. Mede o ganho estrutural do prefill único.
3. **Deduplicação** — perguntas idênticas em fan-out especulativo.
4. **Memória** — RSS via `resource`, e alocação de pico via `tracemalloc`.

Nada aqui mede inteligência do modelo: o backend é heurístico.

Uso:
    python tools/bench.py [--iterations 300] [--json saida.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import resource
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from specter_decision import (  # noqa: E402
    Question,
    SimpleCalibration,
    SpecterDecisionEngine,
    SystemOneCalibration,
    SystemOneLocalBackend,
)


class NullBackend:
    """Backend de custo ~zero: mede só o que o kernel gasta."""

    model_id = "null-backend"

    async def logits(self, state_json: str, question: Question) -> list[float]:
        return [0.0] * question.cardinality


class NullBatchBackend(NullBackend):
    """Mesmo custo zero, porém com protocolo de lote."""

    model_id = "null-backend"

    async def logits_batch(self, state_json: str, questions: Sequence[Question]):
        return [[0.0] * question.cardinality for question in questions]


class NoBatch:
    """Envolve um backend de lote escondendo `logits_batch`."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.model_id = inner.model_id

    async def logits(self, state_json: str, question: Question):
        return await self._inner.logits(state_json, question)


def _state(repeat: int = 4) -> dict[str, Any]:
    sentence = (
        "Cliente relata que a integração de pagamentos falha desde ontem, "
        "está perdendo vendas e pede retorno urgente. "
    )
    return {"ticket": sentence * repeat, "plan": "pro", "tickets_abertos": 3}


def _mixed(count: int) -> list[Question]:
    questions: list[Question] = []
    for index in range(count):
        kind = index % 3
        if kind == 0:
            questions.append(
                Question.choice(
                    f"rota{index}",
                    "Qual equipe deve atender",
                    {
                        "billing": "Cobrança, faturas, reembolso",
                        "technical": "Erros, integrações, indisponibilidade",
                        "sales": "Preço, upgrade, contrato",
                    },
                )
            )
        elif kind == 1:
            questions.append(
                Question.score(
                    f"severidade{index}",
                    "Qual a severidade do relato",
                    ["cosmético", "falha com alternativa", "bloqueio sem alternativa"],
                )
            )
        else:
            questions.append(Question.noul(f"urgente{index}", "O relato indica urgência"))
    return questions


async def _timed(engine: SpecterDecisionEngine, state: Any, questions, iterations: int):
    await engine.decide(state, questions)  # aquecimento
    samples: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        await engine.decide(state, questions)
        samples.append((time.perf_counter() - started) * 1000.0)
    samples.sort()
    elapsed = sum(samples) / 1000.0
    return {
        "p50_ms": round(statistics.median(samples), 4),
        "p95_ms": round(samples[min(len(samples) - 1, int(len(samples) * 0.95))], 4),
        "p99_ms": round(samples[min(len(samples) - 1, int(len(samples) * 0.99))], 4),
        "decisions_per_s": round(len(questions) * iterations / elapsed, 1) if elapsed else None,
    }


def _engine(backend: Any, *, calibration: Any, concurrency: int = 8) -> SpecterDecisionEngine:
    return SpecterDecisionEngine(
        backend, calibration, domain="bench", concurrency=concurrency, timeout=10.0
    )


async def run(iterations: int) -> dict[str, Any]:
    """Executa todas as medições e devolve o relatório."""
    state = _state()
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "iterations": iterations,
        "state_bytes": len(json.dumps(state)),
        "note": "backend heurístico/nulo: mede engenharia, não inteligência",
    }

    simple = SimpleCalibration(model_id="null-backend")
    report["kernel_overhead"] = {
        str(batch): await _timed(
            _engine(NullBatchBackend(), calibration=simple), state, _mixed(batch), iterations
        )
        for batch in (1, 8, 32, 128)
    }

    heuristic = {}
    for batch in (1, 8, 32, 128):
        questions = _mixed(batch)
        batched = await _timed(
            _engine(
                SystemOneLocalBackend(),
                calibration=SystemOneCalibration(domain="bench"),
            ),
            state,
            questions,
            iterations,
        )
        serial = await _timed(
            _engine(
                NoBatch(SystemOneLocalBackend()),
                calibration=SystemOneCalibration(domain="bench"),
            ),
            state,
            questions,
            iterations,
        )
        heuristic[str(batch)] = {
            "batch": batched,
            "fanout": serial,
            "speedup_p50": round(serial["p50_ms"] / batched["p50_ms"], 2)
            if batched["p50_ms"]
            else None,
        }
    report["batch_vs_fanout"] = heuristic

    repeated = [Question(f"q{index}", "noul", "O relato indica urgência") for index in range(64)]
    distinct = [
        Question(f"q{index}", "noul", f"O relato indica urgência do tipo {index}")
        for index in range(64)
    ]
    engine = _engine(
        SystemOneLocalBackend(), calibration=SystemOneCalibration(domain="bench")
    )
    report["deduplication"] = {
        "repeated_64": await _timed(engine, state, repeated, iterations),
        "distinct_64": await _timed(engine, state, distinct, iterations),
    }

    tracemalloc.start()
    engine = _engine(
        SystemOneLocalBackend(), calibration=SystemOneCalibration(domain="bench")
    )
    for _ in range(50):
        await engine.decide(state, _mixed(128))
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    report["memory"] = {
        "traced_current_kib": round(current / 1024, 1),
        "traced_peak_kib": round(peak / 1024, 1),
        "process_max_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }
    return report


def main() -> int:
    """Ponto de entrada da linha de comando."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--json", dest="output")
    args = parser.parse_args()
    report = asyncio.run(run(args.iterations))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
