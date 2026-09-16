"""Interface de linha de comando: `specter-decision`.

Subcomandos:
    evaluate      avalia um documento JSON (arquivo ou stdin)
    capabilities  imprime a configuração corrente
    selftest      roda a suíte de contrato embarcada
    bench         mede latência/throughput no hardware real
    calibrate     ajusta um artefato a partir de registros rotulados

Saída sempre em JSON em stdout; erros em stderr com código de saída != 0.
Nenhum subcomando faz rede ou grava fora do caminho pedido.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from typing import Any, Sequence

from . import __version__
from .backends.system_one import SystemOneCalibration, SystemOneLocalBackend
from .calibration import CalibrationArtifact
from .engine import SpecterDecisionEngine
from .service import parse_questions
from .types import Question

__all__ = ["main"]


def _build_engine(domain: str, timeout: float, concurrency: int) -> SpecterDecisionEngine:
    return SpecterDecisionEngine(
        SystemOneLocalBackend(),
        SystemOneCalibration(domain=domain),
        domain=domain,
        concurrency=concurrency,
        timeout=timeout,
    )


def _load(path: str | None) -> Any:
    raw = sys.stdin.read() if path in (None, "-") else open(path, encoding="utf-8").read()
    return json.loads(raw)


def _emit(payload: Any) -> None:
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2, allow_nan=False)
    sys.stdout.write("\n")


def _cmd_evaluate(args: argparse.Namespace) -> int:
    document = _load(args.input)
    questions = parse_questions(document.get("questions"))
    engine = _build_engine(
        str(document.get("domain", args.domain)), args.timeout, args.concurrency
    )
    answers = asyncio.run(engine.decide(document.get("state"), questions))
    _emit({"status": "ok", "domain": engine.domain, "answers": answers})
    return 0


def _cmd_capabilities(args: argparse.Namespace) -> int:
    engine = _build_engine(args.domain, args.timeout, args.concurrency)
    _emit({"version": __version__, **engine.capabilities()})
    return 0


def _cmd_selftest(_: argparse.Namespace) -> int:
    import unittest

    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def _bench_state(size: int) -> dict[str, Any]:
    sentence = (
        "Cliente relata que a integração de pagamentos falha desde ontem, "
        "está perdendo vendas e pede retorno urgente. "
    )
    return {"ticket": sentence * max(1, size), "plan": "pro", "tickets_abertos": 3}


def _bench_questions(count: int) -> list[Question]:
    pool: list[Question] = []
    for index in range(count):
        kind = index % 3
        if kind == 0:
            pool.append(
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
            pool.append(
                Question.score(
                    f"severidade{index}",
                    "Qual a severidade do relato",
                    ["cosmético", "falha com alternativa", "bloqueio sem alternativa"],
                )
            )
        else:
            pool.append(Question.noul(f"urgente{index}", "O relato indica urgência"))
    return pool


async def _bench(batch: int, iterations: int, concurrency: int, state_size: int) -> dict[str, Any]:
    engine = _build_engine("bench", 5.0, concurrency)
    state = _bench_state(state_size)
    questions = _bench_questions(batch)
    await engine.decide(state, questions)  # aquece caches
    samples: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        await engine.decide(state, questions)
        samples.append((time.perf_counter() - started) * 1000.0)
    samples.sort()
    total = sum(samples) / 1000.0
    return {
        "batch": batch,
        "iterations": iterations,
        "concurrency": concurrency,
        "state_bytes": len(json.dumps(state)),
        "p50_ms": round(statistics.median(samples), 4),
        "p95_ms": round(samples[min(len(samples) - 1, int(len(samples) * 0.95))], 4),
        "p99_ms": round(samples[min(len(samples) - 1, int(len(samples) * 0.99))], 4),
        "mean_ms": round(statistics.fmean(samples), 4),
        "decisions_per_s": round(batch * iterations / total, 1) if total else None,
    }


def _cmd_bench(args: argparse.Namespace) -> int:
    results = [
        asyncio.run(_bench(batch, args.iterations, args.concurrency, args.state_size))
        for batch in args.batches
    ]
    _emit({"version": __version__, "note": "backend heurístico local; não mede inteligência", "results": results})
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    document = _load(args.input)
    records = []
    for entry in document["records"]:
        question = parse_questions([entry["question"]])[0]
        records.append((question, [float(v) for v in entry["logits"]], int(entry["truth"])))
    artifact = CalibrationArtifact.fit(
        records,
        domain=args.domain,
        model_id=args.model_id,
        holdout=args.holdout,
        bind_signatures=args.bind_signatures,
        notes=args.notes,
    )
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(artifact.to_json())
    _emit({"status": "ok", "metrics": artifact.metrics, "temperature": artifact.temperature_value})
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Ponto de entrada do console script.

    Args:
        argv: argumentos; None usa `sys.argv[1:]`.

    Returns:
        Código de saída do processo.
    """
    parser = argparse.ArgumentParser(prog="specter-decision", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--domain", default="default")
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--concurrency", type=int, default=8)
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate", help="avalia um documento JSON")
    evaluate.add_argument("input", nargs="?", default="-")
    evaluate.set_defaults(handler=_cmd_evaluate)

    capabilities = subparsers.add_parser("capabilities", help="configuração corrente")
    capabilities.set_defaults(handler=_cmd_capabilities)

    selftest = subparsers.add_parser("selftest", help="roda a suíte de contrato")
    selftest.set_defaults(handler=_cmd_selftest)

    bench = subparsers.add_parser("bench", help="mede latência e throughput")
    bench.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    bench.add_argument("--iterations", type=int, default=200)
    bench.add_argument("--state-size", type=int, default=4)
    bench.set_defaults(handler=_cmd_bench)

    calibrate = subparsers.add_parser("calibrate", help="ajusta artefato de calibração")
    calibrate.add_argument("input")
    calibrate.add_argument("--output")
    calibrate.add_argument("--model-id", default="unknown")
    calibrate.add_argument("--holdout", type=float, default=0.3)
    calibrate.add_argument("--bind-signatures", action="store_true")
    calibrate.add_argument("--notes", default="")
    calibrate.set_defaults(handler=_cmd_calibrate)

    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception as error:  # fronteira do processo: erro legível, sem traceback
        print(f"erro: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
