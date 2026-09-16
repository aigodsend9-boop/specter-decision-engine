"""Exemplo 3 — escrever o seu backend com prefill único (o ganho estrutural).

Este é o exemplo mais importante do pacote. Ele mostra a diferença entre
implementar só `logits` (o kernel chama N vezes) e implementar também
`logits_batch` (o kernel chama uma vez e você codifica o estado uma vez).

Em um modelo real, codificar o estado é o custo dominante: um estado de 4 KB
codificado 128 vezes é 128× trabalho desperdiçado. Aqui o custo é simulado
com trabalho de CPU de verdade (não `sleep`), porque é assim que um encoder
se comporta: ele ocupa o processador e **não** se sobrepõe com outras
chamadas. É justamente por isso que concorrência de chamadas não substitui
prefill único — e por isso um backend bloqueante precisa de executor.

    python examples/03_backend_proprio.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from specter_decision import (  # noqa: E402
    Question,
    SimpleCalibration,
    SpecterDecisionEngine,
)

CUSTO_PREFILL = 0.010  # 10 ms: codificar o estado (na vida real, o encoder)
CUSTO_CABECA = 0.001  # 1 ms: pontuar uma pergunta contra o estado já codificado


def _queimar(segundos: float) -> None:
    """Consome CPU pelo tempo pedido — simula trabalho real de encoder."""
    limite = time.perf_counter() + segundos
    while time.perf_counter() < limite:
        pass


class BackendSemLote:
    """Só implementa `logits`: paga o prefill em toda pergunta."""

    model_id = "exemplo/sem-lote"

    def __init__(self) -> None:
        self.prefills = 0

    async def logits(self, state_json: str, question: Question) -> list[float]:
        _queimar(CUSTO_PREFILL)  # codifica o estado DE NOVO, a cada pergunta
        self.prefills += 1
        _queimar(CUSTO_CABECA)
        await asyncio.sleep(0)  # devolve o controle ao loop
        return self._pontuar(question)

    @staticmethod
    def _pontuar(question: Question) -> list[float]:
        # Substitua por cabeças numéricas reais. O contrato exige apenas
        # `cardinality` floats finitos, na ordem dos rótulos.
        return [float(indice) for indice in range(question.cardinality)]


class BackendComLote(BackendSemLote):
    """Implementa `logits_batch`: um prefill para todas as perguntas."""

    model_id = "exemplo/com-lote"

    async def logits_batch(
        self, state_json: str, questions: Sequence[Question]
    ) -> list[list[float]]:
        _queimar(CUSTO_PREFILL)  # UMA vez, qualquer que seja N
        self.prefills += 1
        _queimar(CUSTO_CABECA * len(questions))
        await asyncio.sleep(0)
        return [self._pontuar(question) for question in questions]


async def medir(backend: BackendSemLote, perguntas: list[Question]) -> tuple[float, int]:
    engine = SpecterDecisionEngine(
        backend,
        SimpleCalibration(model_id=backend.model_id),
        domain="exemplo",
        concurrency=8,
        timeout=30.0,
    )
    inicio = time.perf_counter()
    decisoes = await engine.decide({"ticket": "estado de exemplo"}, perguntas)
    duracao = (time.perf_counter() - inicio) * 1000
    assert all(item["status"] == "ok" for item in decisoes)
    return duracao, backend.prefills


async def main() -> None:
    perguntas = [
        Question.noul(f"q{indice}", f"Afirmação número {indice} é verdadeira?")
        for indice in range(32)
    ]
    print(f"{len(perguntas)} perguntas, prefill de {CUSTO_PREFILL * 1000:.0f} ms\n")

    sem, prefills_sem = await medir(BackendSemLote(), perguntas)
    com, prefills_com = await medir(BackendComLote(), perguntas)

    print(f"sem logits_batch : {sem:8.1f} ms   prefills = {prefills_sem}")
    print(f"com logits_batch : {com:8.1f} ms   prefills = {prefills_com}")
    print(f"ganho            : {sem / com:8.1f}×")
    print(
        "\nO kernel detecta `logits_batch` sozinho: você não configura nada, "
        "não muda a chamada de `decide()` e backends antigos continuam "
        "funcionando pelo caminho de fan-out."
    )

    print("\nChecklist do contrato do backend:")
    for linha in (
        "logits/logits_batch são async e cooperativos (não bloqueiam o event loop)",
        "retornam exatamente `question.cardinality` floats finitos, na ordem dos rótulos",
        "não leem `question.id` — o kernel já o substituiu por '_'",
        "não guardam estado entre perguntas: o isolamento é parte do contrato",
        "propagam CancelledError; o kernel cancela o que passou do deadline",
        "trabalho pesado de CPU vai para executor ou processo separado",
    ):
        print(f"  - {linha}")


if __name__ == "__main__":
    asyncio.run(main())
