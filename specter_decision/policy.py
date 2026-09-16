"""Políticas de consumo: o que o CÓDIGO faz com uma decisão tipada.

Uma decisão não autoriza nada. Quem decide agir é o programa, e isso deve
ser explícito. Este módulo concentra os três padrões que aparecem em
praticamente todo fluxo real:

* **Gate por confiança** — três faixas (agir / verificar / humano), com
  limiar proporcional ao risco da ação.
* **Score composto** — quebrar um julgamento complexo em notas atômicas e
  combinar com pesos que ficam no seu código, não no modelo.
* **Alta cardinalidade** — Choice tem teto de 255 opções; acima disso,
  pontuar em blocos e decidir em segundo estágio.

Nada aqui chama backend por conta própria: as funções recebem decisões
prontas ou um engine explícito.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .types import Decision, Question

__all__ = ["ConfidenceGate", "Route", "composite_score", "is_ok", "rank_then_choose", "require"]


class Route:
    """Rótulos de roteamento produzidos pelo gate."""

    ACT = "act"
    VERIFY = "verify"
    HUMAN = "human"
    UNAVAILABLE = "unavailable"


def is_ok(decision: Decision) -> bool:
    """True se a decisão tem valor utilizável."""
    return decision.get("status") == "ok" and decision.get("error") is None


@dataclass(frozen=True, slots=True)
class ConfidenceGate:
    """Faixas de confiança calibradas para uma ação específica.

    Attributes:
        act: confiança mínima para agir sem supervisão.
        verify: confiança mínima para agir com confirmação.

    Limiares dependem do risco: uma leitura pode agir em 0.6; uma
    transferência de dinheiro talvez exija 0.95.
    """

    act: float = 0.85
    verify: float = 0.60

    def __post_init__(self) -> None:
        if not 0.0 <= self.verify <= self.act <= 1.0:
            raise ValueError("limiares devem satisfazer 0 <= verify <= act <= 1")

    def route(self, decision: Decision) -> str:
        """Classifica a decisão em act / verify / human / unavailable."""
        if not is_ok(decision):
            return Route.UNAVAILABLE
        confidence = decision.get("confidence")
        if confidence is None:
            return Route.HUMAN
        if confidence >= self.act:
            return Route.ACT
        if confidence >= self.verify:
            return Route.VERIFY
        return Route.HUMAN

    def partition(self, decisions: Iterable[Decision]) -> dict[str, list[Decision]]:
        """Agrupa decisões por rota, preservando a ordem de entrada."""
        buckets: dict[str, list[Decision]] = {
            Route.ACT: [],
            Route.VERIFY: [],
            Route.HUMAN: [],
            Route.UNAVAILABLE: [],
        }
        for decision in decisions:
            buckets[self.route(decision)].append(decision)
        return buckets


def require(decision: Decision, minimum: float = 0.0) -> Any:
    """Extrai o valor exigindo sucesso e confiança mínima.

    Args:
        decision: decisão devolvida pelo kernel.
        minimum: confiança mínima aceitável.

    Returns:
        O valor tipado da decisão.

    Raises:
        ValueError: se a decisão falhou ou a confiança é insuficiente.
    """
    if not is_ok(decision):
        raise ValueError(f"decisão indisponível: {decision.get('error')}")
    confidence = decision.get("confidence")
    if minimum and (confidence is None or confidence < minimum):
        raise ValueError(f"confiança {confidence} abaixo do mínimo {minimum}")
    return decision["value"]


def composite_score(
    decisions: Sequence[Decision],
    weights: Mapping[str, float],
    *,
    normalize: bool = True,
) -> dict[str, float]:
    """Combina notas atômicas em um índice, com pesos do seu código.

    Cada Score é normalizado para [0,1] dividindo pela cardinalidade menos
    um, de modo que rubricas com números diferentes de níveis possam ser
    somadas. A confiança agregada é a MENOR confiança entre os componentes
    usados — o elo mais fraco governa, o que é conservador de propósito.

    Args:
        decisions: decisões do tipo score/noul já avaliadas.
        weights: peso por id de decisão; ids ausentes são ignorados.
        normalize: divide pelo peso total utilizado.

    Returns:
        Dicionário com `value`, `confidence`, `coverage` e `missing`.
    """
    total_weight = 0.0
    accumulated = 0.0
    confidences: list[float] = []
    missing: list[str] = []
    for decision in decisions:
        weight = weights.get(decision["id"])
        if weight is None:
            continue
        if not is_ok(decision):
            missing.append(decision["id"])
            continue
        legend = decision.get("legend") or []
        value = decision["value"]
        if decision["type"] == "score" and len(legend) > 1:
            unit = float(value) / (len(legend) - 1)
        elif decision["type"] == "noul":
            unit = float(value)
        else:
            missing.append(decision["id"])
            continue
        accumulated += weight * unit
        total_weight += abs(weight)
        confidence = decision.get("confidence")
        if confidence is not None:
            confidences.append(float(confidence))
    requested = math.fsum(abs(value) for value in weights.values()) or 1.0
    value = accumulated / total_weight if (normalize and total_weight) else accumulated
    return {
        "value": value,
        "confidence": min(confidences) if confidences else 0.0,
        "coverage": total_weight / requested,
        "missing": missing,
    }


async def rank_then_choose(
    engine: Any,
    state: Any,
    instructions: str,
    options: Sequence[str],
    *,
    criteria: Mapping[str, str] | None = None,
    shortlist: int = 8,
    block_size: int = 255,
) -> Decision:
    """Escolha com cardinalidade acima do teto, em dois estágios.

    Estágio 1: divide as opções em blocos de até `block_size`, pontua cada
    bloco em paralelo e junta as melhores. Estágio 2: uma única Choice
    entre as finalistas, que é a decisão devolvida.

    As probabilidades do estágio 2 são condicionais à lista curta — isso é
    uma propriedade do método, não um defeito, e deve ser considerada ao
    interpretar a confiança.

    Args:
        engine: instância de `SpecterDecisionEngine`.
        state: estado a avaliar.
        instructions: enunciado da escolha.
        options: universo de opções (pode passar de 255).
        criteria: rubrica opcional por opção.
        shortlist: quantas finalistas levar ao segundo estágio.
        block_size: tamanho máximo de cada bloco do primeiro estágio.

    Returns:
        Decisão do tipo choice sobre a lista curta.

    Raises:
        ValueError: se `options` tiver menos de duas entradas.
    """
    unique = list(dict.fromkeys(options))
    if len(unique) < 2:
        raise ValueError("INVALID_INPUT: é preciso ao menos duas opções")
    if len(unique) <= block_size:
        blocks = [unique]
    else:
        blocks = [
            unique[start : start + block_size]
            for start in range(0, len(unique), block_size)
        ]

    def build(identifier: str, labels: Sequence[str]) -> Question:
        if criteria:
            return Question.choice(
                identifier, instructions, {label: criteria.get(label) for label in labels}
            )
        return Question.choice(identifier, instructions, tuple(labels))

    finalists: list[tuple[float, str]] = []
    if len(blocks) == 1:
        finalists = [(1.0, label) for label in blocks[0]]
    else:
        questions = [build(f"_stage1_{index}", block) for index, block in enumerate(blocks)]
        for decision, block in zip(await engine.decide(state, questions), blocks):
            if not is_ok(decision):
                continue
            probabilities = decision["probabilities"] or []
            finalists.extend(zip(probabilities, block))
        finalists.sort(key=lambda pair: (-pair[0], pair[1]))
        finalists = finalists[: max(2, shortlist)]
        if len(finalists) < 2:
            finalists = [(1.0, label) for label in unique[:2]]

    labels = [label for _, label in finalists]
    decision = (await engine.decide(state, [build("_stage2", labels)]))[0]
    decision["id"] = "final"
    return decision
