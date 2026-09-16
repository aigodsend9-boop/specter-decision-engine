"""Exemplo 1 — triagem de suporte com gate por confiança.

Mostra o padrão completo: perguntas com rubrica, fan-out especulativo em uma
única chamada, roteamento por faixa de confiança e índice composto calculado
no seu código (não no modelo).

    python examples/01_triagem_suporte.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from specter_decision import (  # noqa: E402
    ConfidenceGate,
    Question,
    Route,
    SpecterDecisionEngine,
    SystemOneCalibration,
    SystemOneLocalBackend,
    composite_score,
)

TICKET = {
    "canal": "email",
    "plano": "pro",
    "mensagem": (
        "Boa tarde. A integração de pagamentos parou de funcionar ontem à noite, "
        "as cobranças estão falhando e já perdi vendas hoje. Preciso de retorno urgente."
    ),
    "tickets_abertos": 2,
}

PERGUNTAS = [
    Question.choice(
        "equipe",
        "Qual equipe deve assumir este atendimento?",
        {
            "billing": "Cobrança, faturas, reembolso, assinatura",
            "technical": "Erros, integrações, indisponibilidade, API",
            "sales": "Preço, upgrade, contrato, proposta",
        },
    ),
    Question.score(
        "severidade",
        "Qual a severidade operacional do relato?",
        [
            "cosmético, sem impacto",
            "falha com alternativa disponível",
            "bloqueio sem alternativa",
        ],
        tolerance=0.5,
    ),
    Question.score(
        "frustracao",
        "Qual o nível de frustração aparente do cliente?",
        ["calmo", "incomodado", "muito irritado"],
    ),
    Question.noul("urgente", "O relato indica urgência ou prazo apertado?"),
    # Pergunta especulativa: custa quase nada no mesmo lote e talvez nem
    # seja lida. Esse é o padrão "speculative fan-out".
    Question.noul("risco_churn", "O relato sugere risco de cancelamento?"),
]


async def main() -> None:
    engine = SpecterDecisionEngine(
        SystemOneLocalBackend(),
        SystemOneCalibration(domain="suporte-v1"),
        domain="suporte-v1",
        concurrency=8,
        timeout=2.0,
    )
    decisoes = await engine.decide(TICKET, PERGUNTAS)
    por_id = {item["id"]: item for item in decisoes}

    print(json.dumps(decisoes, ensure_ascii=False, indent=2))

    # Limiares por risco da ação: encaminhar é barato, escalar acorda gente.
    encaminhar = ConfidenceGate(act=0.70, verify=0.45)
    escalar = ConfidenceGate(act=0.90, verify=0.75)

    rota = encaminhar.route(por_id["equipe"])
    print(f"\nequipe = {por_id['equipe']['value']!r} → rota {rota}")
    if rota == Route.ACT:
        print("  ação: encaminhar automaticamente para a fila")
    elif rota == Route.VERIFY:
        print("  ação: encaminhar com confirmação do agente")
    else:
        print("  ação: fila geral, decisão humana")

    # Prioridade composta: pesos são SEUS, ficam no código e são auditáveis.
    prioridade = composite_score(
        decisoes,
        {"severidade": 0.5, "urgente": 0.3, "frustracao": 0.2},
    )
    print(
        f"\nprioridade = {prioridade['value']:.3f} "
        f"(confiança mínima {prioridade['confidence']:.3f}, "
        f"cobertura {prioridade['coverage']:.0%})"
    )
    if prioridade["value"] > 0.66 and escalar.route(por_id["severidade"]) == Route.ACT:
        print("  ação: escalar para plantão")
    else:
        print("  ação: fila normal — sem confiança suficiente para acordar o plantão")

    print("\nbackend:", json.dumps(engine.backend.stats(), ensure_ascii=False))
    print(
        "\nNota: este ticket mistura vocabulário de cobrança e de integração, "
        "então o baseline léxico fica dividido e a confiança cai. O gate mandar "
        "para revisão humana é o comportamento correto — é exatamente para isso "
        "que a confiança existe. Com um backend treinado, a mesma estrutura "
        "decide sozinha e o gate só intervém nos casos realmente ambíguos."
    )


if __name__ == "__main__":
    asyncio.run(main())
