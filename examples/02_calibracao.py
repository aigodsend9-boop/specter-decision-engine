"""Exemplo 2 — ajustar, auditar e usar um artefato de calibração.

O kernel se recusa a decidir sem calibração compatível. Este exemplo mostra
o ciclo inteiro com dados sintéticos: ajustar temperatura, ajustar o mapa de
confiança, ler as métricas, salvar em disco e recarregar.

    python examples/02_calibracao.py

Os dados aqui são gerados com semente fixa e servem para exercitar o
processo. Em produção, os rótulos vêm do seu domínio, separados por entidade
e por tempo — e essa separação é responsabilidade sua, não do código.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from specter_decision import Question, SpecterDecisionEngine  # noqa: E402
from specter_decision.calibration import CalibrationArtifact  # noqa: E402
from specter_decision.numerics import softmax  # noqa: E402

PERGUNTA = Question.choice(
    "equipe",
    "Qual equipe deve assumir?",
    {"billing": "Cobrança", "technical": "Erros", "sales": "Comercial"},
)


def registros_rotulados(quantidade: int = 600) -> list[tuple[Question, list[float], int]]:
    """Simula um backend com sinal real, porém exageradamente confiante."""
    gerador = random.Random(2026)
    dados: list[tuple[Question, list[float], int]] = []
    for _ in range(quantidade):
        verdade = gerador.randrange(3)
        base = [gerador.gauss(0.0, 0.7) for _ in range(3)]
        base[verdade] += 1.2
        dados.append((PERGUNTA, [valor * 3.0 for valor in base], verdade))
    return dados


class BackendGravado:
    """Backend que devolve logits pré-gravados — determinístico para o exemplo."""

    model_id = "backend-exemplo/1"

    def __init__(self, logits: list[float]) -> None:
        self._logits = logits

    async def logits(self, state_json: str, question: Question) -> list[float]:
        return self._logits


def main() -> None:
    dados = registros_rotulados()

    artefato = CalibrationArtifact.fit(
        dados,
        domain="suporte-v1",
        model_id="backend-exemplo/1",
        holdout=0.3,
        bind_signatures=True,
        notes="exemplo sintético — não usar em produção",
    )

    metricas = artefato.metrics
    print(f"temperatura ajustada : {artefato.temperature_value:.4f}")
    print(f"NLL antes / depois   : {metricas['nll_before']:.4f} → {metricas['nll_after']:.4f}")
    print(f"Brier (teste)        : {metricas['brier']:.4f}")
    print(f"ECE (teste)          : {metricas['ece']:.4f}")
    baixo, alto = metricas["accuracy_ci95"]
    print(f"acurácia (IC95)      : {metricas['accuracy']:.3f}  [{baixo:.3f}, {alto:.3f}]")
    print(f"hash do conjunto     : {artefato.dataset_hash}")

    print("\ntabela de confiabilidade (confiança declarada vs acerto medido):")
    for linha in metricas["reliability"]:
        largura = int(linha["n"] / max(1, metricas["n_test"]) * 40)
        print(
            f"  [{linha['lower']:.1f},{linha['upper']:.1f}) "
            f"n={int(linha['n']):3d}  conf={linha['confidence']:.3f}  "
            f"acc={linha['accuracy']:.3f}  {'█' * largura}"
        )

    destino = Path(__file__).with_name("suporte-v1.calibracao.json")
    destino.write_text(artefato.to_json(), encoding="utf-8")
    recarregado = CalibrationArtifact.from_json(destino.read_text(encoding="utf-8"))
    print(f"\nartefato salvo e recarregado: {destino.name}")

    # O artefato recusa o que não cobre — e recusar é a resposta certa.
    print("cobre o domínio certo   :", recarregado.supports("suporte-v1", "backend-exemplo/1", PERGUNTA))
    print("cobre outro domínio     :", recarregado.supports("outro", "backend-exemplo/1", PERGUNTA))
    print("cobre outro modelo      :", recarregado.supports("suporte-v1", "outro-modelo", PERGUNTA))
    outra = Question.choice("x", "Pergunta diferente?", {"a": None, "b": None, "c": None})
    print("cobre outra pergunta    :", recarregado.supports("suporte-v1", "backend-exemplo/1", outra))

    import asyncio

    # Dois casos: um em que o modelo acerta e outro em que erra confiante.
    # Calibração não elimina o erro — ela faz o número 0.93 significar
    # "certo em ~93% das vezes", o que permite ao seu código decidir
    # quando confiar e quando escalar.
    def avaliar(indice: int) -> dict:
        pergunta, logits, verdade = dados[indice]
        engine = SpecterDecisionEngine(
            BackendGravado(logits), recarregado, domain="suporte-v1", timeout=2.0
        )
        decisao = asyncio.run(engine.decide({"ticket": "exemplo"}, [pergunta]))[0]
        decisao["_verdade"] = pergunta.labels[verdade]
        decisao["_cru"] = [round(v, 3) for v in softmax(logits, 1.0)]
        return decisao

    acerto = next(d for d in map(avaliar, range(40)) if d["value"] == d["_verdade"])
    erro = next(d for d in map(avaliar, range(40)) if d["value"] != d["_verdade"])
    for rotulo, decisao in (("acerto", acerto), ("erro", erro)):
        print(f"\n{rotulo}:")
        print(f"  probabilidades sem calibração : {decisao['_cru']}")
        print(f"  probabilidades com calibração : {[round(v, 3) for v in decisao['probabilities']]}")
        print(
            f"  valor={decisao['value']!r}  confiança={decisao['confidence']:.3f}  "
            f"verdade={decisao['_verdade']!r}"
        )

    print(
        "\nLeitura correta: a confiança de 0.9 não promete acerto neste caso, "
        "promete a frequência de acerto na faixa. Se a tabela de confiabilidade "
        "mostrar acurácia abaixo da confiança declarada (como no bin [0.8,0.9) "
        "acima), o artefato ainda está otimista e precisa de mais dados ou de "
        "limiar mais alto para agir sozinho."
    )

    print("\nenvelope completo (caso de acerto):")
    limpo = {chave: valor for chave, valor in acerto.items() if not chave.startswith("_")}
    print(json.dumps(limpo, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
