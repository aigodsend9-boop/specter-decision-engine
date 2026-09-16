"""Exemplo 4 — ativar a superfície HTTP sem abrir porta nenhuma.

O hook `dispatch_decision` é chamado ANTES do leitor de corpo legado do
Core: ele responde às três rotas de decisão e devolve None para qualquer
outra coisa. Este exemplo exercita o caminho inteiro em memória —
autenticação, envelope, limites, erros — sem socket, sem thread, sem
alterar nada no seu servidor.

    python examples/04_servico_http.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from specter_decision import (  # noqa: E402
    SpecterDecisionEngine,
    SystemOneCalibration,
    SystemOneLocalBackend,
)
from specter_decision.http import configure, dispatch_decision  # noqa: E402
from specter_decision.service import DecisionService  # noqa: E402

TOKEN = os.environ.get("SPECTER_DECISION_TOKEN", "token-de-exemplo-nao-use-em-producao")


def fabrica(dominio: str) -> SpecterDecisionEngine:
    """Uma instância nova por requisição: nada de semáforo entre event loops.

    Em produção, `backend` e `calibration` vêm de um registro allowlist do
    operador. NUNCA importe caminho ou módulo vindo da requisição.
    """
    return SpecterDecisionEngine(
        SystemOneLocalBackend(),
        SystemOneCalibration(domain=dominio),
        domain=dominio,
        concurrency=8,
        timeout=2.0,
    )


DOCUMENTO = {
    "domain": "suporte-v1",
    "state": {"ticket": "A integração parou e as cobranças estão falhando desde ontem."},
    "questions": {
        "equipe": {
            "type": "choice",
            "instructions": "Qual equipe deve atender?",
            "criteria": {
                "billing": "Cobrança, faturas, reembolso",
                "technical": "Erros, integrações, indisponibilidade",
            },
        },
        "urgente": {"type": "noul", "instructions": "O relato indica urgência?"},
    },
}


def chamar(metodo: str, caminho: str, *, token: str | None = TOKEN, corpo=None):
    """Chama o hook como o servidor legado chamaria."""
    cabecalhos = {"Authorization": f"Bearer {token}"} if token else {}
    dados = json.dumps(corpo).encode("utf-8") if corpo is not None else None
    resposta = dispatch_decision(metodo, caminho, cabecalhos, dados)
    if resposta is None:
        return None, None
    status, _, body = resposta
    return status, json.loads(body) if body else {}


def main() -> None:
    print("1. Sem serviço configurado — instalado em disco não é ativado:")
    configure(None)
    print("  ", chamar("GET", "/v1/decision/capabilities")[1], "\n")

    print("2. Rota que não é nossa passa adiante (None = siga o fluxo legado):")
    configure(DecisionService(token=TOKEN, engine_factory=fabrica, max_requests=2))
    print("   /status →", chamar("GET", "/status")[0], "\n")

    print("3. Sem token ou com token errado:")
    print("   sem token   →", chamar("GET", "/v1/decision/capabilities", token=None))
    print("   token errado→", chamar("GET", "/v1/decision/capabilities", token="errado"), "\n")

    print("4. Capacidades (não prometem prontidão produtiva):")
    _, capacidades = chamar("GET", "/v1/decision/capabilities")
    print("  ", json.dumps(capacidades, ensure_ascii=False), "\n")

    print("5. Avaliação:")
    status, envelope = chamar("POST", "/v1/decision/evaluate", corpo=DOCUMENTO)
    print(f"   HTTP {status}  contrato={envelope['contract']}  modelo={envelope['model']}")
    for decisao in envelope["answers"]:
        print(
            f"   - {decisao['id']:8s} {decisao['status']:5s} "
            f"valor={decisao['value']!r} confiança={decisao['confidence']}"
        )
    print()

    print("6. Erros tipados (nenhum vira decisão inventada):")
    quebrado = dispatch_decision(
        "POST", "/v1/decision/evaluate", {"Authorization": f"Bearer {TOKEN}"}, b"{nao-e-json"
    )
    print("   JSON quebrado   →", quebrado[0], json.loads(quebrado[2])["error"])
    grande = dispatch_decision(
        "POST", "/v1/decision/evaluate", {"Authorization": f"Bearer {TOKEN}"}, b"x" * (300 * 1024)
    )
    print("   corpo gigante   →", grande[0], json.loads(grande[2])["error"])
    print("   método errado   →", chamar("DELETE", "/v1/decision/evaluate", corpo={})[0])
    print(
        "   pergunta inválida→",
        chamar(
            "POST",
            "/v1/decision/evaluate",
            corpo={"state": "x", "questions": [{"id": "a", "type": "extract"}]},
        ),
    )

    print(
        "\nPara expor isto fora da máquina: proxy TLS com allowlist só destas "
        "rotas, autenticação e rate limit no proxy, token rotacionado. "
        "Não publique o Core inteiro para lançar esta API."
    )
    configure(None)


if __name__ == "__main__":
    main()
