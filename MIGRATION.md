# Migração 0.2.x → 0.3.0

Tempo estimado: minutos. O contrato público foi preservado de propósito.

## O que continua funcionando sem tocar em nada

- `from engine import Question, SpecterDecisionEngine` (shim de
  compatibilidade preservado).
- `SpecterDecisionEngine(backend, calibration, domain=..., concurrency=...,
  timeout=...)` — mesma assinatura.
- Backends que implementam apenas `async logits(state_json, question)`.
- Calibrações com `supports(*args)`, `temperature(x)`, `confidence(q, p, v)`.
- Formato do `Decision`: `id`, `type`, `status`, `value`, `probabilities`,
  `confidence`, `legend`, `error`.
- Códigos `UNCALIBRATED`, `TIMEOUT`, `BACKEND_ERROR`, `INVALID_OUTPUT`.
- Rotas `/v1/decision/evaluate|capabilities|openapi.json` com Bearer token.

## A única mudança incompatível

**Perguntas inválidas falham mais cedo.** Em 0.2, `Question('x','choice','?',
('a','a'))` era construída e só explodia em `decide()`. Em 0.3 a validação
acontece no construtor.

```python
# 0.2 — validação na avaliação
casos = [("x", [Question("x", "choice", "?", ("a", "a"))])]   # construía
for estado, perguntas in casos:
    with assertRaises(ValueError):
        await engine.decide(estado, perguntas)

# 0.3 — validação na construção (mesma exceção, mesmo prefixo)
with assertRaises(ValueError):          # ValueError: INVALID_INPUT: rótulos duplicados
    Question("x", "choice", "?", ("a", "a"))
```

Se algum teste seu constrói perguntas inválidas fora do bloco
`assertRaises`, mova a construção para dentro. Nenhuma outra alteração é
necessária.

## Ganhos que exigem uma linha sua

**1. Lote (o mais importante).** Adicione `logits_batch` ao seu backend:

```python
class MeuBackend:
    model_id = "meu-modelo/1"

    async def logits(self, state_json, question):        # continua existindo
        ...

    async def logits_batch(self, state_json, questions):  # novo, opcional
        prefill = self.encode(state_json)                 # UMA vez
        return [self.score(prefill, q) for q in questions]
```

O kernel detecta sozinho. Sem isso, nada quebra — você só não ganha o
prefill único.

**2. Calibração real.** Troque a fixture por um artefato ajustado:

```python
from specter_decision.calibration import CalibrationArtifact
artefato = CalibrationArtifact.from_json(open("suporte-v1.json").read())
engine = SpecterDecisionEngine(backend, artefato, domain="suporte-v1")
```

O artefato recusa domínio ou `model_id` diferentes — e recusar é o
comportamento correto: melhor `UNCALIBRATED` do que confiança inventada.

**3. Rubricas nas opções.** `Question.choice("dept", "Qual equipe?", {"billing":
"Cobrança e faturas", "technical": "Erros e integrações"})` decide melhor que
rótulos nus. Há teste medindo essa diferença.

**4. Políticas.** Substitua `if confidence >= 0.8` espalhado pelo código por
`ConfidenceGate(act=..., verify=...)`, com limiar por risco da ação.

## Se você usava o serviço HTTP

`DecisionService(token=..., engine_factory=..., max_requests=2)` mantém a
assinatura. Novidades: `parse_questions` aceita o formato de mapa
(`{"id": {"type": ..., "instructions": ..., "criteria": ...}}`) além da
lista, e o envelope agora traz `contract`, `request_id` e `elapsed_ms`.
Clientes antigos ignoram campos extras sem problema.

## Rollback

O pacote 0.3 é autocontido em `specter_decision/`. Para voltar, restaure o
diretório anterior — não há migração de banco, de arquivo de configuração
nem de estado persistido, porque o kernel não persiste nada.
