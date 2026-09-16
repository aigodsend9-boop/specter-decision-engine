# Specter Decision Engine 0.3.0

**Decisões tipadas, não texto.** Kernel numérico para avaliação de decisões
limitadas e independentes dentro de software e agentes.

Autor do projeto original: **Guilherme Peralta Novaes**. Esta versão 0.3.0 é
um refinamento do desenho 0.1/0.2, feito com assistência de IA, preservando
o contrato público e a postura de honestidade do repositório original.

> **Status honesto.** O núcleo é executável, testado e medido. **Não há
> modelo treinado embarcado.** O backend local (`SystemOneLocalBackend`) é
> um pontuador léxico determinístico — um *baseline de regras* auditável,
> não um modelo semântico. Calibração real, avaliação semântica e gates de
> produção continuam pendentes e dependem do operador. Este projeto não
> reproduz os pesos nem o treinamento do Jev.

---

## 1. Por que este formato existe

Um LLM devolve string. Para o software usar, alguém precisa parsear,
validar e torcer para o modelo não sair dos trilhos. A alternativa é
inverter o contrato: o programa declara **antecipadamente** quais respostas
são possíveis, e o modelo devolve **números** sobre esse conjunto fechado.

| | LLM de chat | Decisão tipada |
|---|---|---|
| Saída | string arbitrária | valor do conjunto declarado |
| Erro de tipo | possível | impossível por construção |
| Incerteza | opcional e mal calibrada | distribuição + confiança em todo retorno |
| Falha | vira texto plausível | vira erro tipado, nunca uma decisão |
| Amostragem | sequencial por token | projeção paralela e independente |

Três primitivas, iguais às do desenho original:

| Tipo | `value` | `probabilities` / `legend` | Evento calibrado da `confidence` |
|---|---|---|---|
| `noul` | P(sim) em [0,1] | `[P(não), P(sim)]` / `[false,true]` | classificação pelo limiar 0.5 correta |
| `choice` | opção do argmax; empate resolve pelo menor índice | ordem das opções de entrada | opção escolhida correta |
| `score` | soma(i × P(i)) em [0, K−1] | ordem da legenda de entrada | erro absoluto ≤ `tolerance` |

`confidence` é P(evento acima) — não é entropia, não é `max(probabilities)`
cru e não é declaração do modelo sobre si mesmo.

## 2. Começo rápido

Requer apenas Python 3.11+. Sem dependências em runtime.

```bash
python -m unittest discover -s tests -v     # 126 testes
python -m specter_decision.cli capabilities
python tools/bench.py --iterations 300
```

```python
import asyncio
from specter_decision import (
    Question, SpecterDecisionEngine, SystemOneLocalBackend, SystemOneCalibration,
)

engine = SpecterDecisionEngine(
    SystemOneLocalBackend(),
    SystemOneCalibration(domain="suporte-v1"),   # troque por artefato real
    domain="suporte-v1",
    concurrency=8,
    timeout=2.0,
)

answers = asyncio.run(engine.decide(
    {"ticket": "A integração de pagamentos falhou e estou perdendo vendas."},
    [
        Question.choice("equipe", "Qual equipe deve atender?", {
            "billing":   "Cobrança, faturas, reembolso",
            "technical": "Erros, integrações, indisponibilidade",
            "sales":     "Preço, upgrade, contrato",
        }),
        Question.score("severidade", "Qual a severidade?",
                       ["cosmético", "falha com alternativa", "bloqueio sem alternativa"]),
        Question.noul("urgente", "O relato indica urgência?"),
    ],
))

rota = answers[0]
if rota["status"] == "ok" and rota["confidence"] >= 0.85:
    fila = rota["value"]        # a decisão NÃO autoriza nem executa ação
```

Descrever as opções (`criteria`) melhora materialmente a qualidade: é a
diferença entre o modelo ver `"alpha"` e ver `"alpha: cobrança, faturas e
reembolso"`. Há teste cobrindo exatamente esse ganho.

## 2.1 Exemplos executáveis

Todos rodam sem rede e sem configuração:

| Arquivo | O que demonstra |
|---|---|
| `examples/01_triagem_suporte.py` | triagem completa: rubricas, fan-out especulativo, gate por confiança, índice composto |
| `examples/02_calibracao.py` | ajustar artefato, ler tabela de confiabilidade, salvar/recarregar, ver acerto e erro confiante |
| `examples/03_backend_proprio.py` | escrever seu backend com prefill único — **8,3× medido** contra o caminho sem lote |
| `examples/04_servico_http.py` | ativar a superfície HTTP sem abrir porta, com erros tipados |

```bash
python examples/03_backend_proprio.py
# 32 perguntas, prefill de 10 ms
# sem logits_batch :    353.5 ms   prefills = 32
# com logits_batch :     42.5 ms   prefills = 1
# ganho            :      8.3x
```

## 3. Arquitetura

```
state → snapshot canônico (1x) → validação → planejamento
      → prefill único do estado
      → logits por pergunta        ┌ lote: 1 chamada, N cabeças
                                   └ fan-out: N chamadas, semáforo global
      → temperatura versionada → softmax POR SEGMENTO (nunca global)
      → projeção tipada → validação → Decision
```

- O mesmo snapshot vai para todas as perguntas. Nenhuma pergunta enxerga a
  resposta de outra. O `id` escolhido pela aplicação **nunca** chega ao
  backend (é substituído por `_`).
- Concorrência de chamadas não é paralelismo de GPU. O backend precisa ser
  cooperativo; adaptador bloqueante exige executor ou isolamento de
  processo próprio. O contrato sozinho não impede estado interno
  compartilhado.
- Requisição inválida é rejeitada com `ValueError('INVALID_INPUT: ...')`
  **antes** de qualquer inferência. Falha nunca vira decisão.

### Módulos

| Arquivo | Responsabilidade |
|---|---|
| `numerics.py` | softmax estável, temperatura, simplex exato, esperança |
| `canonical.py` | snapshot canônico, validação iterativa, fingerprint |
| `types.py` | `Question`/`Decision`, limites, assinatura criptográfica |
| `protocols.py` | `Backend`, `BatchBackend`, `Calibration` |
| `engine.py` | planejamento, agendamento com deadline, projeção |
| `calibration.py` | temperatura por NLL, isotônica PAV, métricas, artefato |
| `policy.py` | gate por confiança, score composto, alta cardinalidade |
| `backends/system_one.py` | baseline léxico local com prefill cacheado |
| `backends/jev.py` | adaptador para a API System One da TypeSafe |
| `service.py` / `http.py` | envelope, autenticação, rotas, cliente |
| `cli.py` | `evaluate`, `capabilities`, `selftest`, `bench`, `calibrate` |

## 4. O que mudou em relação a 0.2 — e por quê

1. **Lote nativo (`BatchBackend`).** Se o backend expõe `logits_batch`, o
   estado é codificado **uma vez** para N perguntas. É o ganho estrutural do
   desenho System One; fan-out virou o caminho de compatibilidade.
2. **Deduplicação por assinatura.** Perguntas de conteúdo idêntico (comuns
   em fan-out especulativo) são computadas uma vez. Medido: **2,2× mais
   rápido** em 64 perguntas repetidas.
3. **Admissão consciente do deadline.** Trabalho nascido vencido não é
   iniciado: devolve `TIMEOUT` sem custo de backend.
4. **Gate de calibração por pergunta.** Uma pergunta descoberta pelo
   artefato não derruba as outras.
5. **Semáforo por (instância, event loop).** Recriado quando o loop muda —
   elimina a classe de bug de semáforo compartilhado entre loops.
6. **Calibração de verdade.** 0.2 definia o protocolo mas não entregava como
   ajustar. Agora há temperatura por minimização de NLL, isotônica por PAV,
   Brier/NLL/ECE/confiabilidade com bootstrap e artefato versionado que
   **recusa** domínio, modelo ou pergunta fora do registro.
7. **Validação antecipada.** Pergunta inválida falha na construção, não na
   avaliação (mesmo `ValueError`, mesmo prefixo `INVALID_INPUT`).
8. **Simplex exato.** A soma das probabilidades é 1.0 em ponto flutuante,
   não "1.0 ± 1e-16".
9. **Abstenção e recibos, opcionais.** `abstain_below` transforma confiança
   baixa em `ABSTAINED`; `receipts=True` anexa hash auditável que vincula a
   decisão ao estado sem expor o conteúdo do estado.
10. **Políticas explícitas** (`ConfidenceGate`, `composite_score`,
    `rank_then_choose` para mais de 255 opções) — os padrões que todo fluxo
    real reimplementa errado.
11. **Adaptador Jev.** O mesmo contrato tipado, com um modelo treinado real
    por trás, quando o operador tem credencial.

Detalhes de migração em `MIGRATION.md`. Medições em `BENCHMARKS.md`.

## 5. Calibração

Sem artefato compatível, o kernel devolve `UNCALIBRATED` **sem chamar o
modelo**. O caminho para um artefato legítimo:

```python
from specter_decision.calibration import CalibrationArtifact

artefato = CalibrationArtifact.fit(
    registros,                    # [(Question, logits, verdade), ...]
    domain="suporte-v1",
    model_id=backend.model_id,
    holdout=0.3,
    bind_signatures=True,         # vincula às perguntas exatas
)
open("suporte-v1.json", "w").write(artefato.to_json())
print(artefato.metrics)           # nll, brier, ece, acurácia com IC95
```

Regras que o código não tem como impor por você:

1. Separe treino / calibração / teste **por entidade e por tempo**, sem
   duplicatas vazando entre os conjuntos.
2. Avalie Brier, NLL, diagramas de confiabilidade e ECE **com incerteza**.
   ECE isolado não basta.
3. Modelo novo, rubrica nova ou domínio novo **invalidam** o artefato.
4. Drift ou cobertura insuficiente → abstenção ou revisão humana.
5. Calibração é propriedade estatística no domínio medido, não garantia
   universal. O kernel não autentica a veracidade de um artefato fornecido
   pela aplicação.

## 6. Limites e postura de segurança

- 128 perguntas por requisição, 255 opções em `choice`, 10 níveis em
  `score`, 256 KiB de estado, profundidade 32.
- JSON com NaN/Infinity, chave não-string, ciclo, tipo não serializável ou
  cardinalidade inválida é **rejeitado**.
- Sem geração textual. Rótulos vêm da entrada; erros são enums
  (`UNCALIBRATED`, `TIMEOUT`, `BACKEND_ERROR`, `INVALID_OUTPUT`,
  `ABSTAINED`).
- `status=ok` no envelope significa requisição processada — **não** que
  todas as perguntas decidiram.
- A decisão não concede autorização. O engine não executa ações, não importa
  código recebido e não envia mensagens.
- Nenhum conteúdo de estado é registrado em log; só fingerprints.
- Exposição externa exige proxy TLS com allowlist das rotas de decisão,
  autenticação e rate limit **no proxy**. Não publique o Core inteiro para
  lançar esta API.

## 7. Gates obrigatórios antes de produção

1. Modelo numérico com licença e dataset definidos e avaliação semântica
   real (o backend local **não** atende a este gate).
2. Calibração fora do treino, artefato vinculado a hash de pergunta, modelo
   e domínio; Brier/NLL/ECE com incerteza, drift e política de abstenção.
3. Teste de carga no hardware real: p50/p95, erro, memória, custo e
   concorrência. Fixtures não medem inteligência nem latência de modelo.
4. Revisão do backend, do transporte TLS e dos segredos; rotação de token e
   isolamento entre clientes.
5. Ativação em janela de manutenção, smoke test autenticado, rollback em
   caso de regressão.
6. Licença de distribuição e destino de publicação definidos.

`production_ready` permanece `false` nesta entrega, mesmo com fábrica
registrada. Não prometemos perfeição estrutural, confiança universal nem
desempenho equivalente ao Jev.

## 8. Fontes

- [TypeSafe: introdução e primitivas](https://docs.typesafe.ai/introduction)
  — decisões tipadas e perguntas independentes. A interface daqui usa lista
  + id e não pretende compatibilidade binária com o SDK oficial.
- [Jev / System One](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
  — arquitetura, amostrador paralelo e RLCD são descritos pelo fabricante;
  os ganhos publicados não são benchmarks deste projeto nem evidência de
  pesos disponíveis.
- [Confiança](https://docs.typesafe.ai/confidence) — no Jev é estatística
  derivada da distribuição, e `noul` não inclui confiança. O Specter estende
  esse contrato e define o evento de acerto explicitamente.
- [Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html) —
  temperature scaling como método pós-treino; não garante calibração em
  qualquer domínio.
- [OpenAPI 3.1.1](https://spec.openapis.org/oas/v3.1.1.html) — contrato do
  transporte.
