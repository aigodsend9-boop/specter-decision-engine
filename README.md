# Specter Decision Engine — desenho original MVP 0.1

Autor do projeto: **Guilherme Peralta Novaes**. Desenvolvido com assistência de IA.

Atualização: release candidate 0.2.0rc1 integrada em disco ao Core. Ver `RELEASE.md` para API, SDK remoto, limites, ativação e gates de produção. Este documento preserva o desenho inicial; o módulo `engine.py` agora é um import de compatibilidade para `specter_decision.engine`.

Status: núcleo Python executável; modelo semântico, calibração real e integração de produção PENDENTES. Sem geração de texto, rede, instalação, daemon ou mudanças no Core ativo. Não é reprodução dos pesos/treinamento proprietário do Jev.

## 1. SDK

Python 3.11+, biblioteca padrão. `engine.py`: `SpecterDecisionEngine`, `Question`, protocolos `Backend` e `Calibration`.

```python
from engine import Question, SpecterDecisionEngine

# Objetos fornecidos pela aplicação; não vêm implementados como modelo neste MVP.
engine = SpecterDecisionEngine(
    backend=trained_numeric_backend,
    calibration=validated_calibration,
    domain="support-v1", concurrency=8, timeout=2.0,
)
answers = await engine.decide(
    state={"ticket": "Exportação falha no Safari; funciona no Chrome."},
    questions=[
        Question("workaround", "noul", "Existe alternativa operacional?"),
        Question("route", "choice", "Qual equipe deve analisar?",
                 ("billing", "engineering", "support")),
        Question("severity", "score", "Qual a severidade?",
                 ("cosmético", "falha com alternativa", "bloqueio sem alternativa"),
                 tolerance=0.5),
    ],
)
route = answers[1]
if route["status"] == "ok" and route["confidence"] >= approved_threshold:
    selected_queue = route["value"]  # Decisão não autoriza nem executa ação.
```

Execução local reproduzível: `python -m unittest -v`. As fixtures de `test_engine.py` são exclusivamente sintéticas; não usar suas probabilidades/confiança em produção.

Resposta de sucesso, ilustrativa (não inferência real):

```json
{"id":"route","type":"choice","status":"ok","value":"engineering","probabilities":[0.05,0.85,0.10],"confidence":0.81,"legend":["billing","engineering","support"],"error":null}
```

| Tipo | value | probabilities / legend | Evento calibrado para confidence |
|---|---|---|---|
| noul | P(sim), [0,1] | [P(não), P(sim)] / [false,true] | classificação pelo limiar 0.5 correta |
| choice | opção argmax; empate resolve pela primeira | ordem das opções de entrada | opção escolhida correta |
| score | soma(i × P(i)), [0,K−1] | ordem da legenda de entrada | erro absoluto da nota ≤ tolerance |

`confidence` é P(evento definido acima), não entropia nem declaração do modelo. Noul probabilístico não representa uma probabilidade de segunda ordem.

Erros por pergunta mantêm os campos: `status=error`, `value/probabilities/confidence=null`, `error` enum (`UNCALIBRATED`, `TIMEOUT`, `BACKEND_ERROR`, `INVALID_OUTPUT`). Não inventar uma decisão para preencher falhas. Requisições inválidas são rejeitadas com `ValueError('INVALID_INPUT')` antes da inferência; um futuro transporte HTTP deve mapear para erro JSON tipado, não HTTP 200. Cancelamento pelo chamador propaga.

## 2. Arquitetura

```text
state → snapshot JSON imutável → validação → parallel sampler (limite/deadline)
                                              ├─ state + pergunta A → logits A
                                              ├─ state + pergunta B → logits B
                                              └─ state + pergunta C → logits C
          logits → calibração versionada → projeção tipada → validação → SDK
```

- Implementado: chamadas async isoladas, mesmo snapshot, IDs não enviados ao modelo, concorrência global por instância/event loop, deadline incluindo espera, resultado na ordem original. Sem contexto compartilhado de respostas, retries ou logs de state.
- Não confundir concorrência de chamadas com paralelismo GPU: o backend deve ser cooperativo e não bloquear o event loop. CPU exige worker/executor; batches tensoriais exigem backend próprio. Adaptador não confiável precisa isolamento de processo: o contrato sozinho não impede estado interno compartilhado.
- Modelo alvo: encoder discriminativo compacto + scorer de pares `(state, pergunta, candidato)`; candidato sim/não para Noul, opções para Choice, níveis para Score. Cabeças numéricas, nenhuma cabeça autoregressiva. Treinar com rótulos do domínio, não presumir generalização de perguntas arbitrárias.
- Produção: codificar state uma vez, attention das perguntas para state, sem attention entre perguntas. Microbatch com máscara de candidatos, segment-softmax por pergunta e índices para recompor respostas. Nunca softmax global. Não prometer latência constante quando o batch excede capacidade.
- Sampler aqui significa projeção probabilística independente, não múltiplas amostras textuais. Argmax/esperança determinísticos reduzem custo e variância; amostragem estocástica seria opção futura separada.
- Formato: construir objetos no código a partir de números finitos e labels permitidos; não pedir JSON ao modelo. Validar cardinalidade, simplex, limites, discriminação do tipo. Garantia de formato não é garantia de acerto semântico, disponibilidade ou ausência de falhas de software.
- Limites atuais: até 128 perguntas, 255 opções Choice, 10 níveis Score, state JSON de até 256 KiB. Sem porta pública nem comandos executáveis.

## 3. Calibração e aceitação

1. Separar treino / ajuste de calibração / teste final por entidade e tempo, sem vazamento de duplicatas.
2. Ajustar temperatura positiva minimizando NLL em calibração. Avaliar Brier, NLL, reliability diagrams e ECE com intervalos de confiança; ECE isolado não basta.
3. Para confidence, ajustar mapa de probabilidade de acerto (por exemplo isotônico) com predições out-of-fold e eventos da tabela; Score exige rótulos numéricos e tolerance fixada. Não chamar max(probabilities) ou 1−entropia de confiança empiricamente calibrada.
4. Registrar modelo/tokenizador, domínio, hash da pergunta/legenda, método, dataset e métricas. O protocolo `Calibration.supports` deve conferir esse registro; o kernel não autentica a veracidade de um artefato fornecido pela aplicação.
5. Novo modelo/rubrica/domínio invalida artefato. Drift ou cobertura insuficiente → abstenção/revisão. Calibração é propriedade estatística no domínio medido, não garantia universal.

Sem artefato compatível, o SDK retorna `UNCALIBRATED` sem sequer chamar o modelo. **Nenhum calibrador real foi treinado nesta entrega.**

## 4. MVP implementável agora

| Etapa | Entrega / gate |
|---|---|
| 0 — entregue | kernel + exemplos + testes sintéticos; validar distribuição, erros, isolamento e concorrência |
| 1 | selecionar dataset rotulado do domínio e encoder com licença adequada; treinar scorer numérico; medir baseline de regras |
| 2 | ajustar artefatos de calibração e aceitar somente após teste independente; limiares definidos pelo risco e cobertura |
| 3 | backend em batch, cache limitado de features por hash/versão/tenant; comparar batch 1/8/32, p50/p95, decisões/s, RAM e custo por 1k decisões |
| 4 | importar SDK no fluxo Specter; inicialmente shadow/read-only; persistir apenas recibos autorizados, nunca gerar mensagens automáticas de decisão no feed |

Critérios: sucesso sempre tipado; falhas nunca viram decisões; pergunta isolada = mesma pergunta no batch (tolerância numérica declarada); validade estatística documentada; comparar custo/latência sob mesma carga. Testes locais sintéticos não certificam desempenho do modelo. Sem metas numéricas de latência antes de medir no hardware real.

## 5. Atualização Specter consultada em 16/09/2026

- ID Antigravity `60ca1d19-eaed-4970-bcd5-3f5b988ef578` não é thread Codex legível. Foram lidos seus artefatos locais, não alegado acesso integral à transcrição.
- `implementation_plan.md` e `walkthrough.md` (15/09): console PowerShell, histórico TXT, remoção de pulses/polling. Os 97 testes e túnel nesses documentos são relatos anteriores, não testes reexecutados aqui.
- Código atual lido: `specter_terminal.py` contém SessionLogger/TerminalHub; `specter_core_v3.py` só chama telemetria interativa em `/intel` e `/status`; inicialização não inicia `system_intel_background_loop`. Outros loops continuam presentes.
- Artefato `scratch/sync_core_modules.py` de 16/09 parametriza caminhos de distribuição por `SPECTER_ROOT`, `SPECTER_GATEWAY_URL`, `SPECTER_CORE_DIR`, `SPECTER_WORK_DIR`. Script apenas lido, não executado; não comprova que a distribuição foi atualizada.
- Integração proposta: biblioteca interna chamada sob demanda. Não ampliar o executor/federação nem usar mensagens externas como autorização. Core, banco, serviços e túneis não foram alterados.

## Fontes primárias

- [TypeSafe: visão e primitives](https://docs.typesafe.ai/introduction): decisões tipadas e perguntas independentes. Nossa interface usa lista + id, não pretende compatibilidade binária com o SDK oficial.
- [Jev/System One](https://typesafe.ai/blog/introducing-system-one-models-and-jev): arquitetura/sampler/RLCD são descritos pelo fabricante; os ganhos publicados não são benchmarks do Specter nem evidência de pesos disponíveis.
- [Score](https://docs.typesafe.ai/primitives/score): legenda ordinal e média ponderada.
- [Confidence](https://docs.typesafe.ai/confidence): no Jev, estatística derivada da distribuição; Noul não inclui confidence. Specter estende esse contrato conforme o pedido e define evento de acerto explícito.
- [Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html): temperature scaling como método pós-treino; não garante calibração em qualquer domínio.
