# Changelog

Formato: mudanças agrupadas por impacto. Datas em UTC.

## 0.3.0 — 2026-09-16

### Estrutural
- **`BatchBackend`**: protocolo opcional `logits_batch(state_json, questions)`.
  Quando presente, o estado é codificado uma vez para N perguntas. Detecção
  em tempo de execução; backends 0.2 continuam funcionando sem alteração.
- **Deduplicação por assinatura** de pergunta dentro da requisição: conteúdo
  idêntico é computado uma vez (2,17× medido em 64 perguntas repetidas).
- **Admissão consciente do deadline**: trabalho já vencido devolve `TIMEOUT`
  sem chamar o backend.
- **Gate de calibração por pergunta** em vez de tudo-ou-nada.
- **Semáforo por (instância, event loop)**, recriado quando o loop muda.
- **Cancelamento** do chamador cancela o trabalho em voo e propaga, com os
  `finally` do backend garantidamente executados.

### Calibração
- `TemperatureScaler`: ajuste de T por minimização de NLL (seção áurea em
  log T), com garantia de nunca piorar a NLL de partida.
- `IsotonicCalibrator`: regressão isotônica por PAV, serializável.
- Métricas: NLL, Brier multiclasse, ECE por bins, tabela de confiabilidade,
  intervalo por bootstrap com semente fixa.
- `CalibrationArtifact`: versionado, com vínculo a domínio, `model_id` e
  (opcionalmente) assinaturas de pergunta; `to_json`/`from_json` com recusa
  de versão incompatível; `fit()` com holdout determinístico e hash do
  conjunto.

### Contrato
- `Question` agora aceita `criteria` (rubrica por rótulo), no estilo da API
  System One, e expõe `signature`, `cardinality`, `anonymous()`, `describe()`.
- Construtores `Question.noul/choice/score`.
- Novo código de erro `ABSTAINED`, emitido apenas com `abstain_below` ativo.
- `receipts=True` anexa hash auditável que vincula decisão ao estado sem
  expor o conteúdo do estado.
- `capabilities()` no engine e no serviço.

### Numérico
- Soma do simplex corrigida para **exatamente** 1.0 em ponto flutuante.
- Softmax com saturação defensiva: logits de magnitude extrema não geram
  `inf`/`NaN`; underflow total degenera para uniforme em vez de explodir.
- `math.fsum` em toda soma acumulativa (esperança, normalização, métricas).

### Snapshot
- Validação **iterativa** (sem recursão): estado profundo não estoura a
  pilha. Detecção de ciclo, limite de nós, profundidade e bytes.
- JSON canônico com chaves ordenadas: mesma estrutura → mesma fingerprint,
  independentemente da ordem de inserção.

### Backend local
- `SystemOneLocalBackend` v2: prefill do estado cacheado por fingerprint,
  sketch de rótulo cacheado por assinatura, glossário de domínio
  normalizado pelo mesmo tokenizador (PT/EN), singularização de plurais
  portugueses (`-ções` → `-ção`), sinal ordinal de intensidade para `score`.
- `stats()` com contadores e eficiência de cache.

### Novo
- `backends/jev.py`: adaptador para a API System One da TypeSafe. Converte a
  distribuição do provedor em log-probabilidades, preservando os números
  originais e permitindo recalibrar temperatura localmente.
- `policy.py`: `ConfidenceGate`, `composite_score`, `rank_then_choose`
  (cardinalidade acima de 255 em dois estágios), `require`, `is_ok`.
- `cli.py`: `evaluate`, `capabilities`, `selftest`, `bench`, `calibrate`.
- `tools/bench.py`: banco de medição reprodutível.
- Suíte de 126 testes (era 8 no kernel 0.2) e `BENCHMARKS.md` com números
  medidos neste hardware.

### Mudança incompatível
- Pergunta inválida é rejeitada **na construção** de `Question`, não na
  avaliação. Mesma exceção (`ValueError`) e mesmo prefixo (`INVALID_INPUT`).
  Ver `MIGRATION.md`.

### Sem mudança
- Formato do `Decision`, nomes dos códigos de erro existentes, semântica dos
  três tipos, limites (128/255/10/256 KiB), rotas HTTP e postura de
  segurança. `production_ready` continua `false`.

## 0.2.0rc5 — anterior
Kernel com fan-out por pergunta, protocolos `Backend`/`Calibration`,
serviço HTTP com token, OpenAPI 3.1.1, backend local heurístico v1.

## 0.1 — desenho original
SDK, arquitetura, tabela de tipos e plano de MVP por Guilherme Peralta
Novaes.
