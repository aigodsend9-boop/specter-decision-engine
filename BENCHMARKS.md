# Medições — 0.3.0

Todas as medições abaixo foram **executadas**, não estimadas. Reproduza com:

```bash
python tools/bench.py --iterations 300 --json docs/benchmark.json
```

**Hardware desta execução:** Linux x86_64 (glibc 2.39), 1 vCPU, 4 GiB RAM,
CPython 3.12.3. Estado de 551 bytes. Um servidor real dá números melhores;
1 vCPU é deliberadamente o pior caso.

**O que estes números NÃO são:** medida de inteligência, de qualidade
semântica ou de latência de modelo treinado. O backend aqui é heurístico ou
nulo. Eles medem **engenharia**: quanto o kernel custa e quanto as
otimizações estruturais devolvem.

## 1. Custo do kernel (backend de custo zero)

Isola snapshot + validação + agendamento + softmax + projeção.

| Perguntas | p50 | p95 | p99 | decisões/s |
|---|---|---|---|---|
| 1 | 0,029 ms | 0,049 ms | 0,065 ms | 31.235 |
| 8 | 0,071 ms | 0,093 ms | 0,106 ms | 107.934 |
| 32 | 0,171 ms | 0,216 ms | 0,295 ms | 179.963 |
| 128 | 0,550 ms | 0,601 ms | 0,679 ms | 231.293 |

O overhead por decisão cai de ~29 µs (pergunta isolada) para ~4,3 µs
(lote de 128): o custo fixo do snapshot é amortizado, exatamente o
comportamento esperado do desenho.

## 2. Lote vs fan-out (mesmo backend, mesma carga)

`batch` usa `logits_batch` (um prefill do estado). `fanout` esconde o
protocolo de lote e força uma chamada por pergunta.

| Perguntas | p50 lote | p50 fan-out | ganho |
|---|---|---|---|
| 1 | 0,041 ms | 0,061 ms | 1,52× |
| 8 | 0,093 ms | 0,132 ms | 1,42× |
| 32 | 0,191 ms | 0,230 ms | 1,20× |
| 128 | 0,577 ms | 0,615 ms | 1,07× |

O ganho aqui é modesto **porque o backend local é barato**: o prefill custa
quase nada, então sobra pouco para economizar. Em um backend real — onde
codificar o estado domina o custo — a economia é proporcional ao tamanho do
estado, não ao número de perguntas. A conta que importa: com fan-out, um
estado de 4 KB é codificado 128 vezes; com lote, uma.

## 3. Deduplicação por assinatura

64 perguntas em fan-out especulativo, mesmo estado.

| Cenário | p50 | decisões/s |
|---|---|---|
| 64 perguntas **idênticas** | 0,242 ms | 258.194 |
| 64 perguntas distintas | 0,525 ms | 121.154 |

**2,17× mais rápido** quando o conteúdo se repete, com resultado idêntico —
o backend é chamado uma vez e a projeção acontece 64 vezes. Nenhuma
aplicação precisa saber disso para se beneficiar.

## 4. Memória

| Métrica | Valor |
|---|---|
| Alocação rastreada, em regime | 29,1 KiB |
| Pico rastreado (50 × lote de 128) | 129,0 KiB |
| RSS máximo do processo | 25,4 MiB |

O RSS é quase todo interpretador CPython. O kernel em si opera na casa das
dezenas de KiB, bem dentro do teto de 50 MB por módulo adotado no projeto
Specter. Os caches são limitados por construção (LRU com capacidade fixa),
então não há crescimento não limitado com o tempo.

## 5. Suíte de testes

126 testes, 100% verdes, ~0,52 s no mesmo hardware:

| Arquivo | Testes | Cobre |
|---|---|---|
| `test_contract.py` | 21 | contrato do kernel (superconjunto da suíte 0.2) |
| `test_contract_types.py` | 17 | snapshot canônico e validação de perguntas |
| `test_calibration.py` | 25 | temperatura, isotônica, métricas, artefato |
| `test_system_one.py` | 16 | backend local, prefill, cache, semântica |
| `test_policy.py` | 15 | gate, score composto, alta cardinalidade |
| `test_service_http.py` | 20 | serviço, rotas, autenticação, cliente |
| `test_numerics.py` | 12 | estabilidade numérica e simplex |

## 6. O que ainda não foi medido

- Qualidade semântica contra rótulos humanos em domínio real.
- Latência e custo com backend treinado (local ou remoto).
- Comportamento sob carga concorrente sustentada com backend lento.
- Consumo com estados próximos do teto de 256 KiB em produção.

Nenhuma meta numérica deve ser prometida antes dessas medições no hardware
e no domínio reais.
