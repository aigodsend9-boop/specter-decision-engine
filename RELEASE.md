# Specter Decision Engine 0.2.0rc1

Release candidate para integração; **não é lançamento de um modelo treinado**.

## Instalado no Core, sem ativar inferência

- Pacote: `C:/Specter/Core/specter_decision/`.
- Hooks: `specter_core_v3.py`, interceptação de GET/POST/OPTIONS antes do leitor de corpo legado.
- Nenhum banco, túnel, executor, worker, firewall ou serviço reiniciado/alterado.
- O processo Core já em execução não recarrega Python automaticamente. Instalação em disco não prova ativação no processo.
- Ausência de `SPECTER_DECISION_TOKEN` → HTTP 503 `NOT_CONFIGURED`. Token válido sem factory → 503 `NOT_READY` na avaliação.
- A configuração padrão não carrega modelos, não cria threads, não faz rede nem escreve logs de conteúdo.

## Interface universal

| Método | Caminho | Uso |
|---|---|---|
| POST | `/v1/decision/evaluate` | state + questions + domain → decisões tipadas |
| GET | `/v1/decision/capabilities` | capacidades/configuração, sem alegar prontidão produtiva |
| GET | `/v1/decision/openapi.json` | contrato OpenAPI 3.1.1 para clientes/ferramentas |

Todas exigem Bearer token. Não há CORS aberto para estas rotas. Métodos documentados e erros de aplicação retornam JSON; erros de protocolo HTTP anteriores ao roteamento dependem do servidor legado.

SDK remoto, síncrono:

```python
import os
from specter_decision import Question
from specter_decision.client import DecisionClient

client = DecisionClient("http://127.0.0.1:8888", os.environ["SPECTER_DECISION_TOKEN"])
result = client.decide(
    {"ticket": "Exportação indisponível"},
    [Question("route", "choice", "Qual equipe?", ("billing", "engineering"))],
    domain="support-v1",
)
# Checar status do envelope E de cada decisão antes de consumir valores.
```

Clientes remotos exigem HTTPS; redirects são recusados, não há retries automáticos. O cliente valida labels, ordem, simplex, nota e formato da resposta. Token pertence ao operador e nunca é colocado no state.

Aplicações/agentes podem gerar bindings a partir de `specter_decision/openapi.json` ou usar HTTP direto. Não foram instalados conectores em serviços externos; OpenAPI não equivale a adesão automática de todo ecossistema. MCP/provider-specific tools são integrações futuras, sem ferramentas de shell nesta API.

## Ativação pelo operador, após validação do modelo

Na inicialização confiável do Core, antes de aceitar tráfego:

```python
from specter_decision import SpecterDecisionEngine
from specter_decision.http import configure
from specter_decision.service import DecisionService

def factory(domain):
    # Registro local allowlist; nunca importar caminho/modulo vindo da requisição.
    backend, calibration = approved_registry.lookup(domain)
    return SpecterDecisionEngine(backend, calibration, domain=domain,
                                 concurrency=8, timeout=2.0)

configure(DecisionService(token=operator_secret, engine_factory=factory, max_requests=2))
```

`approved_registry`, backend, calibration e operator_secret são dependências reais que o operador deve fornecer, não fixtures entregues como modelo. Uma instância nova de engine por requisição evita compartilhar semáforos entre event loops. Pesos somente-leitura podem ser compartilhados por adaptadores thread-safe.

## Limites e segurança operacional

- Até 256 KiB por corpo completo, 128 perguntas, quatro leitores de corpo e dois requests de inferência por serviço (defaults). Sob carga → 429, sem fila ilimitada.
- Deadline absoluto de leitura de corpo: 5s. Timeout de perguntas inclui espera. O backend deve cooperar com cancelamento; adaptadores bloqueantes precisam isolamento próprio de processo.
- JSON duplicado, NaN/Infinity, campos desconhecidos, tipo inválido, corpo comprimido/chunked e cardinalidade inválida são rejeitados.
- Sem geração textual. Labels vêm da entrada; erros são enums. `status=ok` no envelope significa request processado, não que todas as perguntas deram decisão válida.
- Sem artefato de calibração compatível → decisão `UNCALIBRATED`, valores nulos. Não usar as fixtures em produção.
- Decisão não concede autorização. O engine não executa ações, não importa código recebido e não envia mensagens ao feed.
- O Core legado contém outras superfícies fora desta revisão. **Não publicar o Core inteiro para lançar esta API.** Para acesso externo, usar proxy TLS com allowlist apenas das rotas de decisão, autenticação/rate limits e proteção de conexões/headers no proxy. Isso ainda não foi implantado.

## Gates obrigatórios antes de produção

1. Modelo numérico com licença/dataset definidos e avaliação semântica real.
2. Calibração fora do treino; artefato vinculado ao hash da pergunta, modelo e domínio; Brier/NLL/ECE com incerteza, drift e política de abstenção.
3. Teste de carga no hardware real: p50/p95, erro, memória, custo e concorrência. Resultados com fixtures não medem inteligência ou latência do modelo.
4. Revisão do backend e do transporte TLS/segredos; rotação do token e isolamento entre clientes conforme necessidade.
5. Ativação em janela de manutenção; smoke test autenticado; rollback se regressão. Sem reinício automático nesta entrega.
6. Definir licença de distribuição e destino de publicação. Pacote preparado localmente, não publicado em PyPI/GitHub.

`production_ready` permanece false nesta release candidate, mesmo com factory: sua presença não certifica estes gates. Não prometemos perfeição estrutural, confiança universal ou desempenho equivalente ao Jev.

## Testes e rollback

Runtime sem dependências externas. Testes de schema usam `jsonschema` (dependência apenas de desenvolvimento). Rodar `python -m unittest -v` no projeto. Teste do hook extrai somente a classe do Core via AST: não importa Core, não inicia workers nem acessa seus bancos.

Backup pré-integração em `release/rollback/specter_core_v3.py.before`. Rollback cirúrgico: retirar o import `dispatch_decision` e os três guards correspondentes; preservar edições posteriores de terceiros. Não sobrescrever todo Core com backup se houve alterações concorrentes. Pacote pode permanecer inerte em disco; reiniciar apenas em janela aprovada.

Fontes: [OpenAPI 3.1.1](https://spec.openapis.org/oas/v3.1.1.html). Pesquisa Jev e desenho do MVP em `README.md`.
