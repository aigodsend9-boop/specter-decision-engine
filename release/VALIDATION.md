# Validação — 16/09/2026 — 0.2.0rc1

- VERIFICADO: 25 testes passaram. Inferência e calibração nesses testes são fixtures sintéticas, não modelos reais.
- VERIFICADO: testes de autenticação, contrato JSON Schema, SDK HTTP, limites, timeout, erros, isolamento, semáforo, no-calibration e handler real do Core em servidor loopback temporário.
- VERIFICADO: Core modificado apenas por 1 import e 3 guards (7 linhas adicionadas), comparado ao conteúdo original preservado em `rollback/specter_core_v3.py.before`.
- VERIFICADO: sete arquivos do pacote instalado em `C:/Specter/Core/specter_decision` têm SHA-256 igual aos arquivos fonte.
- VERIFICADO: Core continua sintaticamente válido; aviso pré-existente sobre escape `\\S` no HTML legado não foi alterado fora do escopo.
- VERIFICADO: wheel construído offline, sem dependências de runtime, instalado em diretório isolado e importado em processo Python isolado; versão 0.2.0rc1, serviço sem configuração retorna 503.
- VERIFICADO: wheel contém somente os sete arquivos do pacote e metadados de distribuição; não inclui backup do Core, testes, credenciais ou fixtures.
- SHA-256 do wheel: `777f8468e26f29ee16a49a7fe914b81eafc587bdde827621ebdc234e977f704c`.
- NÃO EXECUTADO: reinício do Core, instalação de modelos, treinamento/calibração, publicação em registro, abertura de porta/firewall, configuração de proxy ou divulgação externa.
- VERIFICADO: o processo ativo na porta 8888 retornou 404 para `/v1/decision/capabilities`; os novos hooks ainda não foram carregados nesse processo.
- PENDENTE: seleção do backend e dados, calibração independente, testes de carga reais, revisão do transporte público e decisão de licença/publicação.

Não considerar a configuração do backend, sucesso de import ou saúde HTTP como prova de inteligência, calibração ou prontidão produtiva.
