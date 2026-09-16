"""Backends de referência.

`system_one` é um baseline heurístico local, determinístico e sem pesos
treinados — serve para medir latência real e comparar contra regras.

`jev` é o adaptador para a API System One da TypeSafe, para quando o
operador tem credencial e quer um modelo treinado por trás do mesmo
contrato tipado. Importe-o explicitamente
(`from specter_decision.backends.jev import JevRemoteBackend`): ele não é
carregado por padrão, para que o pacote continue sem qualquer caminho de
rede ativo por omissão.
"""

from .system_one import SystemOneCalibration, SystemOneLocalBackend, tokenize

__all__ = ["SystemOneCalibration", "SystemOneLocalBackend", "tokenize"]
