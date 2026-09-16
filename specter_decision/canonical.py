"""Snapshot canônico do `state`: validação estrita e impressão digital estável.

O kernel serializa o estado UMA vez por requisição e entrega exatamente a
mesma string a todas as perguntas. Isso garante três propriedades:

1. Isolamento: nenhuma pergunta influencia o contexto de outra.
2. Reprodutibilidade: mesma entrada -> mesmos bytes -> mesma fingerprint.
3. Cache: a fingerprint é a chave natural de prefill/feature cache.

Validação é feita por travessia iterativa (sem recursão) para que um estado
profundo maliciosamente construído não estoure a pilha do interpretador.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Final

__all__ = ["Snapshot", "SNAPSHOT_FORMAT", "canonical_snapshot", "fingerprint_of"]

SNAPSHOT_FORMAT: Final[str] = "specter-canonical-json/1"
_DIGEST_SIZE: Final[int] = 16
_DEFAULT_MAX_BYTES: Final[int] = 256 * 1024
_DEFAULT_MAX_DEPTH: Final[int] = 32
_DEFAULT_MAX_NODES: Final[int] = 200_000


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Estado imutável pronto para inferência.

    Attributes:
        json: JSON canônico (chaves ordenadas, sem espaços, sem NaN).
        fingerprint: blake2b-128 hexadecimal dos bytes canônicos.
        nbytes: tamanho em bytes UTF-8.
        nodes: quantidade de nós percorridos na validação.
    """

    json: str
    fingerprint: str
    nbytes: int
    nodes: int

    def __len__(self) -> int:
        return self.nbytes


def fingerprint_of(payload: str | bytes, *, prefix: str = "") -> str:
    """Impressão digital blake2b-128 estável entre processos e plataformas.

    Args:
        payload: texto ou bytes a resumir.
        prefix: domínio de separação (evita colisão entre usos distintos).

    Returns:
        Hexadecimal de 32 caracteres.
    """
    data = payload.encode("utf-8") if isinstance(payload, str) else payload
    digest = hashlib.blake2b(data, digest_size=_DIGEST_SIZE, person=b"specter-fp")
    if prefix:
        digest.update(prefix.encode("utf-8"))
    return digest.hexdigest()


def _reject(reason: str) -> None:
    raise ValueError(f"INVALID_INPUT: {reason}")


def _validate(state: Any, max_depth: int, max_nodes: int) -> int:
    """Percorre o estado rejeitando tipos e valores não serializáveis.

    Returns:
        Número de nós visitados.

    Raises:
        ValueError: com prefixo INVALID_INPUT em qualquer violação.
    """
    stack: list[tuple[Any, int]] = [(state, 0)]
    seen: set[int] = set()
    nodes = 0
    while stack:
        node, depth = stack.pop()
        nodes += 1
        if nodes > max_nodes:
            _reject(f"estado com mais de {max_nodes} nós")
        if depth > max_depth:
            _reject(f"profundidade do estado acima de {max_depth}")
        if node is None or isinstance(node, (str, bool)):
            continue
        if isinstance(node, int):
            continue
        if isinstance(node, float):
            if not math.isfinite(node):
                _reject("NaN/Infinity não são representáveis em JSON")
            continue
        if isinstance(node, (dict, list, tuple)):
            identity = id(node)
            if identity in seen:
                _reject("referência circular no estado")
            seen.add(identity)
            if isinstance(node, dict):
                for key, value in node.items():
                    if not isinstance(key, str):
                        _reject("chaves do estado devem ser strings")
                    stack.append((value, depth + 1))
            else:
                for value in node:
                    stack.append((value, depth + 1))
            continue
        _reject(f"tipo não serializável no estado: {type(node).__name__}")
    return nodes


def canonical_snapshot(
    state: Any,
    *,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    max_depth: int = _DEFAULT_MAX_DEPTH,
    max_nodes: int = _DEFAULT_MAX_NODES,
) -> Snapshot:
    """Valida e serializa o estado em forma canônica.

    Args:
        state: qualquer estrutura JSON (str, número, bool, None, list, dict).
        max_bytes: limite de tamanho do snapshot serializado.
        max_depth: profundidade máxima de aninhamento.
        max_nodes: número máximo de nós.

    Returns:
        Snapshot imutável com JSON canônico e fingerprint.

    Raises:
        ValueError: prefixada com INVALID_INPUT quando o estado é inválido.
    """
    nodes = _validate(state, max_depth, max_nodes)
    try:
        text = json.dumps(
            state,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:  # pragma: no cover - defesa dupla
        _reject(f"estado não serializável ({exc})")
    encoded = text.encode("utf-8")
    if len(encoded) > max_bytes:
        _reject(f"estado com {len(encoded)} bytes acima do limite de {max_bytes}")
    return Snapshot(
        json=text,
        fingerprint=fingerprint_of(encoded, prefix=SNAPSHOT_FORMAT),
        nbytes=len(encoded),
        nodes=nodes,
    )
