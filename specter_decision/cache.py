"""Cache LRU limitado, usado para prefill de estado e sketches de rótulo.

Requisitos do projeto: teto de memória previsível, zero dependências e
custo O(1) por operação. `OrderedDict.move_to_end` dá exatamente isso.

O cache é deliberadamente *não* thread-safe por padrão: o kernel é async e
roda em um único event loop. Backends que usem executores devem criar uma
instância por worker ou proteger o acesso externamente.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

__all__ = ["CacheStats", "LRUCache"]

K = TypeVar("K")
V = TypeVar("V")


@dataclass(slots=True)
class CacheStats:
    """Contadores de uso do cache."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        """Taxa de acerto em [0,1]; 0.0 quando não houve consulta."""
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def as_dict(self) -> dict[str, float]:
        """Snapshot serializável dos contadores."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": round(self.hit_rate, 6),
        }


class LRUCache(Generic[K, V]):
    """Mapa com capacidade máxima e descarte do menos usado recentemente."""

    __slots__ = ("_data", "_capacity", "stats")

    def __init__(self, capacity: int = 128) -> None:
        """Inicializa o cache.

        Args:
            capacity: número máximo de entradas (>= 1).
        """
        if capacity < 1:
            raise ValueError("capacidade deve ser >= 1")
        self._data: OrderedDict[K, V] = OrderedDict()
        self._capacity = capacity
        self.stats = CacheStats()

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def get(self, key: K) -> V | None:
        """Retorna o valor e o promove a mais recente, ou None."""
        item = self._data.get(key)
        if item is None:
            self.stats.misses += 1
            return None
        self._data.move_to_end(key)
        self.stats.hits += 1
        return item

    def put(self, key: K, value: V) -> V:
        """Insere/atualiza uma entrada, descartando a mais antiga se preciso."""
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        while len(self._data) > self._capacity:
            self._data.popitem(last=False)
            self.stats.evictions += 1
        return value

    def get_or_compute(self, key: K, factory: Callable[[], V]) -> V:
        """Retorna o valor existente ou calcula, armazena e devolve."""
        found = self.get(key)
        if found is not None:
            return found
        return self.put(key, factory())

    def clear(self) -> None:
        """Esvazia o cache preservando os contadores."""
        self._data.clear()
