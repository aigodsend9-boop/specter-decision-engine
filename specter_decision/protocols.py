"""Protocolos que a aplicação implementa. O kernel não fornece modelo.

Três contratos:

* `Backend` — uma pergunta por chamada (compatível com 0.1/0.2).
* `BatchBackend` — todas as perguntas em uma chamada, com o estado
  codificado uma única vez. É o caminho que reproduz o "parallel sampler":
  um prefill do state, N cabeças numéricas, softmax por segmento.
* `Calibration` — converte logits em probabilidades honestas e define o
  evento cuja probabilidade é reportada como `confidence`.

O kernel detecta `logits_batch` em tempo de execução; implementar o
protocolo de lote é opcional e não quebra backends antigos.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from .types import Question

__all__ = ["Backend", "BatchBackend", "Calibration"]


@runtime_checkable
class Backend(Protocol):
    """Modelo numérico discriminativo, sem cabeça autoregressiva."""

    model_id: str

    async def logits(self, state_json: str, question: Question) -> Sequence[float]:
        """Retorna um logit por rótulo da pergunta.

        Args:
            state_json: snapshot canônico, idêntico para todas as perguntas.
            question: pergunta anônima (id == "_").

        Returns:
            Sequência de floats finitos com `question.cardinality` elementos.
        """
        ...


@runtime_checkable
class BatchBackend(Protocol):
    """Backend que pontua várias perguntas com um único prefill do estado."""

    model_id: str

    async def logits_batch(
        self, state_json: str, questions: Sequence[Question]
    ) -> Sequence[Sequence[float]]:
        """Retorna uma matriz ragged: uma linha de logits por pergunta.

        A ordem da saída deve espelhar a ordem da entrada. Softmax é sempre
        por segmento (por pergunta) — nunca global sobre o lote.
        """
        ...


@runtime_checkable
class Calibration(Protocol):
    """Artefato estatístico versionado, ajustado fora do treino."""

    def supports(self, domain: str, model_id: str, question: Question) -> bool:
        """Informa se este artefato cobre (domínio, modelo, pergunta)."""
        ...

    def temperature(self, question: Question) -> float:
        """Temperatura positiva aplicada aos logits antes do softmax."""
        ...

    def confidence(
        self, question: Question, probabilities: Sequence[float], value: object
    ) -> float:
        """Probabilidade calibrada do evento de acerto definido por tipo."""
        ...
