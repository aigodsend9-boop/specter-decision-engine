"""Contrato tipado: perguntas, decisões, limites e códigos de erro.

`Question` é imutável, validada na construção e carrega uma assinatura
estável (`signature`) usada para três coisas distintas:

* deduplicação de trabalho dentro de uma mesma requisição;
* chave de cache de features/prefill no backend;
* vínculo criptográfico com o artefato de calibração.

Nada aqui depende de bibliotecas externas.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Final, Literal, Mapping, Sequence, TypedDict

from .canonical import fingerprint_of

__all__ = [
    "Decision",
    "ErrorCode",
    "Limits",
    "Question",
    "QuestionType",
    "DEFAULT_LIMITS",
    "NOUL_LEGEND",
]

QuestionType = Literal["noul", "choice", "score"]

NOUL_LEGEND: Final[tuple[str, str]] = ("false", "true")
_QUESTION_TYPES: Final[frozenset[str]] = frozenset({"noul", "choice", "score"})
_SIGNATURE_DOMAIN: Final[str] = "specter-question/2"


class ErrorCode:
    """Enum fechado de erros por pergunta (strings estáveis na API)."""

    UNCALIBRATED: Final[str] = "UNCALIBRATED"
    TIMEOUT: Final[str] = "TIMEOUT"
    BACKEND_ERROR: Final[str] = "BACKEND_ERROR"
    INVALID_OUTPUT: Final[str] = "INVALID_OUTPUT"
    ABSTAINED: Final[str] = "ABSTAINED"

    ALL: Final[frozenset[str]] = frozenset(
        {"UNCALIBRATED", "TIMEOUT", "BACKEND_ERROR", "INVALID_OUTPUT", "ABSTAINED"}
    )


@dataclass(frozen=True, slots=True)
class Limits:
    """Limites duros aplicados antes de qualquer inferência.

    Attributes:
        max_questions: perguntas por requisição.
        max_choice_options: cardinalidade de Choice (paridade com Jev: 255).
        max_score_levels: níveis de Score.
        max_state_bytes: tamanho do snapshot canônico.
        max_state_depth: profundidade do estado.
        max_text_chars: tamanho do enunciado.
        max_label_chars: tamanho de cada rótulo/critério.
        max_id_chars: tamanho do identificador da pergunta.
    """

    max_questions: int = 128
    max_choice_options: int = 255
    max_score_levels: int = 10
    max_state_bytes: int = 256 * 1024
    max_state_depth: int = 32
    max_text_chars: int = 4096
    max_label_chars: int = 512
    max_id_chars: int = 64


DEFAULT_LIMITS: Final[Limits] = Limits()


class Decision(TypedDict):
    """Resposta tipada de uma pergunta.

    `status` é "ok" ou "error". Em erro, value/probabilities/confidence são
    None e `error` carrega um código de `ErrorCode`. Nunca há texto gerado.
    """

    id: str
    type: str
    status: str
    value: Any
    probabilities: list[float] | None
    confidence: float | None
    legend: list[str] | None
    error: str | None


def _reject(reason: str) -> None:
    raise ValueError(f"INVALID_INPUT: {reason}")


def _check_text(value: Any, what: str, limit: int) -> str:
    if not isinstance(value, str):
        _reject(f"{what} deve ser string")
    text = value.strip()
    if not text:
        _reject(f"{what} vazio")
    if len(text) > limit:
        _reject(f"{what} com {len(text)} caracteres acima de {limit}")
    return text


@dataclass(frozen=True, slots=True)
class Question:
    """Pergunta independente e tipada.

    Args:
        id: identificador escolhido pela aplicação. Nunca é enviado ao
            modelo (o kernel substitui por "_" antes da inferência).
        type: "noul", "choice" ou "score".
        text: enunciado / instruções.
        labels: opções (choice) ou níveis ordinais (score). Para noul é
            derivado automaticamente como ("false", "true").
        tolerance: erro absoluto aceito em Score para o evento de acerto.
        criteria: descrições por rótulo, alinhadas a `labels`. Aceita
            mapeamento {rótulo: descrição} ou sequência na mesma ordem.
    """

    id: str
    type: str
    text: str
    labels: tuple[str, ...] = ()
    tolerance: float = 0.5
    criteria: tuple[str, ...] = ()
    signature: str = field(default="", compare=True)

    def __post_init__(self) -> None:
        limits = DEFAULT_LIMITS
        object.__setattr__(self, "id", _check_text(self.id, "id", limits.max_id_chars))
        if self.type not in _QUESTION_TYPES:
            _reject(f"tipo desconhecido: {self.type!r}")
        object.__setattr__(
            self, "text", _check_text(self.text, "enunciado", limits.max_text_chars)
        )
        object.__setattr__(self, "labels", self._normalize_labels(limits))
        object.__setattr__(self, "criteria", self._normalize_criteria(limits))
        if not isinstance(self.tolerance, (int, float)) or isinstance(self.tolerance, bool):
            _reject("tolerance deve ser numérico")
        if not math.isfinite(float(self.tolerance)) or float(self.tolerance) < 0.0:
            _reject("tolerance deve ser finito e >= 0")
        object.__setattr__(self, "tolerance", float(self.tolerance))
        if not self.signature:
            object.__setattr__(self, "signature", self._compute_signature())

    def _normalize_labels(self, limits: Limits) -> tuple[str, ...]:
        raw = self.labels
        if raw is None:
            raw = ()
        if isinstance(raw, str):
            _reject("labels deve ser uma sequência, não string")
        if isinstance(raw, Mapping):
            raw = tuple(raw.keys())
        labels = tuple(
            _check_text(item, "label", limits.max_label_chars) for item in tuple(raw)
        )
        if self.type == "noul":
            if labels and labels != NOUL_LEGEND:
                if len(labels) != 2:
                    _reject("noul aceita exatamente 2 rótulos")
            return labels or NOUL_LEGEND
        if self.type == "choice":
            if not 2 <= len(labels) <= limits.max_choice_options:
                _reject(f"choice exige 2..{limits.max_choice_options} opções")
        else:  # score
            if not 2 <= len(labels) <= limits.max_score_levels:
                _reject(f"score exige 2..{limits.max_score_levels} níveis")
        if len(set(labels)) != len(labels):
            _reject("rótulos duplicados")
        return labels

    def _normalize_criteria(self, limits: Limits) -> tuple[str, ...]:
        raw = self.criteria
        if not raw:
            return ()
        if isinstance(raw, Mapping):
            ordered = [raw.get(label) or "" for label in self.labels]
        else:
            ordered = list(raw)
        if len(ordered) != len(self.labels):
            _reject("criteria deve ter o mesmo tamanho de labels")
        cleaned: list[str] = []
        for item in ordered:
            if item is None:
                cleaned.append("")
                continue
            if not isinstance(item, str):
                _reject("criteria deve conter strings ou None")
            if len(item) > limits.max_label_chars:
                _reject("descrição de critério longa demais")
            cleaned.append(item.strip())
        return tuple(cleaned)

    def _compute_signature(self) -> str:
        payload = json.dumps(
            {
                "v": _SIGNATURE_DOMAIN,
                "type": self.type,
                "text": self.text,
                "labels": list(self.labels),
                "criteria": list(self.criteria),
                "tolerance": round(self.tolerance, 12) if self.type == "score" else None,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return fingerprint_of(payload, prefix=_SIGNATURE_DOMAIN)

    @property
    def cardinality(self) -> int:
        """Número de saídas possíveis (2 para noul)."""
        return len(self.labels)

    def anonymous(self) -> "Question":
        """Cópia com id neutro, usada para chamar o backend.

        O identificador escolhido pela aplicação carrega semântica de negócio
        e não deve influenciar a inferência; o kernel sempre envia "_".
        """
        if self.id == "_":
            return self
        return Question(
            id="_",
            type=self.type,
            text=self.text,
            labels=self.labels,
            tolerance=self.tolerance,
            criteria=self.criteria,
            signature=self.signature,
        )

    def describe(self, index: int) -> str:
        """Texto descritivo de um rótulo (rótulo + critério, se houver)."""
        label = self.labels[index]
        if index < len(self.criteria) and self.criteria[index]:
            return f"{label}: {self.criteria[index]}"
        return label

    # --- construtores no estilo Jev -------------------------------------
    @classmethod
    def noul(
        cls, id: str, instructions: str, criteria: Mapping[str, str] | None = None
    ) -> "Question":
        """Pergunta sim/não. `criteria` aceita {"true": ..., "false": ...}."""
        pairs = ()
        if criteria:
            pairs = (str(criteria.get("false", "")), str(criteria.get("true", "")))
        return cls(id, "noul", instructions, NOUL_LEGEND, criteria=pairs)

    @classmethod
    def choice(
        cls,
        id: str,
        instructions: str,
        criteria: Mapping[str, str | None] | Sequence[str],
    ) -> "Question":
        """Escolha de uma opção. Aceita mapa {opção: rubrica} ou lista."""
        if isinstance(criteria, Mapping):
            labels = tuple(criteria.keys())
            descriptions = tuple((criteria[key] or "") for key in labels)
            return cls(id, "choice", instructions, labels, criteria=descriptions)
        return cls(id, "choice", instructions, tuple(criteria))

    @classmethod
    def score(
        cls,
        id: str,
        instructions: str,
        criteria: Sequence[str],
        tolerance: float = 0.5,
    ) -> "Question":
        """Nota ordinal sobre níveis descritivos e ordenados."""
        return cls(id, "score", instructions, tuple(criteria), tolerance=tolerance)
