"""Calibração: temperatura, mapa isotônico, métricas e artefato versionado.

O repositório 0.2 definia o *protocolo* de calibração mas não entregava
como ajustá-la. Aqui está a maquinaria completa, em Python puro:

* `TemperatureScaler` — ajusta T > 0 minimizando NLL (busca por seção
  áurea em log T, convexa no caso multiclasse com rótulo fixo).
* `IsotonicCalibrator` — regressão isotônica por PAV (Pool Adjacent
  Violators), o mapa monótono padrão para transformar uma estatística
  bruta em probabilidade empírica de acerto.
* métricas — NLL, Brier, ECE por bins, tabela de confiabilidade e
  intervalo por bootstrap determinístico.
* `CalibrationArtifact` — vincula o ajuste a (domínio, model_id,
  assinaturas de pergunta) e serializa para JSON.

Nada disso certifica um modelo: calibração é propriedade estatística no
domínio medido. O artefato recusa domínio/modelo/pergunta fora do
registro, e o kernel devolve UNCALIBRATED quando a recusa acontece.
"""

from __future__ import annotations

import bisect
import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .canonical import fingerprint_of
from .numerics import argmax, clamp, expectation, softmax
from .types import Question

__all__ = [
    "CalibrationArtifact",
    "IsotonicCalibrator",
    "SimpleCalibration",
    "TemperatureScaler",
    "accuracy_event",
    "brier_score",
    "bootstrap_interval",
    "confidence_statistic",
    "expected_calibration_error",
    "negative_log_likelihood",
    "reliability_table",
]

_EPSILON = 1e-12
_ARTIFACT_VERSION = "specter-calibration/1"


# --------------------------------------------------------------- estatísticas
def confidence_statistic(
    question: Question, probabilities: Sequence[float], value: Any
) -> float:
    """Estatística bruta de confiança, por tipo de pergunta.

    Definições (iguais às da tabela do desenho original):
        noul: P(classificação pelo limiar 0.5 correta) = max(p).
        choice: P(opção escolhida correta) = p do argmax.
        score: P(|nota - verdade| <= tolerance) = massa dos níveis dentro
            da tolerância em torno da esperança.

    Args:
        question: pergunta avaliada.
        probabilities: distribuição já calibrada em temperatura.
        value: valor projetado (probabilidade, rótulo ou nota).

    Returns:
        Número em [0,1]. Não é confiança calibrada — é a entrada do mapa.
    """
    kind = question.type
    if kind == "score":
        center = float(value) if isinstance(value, (int, float)) else expectation(probabilities)
        tolerance = question.tolerance
        total = math.fsum(
            probability
            for index, probability in enumerate(probabilities)
            if abs(index - center) <= tolerance + 1e-12
        )
        return clamp(total, 0.0, 1.0)
    return clamp(max(probabilities), 0.0, 1.0)


def accuracy_event(
    question: Question, probabilities: Sequence[float], truth: Any
) -> int:
    """Indicador 0/1 do evento de acerto correspondente ao tipo.

    Args:
        question: pergunta avaliada.
        probabilities: distribuição calibrada.
        truth: índice verdadeiro (noul/choice) ou nível verdadeiro (score).

    Returns:
        1 se o evento ocorreu, 0 caso contrário.
    """
    if question.type == "score":
        predicted = expectation(probabilities)
        return int(abs(predicted - float(truth)) <= question.tolerance + 1e-12)
    return int(argmax(probabilities) == int(truth))


# ------------------------------------------------------------------ métricas
def negative_log_likelihood(
    probability_rows: Sequence[Sequence[float]], truths: Sequence[int]
) -> float:
    """NLL média do rótulo verdadeiro (menor é melhor)."""
    if not probability_rows:
        return float("nan")
    total = math.fsum(
        -math.log(max(row[int(truth)], _EPSILON))
        for row, truth in zip(probability_rows, truths)
    )
    return total / len(probability_rows)


def brier_score(
    probability_rows: Sequence[Sequence[float]], truths: Sequence[int]
) -> float:
    """Brier multiclasse (soma dos quadrados dos resíduos), menor é melhor."""
    if not probability_rows:
        return float("nan")
    total = 0.0
    for row, truth in zip(probability_rows, truths):
        total += math.fsum(
            (probability - (1.0 if index == int(truth) else 0.0)) ** 2
            for index, probability in enumerate(row)
        )
    return total / len(probability_rows)


def reliability_table(
    confidences: Sequence[float], outcomes: Sequence[int], bins: int = 10
) -> list[dict[str, float]]:
    """Agrupa (confiança, acerto) em bins de largura igual.

    Args:
        confidences: valores em [0,1].
        outcomes: 1 acerto / 0 erro, mesmo comprimento.
        bins: número de faixas.

    Returns:
        Lista de dicionários com bin, n, confiança média e acurácia.
    """
    buckets: list[list[float]] = [[0.0, 0.0, 0.0] for _ in range(bins)]
    for confidence, outcome in zip(confidences, outcomes):
        index = min(bins - 1, max(0, int(confidence * bins)))
        buckets[index][0] += 1.0
        buckets[index][1] += confidence
        buckets[index][2] += float(outcome)
    table: list[dict[str, float]] = []
    for index, (count, confidence_sum, hits) in enumerate(buckets):
        if count == 0:
            continue
        table.append(
            {
                "bin": index,
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "n": count,
                "confidence": confidence_sum / count,
                "accuracy": hits / count,
            }
        )
    return table


def expected_calibration_error(
    confidences: Sequence[float], outcomes: Sequence[int], bins: int = 10
) -> float:
    """ECE ponderado pelo tamanho dos bins.

    ECE isolado não basta para aceitar um artefato: use junto de Brier,
    NLL e da tabela de confiabilidade com incerteza.
    """
    total = len(confidences)
    if total == 0:
        return float("nan")
    return math.fsum(
        row["n"] / total * abs(row["accuracy"] - row["confidence"])
        for row in reliability_table(confidences, outcomes, bins)
    )


def bootstrap_interval(
    values: Sequence[float],
    *,
    samples: int = 500,
    level: float = 0.95,
    seed: int = 20260916,
) -> tuple[float, float]:
    """Intervalo percentil por bootstrap com semente fixa (reprodutível).

    Args:
        values: amostra (por exemplo, indicadores de acerto).
        samples: número de reamostragens.
        level: nível de confiança em (0,1).
        seed: semente do gerador.

    Returns:
        Par (limite inferior, limite superior).
    """
    size = len(values)
    if size == 0:
        return (float("nan"), float("nan"))
    generator = random.Random(seed)
    means: list[float] = []
    for _ in range(samples):
        total = 0.0
        for _ in range(size):
            total += values[generator.randrange(size)]
        means.append(total / size)
    means.sort()
    tail = (1.0 - level) / 2.0
    lower = means[min(size - 1, int(tail * samples))] if samples else float("nan")
    upper = means[min(samples - 1, int((1.0 - tail) * samples))]
    return (lower, upper)


# ------------------------------------------------------------- ajuste de T
class TemperatureScaler:
    """Ajuste de temperatura por minimização de NLL (Guo et al., 2017)."""

    __slots__ = ("temperature", "nll_before", "nll_after", "iterations")

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = float(temperature)
        self.nll_before = float("nan")
        self.nll_after = float("nan")
        self.iterations = 0

    def fit(
        self,
        logit_rows: Sequence[Sequence[float]],
        truths: Sequence[int],
        *,
        low: float = 0.05,
        high: float = 20.0,
        tolerance: float = 1e-4,
    ) -> "TemperatureScaler":
        """Busca T em [low, high] minimizando a NLL no conjunto fornecido.

        Usa seção áurea em log T: robusta, sem gradiente e com número de
        iterações previsível (~35 avaliações para tolerância 1e-4).
        """
        if not logit_rows:
            raise ValueError("conjunto de calibração vazio")
        if len(logit_rows) != len(truths):
            raise ValueError("logits e rótulos com tamanhos diferentes")

        def objective(temperature: float) -> float:
            rows = [softmax(row, temperature) for row in logit_rows]
            return negative_log_likelihood(rows, truths)

        self.nll_before = objective(1.0)
        left, right = math.log(low), math.log(high)
        phi = (math.sqrt(5.0) - 1.0) / 2.0
        c = right - phi * (right - left)
        d = left + phi * (right - left)
        fc, fd = objective(math.exp(c)), objective(math.exp(d))
        iterations = 0
        while abs(right - left) > tolerance and iterations < 200:
            iterations += 1
            if fc < fd:
                right, d, fd = d, c, fc
                c = right - phi * (right - left)
                fc = objective(math.exp(c))
            else:
                left, c, fc = c, d, fd
                d = left + phi * (right - left)
                fd = objective(math.exp(d))
        self.iterations = iterations
        self.temperature = math.exp((left + right) / 2.0)
        self.nll_after = objective(self.temperature)
        if self.nll_after > self.nll_before:  # nunca piorar
            self.temperature = 1.0
            self.nll_after = self.nll_before
        return self


# ------------------------------------------------------------- isotônica
@dataclass(slots=True)
class IsotonicCalibrator:
    """Mapa monótono não decrescente ajustado por PAV.

    Attributes:
        knots_x: fronteiras crescentes dos blocos.
        knots_y: valor ajustado de cada bloco, em [0,1].
    """

    knots_x: list[float] = field(default_factory=list)
    knots_y: list[float] = field(default_factory=list)

    def fit(
        self, scores: Sequence[float], outcomes: Sequence[int]
    ) -> "IsotonicCalibrator":
        """Ajusta o mapa estatística -> probabilidade de acerto.

        Args:
            scores: estatística bruta (out-of-fold, nunca do treino).
            outcomes: indicador 0/1 do evento de acerto.

        Returns:
            self, com os nós preenchidos.
        """
        if not scores:
            raise ValueError("conjunto isotônico vazio")
        if len(scores) != len(outcomes):
            raise ValueError("scores e outcomes com tamanhos diferentes")
        pairs = sorted(zip((float(s) for s in scores), (float(o) for o in outcomes)))
        blocks: list[list[float]] = []  # [soma_y, peso, x_final]
        for x, y in pairs:
            blocks.append([y, 1.0, x])
            while len(blocks) >= 2 and (
                blocks[-2][0] / blocks[-2][1] >= blocks[-1][0] / blocks[-1][1]
            ):
                last = blocks.pop()
                previous = blocks[-1]
                previous[0] += last[0]
                previous[1] += last[1]
                previous[2] = last[2]
        self.knots_x = [block[2] for block in blocks]
        self.knots_y = [clamp(block[0] / block[1], 0.0, 1.0) for block in blocks]
        return self

    def predict(self, score: float) -> float:
        """Aplica o mapa; extrapolação é constante nas pontas."""
        if not self.knots_x:
            return clamp(float(score), 0.0, 1.0)
        index = bisect.bisect_left(self.knots_x, float(score))
        if index >= len(self.knots_y):
            return self.knots_y[-1]
        return self.knots_y[index]

    def to_dict(self) -> dict[str, list[float]]:
        """Serializa os nós."""
        return {"x": list(self.knots_x), "y": list(self.knots_y)}

    @classmethod
    def from_dict(cls, payload: dict[str, Sequence[float]]) -> "IsotonicCalibrator":
        """Reconstrói a partir de `to_dict`."""
        return cls(
            knots_x=[float(v) for v in payload.get("x", [])],
            knots_y=[float(v) for v in payload.get("y", [])],
        )


# ------------------------------------------------------------- calibrações
class SimpleCalibration:
    """Calibração mínima e honesta: temperatura fixa + teto de confiança.

    Serve para desenvolvimento e para backends heurísticos, onde afirmar
    confiança alta seria desonesto. Implementa o protocolo `Calibration`.
    """

    __slots__ = ("_temperature", "_domain", "_model_id", "_ceiling", "_mapping")

    def __init__(
        self,
        temperature: float = 1.0,
        *,
        domain: str | None = None,
        model_id: str | None = None,
        ceiling: float = 1.0,
        mapping: IsotonicCalibrator | None = None,
    ) -> None:
        if temperature <= 0 or not math.isfinite(temperature):
            raise ValueError("temperatura deve ser finita e > 0")
        if not 0.0 < ceiling <= 1.0:
            raise ValueError("ceiling deve estar em (0,1]")
        self._temperature = float(temperature)
        self._domain = domain
        self._model_id = model_id
        self._ceiling = float(ceiling)
        self._mapping = mapping

    def supports(self, domain: Any = None, model_id: Any = None, question: Any = None) -> bool:
        """Aceita a requisição se domínio e modelo baterem com o registro."""
        if self._domain is not None and domain is not None and domain != self._domain:
            return False
        if self._model_id is not None and model_id is not None and model_id != self._model_id:
            return False
        return True

    def temperature(self, question: Any = None) -> float:
        """Temperatura única para todas as perguntas."""
        return self._temperature

    def confidence(
        self, question: Question, probabilities: Sequence[float], value: Any
    ) -> float:
        """Estatística do evento de acerto, mapeada e limitada pelo teto."""
        raw = confidence_statistic(question, probabilities, value)
        if self._mapping is not None:
            raw = self._mapping.predict(raw)
        return clamp(raw * self._ceiling, 0.0, 1.0)


@dataclass(slots=True)
class CalibrationArtifact:
    """Artefato versionado e vinculado a (domínio, modelo, perguntas).

    Attributes:
        domain: domínio lógico coberto.
        model_id: identificador exato do backend calibrado.
        temperature_value: T ajustado.
        mapping: mapa isotônico opcional para a confiança.
        signatures: assinaturas de pergunta cobertas; vazio = qualquer uma
            dentro do domínio (registrar isso é decisão do operador).
        metrics: métricas medidas fora do treino.
        dataset_hash: impressão digital do conjunto usado.
        created_at: epoch UTC do ajuste.
        notes: anotações livres do operador.
    """

    domain: str
    model_id: str
    temperature_value: float = 1.0
    mapping: IsotonicCalibrator | None = None
    signatures: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    dataset_hash: str = ""
    created_at: float = field(default_factory=lambda: time.time())
    notes: str = ""
    version: str = _ARTIFACT_VERSION

    # -- protocolo Calibration ------------------------------------------
    def supports(self, domain: Any = None, model_id: Any = None, question: Any = None) -> bool:
        """Recusa domínio, modelo ou pergunta fora do registro."""
        if domain is not None and domain != self.domain:
            return False
        if model_id is not None and model_id != self.model_id:
            return False
        if self.signatures and isinstance(question, Question):
            return question.signature in self.signatures
        return True

    def temperature(self, question: Any = None) -> float:
        """T ajustado no conjunto de calibração."""
        return self.temperature_value

    def confidence(
        self, question: Question, probabilities: Sequence[float], value: Any
    ) -> float:
        """Confiança calibrada: estatística bruta -> mapa isotônico."""
        raw = confidence_statistic(question, probabilities, value)
        if self.mapping is not None:
            raw = self.mapping.predict(raw)
        return clamp(raw, 0.0, 1.0)

    # -- persistência ----------------------------------------------------
    def to_json(self, *, indent: int | None = 2) -> str:
        """Serializa o artefato completo em JSON."""
        return json.dumps(
            {
                "version": self.version,
                "domain": self.domain,
                "model_id": self.model_id,
                "temperature": self.temperature_value,
                "mapping": self.mapping.to_dict() if self.mapping else None,
                "signatures": list(self.signatures),
                "metrics": self.metrics,
                "dataset_hash": self.dataset_hash,
                "created_at": self.created_at,
                "notes": self.notes,
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
        )

    @classmethod
    def from_json(cls, payload: str) -> "CalibrationArtifact":
        """Reconstrói a partir de `to_json`, validando a versão."""
        data = json.loads(payload)
        if data.get("version") != _ARTIFACT_VERSION:
            raise ValueError(f"versão de artefato incompatível: {data.get('version')!r}")
        mapping = data.get("mapping")
        return cls(
            domain=data["domain"],
            model_id=data["model_id"],
            temperature_value=float(data.get("temperature", 1.0)),
            mapping=IsotonicCalibrator.from_dict(mapping) if mapping else None,
            signatures=tuple(data.get("signatures", ())),
            metrics=data.get("metrics", {}),
            dataset_hash=data.get("dataset_hash", ""),
            created_at=float(data.get("created_at", 0.0)),
            notes=data.get("notes", ""),
        )

    # -- ajuste ----------------------------------------------------------
    @classmethod
    def fit(
        cls,
        records: Iterable[tuple[Question, Sequence[float], int]],
        *,
        domain: str,
        model_id: str,
        holdout: float = 0.3,
        bins: int = 10,
        bind_signatures: bool = False,
        notes: str = "",
    ) -> "CalibrationArtifact":
        """Ajusta temperatura e mapa de confiança a partir de rótulos reais.

        O conjunto é dividido de forma determinística: os primeiros
        (1 - holdout) registros ajustam, o restante mede. Separação por
        entidade e por tempo é responsabilidade de quem monta a lista —
        o artefato registra o hash do conjunto para auditoria.

        Args:
            records: triplas (pergunta, logits do backend, índice/nível
                verdadeiro).
            domain: domínio lógico.
            model_id: identificador do backend.
            holdout: fração final reservada para medição.
            bins: bins do ECE.
            bind_signatures: vincula o artefato às assinaturas vistas.
            notes: anotação livre.

        Returns:
            Artefato pronto para uso pelo kernel.

        Raises:
            ValueError: conjunto vazio ou holdout fora de (0,1).
        """
        items = list(records)
        if not items:
            raise ValueError("nenhum registro para calibrar")
        if not 0.0 < holdout < 1.0:
            raise ValueError("holdout deve estar em (0,1)")
        split = max(1, int(len(items) * (1.0 - holdout)))
        train, test = items[:split], items[split:] or items[-1:]

        scaler = TemperatureScaler().fit(
            [row for _, row, _ in train], [truth for _, _, truth in train]
        )
        temperature = scaler.temperature

        statistics: list[float] = []
        outcomes: list[int] = []
        for question, row, truth in train:
            probabilities = softmax(row, temperature)
            value = _project_value(question, probabilities)
            statistics.append(confidence_statistic(question, probabilities, value))
            outcomes.append(accuracy_event(question, probabilities, truth))
        mapping = IsotonicCalibrator().fit(statistics, outcomes)

        eval_rows: list[Sequence[float]] = []
        eval_truths: list[int] = []
        eval_confidences: list[float] = []
        eval_outcomes: list[int] = []
        for question, row, truth in test:
            probabilities = softmax(row, temperature)
            value = _project_value(question, probabilities)
            eval_rows.append(probabilities)
            eval_truths.append(int(truth))
            eval_confidences.append(
                mapping.predict(confidence_statistic(question, probabilities, value))
            )
            eval_outcomes.append(accuracy_event(question, probabilities, truth))

        low, high = bootstrap_interval([float(o) for o in eval_outcomes])
        metrics = {
            "n_train": len(train),
            "n_test": len(test),
            "nll_before": scaler.nll_before,
            "nll_after": scaler.nll_after,
            "brier": brier_score(eval_rows, eval_truths),
            "ece": expected_calibration_error(eval_confidences, eval_outcomes, bins),
            "accuracy": (
                math.fsum(eval_outcomes) / len(eval_outcomes) if eval_outcomes else float("nan")
            ),
            "accuracy_ci95": [low, high],
            "reliability": reliability_table(eval_confidences, eval_outcomes, bins),
        }
        digest = fingerprint_of(
            json.dumps(
                [[q.signature, list(map(float, row)), int(truth)] for q, row, truth in items],
                sort_keys=True,
                separators=(",", ":"),
            ),
            prefix="specter-dataset/1",
        )
        signatures = tuple(sorted({q.signature for q, _, _ in items})) if bind_signatures else ()
        return cls(
            domain=domain,
            model_id=model_id,
            temperature_value=temperature,
            mapping=mapping,
            signatures=signatures,
            metrics=metrics,
            dataset_hash=digest,
            notes=notes,
        )


def _project_value(question: Question, probabilities: Sequence[float]) -> Any:
    """Reprojeta o valor tipado a partir da distribuição (uso interno)."""
    if question.type == "noul":
        return probabilities[1]
    if question.type == "choice":
        return question.labels[argmax(probabilities)]
    return expectation(probabilities)
