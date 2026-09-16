"""Specter Decision Engine — decisões tipadas, não texto.

Kernel numérico para avaliação de decisões limitadas e independentes
dentro de software e agentes. Sem geração de texto, sem rede no caminho
quente, sem dependências externas em tempo de execução.

Uso mínimo::

    from specter_decision import Question, SpecterDecisionEngine
    from specter_decision import SystemOneLocalBackend, SystemOneCalibration

    engine = SpecterDecisionEngine(
        SystemOneLocalBackend(),
        SystemOneCalibration(domain="support-v1"),
        domain="support-v1",
        concurrency=8,
        timeout=2.0,
    )
    answers = await engine.decide(state, questions)

O backend local é um baseline heurístico auditável: ele NÃO é um modelo
treinado e não substitui calibração no seu domínio.
"""

from __future__ import annotations

from .calibration import (
    CalibrationArtifact,
    IsotonicCalibrator,
    SimpleCalibration,
    TemperatureScaler,
    accuracy_event,
    brier_score,
    confidence_statistic,
    expected_calibration_error,
    negative_log_likelihood,
    reliability_table,
)
from .canonical import Snapshot, canonical_snapshot, fingerprint_of
from .engine import SpecterDecisionEngine
from .policy import ConfidenceGate, Route, composite_score, is_ok, rank_then_choose, require
from .protocols import Backend, BatchBackend, Calibration
from .types import DEFAULT_LIMITS, Decision, ErrorCode, Limits, Question
from .backends.system_one import SystemOneCalibration, SystemOneLocalBackend

__version__ = "0.3.0"
__contract__ = "specter-decision/0.3"

__all__ = [
    "Backend",
    "BatchBackend",
    "Calibration",
    "CalibrationArtifact",
    "ConfidenceGate",
    "DEFAULT_LIMITS",
    "Decision",
    "ErrorCode",
    "IsotonicCalibrator",
    "Limits",
    "Question",
    "Route",
    "SimpleCalibration",
    "Snapshot",
    "SpecterDecisionEngine",
    "SystemOneCalibration",
    "SystemOneLocalBackend",
    "TemperatureScaler",
    "__version__",
    "accuracy_event",
    "brier_score",
    "canonical_snapshot",
    "composite_score",
    "confidence_statistic",
    "expected_calibration_error",
    "fingerprint_of",
    "is_ok",
    "negative_log_likelihood",
    "rank_then_choose",
    "reliability_table",
    "require",
]
