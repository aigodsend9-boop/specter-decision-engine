"""Import de compatibilidade com o protótipo 0.1/0.2.

Mantido para que `from engine import Question, SpecterDecisionEngine`
continue funcionando. Código novo deve importar de `specter_decision`.
"""

from specter_decision import (  # noqa: F401
    Backend,
    BatchBackend,
    Calibration,
    Decision,
    ErrorCode,
    Question,
    SpecterDecisionEngine,
)

__all__ = [
    "Backend",
    "BatchBackend",
    "Calibration",
    "Decision",
    "ErrorCode",
    "Question",
    "SpecterDecisionEngine",
]
