"""Specter Decision Engine: bounded numeric decisions, never action execution."""
from .engine import Backend, Calibration, Decision, Question, SpecterDecisionEngine
from .backends import (
    LocalLexicalBackend,
    SystemOneCalibration,
    SystemOneLocalBackend,
    TypeSafeJevBackend,
)
from .calibration import DomainCalibration, SimpleCalibration

__version__ = "0.2.0rc5"
__author__ = "Guilherme Peralta Novaes"
__all__ = [
    "Backend",
    "Calibration",
    "Decision",
    "Question",
    "SpecterDecisionEngine",
    "LocalLexicalBackend",
    "SystemOneLocalBackend",
    "SystemOneCalibration",
    "TypeSafeJevBackend",
    "SimpleCalibration",
    "DomainCalibration",
]
