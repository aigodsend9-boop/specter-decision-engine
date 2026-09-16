"""Specter Decision Engine: bounded numeric decisions, never action execution."""
from .engine import Backend, Calibration, Decision, Question, SpecterDecisionEngine

__version__ = '0.2.0rc2'
__all__ = ['Backend', 'Calibration', 'Decision', 'Question', 'SpecterDecisionEngine']
