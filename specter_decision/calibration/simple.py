"""Usable calibration artifacts for Specter.

Honest scope:
- temperature() is a fixed or per-signature scalar (not learned from held-out yet)
- confidence() maps peak probability / score residual into [0,1]
- supports() gates by model_id + domain so UNCALIBRATED stays meaningful

This is *operational* calibration so the engine runs. It is not a substitute for
held-out domain validation required before production claims.
"""
from __future__ import annotations

import math
from typing import Mapping


class SimpleCalibration:
    """Accept any signature for a pinned model_id + domain."""

    def __init__(
        self,
        model_id: str,
        domain: str,
        *,
        temperature: float = 1.0,
        confidence_floor: float = 0.05,
    ):
        if not model_id or not domain:
            raise ValueError("model_id and domain required")
        if not (0 < temperature < 100):
            raise ValueError("invalid temperature")
        self._model_id = model_id
        self._domain = domain
        self._temperature = float(temperature)
        self._floor = float(confidence_floor)

    def supports(self, model_id: str, domain: str, question_signature: str) -> bool:
        return model_id == self._model_id and domain == self._domain

    def temperature(self, question_signature: str) -> float:
        return self._temperature

    def confidence(self, question, probabilities: tuple[float, ...], value) -> float:
        if not probabilities:
            return self._floor
        peak = max(probabilities)
        if question.type == "noul":
            # Distance from 0.5 — extreme nouls are more "confident"
            p_yes = probabilities[1] if len(probabilities) > 1 else peak
            conf = abs(p_yes - 0.5) * 2.0
        elif question.type == "choice":
            # Gap between top-1 and top-2
            ordered = sorted(probabilities, reverse=True)
            gap = ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)
            conf = 0.5 * peak + 0.5 * gap
        else:
            # score: confidence falls when mass is spread or residual to nearest int is high
            residual = abs(float(value) - round(float(value))) if isinstance(value, (int, float)) else 0.5
            conf = peak * (1.0 - 0.5 * residual)
        return max(self._floor, min(1.0, conf))


class DomainCalibration:
    """Per-signature temperature overrides (e.g. loaded from a JSON artifact)."""

    def __init__(
        self,
        model_id: str,
        domain: str,
        *,
        default_temperature: float = 1.0,
        temperatures: Mapping[str, float] | None = None,
        confidence_floor: float = 0.05,
    ):
        self._model_id = model_id
        self._domain = domain
        self._default_t = float(default_temperature)
        self._temps = dict(temperatures or {})
        self._floor = float(confidence_floor)

    def supports(self, model_id: str, domain: str, question_signature: str) -> bool:
        return model_id == self._model_id and domain == self._domain

    def temperature(self, question_signature: str) -> float:
        t = self._temps.get(question_signature, self._default_t)
        return float(t) if t > 0 else self._default_t

    def confidence(self, question, probabilities: tuple[float, ...], value) -> float:
        # Reuse SimpleCalibration logic without duplicating state
        return SimpleCalibration(
            self._model_id, self._domain, confidence_floor=self._floor
        ).confidence(question, probabilities, value)
