"""Núcleo numérico determinístico e livre de dependências.

Todas as funções aqui são puras, sem estado global e sem alocação
desnecessária. Elas são o caminho quente do kernel: qualquer decisão
passa por `softmax` uma vez e por `expectation` no máximo uma vez.

Garantias:
    * nenhuma saída contém NaN/Inf (entradas inválidas levantam ValueError);
    * `softmax` é numericamente estável (subtração do máximo);
    * a soma do simplex retornado é 1.0 dentro de 1 ulp (correção de resíduo
      aplicada no índice de maior massa, que é o mais robusto a erro relativo);
    * desempate é sempre pelo menor índice (documentado e testado).
"""

from __future__ import annotations

import math
from typing import Final, Sequence

__all__ = [
    "SIMPLEX_TOLERANCE",
    "argmax",
    "clamp",
    "entropy_normalized",
    "expectation",
    "is_finite_sequence",
    "margin",
    "renormalize",
    "softmax",
    "top_mass",
    "validate_simplex",
]

SIMPLEX_TOLERANCE: Final[float] = 1e-9
_MAX_EXP_SPAN: Final[float] = 700.0  # exp(709) ainda é finito em float64.


def is_finite_sequence(values: Sequence[float]) -> bool:
    """Retorna True se todos os elementos forem floats finitos.

    Args:
        values: sequência numérica candidata.

    Returns:
        True quando todo elemento é finito e convertível para float.
    """
    for value in values:
        if value is True or value is False:
            return False
        if not isinstance(value, (int, float)):
            return False
        if not math.isfinite(value):
            return False
    return True


def clamp(value: float, low: float, high: float) -> float:
    """Limita `value` ao intervalo fechado [low, high]."""
    if value < low:
        return low
    if value > high:
        return high
    return value


def argmax(values: Sequence[float]) -> int:
    """Índice do maior valor; empate resolve pelo menor índice.

    Args:
        values: sequência não vazia.

    Returns:
        Índice do máximo.

    Raises:
        ValueError: se a sequência for vazia.
    """
    if not values:
        raise ValueError("argmax de sequência vazia")
    best = 0
    best_value = values[0]
    for index in range(1, len(values)):
        current = values[index]
        if current > best_value:
            best = index
            best_value = current
    return best


def softmax(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    """Softmax estável com escalonamento por temperatura.

    Args:
        logits: vetor de logits finitos (tamanho >= 1).
        temperature: temperatura positiva e finita. T < 1 concentra a massa,
            T > 1 achata a distribuição.

    Returns:
        Lista de probabilidades no simplex, com soma 1.0.

    Raises:
        ValueError: logits vazios/não finitos ou temperatura inválida.
    """
    size = len(logits)
    if size == 0:
        raise ValueError("softmax de vetor vazio")
    if not isinstance(temperature, (int, float)) or not math.isfinite(temperature):
        raise ValueError("temperatura não finita")
    if temperature <= 0.0:
        raise ValueError("temperatura deve ser > 0")
    if not is_finite_sequence(logits):
        raise ValueError("logits não finitos")

    inverse = 1.0 / float(temperature)
    scaled = [float(value) * inverse for value in logits]
    highest = max(scaled)
    # Saturação defensiva: mantém exp() no domínio seguro mesmo com
    # logits legítimos de magnitude extrema vindos de um backend externo.
    exponentials = [
        math.exp(value - highest) if value - highest > -_MAX_EXP_SPAN else 0.0
        for value in scaled
    ]
    total = math.fsum(exponentials)
    if total <= 0.0 or not math.isfinite(total):
        # Degenerescência só é possível com underflow total; distribuição
        # uniforme é a única resposta honesta e mantém o contrato do simplex.
        uniform = 1.0 / size
        return [uniform] * size
    inverse_total = 1.0 / total
    probabilities = [value * inverse_total for value in exponentials]
    return renormalize(probabilities)


def renormalize(probabilities: list[float]) -> list[float]:
    """Corrige o resíduo de ponto flutuante para que a soma seja 1.0.

    O resíduo (ordem de 1e-16) é somado ao maior componente, onde o erro
    relativo introduzido é mínimo.

    Args:
        probabilities: lista já aproximadamente normalizada.

    Returns:
        A mesma lista, corrigida in place e devolvida por conveniência.
    """
    residual = 1.0 - math.fsum(probabilities)
    if residual:
        index = argmax(probabilities)
        corrected = probabilities[index] + residual
        probabilities[index] = clamp(corrected, 0.0, 1.0)
    return probabilities


def validate_simplex(
    probabilities: Sequence[float],
    *,
    size: int | None = None,
    tolerance: float = SIMPLEX_TOLERANCE,
) -> bool:
    """Verifica cardinalidade, domínio [0,1] e soma unitária.

    Args:
        probabilities: distribuição candidata.
        size: cardinalidade esperada; None ignora a checagem.
        tolerance: desvio máximo aceito na soma.

    Returns:
        True se a distribuição é válida.
    """
    if size is not None and len(probabilities) != size:
        return False
    if not probabilities or not is_finite_sequence(probabilities):
        return False
    for value in probabilities:
        if value < -tolerance or value > 1.0 + tolerance:
            return False
    return abs(math.fsum(probabilities) - 1.0) <= tolerance


def expectation(probabilities: Sequence[float]) -> float:
    """Valor esperado do índice ordinal: soma(i * p_i).

    Usa `math.fsum` para evitar acúmulo de erro com muitos níveis.

    Args:
        probabilities: distribuição sobre níveis ordinais 0..K-1.

    Returns:
        Esperança em [0, K-1].
    """
    return math.fsum(index * value for index, value in enumerate(probabilities))


def entropy_normalized(probabilities: Sequence[float]) -> float:
    """Entropia de Shannon normalizada em [0,1].

    Args:
        probabilities: distribuição válida.

    Returns:
        0.0 para distribuição degenerada, 1.0 para uniforme.
    """
    size = len(probabilities)
    if size <= 1:
        return 0.0
    total = math.fsum(-value * math.log(value) for value in probabilities if value > 0.0)
    return clamp(total / math.log(size), 0.0, 1.0)


def margin(probabilities: Sequence[float]) -> float:
    """Diferença entre a maior e a segunda maior probabilidade."""
    if len(probabilities) < 2:
        return 1.0
    ordered = sorted(probabilities, reverse=True)
    return clamp(ordered[0] - ordered[1], 0.0, 1.0)


def top_mass(probabilities: Sequence[float]) -> float:
    """Massa do componente dominante."""
    return max(probabilities) if probabilities else 0.0
