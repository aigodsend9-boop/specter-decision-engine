"""Backend local System-One: prefill do estado + cabeças numéricas por pergunta.

**O que este backend é.** Um pontuador léxico determinístico, sem pesos
treinados, sem rede e sem dependências. Ele existe para (a) dar um baseline
de regras mensurável — o passo 1 do plano do MVP —, (b) exercitar o
caminho de lote do kernel e (c) permitir medir latência/memória reais.

**O que este backend NÃO é.** Não é um modelo semântico. Não reproduz Jev.
Não deve ser usado para decisões de produção sem calibração no domínio e
sem comparação contra um backend treinado.

Arquitetura (espelha o desenho de produção):

    state_json --(1x)--> Prefill (tokens, bigramas, sinais)   [cache LRU]
                            |
    pergunta ---(sketch)----+--> evidência por rótulo --> logits

O prefill é calculado uma única vez por fingerprint de estado e reusado
por todas as perguntas — é exatamente o ganho que o amostrador paralelo
explora. O sketch de cada rótulo é memorizado por assinatura de pergunta.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Final, Iterable, Sequence

from ..cache import LRUCache
from ..calibration import SimpleCalibration
from ..canonical import fingerprint_of
from ..types import Question

__all__ = ["Prefill", "SystemOneCalibration", "SystemOneLocalBackend", "tokenize"]

MODEL_ID: Final[str] = "system-one-local/heuristic-v2"

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_STOPWORDS: Final[frozenset[str]] = frozenset(
    """
    a an and are as at be been but by can cant did do does for from had has have he her his
    how i if in is it its me my no not of on or our out so than that the their them then there
    these they this to too us was we were what when where which who why will with would you your
    ao aos as com como da das de dela dele do dos e ela ele eles em entre era essa esse esta este
    eu foi for isso ja la mais mas me mesmo meu na nao nas nem no nos nossa nosso num numa o os ou
    para pela pelo por qual quando que quem se sem ser seu sua tem um uma voce vocês
    """.split()
)

# Expansões de domínio. Uma chave casa com um token do rótulo/enunciado e
# adiciona termos relacionados com peso menor. Curto de propósito: cada
# entrada é auditável e o operador deve ajustá-la ao seu domínio.
_GLOSS: Final[dict[str, tuple[str, ...]]] = {
    "technical": ("bug", "error", "errors", "fail", "fails", "failing", "failed", "failure",
                  "broken", "break", "crash", "integration", "integrate", "api", "connect",
                  "connection", "connecting", "sync", "login", "timeout", "outage", "webhook",
                  "install", "setup", "token", "endpoint", "server", "debug", "stripe",
                  "account", "500", "403", "erro", "falha", "falhando", "conectar", "integracao"),
    "engineering": ("bug", "crash", "regression", "deploy", "code", "stack", "exception", "api"),
    "billing": ("invoice", "invoices", "charge", "charged", "refund", "payment", "payments",
                "subscription", "receipt", "overcharge", "card", "billed", "fatura", "cobranca",
                "reembolso", "pagamento", "assinatura", "boleto"),
    "sales": ("demo", "quote", "pricing", "price", "upgrade", "trial", "purchase", "contract",
              "renewal", "discount", "buy", "vendas", "orcamento", "proposta"),
    "support": ("help", "question", "assist", "issue", "ticket", "ajuda", "duvida", "suporte"),
    "urgency": ("asap", "urgent", "urgently", "immediately", "now", "today", "emergency",
                "critical", "blocked", "blocking", "losing", "lost", "deadline", "waiting",
                "days", "hours", "urgente", "imediato", "agora", "hoje", "parado", "bloqueado"),
    "urgent": ("asap", "urgent", "immediately", "now", "today", "emergency", "critical",
               "blocked", "losing", "deadline", "waiting", "days", "urgente", "agora"),
    "time": ("days", "hours", "minutes", "deadline", "today", "tomorrow", "now", "asap",
             "dias", "horas", "prazo", "hoje", "amanha"),
    "angry": ("angry", "furious", "unacceptable", "ridiculous", "terrible", "worst", "awful",
              "outrageous", "absurdo", "inaceitavel", "pessimo", "revoltado"),
    "frustrated": ("frustrated", "frustrating", "annoyed", "upset", "disappointed", "tired",
                   "again", "still", "keeps", "repeatedly", "frustrado", "cansado", "denovo"),
    "calm": ("hello", "hi", "thanks", "thank", "please", "wondering", "question", "ola",
             "obrigado", "por favor"),
    "positive": ("great", "love", "excellent", "perfect", "awesome", "otimo", "excelente"),
    "negative": ("bad", "wrong", "broken", "fail", "failing", "issue", "problem", "cant",
                 "unable", "ruim", "errado", "problema", "nao"),
    "risk": ("fraud", "chargeback", "breach", "leak", "attack", "abuse", "risco", "fraude"),
    "security": ("password", "token", "leak", "breach", "phishing", "malware", "senha"),
    "spam": ("free", "click", "winner", "offer", "unsubscribe", "promo"),
}

# Marcadores fortes: contam com peso cheio no sinal de intensidade.
_STRONG_CUES: Final[frozenset[str]] = frozenset(
    "asap urgent urgently immediately emergency critical blocked outage down unacceptable "
    "furious refund cancel churn lawsuit urgente imediato inaceitavel cancelar".split()
)

_GLOSS_WEIGHT: Final[float] = 0.6
_LABEL_WEIGHT: Final[float] = 1.0
_CHOICE_GAIN: Final[float] = 4.0
_CHOICE_SATURATION: Final[float] = 1.0
_NOUL_GAIN: Final[float] = 4.0
_NOUL_OFFSET: Final[float] = 0.30
_NOUL_SATURATION: Final[float] = 1.5
_SCORE_LEX_GAIN: Final[float] = 2.5
_SCORE_ORDINAL_GAIN: Final[float] = 2.0


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _singularize(token: str) -> str:
    """Reduz plurais comuns de PT e EN a uma forma canônica.

    Sem a regra de "-ções -> -ção", "integrações" e "integração" viram
    tokens distintos e o casamento léxico simplesmente não acontece — em
    português esse é o erro de recall mais caro do pipeline.
    """
    if len(token) > 4:
        if token.endswith("coes"):
            return token[:-4] + "cao"
        if token.endswith("oes") or token.endswith("aes"):
            return token[:-3] + "ao"
        if token.endswith("ais"):
            return token[:-3] + "al"
        if token.endswith("eis"):
            return token[:-3] + "el"
        if token.endswith("ois"):
            return token[:-3] + "ol"
        if token.endswith("s") and not token.endswith("ss"):
            return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Normaliza e quebra um texto em tokens comparáveis.

    Minúsculas, remoção de acentos, descarte de stopwords PT/EN e plural
    simples removido quando a palavra tem mais de quatro letras.

    Args:
        text: texto livre.

    Returns:
        Lista de tokens na ordem de ocorrência.
    """
    normalized = _strip_accents(text).lower()
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(normalized):
        token = match.group(0)
        if token in _STOPWORDS or len(token) == 1:
            continue
        tokens.append(_singularize(token))
    return tokens


def _normalized_gloss() -> dict[str, tuple[str, ...]]:
    """Normaliza chaves e valores do glossário com o mesmo tokenizador.

    Sem isso, "sales" (chave) e "sale" (token do rótulo já normalizado)
    nunca casariam — a classe de bug mais silenciosa deste backend.
    """
    table: dict[str, tuple[str, ...]] = {}
    for key, values in _GLOSS.items():
        key_tokens = tokenize(key)
        if not key_tokens:
            continue
        expanded: list[str] = []
        for value in values:
            expanded.extend(tokenize(value))
        table[key_tokens[0]] = tuple(dict.fromkeys(expanded))
    return table


_GLOSS_N: Final[dict[str, tuple[str, ...]]] = _normalized_gloss()


def _expand(tokens: Iterable[str]) -> dict[str, float]:
    """Constrói o sketch ponderado de um descritor (rótulo, critério...)."""
    sketch: dict[str, float] = {}
    for token in tokens:
        sketch[token] = max(sketch.get(token, 0.0), _LABEL_WEIGHT)
        for related in _GLOSS_N.get(token, ()):  # expansão de domínio
            sketch.setdefault(related, _GLOSS_WEIGHT)
    return sketch


@dataclass(frozen=True, slots=True)
class Prefill:
    """Codificação reutilizável do estado.

    Attributes:
        fingerprint: hash do `state_json` que gerou este prefill.
        weights: token -> peso sublinear (1 + ln(tf)).
        bigrams: bigramas presentes, para casamento de expressões.
        size: número de tokens úteis.
        intensity: sinal em [0,1] de urgência/severidade percebida.
        strong_cues: quantos marcadores fortes apareceram.
    """

    fingerprint: str
    weights: dict[str, float]
    bigrams: frozenset[str]
    size: int
    intensity: float
    strong_cues: int


class SystemOneLocalBackend:
    """Backend local, determinístico, que implementa `Backend` e `BatchBackend`.

    Attributes:
        model_id: identificador estável, versionado junto com a heurística.
    """

    model_id: str = MODEL_ID

    __slots__ = ("_prefills", "_sketches", "_last_prefill", "calls", "batch_calls")

    def __init__(self, *, prefill_cache: int = 64, sketch_cache: int = 512) -> None:
        """Inicializa caches limitados.

        Args:
            prefill_cache: estados distintos mantidos em memória.
            sketch_cache: perguntas distintas com sketch memorizado.
        """
        self._prefills: LRUCache[str, Prefill] = LRUCache(prefill_cache)
        self._sketches: LRUCache[str, list[dict[str, float]]] = LRUCache(sketch_cache)
        self._last_prefill: Prefill | None = None
        self.calls = 0
        self.batch_calls = 0

    # ------------------------------------------------------------- protocolo
    async def logits(self, state_json: str, question: Question) -> list[float]:
        """Pontua uma pergunta (caminho de compatibilidade)."""
        self.calls += 1
        return self._score(self._prefill(state_json), question)

    async def logits_batch(
        self, state_json: str, questions: Sequence[Question]
    ) -> list[list[float]]:
        """Pontua todas as perguntas com um único prefill do estado."""
        self.batch_calls += 1
        self.calls += len(questions)
        prefill = self._prefill(state_json)
        return [self._score(prefill, question) for question in questions]

    # ---------------------------------------------------------------- prefill
    def _prefill(self, state_json: str) -> Prefill:
        fingerprint = fingerprint_of(state_json, prefix="system-one-prefill/2")
        cached = self._prefills.get(fingerprint)
        if cached is None:
            cached = self._prefills.put(fingerprint, self._encode(state_json, fingerprint))
        self._last_prefill = cached
        return cached

    @staticmethod
    def _encode(state_json: str, fingerprint: str) -> Prefill:
        try:
            payload: Any = json.loads(state_json)
        except ValueError:
            payload = state_json
        chunks: list[tuple[str, float]] = []
        stack: list[tuple[Any, float]] = [(payload, 1.0)]
        while stack:
            node, weight = stack.pop()
            if isinstance(node, str):
                chunks.append((node, weight))
            elif isinstance(node, dict):
                for key, value in node.items():
                    chunks.append((str(key), 0.3))  # nomes de campo são contexto fraco
                    stack.append((value, weight))
            elif isinstance(node, (list, tuple)):
                for value in node:
                    stack.append((value, weight))
            elif isinstance(node, (int, float)) and not isinstance(node, bool):
                chunks.append((str(node), 0.5))

        counts: dict[str, float] = {}
        ordered: list[str] = []
        strong = 0
        for text, weight in chunks:
            tokens = tokenize(text)
            ordered.extend(tokens)
            for token in tokens:
                counts[token] = counts.get(token, 0.0) + weight
                if token in _STRONG_CUES:
                    strong += 1
        weights = {
            token: (1.0 + math.log(count)) if count >= 1.0 else count
            for token, count in counts.items()
            if count > 0
        }
        bigrams = frozenset(
            f"{ordered[index]}_{ordered[index + 1]}" for index in range(len(ordered) - 1)
        )
        intensity = _intensity(weights, strong)
        return Prefill(
            fingerprint=fingerprint,
            weights=weights,
            bigrams=bigrams,
            size=len(ordered),
            intensity=intensity,
            strong_cues=strong,
        )

    # --------------------------------------------------------------- pontuação
    def _sketch(self, question: Question) -> list[dict[str, float]]:
        cached = self._sketches.get(question.signature)
        if cached is not None:
            return cached
        if question.type == "noul":
            tokens = tokenize(question.text)
            if question.criteria and question.criteria[1]:
                tokens += tokenize(question.criteria[1])
            sketches = [_expand(tokens)]
        else:
            sketches = [
                _expand(tokenize(question.describe(index)))
                for index in range(question.cardinality)
            ]
        return self._sketches.put(question.signature, sketches)

    def _score(self, prefill: Prefill, question: Question) -> list[float]:
        sketches = self._sketch(question)
        if question.type == "noul":
            evidence = _evidence(prefill, sketches[0])
            saturated = evidence / (evidence + _NOUL_SATURATION)
            boost = 0.15 if prefill.strong_cues and _shares_cue(sketches[0]) else 0.0
            delta = _NOUL_GAIN * (saturated + boost - _NOUL_OFFSET)
            return [0.0, _clip(delta)]

        scores = [_evidence(prefill, sketch) for sketch in sketches]
        peak = max(scores) if scores else 0.0
        denominator = peak + _CHOICE_SATURATION
        if question.type == "choice":
            return [_clip(_CHOICE_GAIN * (score / denominator)) for score in scores]

        levels = question.cardinality
        logits: list[float] = []
        for index, score in enumerate(scores):
            position = index / (levels - 1)
            ordinal = -((position - prefill.intensity) ** 2) * _SCORE_ORDINAL_GAIN
            logits.append(_clip(_SCORE_LEX_GAIN * (score / denominator) + ordinal))
        return logits

    # ------------------------------------------------------------- diagnóstico
    def stats(self) -> dict[str, Any]:
        """Contadores de uso e eficiência de cache."""
        return {
            "model_id": self.model_id,
            "calls": self.calls,
            "batch_calls": self.batch_calls,
            "prefill_cache": self._prefills.stats.as_dict(),
            "sketch_cache": self._sketches.stats.as_dict(),
            "last_prefill": self._last_prefill.fingerprint if self._last_prefill else None,
        }


def _shares_cue(sketch: dict[str, float]) -> bool:
    """True se o descritor da pergunta fala de urgência/severidade."""
    return any(token in _STRONG_CUES for token in sketch)


def _intensity(weights: dict[str, float], strong: int) -> float:
    """Sinal ordinal em [0,1] combinando pistas fortes e vocabulário negativo."""
    negative = math.fsum(
        weights.get(token, 0.0)
        for token in _GLOSS_N["negative"] + _GLOSS_N["angry"] + _GLOSS_N["frustrated"]
    )
    urgency = math.fsum(weights.get(token, 0.0) for token in _GLOSS_N["urgency"])
    raw = 0.9 * strong + 0.5 * urgency + 0.5 * negative
    return raw / (raw + 2.0)


def _evidence(prefill: Prefill, sketch: dict[str, float]) -> float:
    """Evidência do descritor no estado: soma ponderada dos tokens presentes."""
    if not sketch:
        return 0.0
    weights = prefill.weights
    total = 0.0
    for token, weight in sketch.items():
        state_weight = weights.get(token)
        if state_weight is not None:
            total += weight * state_weight
    return total


def _clip(value: float) -> float:
    """Mantém o logit em faixa segura para exp()."""
    if value > 30.0:
        return 30.0
    if value < -30.0:
        return -30.0
    return value


class SystemOneCalibration(SimpleCalibration):
    """Calibração de desenvolvimento para o backend heurístico local.

    Aplica teto de confiança porque o backend não tem calibração empírica:
    reportar confiança alta aqui seria desonesto. Substitua por um
    `CalibrationArtifact` ajustado no seu domínio antes de produção.
    """

    def __init__(
        self,
        *,
        domain: str | None = None,
        temperature: float = 1.0,
        ceiling: float = 0.90,
    ) -> None:
        super().__init__(
            temperature,
            domain=domain,
            model_id=MODEL_ID,
            ceiling=ceiling,
        )
