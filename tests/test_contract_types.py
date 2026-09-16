"""Testes do snapshot canônico e do contrato de perguntas."""

from __future__ import annotations

import unittest

from specter_decision import Question, canonical_snapshot
from specter_decision.canonical import fingerprint_of


class CanonicalTests(unittest.TestCase):
    def test_key_order_does_not_change_fingerprint(self):
        left = canonical_snapshot({"b": 1, "a": [1, 2]})
        right = canonical_snapshot({"a": [1, 2], "b": 1})
        self.assertEqual(left.json, right.json)
        self.assertEqual(left.fingerprint, right.fingerprint)

    def test_list_order_does_change_fingerprint(self):
        self.assertNotEqual(
            canonical_snapshot([1, 2]).fingerprint, canonical_snapshot([2, 1]).fingerprint
        )

    def test_rejects_non_finite_numbers(self):
        for state in (float("nan"), float("inf"), {"x": float("-inf")}, [1, float("nan")]):
            with self.assertRaisesRegex(ValueError, "INVALID_INPUT"):
                canonical_snapshot(state)

    def test_rejects_unsupported_types_and_keys(self):
        with self.assertRaisesRegex(ValueError, "INVALID_INPUT"):
            canonical_snapshot({1: "chave não string"})
        with self.assertRaisesRegex(ValueError, "INVALID_INPUT"):
            canonical_snapshot({"s": {1, 2}})

    def test_rejects_cycles_and_depth(self):
        cyclic: dict = {}
        cyclic["self"] = cyclic
        with self.assertRaisesRegex(ValueError, "INVALID_INPUT"):
            canonical_snapshot(cyclic)
        deep: object = "leaf"
        for _ in range(40):
            deep = [deep]
        with self.assertRaisesRegex(ValueError, "INVALID_INPUT"):
            canonical_snapshot(deep)

    def test_size_limit_is_enforced(self):
        with self.assertRaisesRegex(ValueError, "INVALID_INPUT"):
            canonical_snapshot({"blob": "x" * 5000}, max_bytes=1024)

    def test_unicode_is_preserved_not_escaped(self):
        snapshot = canonical_snapshot({"t": "ação urgente ção"})
        self.assertIn("ação", snapshot.json)
        self.assertEqual(snapshot.nbytes, len(snapshot.json.encode("utf-8")))

    def test_fingerprint_is_deterministic_and_separated(self):
        self.assertEqual(fingerprint_of("abc"), fingerprint_of("abc"))
        self.assertNotEqual(fingerprint_of("abc"), fingerprint_of("abc", prefix="other"))
        self.assertEqual(len(fingerprint_of("abc")), 32)


class QuestionTests(unittest.TestCase):
    def test_noul_gets_default_legend(self):
        question = Question("q", "noul", "urgente?")
        self.assertEqual(question.labels, ("false", "true"))
        self.assertEqual(question.cardinality, 2)

    def test_invalid_questions_are_rejected(self):
        cases = [
            ("", "noul", "x", ()),
            ("q", "unknown", "x", ()),
            ("q", "choice", "x", ("a",)),
            ("q", "choice", "x", ("a", "a")),
            ("q", "score", "x", ("only",)),
            ("q", "score", "x", tuple(str(i) for i in range(11))),
            ("q", "choice", "", ("a", "b")),
        ]
        for identifier, kind, text, labels in cases:
            with self.assertRaisesRegex(ValueError, "INVALID_INPUT"):
                Question(identifier, kind, text, labels)

    def test_choice_accepts_up_to_255_options(self):
        labels = tuple(f"opt{i}" for i in range(255))
        self.assertEqual(Question("q", "choice", "escolha", labels).cardinality, 255)
        with self.assertRaisesRegex(ValueError, "INVALID_INPUT"):
            Question("q", "choice", "escolha", labels + ("overflow",))

    def test_signature_ignores_id_but_tracks_content(self):
        base = Question("a", "choice", "Route?", ("x", "y"))
        same = Question("b", "choice", "Route?", ("x", "y"))
        different = Question("a", "choice", "Route?", ("x", "z"))
        self.assertEqual(base.signature, same.signature)
        self.assertNotEqual(base.signature, different.signature)

    def test_anonymous_keeps_everything_but_id(self):
        question = Question("secreto", "choice", "Route?", ("x", "y"))
        anonymous = question.anonymous()
        self.assertEqual(anonymous.id, "_")
        self.assertEqual(anonymous.signature, question.signature)
        self.assertEqual(anonymous.labels, question.labels)
        self.assertIs(anonymous.anonymous(), anonymous)

    def test_criteria_align_with_labels(self):
        question = Question.choice(
            "dept", "Qual equipe?", {"billing": "Faturas", "technical": None}
        )
        self.assertEqual(question.labels, ("billing", "technical"))
        self.assertEqual(question.describe(0), "billing: Faturas")
        self.assertEqual(question.describe(1), "technical")
        with self.assertRaisesRegex(ValueError, "INVALID_INPUT"):
            Question("q", "choice", "x", ("a", "b"), criteria=("apenas uma",))

    def test_noul_constructor_maps_true_false_criteria(self):
        question = Question.noul("u", "urgente?", {"true": "prazo", "false": "sem pressa"})
        self.assertEqual(question.criteria, ("sem pressa", "prazo"))

    def test_score_tolerance_validation(self):
        question = Question.score("s", "nota", ["a", "b", "c"], tolerance=0.25)
        self.assertEqual(question.tolerance, 0.25)
        with self.assertRaisesRegex(ValueError, "INVALID_INPUT"):
            Question("s", "score", "nota", ("a", "b"), tolerance=-1.0)
        with self.assertRaisesRegex(ValueError, "INVALID_INPUT"):
            Question("s", "score", "nota", ("a", "b"), tolerance=float("nan"))

    def test_questions_are_hashable_and_comparable(self):
        left = Question("a", "noul", "x")
        right = Question("a", "noul", "x")
        self.assertEqual(left, right)
        self.assertEqual(len({left, right}), 1)


if __name__ == "__main__":
    unittest.main()
