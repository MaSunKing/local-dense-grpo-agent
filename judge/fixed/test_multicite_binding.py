
import unittest
from final_binding import bind

class MultiCitationTests(unittest.TestCase):
    def ev(self):
        return [
            {"source_id":"SOURCE_A", "text":"Evidence A"},
            {"source_id":"SOURCE_B", "text":"Evidence B"},
        ]

    def answer(self, value):
        return '<answer>Claim. <cite id="'+value+'">sources</cite></answer>'

    def test_single(self):
        _, units, _ = bind(self.answer("SOURCE_A"), self.ev())
        self.assertEqual(units[0]["citation_ids"], ["SOURCE_A"])

    def test_multiple_preserve_original_span(self):
        raw = self.answer("SOURCE_A,SOURCE_B")
        body, units, _ = bind(raw, self.ev())
        self.assertEqual(units[0]["citation_ids"], ["SOURCE_A","SOURCE_B"])
        span = units[0]["span"]
        self.assertEqual(body[span["start"]:span["end"]], span["quote"])
        self.assertIn('id="SOURCE_A,SOURCE_B"', span["quote"])

    def test_unknown_member_rejected(self):
        with self.assertRaises(ValueError):
            bind(self.answer("SOURCE_A,UNKNOWN"), self.ev())

    def test_empty_members_rejected(self):
        for value in ("SOURCE_A,", ",SOURCE_A", "SOURCE_A,,SOURCE_B"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                bind(self.answer(value), self.ev())

    def test_no_alternative_separator_guess(self):
        with self.assertRaises(ValueError):
            bind(self.answer("SOURCE_A;SOURCE_B"), self.ev())

    def test_repeated_reference_not_double_counted(self):
        _, units, _ = bind(self.answer("SOURCE_A,SOURCE_A"), self.ev())
        self.assertEqual(units[0]["citation_ids"], ["SOURCE_A"])

    def test_no_cross_paragraph_attachment(self):
        raw = ('<answer>First. <cite id="SOURCE_A,SOURCE_B">sources</cite>'
               '\n\nSecond paragraph.</answer>')
        _, units, _ = bind(raw, self.ev())
        self.assertEqual(units[0]["citation_ids"], ["SOURCE_A","SOURCE_B"])
        self.assertEqual(units[1]["citation_ids"], [])

    def test_duplicate_evidence_rejected(self):
        with self.assertRaises(ValueError):
            bind(self.answer("SOURCE_A"), self.ev()+self.ev()[:1])

if __name__ == "__main__":
    unittest.main()
