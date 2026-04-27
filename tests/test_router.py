import unittest

from server.router import IntentRouter


class TestIntentRouter(unittest.TestCase):
    def setUp(self) -> None:
        self.router = IntentRouter()

    def test_simple_phrase_routes_to_groq(self) -> None:
        decision = self.router.route("hello")
        self.assertEqual(decision.intent, "simple")
        self.assertEqual(decision.model, "groq")

    def test_complex_query_routes_to_deepseek(self) -> None:
        text = (
            "Can you explain why this fails, compare two solutions, and then provide a "
            "step by step optimization plan with reasoning and tradeoff analysis"
        )
        decision = self.router.route(text)
        self.assertEqual(decision.intent, "complex")
        self.assertEqual(decision.model, "deepseek")


if __name__ == "__main__":
    unittest.main()
