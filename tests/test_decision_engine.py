import unittest

from server.decision_engine import DecisionEngine


class _FailingLLM:
    async def generate(self, **kwargs):
        raise AssertionError("LLM should not be called for simple social phrases")


class TestDecisionEngine(unittest.IsolatedAsyncioTestCase):
    async def test_simple_greeting_uses_local_response(self) -> None:
        engine = DecisionEngine(groq_client=_FailingLLM())

        result = await engine.decide(user_text="Hi", memory={"recent_logs": []})

        self.assertEqual(result.decision_type, "response")
        self.assertEqual(result.model, "local_rule")
        self.assertEqual(result.response_text, "Hello! How can I help you?")
        self.assertIn("local_simple_phrase_response", result.reasons)


if __name__ == "__main__":
    unittest.main()
