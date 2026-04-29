import unittest

from server.decision_engine import DecisionEngine


class _FailingLLM:
    async def generate(self, **kwargs):
        raise AssertionError("LLM should not be called for simple social phrases")


class _UnsupportedActionThenResponseLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return '{"mode":"action","response":null,"action":{"name":"generate_diet_chart","args":{}}}'
        return '{"mode":"response","response":"Here is a practical weekly diet chart.","action":null}'


class TestDecisionEngine(unittest.IsolatedAsyncioTestCase):
    async def test_simple_greeting_uses_local_response(self) -> None:
        engine = DecisionEngine(groq_client=_FailingLLM())

        result = await engine.decide(user_text="Hi", memory={"recent_logs": []})

        self.assertEqual(result.decision_type, "response")
        self.assertEqual(result.model, "local_rule")
        self.assertEqual(result.response_text, "Hello! How can I help you?")
        self.assertIn("local_simple_phrase_response", result.reasons)

    async def test_unsupported_action_is_repaired_as_response(self) -> None:
        llm = _UnsupportedActionThenResponseLLM()
        engine = DecisionEngine(groq_client=llm)

        result = await engine.decide(user_text="give me a detailed diet chart")

        self.assertEqual(result.decision_type, "response")
        self.assertEqual(result.response_text, "Here is a practical weekly diet chart.")
        self.assertIsNone(result.action)
        self.assertEqual(llm.calls, 2)
        self.assertIn("unsupported_action_repaired_as_response", result.reasons)


if __name__ == "__main__":
    unittest.main()
