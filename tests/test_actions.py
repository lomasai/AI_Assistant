import unittest

from server.actions import ActionExecutionEngine


class TestActionExecutionEngine(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.engine = ActionExecutionEngine()

    async def test_mark_medicine_taken(self) -> None:
        result = await self.engine.execute({"name": "mark_medicine_taken", "args": {}}, context={})
        self.assertTrue(result.ok)
        self.assertEqual(result.name, "mark_medicine_taken")

    async def test_log_event(self) -> None:
        result = await self.engine.execute(
            {
                "name": "log_event",
                "args": {
                    "event": "unit_test_event",
                    "message": "Action log event test",
                    "metadata": {"test": True},
                },
            },
            context={"source": "test"},
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.name, "log_event")
        self.assertIn("log_id", result.data)


if __name__ == "__main__":
    unittest.main()
