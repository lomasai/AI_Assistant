import unittest

try:
    import httpx  # noqa: F401
except ModuleNotFoundError:
    import sys
    import types

    class _DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def post(self, *args, **kwargs):
            raise RuntimeError("httpx stub: post not implemented in this test")

        async def aclose(self):
            return None

    sys.modules["httpx"] = types.SimpleNamespace(
        AsyncClient=_DummyAsyncClient,
        HTTPError=Exception,
        HTTPStatusError=Exception,
        RequestError=Exception,
    )

from server.actions import action_engine
from server.decision_engine import DecisionEngine
from server.pipeline import PipelineEngine, PipelineRuntimeConfig


class _FakeSTT:
    async def transcribe_bytes(self, audio_bytes, filename="audio.wav"):
        return "hello"


class _FakeGroq:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, **kwargs):
        self.calls += 1
        return '{"mode":"response","response":"cached hello response","action":null}'


class _FakeDeepSeek:
    async def generate(self, **kwargs):
        return '{"mode":"response","response":"deepseek response","action":null}'


class _FakeTTS:
    def __init__(self) -> None:
        self.calls = 0

    async def synthesize_to_bytes(self, text):
        self.calls += 1
        return ("AUDIO:" + text).encode("utf-8")


class TestPipelineCache(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.groq = _FakeGroq()
        self.tts = _FakeTTS()

        self.engine = PipelineEngine(
            PipelineRuntimeConfig(
                enable_cache=True,
                cache_max_items=64,
                stt_cache_ttl_seconds=60,
                decision_cache_ttl_seconds=60,
                tts_cache_ttl_seconds=60,
                tts_fail_hard=False,
            )
        )
        self.engine.stt_service = _FakeSTT()
        self.engine.decision_engine = DecisionEngine(
            groq_client=self.groq,
            deepseek_client=_FakeDeepSeek(),
            action_executor=action_engine,
        )
        self.engine.tts_service = self.tts

    async def test_cache_hits_on_second_run(self) -> None:
        first = await self.engine.run_from_audio_bytes(audio_bytes=b"abc", filename="sample.wav")
        second = await self.engine.run_from_audio_bytes(audio_bytes=b"abc", filename="sample.wav")

        self.assertFalse(first.cache.get("stt", False))
        self.assertTrue(second.cache.get("stt", False))
        self.assertTrue(second.cache.get("decision", False))
        self.assertTrue(second.cache.get("tts", False))
        self.assertEqual(self.groq.calls, 1)
        self.assertEqual(self.tts.calls, 1)


if __name__ == "__main__":
    unittest.main()
