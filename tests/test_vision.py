import unittest

from api.vision import build_vision_router
from server.vision import VisionError, vision_service


class TestVisionApi(unittest.TestCase):
    def test_vision_status(self) -> None:
        payload = vision_service.status()
        self.assertTrue(payload["ok"])
        self.assertIn("frames_analyzed", payload)

    def test_vision_router_exposes_expected_routes(self) -> None:
        router = build_vision_router()
        paths = {route.path for route in router.routes}
        self.assertIn("/vision/analyze", paths)
        self.assertIn("/vision/status", paths)


class TestVisionService(unittest.TestCase):
    def test_data_url_prefix_is_stripped(self) -> None:
        with self.assertRaises(VisionError):
            vision_service.analyze("data:image/jpeg;base64,not-valid-base64")


if __name__ == "__main__":
    unittest.main()
