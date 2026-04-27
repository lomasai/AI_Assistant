import unittest
from datetime import datetime, timedelta, timezone

from server.reminders import ReminderConfig, ReminderEngine


class TestReminderEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = ReminderEngine(
            ReminderConfig(
                medicine_times=("09:00",),
                medicine_grace_minutes=90,
                water_gap_minutes=120,
                sitting_limit_minutes=60,
                cooldown_minutes=10,
            )
        )

    def test_emits_expected_reminders(self) -> None:
        now = datetime(2026, 4, 22, 9, 30, tzinfo=timezone.utc)
        events = self.engine.evaluate(
            context={
                "water_gap_minutes": 140,
                "posture": "sitting",
                "sitting_minutes": 70,
            },
            now=now,
        )
        event_types = sorted(event.type for event in events)
        self.assertEqual(event_types, ["medicine", "sitting", "water"])

    def test_cooldown_prevents_spam(self) -> None:
        now = datetime(2026, 4, 22, 9, 30, tzinfo=timezone.utc)
        first = self.engine.evaluate(
            context={"water_gap_minutes": 130, "posture": "sitting", "sitting_minutes": 80},
            now=now,
        )
        second = self.engine.evaluate(
            context={"water_gap_minutes": 160, "posture": "sitting", "sitting_minutes": 90},
            now=now + timedelta(minutes=2),
        )
        self.assertGreater(len(first), 0)
        self.assertEqual(second, [])


if __name__ == "__main__":
    unittest.main()
