from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

# The Pi tells you all of this through the kernel and one binary, so there is
# no dependency here on purpose: a diagnostics panel that needs a pip install
# is a diagnostics panel that is missing on the machine you need it on.

PROC_STAT = Path("/proc/stat")
THERMAL = Path("/sys/class/thermal/thermal_zone0/temp")
MEMINFO = Path("/proc/meminfo")
STATUS = Path("/proc/self/status")
VCGENCMD = "vcgencmd"

CPU_PREFIX = "cpu"
MILLIDEGREES = 1000.0
KILOBYTES = 1024.0
IDLE_FIELDS = (3, 4)  # idle and iowait, in the order /proc/stat lists them
PERCENT = 100.0
UNAVAILABLE = None
UNAVAILABLE_MB = 0.0

# What each bit of `vcgencmd get_throttled` means. The sticky ones matter as
# much as the live ones: a Pi that throttled ten minutes ago explains a frame
# rate that has not recovered.
THROTTLE_BITS = {
    0: "under_voltage",
    1: "frequency_capped",
    2: "throttled",
    3: "soft_temperature_limit",
    16: "under_voltage_since_boot",
    17: "frequency_capped_since_boot",
    18: "throttled_since_boot",
    19: "soft_temperature_limit_since_boot",
}


class Host:
    """The machine underneath, sampled between calls.

    CPU is a difference of two readings, so the first call reports nothing
    and every call after it reports the busy share since the previous one.
    """

    def __init__(self) -> None:
        self._previous: dict[str, tuple[int, int]] = {}
        self._at = 0.0

    def snapshot(self) -> dict:
        return {
            "cores": self.cpu(),
            "temperature_c": self.temperature(),
            "throttled": self.throttled(),
            "memory": self.memory(),
            "load": self.load(),
            "uptime_s": round(time.monotonic(), 1),
        }

    def cpu(self) -> list[float]:
        """Busy share per core since the last call, as a percentage."""
        readings = _cpu_times()
        if not readings:
            return []

        busy: list[float] = []
        for name, (total, idle) in readings.items():
            before = self._previous.get(name)
            if before is not None:
                spent = total - before[0]
                resting = idle - before[1]
                busy.append(round(PERCENT * (spent - resting) / spent, 1) if spent > 0 else 0.0)
        self._previous = readings
        return busy

    def temperature(self) -> float | None:
        try:
            return round(int(THERMAL.read_text()) / MILLIDEGREES, 1)
        except (OSError, ValueError):
            return UNAVAILABLE

    def throttled(self) -> list[str]:
        binary = shutil.which(VCGENCMD)
        if binary is None:
            return []
        try:
            output = subprocess.run(
                [binary, "get_throttled"], capture_output=True, text=True, timeout=2
            ).stdout
            value = int(output.strip().split("=")[-1], 16)
        except (OSError, ValueError, subprocess.SubprocessError):
            return []
        return [name for bit, name in THROTTLE_BITS.items() if value & (1 << bit)]

    def memory(self) -> dict:
        try:
            fields = dict(
                (line.split(":")[0], float(line.split()[1]))
                for line in MEMINFO.read_text().splitlines()
                if ":" in line
            )
        except (OSError, ValueError, IndexError):
            return {}
        total = fields.get("MemTotal", 0.0) / KILOBYTES
        available = fields.get("MemAvailable", 0.0) / KILOBYTES
        return {
            "total_mb": round(total),
            "available_mb": round(available),
            "used_mb": round(total - available),
            # The number that actually answers "will this fit on a Pi".
            "process_mb": self.process_mb(),
        }

    def process_mb(self) -> float:
        """Resident memory of this process.

        System free memory says what the whole machine is doing, including
        Chromium. This says what the robot itself costs, which is the figure
        that decides whether a 2 GB Pi is enough.
        """
        try:
            for line in STATUS.read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return round(float(line.split()[1]) / KILOBYTES, 1)
        except (OSError, ValueError, IndexError):
            pass
        return UNAVAILABLE_MB

    def load(self) -> list[float]:
        try:
            return [round(value, 2) for value in os.getloadavg()]
        except (OSError, AttributeError):
            return []


def _cpu_times() -> dict[str, tuple[int, int]]:
    try:
        lines = PROC_STAT.read_text().splitlines()
    except OSError:
        return {}

    readings: dict[str, tuple[int, int]] = {}
    for line in lines:
        parts = line.split()
        if not parts or not parts[0].startswith(CPU_PREFIX) or parts[0] == CPU_PREFIX:
            continue
        values = [int(v) for v in parts[1:]]
        idle = sum(values[i] for i in IDLE_FIELDS if i < len(values))
        readings[parts[0]] = (sum(values), idle)
    return readings
