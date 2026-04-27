"""Edge sensor module for temperature and humidity."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Protocol


class SensorError(Exception):
    """Raised when sensor reads fail."""


SensorProvider = Literal["dht", "mock", "auto"]


class SensorBackendProtocol(Protocol):
    """Protocol for pluggable sensor backends."""

    def read(self) -> tuple[float, float] | None:
        """Return temperature_c, humidity_percent."""


@dataclass(slots=True)
class SensorConfig:
    """Configuration for sensor reading."""

    provider: SensorProvider = "auto"
    pin: int = 4
    dht_model: str = "DHT22"  # DHT11 | DHT22
    retries: int = 3
    retry_delay_s: float = 0.4
    mock_temperature_c: float = 25.0
    mock_humidity_percent: float = 50.0

    @classmethod
    def from_env(cls) -> "SensorConfig":
        """Build config from environment variables."""
        return cls(
            provider=os.getenv("SENSOR_PROVIDER", "auto").strip().lower(),  # type: ignore[arg-type]
            pin=int(os.getenv("SENSOR_PIN", "4")),
            dht_model=os.getenv("SENSOR_DHT_MODEL", "DHT22").strip().upper(),
            retries=int(os.getenv("SENSOR_RETRIES", "3")),
            retry_delay_s=float(os.getenv("SENSOR_RETRY_DELAY_SECONDS", "0.4")),
            mock_temperature_c=float(os.getenv("SENSOR_MOCK_TEMP_C", "25.0")),
            mock_humidity_percent=float(os.getenv("SENSOR_MOCK_HUMIDITY", "50.0")),
        )


@dataclass(slots=True)
class SensorReading:
    """Structured sensor output."""

    temperature_c: float
    humidity_percent: float
    temperature_f: float
    timestamp_utc: str
    source: str
    status: Literal["ok", "degraded"]

    def as_dict(self) -> dict[str, Any]:
        """Return reading as serializable dictionary."""
        return {
            "temperature_c": self.temperature_c,
            "humidity_percent": self.humidity_percent,
            "temperature_f": self.temperature_f,
            "timestamp_utc": self.timestamp_utc,
            "source": self.source,
            "status": self.status,
        }


class MockSensorBackend:
    """Fallback backend for development and testing."""

    def __init__(self, temperature_c: float, humidity_percent: float) -> None:
        self.temperature_c = temperature_c
        self.humidity_percent = humidity_percent

    def read(self) -> tuple[float, float]:
        return self.temperature_c, self.humidity_percent


class DHTSensorBackend:
    """DHT11/DHT22 backend using Adafruit library."""

    def __init__(self, pin: int, model: str) -> None:
        self.pin = pin
        self.model = model.upper()
        try:
            import board  # type: ignore
            import adafruit_dht  # type: ignore
        except ImportError as exc:
            raise SensorError(
                "DHT backend requires `adafruit-circuitpython-dht` and `board` on Raspberry Pi."
            ) from exc

        pin_obj = getattr(board, f"D{pin}", None)
        if pin_obj is None:
            raise SensorError(f"Invalid GPIO pin mapping for pin={pin}.")

        if self.model == "DHT11":
            self._sensor = adafruit_dht.DHT11(pin_obj)
        elif self.model == "DHT22":
            self._sensor = adafruit_dht.DHT22(pin_obj)
        else:
            raise SensorError(f"Unsupported DHT model: {self.model}")

    def read(self) -> tuple[float, float] | None:
        temperature = self._sensor.temperature
        humidity = self._sensor.humidity
        if temperature is None or humidity is None:
            return None
        return float(temperature), float(humidity)


class SensorService:
    """Sensor service for temperature/humidity acquisition."""

    def __init__(self, config: SensorConfig | None = None) -> None:
        self.config = config or SensorConfig.from_env()
        self._backend: SensorBackendProtocol | None = None
        self._source: str = "unknown"

    async def read(self) -> SensorReading:
        """Read sensors and return structured temperature/humidity data."""
        backend = self._get_or_create_backend()
        temp_c: float | None = None
        humidity: float | None = None

        for _ in range(max(1, self.config.retries)):
            result = await asyncio.to_thread(backend.read)
            if result is not None:
                temp_c, humidity = result
                break
            await asyncio.sleep(max(0.0, self.config.retry_delay_s))

        if temp_c is None or humidity is None:
            # Auto-degrade to mock if available for resilience.
            if self.config.provider == "auto":
                mock = MockSensorBackend(
                    temperature_c=self.config.mock_temperature_c,
                    humidity_percent=self.config.mock_humidity_percent,
                )
                temp_c, humidity = await asyncio.to_thread(mock.read)
                return self._build_reading(
                    temperature_c=temp_c,
                    humidity_percent=humidity,
                    source="mock_fallback",
                    status="degraded",
                )
            raise SensorError("Failed to read sensor values.")

        return self._build_reading(
            temperature_c=temp_c,
            humidity_percent=humidity,
            source=self._source,
            status="ok",
        )

    def _get_or_create_backend(self) -> SensorBackendProtocol:
        if self._backend is not None:
            return self._backend

        provider = self.config.provider
        if provider == "mock":
            self._backend = MockSensorBackend(
                temperature_c=self.config.mock_temperature_c,
                humidity_percent=self.config.mock_humidity_percent,
            )
            self._source = "mock"
            return self._backend

        if provider in {"dht", "auto"}:
            try:
                self._backend = DHTSensorBackend(pin=self.config.pin, model=self.config.dht_model)
                self._source = "dht"
                return self._backend
            except SensorError:
                if provider == "dht":
                    raise

        # auto fallback
        self._backend = MockSensorBackend(
            temperature_c=self.config.mock_temperature_c,
            humidity_percent=self.config.mock_humidity_percent,
        )
        self._source = "mock"
        return self._backend

    @staticmethod
    def _build_reading(
        temperature_c: float,
        humidity_percent: float,
        source: str,
        status: Literal["ok", "degraded"],
    ) -> SensorReading:
        temp_c_rounded = round(temperature_c, 2)
        humidity_rounded = round(humidity_percent, 2)
        temp_f = round((temp_c_rounded * 9.0 / 5.0) + 32.0, 2)
        return SensorReading(
            temperature_c=temp_c_rounded,
            humidity_percent=humidity_rounded,
            temperature_f=temp_f,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            source=source,
            status=status,
        )


sensors_service = SensorService()
