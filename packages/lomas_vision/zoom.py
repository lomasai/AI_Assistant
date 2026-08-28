from __future__ import annotations

CENTRE_DIVISOR = 2


def clamp(factor: float, lowest: float, highest: float) -> float:
    return max(lowest, min(highest, factor))


def crop_rectangle(
    factor: float, sensor_width: int, sensor_height: int, lowest: float, highest: float
) -> tuple[int, int, int, int]:
    """Centred sub-rectangle of the sensor, as (x, y, width, height).

    The Pi camera has no optical zoom, but the sensor is far larger than the
    stream we read out of it, so cropping keeps real pixels rather than
    interpolating. picamera2 takes this as ScalerCrop; other sources ignore it.
    """
    factor = clamp(factor, lowest, highest)
    width = int(sensor_width / factor)
    height = int(sensor_height / factor)
    x = (sensor_width - width) // CENTRE_DIVISOR
    y = (sensor_height - height) // CENTRE_DIVISOR
    return x, y, width, height
