# LomasAI ESP32-S3 firmware

## What this board is for

Everything that must happen inside a few milliseconds.

The Pi runs Python. Python can pause for garbage collection at any moment, so
nothing a pause would ruin is allowed to live there. That is the entire reason
this board exists, and it is not a performance argument — it is a safety one.

**This board owns:** servo PWM, keyframe interpolation, the drive motors and
their encoders, and the cliff, tilt, over-current and e-stop cut-outs.

**The Pi owns:** what to say, who to look at, which gesture to play. Intent,
never execution.

If you ever find yourself wanting the Pi to read a cliff sensor and decide,
stop and raise it. That decision is forty milliseconds late by the time it
gets there.

## Files

| file | what it is |
|---|---|
| `protocol.h` | The wire contract. Mirrors `packages/lomas_hal/protocol.py`; a test fails if they drift. |
| `gestures_generated.h` | Generated from `config/hardware/gestures.yaml`. **Never edit.** |
| `main.cpp` | The stub. Four TODOs, and they are the whole job. |

## The four TODOs

1. **`safety_task`** — read the VL53L0X cliff sensors, the MPU6050 and the
   ACS712, and the e-stop pin. Cut the servo rail and the BTS7960B enable
   lines the moment any is past `limits`. **Cut first, report second**:
   `report()` writes to serial, and serial can wait.
2. **`start_gesture`** — interpolate the keyframes onto the PCA9685 at 50 Hz,
   scaled by `speed`. Clamp every channel. The Pi clamps too; both is on
   purpose.
3. **`LOMAS_CMD_LOOK_AT`** — two int16s, decidegrees, to the neck servos.
4. **`LOMAS_CMD_MOVE`** — two int16s, thousandths of full scale, to the drive
   pair. Stop if the ultrasonic ring says an obstacle is inside
   `limits.obstacle_mm`.

## Hardware, as built

```
Servos     PCA9685 16-channel driver, 50 Hz
           2x OT5325M 25 kg   shoulders
           4x MG996R          neck yaw/pitch, elbows
Drive      2x Rhino 60 RPM encoder DC motors on BTS7960B
Sensors    6x HC-SR04 ring, 4x VL53L0X cliff, MPU6050, limit switches
Safety     latching e-stop, ACS712 current sense, fuse
```

The older specification document describes stepper motors. It is out of date.
**The bill of materials is correct: these are servos.**

## Testing before it goes near a class

The Python side ships a simulator that speaks this exact protocol and logs the
exact bytes:

```
python run.py --mode debug --set hardware.enabled=true
```

Every frame appears in the log as hex. Compare it against what this board
receives; if the two disagree, the frame log says why.

Switching from the simulator to this board is one config key and nothing else:

```
--set hardware.backend=esp32
```

Bench-verify all four cut-outs — e-stop, cliff, tilt, over-current — with the
arms moving, before the robot is in the same room as a child.
