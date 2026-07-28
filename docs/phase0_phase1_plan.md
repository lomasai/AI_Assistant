# Phase 0 And Phase 1 Plan

## Phase 0 Findings

- The repository is a single FastAPI application with a vanilla browser UI.
- Chat, STT, TTS, routing, actions, reminders, memory, and browser-frame vision endpoints exist.
- The camera workflow is still browser-owned through `getUserMedia()` and Base64 JPEG posts to `/vision/analyze`.
- `edge/camera.py` provides an OpenCV `VideoCapture` wrapper, not a Picamera2 CSI owner.
- `edge/motor_control.py` is currently empty.
- Configuration is spread across environment variables and dataclass defaults.
- `config/default.yaml` and `config/device.yaml` did not exist before Phase 1.
- Face registration, local face recognition, wake-word inference, teaching state machine, Picamera2 streaming, and physical servo movement remain future phases.
- Gender, age, emotion, health, and sensor diagnostics are still present in the technical UI and should be hidden from the later student UI.

## Baseline Test Result

The test suite could not start in the current Windows environment because every Python entry point failed before importing tests:

```text
A specified logon session does not exist. It may already have been terminated.
```

Commands attempted:

```powershell
python -m unittest discover -s tests -v
py -3.12 -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Phase 1 Implementation Plan

1. Add central validated YAML configuration loaded from `config/default.yaml` and optional `config/device.yaml`.
2. Keep secrets in environment variables referenced by profile names such as `GROQ_API_KEY`.
3. Add provider-neutral LLM protocol and registry.
4. Add a generic OpenAI-compatible chat-completions provider for OpenAI, Groq, DeepSeek, xAI, and local compatible endpoints.
5. Add mock LLM provider for local tests and offline development.
6. Add protocol interfaces for camera, audio input/output, wake word, STT, TTS, face detection, face recognition, sensors, motion, students, and sessions.
7. Add mock implementations for all hardware-facing interfaces.
8. Add application runtime factory and registries without replacing the existing endpoints.
9. Expose runtime configuration status through health/system metadata without printing secrets.
10. Add focused unit tests for configuration, provider selection, and mock drivers.

## Phase 1 Non-Goals

- No Picamera2 implementation.
- No student registration or embeddings.
- No new student UI.
- No wake-word inference.
- No physical servo movement or ESP32 firmware.
