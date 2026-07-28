# AI Robot Assistant

FastAPI-based AI robot assistant with chat, voice input, memory, text-to-speech placeholders, and a browser-to-backend live camera vision pipeline.

Phase 1 adds a validated central runtime configuration, provider-neutral LLM registry, and mock hardware drivers while preserving the current chat and browser-camera workflow.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Fill `.env` with your provider keys if you want live LLM calls.

## Configuration

Runtime behavior is configured through YAML files:

- `config/default.yaml` contains safe defaults and provider profiles.
- `config/device.yaml` is optional and ignored by git; copy `config/device.yaml.example` when configuring a Raspberry Pi or another device.
- `.env` is only for API keys and secrets.

LLM provider selection is controlled by:

```yaml
llm:
  active_provider: mock
```

Available Phase 1 profiles are:

- `mock`
- `openai`
- `groq`
- `deepseek`
- `xai`
- `generic_openai_compatible`

Each non-mock profile names an environment variable such as `GROQ_API_KEY`; the app never stores or exposes actual key values. Existing Groq and DeepSeek behavior remains compatible with the older environment variables.

Startup validates configuration and exposes non-secret runtime details at:

```text
GET /api/v1/health
GET /api/v1/system/info
```

Phase 1 also defines protocol interfaces and mock implementations for camera, audio input/output, wake word, face detection, face recognition, sensors, motion, students, and sessions. Picamera2, physical servo movement, wake-word inference, face registration, and the new student UI are intentionally not implemented yet.

## Backend Camera Preview

Phase 2 adds a backend-owned camera pipeline for mock and Raspberry Pi CSI preview modes.

Camera modes:

- `browser`: preserves the existing `getUserMedia()` workflow and Base64 vision posts.
- `mock`: backend generates test frames for Windows and automated tests.
- `picamera2`: backend owns the Raspberry Pi CSI camera through Picamera2.
- `disabled`: camera preview is unavailable by configuration.

Backend camera endpoints:

```text
GET /camera/status
GET /camera/stream.mjpg
GET /camera/events
```

The MJPEG stream is shared by preview clients; client disconnects do not stop the shared camera owner. The latest-frame buffer keeps only the newest frame and drops stale frames instead of building a queue.

For Windows development, keep `camera.provider: browser` or use `mock` in `config/device.yaml`:

```yaml
camera:
  provider: mock
  width: 640
  height: 480
  preview_fps: 12
  analysis_fps: 5
feature_flags:
  backend_camera_stream: true
```

For Raspberry Pi CSI camera testing, install Raspberry Pi camera packages and Picamera2, then use:

```yaml
camera:
  provider: picamera2
  width: 640
  height: 480
  preview_fps: 12
  analysis_fps: 5
  jpeg_quality: 75
feature_flags:
  browser_camera: false
  backend_camera_stream: true
```

Useful Raspberry Pi verification commands:

```bash
rpicam-hello --list-cameras
python -c "from picamera2 import Picamera2; print(Picamera2.global_camera_info())"
uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Use exactly one Uvicorn worker with Picamera2. Multiple workers can attempt to open the same CSI camera. For RealVNC, open Chromium on the Pi at `http://127.0.0.1:8000`; the backend MJPEG preview renders inside the existing dashboard when backend camera mode is active.

## Teaching Sessions

Phase 3 adds a deterministic teaching-session state machine and a student-facing UI at `/`. The existing technical dashboard remains available from the student UI's Admin button, or by opening:

```text
http://127.0.0.1:8000/?admin
```

Teaching states:

```text
idle -> session_setup -> lesson_ready -> explaining -> asking_question
-> waiting_for_answer -> evaluating -> remediation -> asking_question
-> waiting_for_answer -> evaluating -> recap -> session_complete
```

Pause, resume and stop are deterministic commands. The tutor model may generate lesson text, but it does not control state transitions, camera, audio, hardware or grading. Answer evaluation is conservative and deterministic: `correct`, `partially_correct`, `incorrect` or `unclear`.

Mock teaching works without cloud keys:

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`, enter a name, topic and objective, then start the lesson. Text answers work without microphone/STT.

Teaching API examples:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/teaching/sessions ^
  -H "Content-Type: application/json" ^
  -d "{\"student_display_name\":\"Asha\",\"grade_level\":\"grade_6\",\"topic\":\"Fractions\",\"language\":\"en\",\"objective\":\"Understand what a fraction means.\"}"

curl -X POST http://127.0.0.1:8000/api/v1/teaching/sessions/{session_id}/start
curl -X POST http://127.0.0.1:8000/api/v1/teaching/sessions/{session_id}/answer ^
  -H "Content-Type: application/json" ^
  -d "{\"answer_text\":\"A fraction is part of a whole.\"}"
curl http://127.0.0.1:8000/api/v1/teaching/sessions/{session_id}/summary
```

Configuration options live under `teaching` in `config/default.yaml`:

- `language`
- `level`
- `max_remediation_attempts`
- `max_lesson_turns`
- `provider_timeout_seconds`
- `structured_output_retries`
- `session_inactivity_timeout_seconds`

Phase 3 UI verification status:

- Software baseline is retained at 41 passing tests before Phase 4.
- Manual browser verification from Phase 3 is pending per review direction.

Known Phase 3 limitations:

- One active teaching session is intended for the first version.
- Session recovery across browser refresh uses `localStorage` plus the in-process mock repository; it does not survive server restart yet.
- No wake-word inference, physical audio changes, attention changes, ESP32 or servo control.
- Student UI hides technical diagnostics; admin/debug retains them.

## Student Registration And Recognition

Phase 4 adds admin-controlled student registration, local SQLite persistence and deterministic local face embeddings for mock/runtime verification.

Data stored locally:

- Student profile: name, optional grade and language.
- Consent flag and consent timestamp.
- Registration status.
- Face embeddings as local numeric vectors.
- Teaching-session records table for future persistence migration.

Data intentionally not stored:

- API keys.
- Raw temporary camera frames.
- Raw images through the API.
- Gender, age, emotion, health or ethnicity attributes.

Admin APIs are under:

```text
POST   /api/v1/admin/students
GET    /api/v1/admin/students
GET    /api/v1/admin/students/{student_id}
DELETE /api/v1/admin/students/{student_id}?confirm=true
POST   /api/v1/admin/registrations
POST   /api/v1/admin/registrations/{registration_id}/samples
GET    /api/v1/admin/registrations/{registration_id}
POST   /api/v1/admin/registrations/{registration_id}/complete
POST   /api/v1/admin/registrations/{registration_id}/cancel
POST   /api/v1/admin/recognize
```

Set `ADMIN_API_TOKEN` in `.env` to require the `X-Admin-Token` header for admin routes. When unset, local development remains open.

The student UI uses:

```text
POST /api/v1/student/recognize
```

This endpoint returns only `Guest` or a recognized display name/id. It does not return raw embeddings or confidence values.

Windows mock setup:

```yaml
camera:
  provider: mock
recognition:
  face_detection_provider: mock
  face_recognition_provider: mock
database:
  sqlite_path: memory/app.db
feature_flags:
  student_registration: true
  face_recognition: true
```

Raspberry Pi local-recognition setup still needs physical validation and a real embedding model path:

```yaml
recognition:
  face_detection_provider: opencv
  face_recognition_provider: local
  face_detection_model_path: models/face/face_detection_yunet_2023mar.onnx
  face_recognition_model_path: models/face/face_recognition_sface_2021dec.onnx
  embedding_model_path: ""
  face_match_threshold: 0.72
database:
  sqlite_path: memory/raspberry-pi-students.db
feature_flags:
  student_registration: true
  face_recognition: true
```

Registration rejects dark, blurry, no-face and multi-face samples. The automated implementation uses deterministic mock embeddings so tests do not need Raspberry Pi hardware or a CSI camera.

Phase 4 closure adds an optional real local OpenCV pipeline:

- Detection: OpenCV YuNet, `face_detection_yunet_2023mar.onnx`.
- Recognition: OpenCV SFace, `face_recognition_sface_2021dec.onnx`.
- Runtime behavior: models load only when `recognition.face_detection_provider: opencv` and `recognition.face_recognition_provider: local` are configured.
- If either model is unavailable, startup fails clearly. The app does not download models at runtime and does not silently fall back to mock unless mock is explicitly configured.
- Embeddings are L2-normalized before storage and cosine similarity is compared with `recognition.face_match_threshold`.

Install model files manually under `models/face/`:

```bash
mkdir -p models/face
sha256sum models/face/face_detection_yunet_2023mar.onnx
sha256sum models/face/face_recognition_sface_2021dec.onnx
```

Expected SHA256 values from the OpenCV Zoo Hugging Face mirror:

```text
face_detection_yunet_2023mar.onnx
8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4

face_recognition_sface_2021dec.onnx
0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79
```

Sources and licenses:

- YuNet model source: OpenCV Zoo `models/face_detection_yunet`; the model directory is MIT licensed.
- SFace model source: OpenCV Zoo `models/face_recognition_sface`; the model directory is Apache-2.0 licensed.
- OpenCV Zoo repository: https://github.com/opencv/opencv_zoo
- Hugging Face OpenCV Zoo mirror: https://huggingface.co/opencv/opencv_zoo

Security notes:

- Set `ADMIN_API_TOKEN` before binding to anything other than `127.0.0.1`, `localhost` or `::1`; startup fails otherwise.
- Admin token checks use constant-time comparison.
- Health responses do not include API keys, token values, raw embeddings, images or model filesystem paths.
- Student-facing recognition returns only a display name/id above threshold or `Guest`.

## Run

```bash
uvicorn api.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

The browser will ask for camera and microphone permissions when you start those features. Camera access generally works on `localhost` / `127.0.0.1`; if permission is denied, enable camera access in the browser site settings and try again.

## Voice Interaction

Phase 5 adds a backend-owned audio coordinator for wake-word activation, push-to-talk, VAD, STT, TTS, speech queueing and deterministic teaching turn-taking. Text input remains fully functional when audio is disabled or unavailable.

Voice flow:

```text
wake word or push-to-talk -> listening -> speech detected -> transcription
-> teaching answer submission -> evaluation -> tutor response -> speech output
-> ready for next turn
```

The LLM does not control microphone, speaker, wake-word state or hardware. Pause/stop cancels active audio. Barge-in is disabled by default.

Windows mock setup:

```yaml
audio:
  input_provider: mock
  output_provider: mock
  sample_rate: 16000
  channels: 1
  chunk_duration_ms: 30
  max_recording_seconds: 8
  retain_temporary_audio: false
wake_word:
  provider: mock
vad:
  provider: mock
stt:
  provider: mock
  language: en
tts:
  provider: mock
feature_flags:
  push_to_talk: true
  voice_turns: true
  stt_input: true
  tts_output: true
  wake_word: false
```

Raspberry Pi 5 local provider targets:

- Wake word: openWakeWord.
- STT: whisper.cpp.
- TTS: Piper.

These providers are local/offline when configured with local models. No model files are downloaded at runtime. Install binaries and models manually, then configure `config/device.yaml`.

Example package and device checks:

```bash
arecord -l
aplay -l
python -c "import sounddevice as sd; print(sd.query_devices())"
which whisper-cli
which piper
python -c "import openwakeword; print('openwakeword ok')"
```

Example Raspberry Pi config:

```yaml
audio:
  input_provider: local
  output_provider: piper
  sample_rate: 16000
  channels: 1
  chunk_duration_ms: 30
  half_duplex: true
  barge_in_enabled: false
  retain_temporary_audio: false
wake_word:
  provider: openwakeword
  model_path: models/wake/hey-tutor.onnx
  sensitivity: 0.5
  cooldown_seconds: 2
vad:
  provider: energy
  silence_timeout_ms: 800
stt:
  provider: whisper_cpp
  executable_path: /usr/local/bin/whisper-cli
  model_path: models/stt/ggml-base.en.bin
  language: en
tts:
  provider: piper
  executable_path: /usr/local/bin/piper
  model_path: models/tts/en_US-lessac-medium.onnx
  voice: en_US-lessac-medium
feature_flags:
  wake_word: true
  push_to_talk: true
  voice_turns: true
  stt_input: true
  tts_output: true
  barge_in: false
```

Audio APIs:

```text
GET  /api/v1/audio/health
GET  /api/v1/audio/state
POST /api/v1/audio/push-to-talk/start
POST /api/v1/audio/push-to-talk/cancel
POST /api/v1/audio/wake-word/activate
POST /api/v1/audio/voice-answer
POST /api/v1/audio/tts/start
POST /api/v1/audio/tts/cancel
GET  /api/v1/audio/events
```

Legacy `/stt` and `/tts` endpoints are preserved. If you configure an external STT/TTS provider, document that audio/text may leave the device. Default Phase 5 mock/local paths do not retain raw recordings.

RealVNC note: camera preview through RealVNC is fine, but microphone/speaker routing can depend on the host OS and VNC audio forwarding. Prefer direct Raspberry Pi ALSA/Pulse/PipeWire checks for hardware verification.

Known Phase 5 limitations:

- Physical microphone/speaker verification is pending.
- openWakeWord, whisper.cpp and Piper model quality/performance must be tested on the target Raspberry Pi 5.
- Hindi support is configuration-ready but needs model validation.
- Phase 3 manual browser UI verification remains pending.

## Engagement Cues And Support Policy

Phase 6 adds a backend-owned observable engagement runtime for supportive lesson prompts. It uses only coarse observable cues: face present/absent, multiple faces, outside frame, coarse head orientation, response delay, repeated unclear answers and inactivity counters.

The policy is intentionally non-punitive:

- No emotion, motivation, intelligence, ADHD, health, gender, age or ethnicity inference.
- No raw frames, embeddings, head angles, confidence values, model paths or secrets are exposed through student APIs or student UI.
- A single unusual frame never triggers an intervention. Signals are smoothed over a rolling window.
- Prompts pause when the teaching session is paused/stopped/completed or while TTS is speaking.
- Student choices are always available: Continue, Repeat, Pause and Use text.

Default development config keeps engagement disabled:

```yaml
engagement:
  enabled: false
  provider: mock
  analysis_fps: 2
  rolling_window_seconds: 10
  absence_duration_seconds: 8
  response_inactivity_seconds: 20
  intervention_cooldown_seconds: 15
  max_interventions_per_lesson: 4
  retain_raw_frames: false
feature_flags:
  engagement_analysis: false
```

For Raspberry Pi testing, enable a local provider only after installing the model file manually. The application does not download engagement models at runtime.

```yaml
engagement:
  enabled: true
  provider: local
  model_path: models/engagement/observable-face-cues.onnx
  retain_raw_frames: false
feature_flags:
  engagement_analysis: true
```

Engagement APIs:

```text
GET  /api/v1/engagement/health
GET  /api/v1/engagement/sessions/{session_id}/state
POST /api/v1/engagement/sessions/{session_id}/signals
POST /api/v1/engagement/sessions/{session_id}/enable
POST /api/v1/engagement/sessions/{session_id}/choice
GET  /api/v1/engagement/sessions/{session_id}/history
GET  /api/v1/engagement/events
```

Physical Raspberry Pi verification remains pending for camera-derived engagement cues. Automated tests use deterministic mock signals only.

## Safe Hardware Control

Phase 7A adds a backend-owned hardware-control foundation for simulated motion only. Physical actuators must remain de-energized until a hardware profile is documented, reviewed and approved.

Current intended hardware profile:

```text
ESP32 board: BLOCKER - exact board unknown
Communication method: BLOCKER - serial USB/UART details unknown
Servos/motors: BLOCKER - exact actuator models unknown
Driver board: BLOCKER - exact servo/motor driver unknown
Power supply: BLOCKER - voltage/current/isolation unknown
Mechanical limits: BLOCKER - physical travel and unsafe operating zone unknown
Emergency stop: BLOCKER - physical e-stop method unknown
```

Phase 7A command protocol:

```json
{
  "version": "hardware.v1",
  "command_id": "uuid",
  "timestamp_utc": "2026-07-28T00:00:00+00:00",
  "action": "small_nod",
  "params": {
    "angle_deg": 10,
    "speed_deg_per_second": 20,
    "duration_seconds": 0.4
  }
}
```

Acknowledgements include only command id, status, timestamp and success/failure. APIs do not expose serial paths, network credentials, internal exceptions or secrets.

Safety rules enforced by the backend:

- The backend exclusively owns hardware commands.
- LLM, frontend and engagement providers do not directly control hardware.
- Physical output is disabled by default with `hardware.physical_output_enabled: false`.
- Enabling physical output requires `provider: esp32`, `transport: serial`, approved hardware profile, board, port, driver, power and emergency-stop details.
- Unsupported, malformed, stale, duplicate and out-of-range commands are rejected.
- Emergency stop has highest priority.
- Startup, shutdown, cancel, timeout and lost connection move to a safe stop/neutral state.
- Movement is limited by angle, speed, duration, cooldown and max continuous-motion settings.
- No movement is allowed unless `unsafe_operating_zone_clear: true`.
- No movement is triggered from engagement signals alone.

Default simulation config:

```yaml
hardware:
  enabled: false
  provider: mock
  transport: mock
  physical_output_enabled: false
  hardware_profile_approved: false
  unsafe_operating_zone_clear: false
feature_flags:
  hardware_control: false
```

Protected hardware APIs:

```text
GET  /api/v1/hardware/health
GET  /api/v1/hardware/actions
POST /api/v1/hardware/actions
POST /api/v1/hardware/cancel
POST /api/v1/hardware/emergency-stop
POST /api/v1/hardware/emergency-stop/reset
GET  /api/v1/hardware/history
```

Phase 7B hardware-validation procedure, pending approval:

1. Document the exact ESP32 board, firmware command parser, actuator models, driver board, power supply, wiring, fuse/current limits and physical e-stop.
2. Confirm mechanical travel limits with actuators disconnected from the student-facing mechanism.
3. Flash ESP32 firmware that accepts only `hardware.v1` bounded commands and returns acknowledgements.
4. Test serial communication with physical output disabled.
5. Energize driver board without load and verify emergency stop cuts motion authority.
6. Test neutral, reset and one small gesture under current-limited bench power.
7. Verify lost heartbeat, timeout and process shutdown return to safe stop.
8. Record measured angles, current draw, heat and recovery behavior.

Do not claim physical verification until those steps are performed on the target Raspberry Pi and hardware assembly.

## Laptop PoC Demo

Phase 8 closes the software-only PoC for a Windows laptop. Physical Raspberry Pi, ESP32, camera, microphone, speaker, servo and motor validation is deferred, and `config/laptop-demo.yaml` keeps every provider in mock mode with:

```yaml
hardware:
  provider: mock
  transport: mock
  physical_output_enabled: false
  hardware_profile_approved: false
```

Start the laptop demo from PowerShell:

```powershell
.\scripts\start_laptop_demo.ps1 -AdminToken "choose-a-local-demo-token"
```

Open:

```text
http://127.0.0.1:8000
http://127.0.0.1:8000/?admin
```

For protected admin/debug calls, use the same token as `X-Admin-Token`. Do not commit real tokens to `.env`, YAML or scripts.

Readiness checks:

```powershell
.\scripts\health_check.ps1 -AdminToken "choose-a-local-demo-token"
```

Data reset and cleanup:

```powershell
.\scripts\reset_laptop_demo_data.ps1
.\scripts\reset_laptop_demo_data.ps1 -RemoveDeviceOverlay
```

The reset script removes only `memory\laptop-demo.db` by default. Student profiles can also be deleted from the admin UI or through `DELETE /api/v1/admin/students/{student_id}?confirm=true`.

Laptop demo checklist:

- Student registration uses deterministic mock face embeddings.
- Recognition returns a registered display name only above threshold; otherwise the student UI shows `Guest`.
- Browser camera fallback remains available, while the default laptop demo uses the backend mock MJPEG stream.
- Voice turn-taking uses mock wake word, STT and TTS; no microphone or speaker hardware is required.
- Engagement prompts use mock observable signals only.
- Hardware controls use the mock controller only; emergency stop and explicit reset can be exercised without energizing actuators.

Troubleshooting:

- If port 8000 is busy, run `.\scripts\start_laptop_demo.ps1 -Port 8001` and open that port.
- If admin actions return 403, set `ADMIN_API_TOKEN` or restart with `-AdminToken` and use the same value in the UI prompt.
- If Python reports a Windows logon-session error, recreate `.venv` and reinstall `requirements.txt`.
- If stale demo students remain, run `.\scripts\reset_laptop_demo_data.ps1`.

Deferred validation labels:

- Real biometric accuracy and spoof resistance.
- Physical camera framing and Raspberry Pi CSI performance.
- Physical microphone, wake-word, STT and speaker quality.
- Real engagement model behaviour from camera-derived cues.
- ESP32 firmware, power system, e-stop wiring and servo/motor movement.

## Voice Input And Transcripts

The Mic button records audio from the browser. The latest transcript is shown in the Voice Transcript panel above the chat input, and the same text is also inserted into the chat as `You said: "..."`

In Chrome/Edge, the UI first tries browser speech recognition for a real-time transcript. If browser speech recognition is unavailable or returns no final text, the app sends the recording to `POST /stt`.

Backend STT can use Groq Whisper through the OpenAI-compatible Groq transcription endpoint. Configure `.env` like this:

```text
STT_PROVIDER=groq_whisper
GROQ_API_KEY=your_groq_key
GROQ_WHISPER_MODEL=whisper-large-v3-turbo
STT_LANGUAGE=en
```

For development only, this placeholder mode always returns `Transcription placeholder output.`:

```text
STT_PROVIDER=placeholder
```

## Web UI

The main dashboard uses a 50/50 desktop layout:

- Chat assistant panel
- Voice message button using browser microphone capture
- Live camera preview panel
- Canvas overlay for face boxes, estimated body boxes, eye landmarks, gaze arrow, and attention indicator
- Vision badges for camera state, face detection, posture, tracking direction, FPS, latency, and last analysis time
- Face analysis, eye/attention, posture/body, tracking/motor, health behavior, sensor data, AI decision, and live event timeline cards
- System information modal using `GET /api/v1/system/info`

The frontend captures one JPEG frame about every 750 ms and sends it to the backend vision endpoint. It skips overlapping requests if analysis is still running.

Chat requests include the latest vision context in the `/chat` payload, so the assistant can respond to non-sensitive visual context such as whether a user is visible, centered, looking toward the camera, or has a posture/attention alert.

## Vision API

Analyze one browser camera frame:

```bash
curl -X POST http://127.0.0.1:8000/vision/analyze ^
  -H "Content-Type: application/json" ^
  -d "{\"image_base64\":\"data:image/jpeg;base64,...\",\"include_decision\":true,\"context\":{}}"
```

Request body:

```json
{
  "image_base64": "data:image/jpeg;base64,...",
  "timestamp": "2026-04-27T10:30:00.000Z",
  "include_decision": true,
  "context": {}
}
```

Response includes:

- `face`
- `eyes_attention`
- `body_posture`
- `tracking`
- `health_behavior`
- `sensors`
- `decision`
- `overlays`
- `latency_ms`

Vision service status:

```bash
curl http://127.0.0.1:8000/vision/status
```

Track face position from one browser camera frame:

```bash
curl -X POST http://127.0.0.1:8000/vision/track ^
  -H "Content-Type: application/json" ^
  -d "{\"image_base64\":\"data:image/jpeg;base64,...\",\"context\":{}}"
```

The tracking endpoint also accepts frame aliases such as `imageBase64`, `frame_base64`, `frame`, `image`, `data_url`, and `dataUrl`.

## Existing API Endpoints

- `GET /api/v1/health`
- `GET /api/v1/system/info`
- `POST /stt`
- `POST /route`
- `POST /chat`
- `POST /tts`
- `POST /pipeline`
- `POST /memory/store`
- `POST /memory/retrieve`
- `POST /vision/analyze`
- `POST /vision/track`
- `GET /vision/status`

## Dependencies

Core dependencies are listed in `requirements.txt`.

Important packages:

- `fastapi`
- `uvicorn[standard]`
- `httpx`
- `python-dotenv`
- `numpy`
- `opencv-python-headless`
- `PyYAML`

OpenCV is used for JPEG decoding and Haar-cascade face detection. If OpenCV is unavailable, the vision service returns a safe mock response instead of crashing.

Optional future model hooks are available in `server/vision_analyzer.py` for apparent gender estimate, estimated age range, and expression. They currently return `unknown` unless a validated model is explicitly added.

## Privacy And Accuracy Notes

- The system does not identify real people by name.
- Apparent gender, age, and expression are visual estimates only and default to `unknown`.
- Low-confidence estimates are shown as unknown/uncertain.
- Face images are not stored by default.
- Vision output is not perfect and should not be used for high-stakes decisions without validated models and human review.

## Testing

```bash
python -m unittest discover -s tests -v
```

If the Windows Python launcher fails with a logon-session error, repair or recreate `.venv` and rerun the same command from the project root.
