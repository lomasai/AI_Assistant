# AI Robot Assistant

FastAPI-based AI robot assistant with chat, voice input, memory, text-to-speech placeholders, and a browser-to-backend live camera vision pipeline.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Fill `.env` with your provider keys if you want live LLM calls.

## Run

```bash
uvicorn api.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

The browser will ask for camera and microphone permissions when you start those features. Camera access generally works on `localhost` / `127.0.0.1`; if permission is denied, enable camera access in the browser site settings and try again.

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
