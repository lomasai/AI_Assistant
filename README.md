# AI Robot System

Modular AI robot stack based on `architecture.md`.

## Structure

```text
.
├── api/
│   └── main.py
├── edge/
│   ├── audio_input.py
│   ├── camera.py
│   ├── posture.py
│   ├── sensors.py
│   └── tracking.py
├── server/
│   ├── actions.py
│   ├── cache.py
│   ├── decision_engine.py
│   ├── pipeline.py
│   ├── reminders.py
│   ├── router.py
│   ├── stt.py
│   ├── tts.py
│   ├── llm/
│   │   ├── deepseek_client.py
│   │   └── groq_client.py
│   └── memory/
│       ├── logs.db
│       └── vector_db.py
├── examples/inputs/
├── tests/
├── scripts/run_tests.sh
├── .env.example
├── architecture.md
├── README.md
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Run Commands

Start API server:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Health check:

```bash
curl -s http://127.0.0.1:8000/api/v1/health
```

OpenAPI docs:

```text
http://127.0.0.1:8000/docs
```

## API Commands (All Endpoints)

STT:

```bash
curl -s -X POST http://127.0.0.1:8000/stt \
  -H "Content-Type: application/json" \
  -d @examples/inputs/stt_request.json
```

Route:

```bash
curl -s -X POST http://127.0.0.1:8000/route \
  -H "Content-Type: application/json" \
  -d @examples/inputs/route_request.json
```

Chat:

```bash
curl -s -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d @examples/inputs/chat_request.json
```

TTS:

```bash
curl -s -X POST http://127.0.0.1:8000/tts \
  -H "Content-Type: application/json" \
  -d @examples/inputs/tts_request.json
```

Memory Store:

```bash
curl -s -X POST http://127.0.0.1:8000/memory/store \
  -H "Content-Type: application/json" \
  -d @examples/inputs/memory_store_request.json
```

Memory Retrieve:

```bash
curl -s -X POST http://127.0.0.1:8000/memory/retrieve \
  -H "Content-Type: application/json" \
  -d @examples/inputs/memory_retrieve_request.json
```

Full Pipeline (Audio -> STT -> Router -> LLM -> Action -> TTS):

```bash
curl -s -X POST http://127.0.0.1:8000/pipeline \
  -H "Content-Type: application/json" \
  -d @examples/inputs/pipeline_request.json
```

## Optimization

- Async end-to-end pipeline orchestration in `server/pipeline.py`
- In-memory TTL caches for STT, decision, and TTS stages
- TTS fallback to text-only response when `PIPELINE_TTS_FAIL_HARD=false`
- Cache and warning flags returned in pipeline response

Relevant env vars in `.env.example`:

- `PIPELINE_CACHE_ENABLED`
- `PIPELINE_CACHE_MAX_ITEMS`
- `PIPELINE_STT_CACHE_TTL_SECONDS`
- `PIPELINE_DECISION_CACHE_TTL_SECONDS`
- `PIPELINE_TTS_CACHE_TTL_SECONDS`
- `PIPELINE_TTS_FAIL_HARD`

## Testing

Run full tests:

```bash
./scripts/run_tests.sh
```

Or:

```bash
python3 -m unittest discover -s tests -v
```
