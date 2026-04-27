# 🤖 AI Robot Project — Agent Prompt Plan (20 Steps)

---

# 🧠 GLOBAL INSTRUCTION (Give this ONCE before all steps)

```text
You are an expert AI systems engineer.

You are building a modular AI robot system based on the provided architecture.md file.

Rules:
- Follow the architecture strictly
- Keep modules decoupled
- Write clean, production-level Python code
- Use FastAPI for backend services
- Use async where needed
- Add comments and docstrings
- Do NOT overcomplicate
- Each module must be independently testable

Output:
- Code
- File structure
- Explanation (brief, not verbose)
```

---

# 🧩 PART 1 — PROJECT SETUP

```text
Create the full project folder structure based on architecture.md.

Include:
- edge/
- server/
- api/
- memory/
- llm/

Add:
- requirements.txt
- .env.example
- README.md

Ensure modular separation between edge and server.
```

---

# 🧩 PART 2 — FASTAPI SERVER BASE

```text
Build a FastAPI backend server.

Requirements:
- Async support
- Health check endpoint
- Modular routing system
- Logging enabled

Structure:
api/main.py
```

---

# 🧩 PART 3 — STT MODULE

```text
Implement Speech-to-Text module.

Requirements:
- Accept audio input (file/stream)
- Integrate Whisper (or placeholder API)
- Return clean text

File:
server/stt.py
```

---

# 🧩 PART 4 — TTS MODULE

```text
Implement Text-to-Speech module.

Requirements:
- Input: text
- Output: audio file/stream
- Pluggable backend (Coqui or API)

File:
server/tts.py
```

---

# 🧩 PART 5 — ROUTER (CORE LOGIC)

```text
Build intent router system.

Requirements:
- Classify input into:
  simple / medium / complex
- Use:
  - rule-based logic
  - scoring system
- Return:
  route decision (groq / deepseek)

File:
server/router.py
```

---

# 🧩 PART 6 — LLM CLIENT: GROQ

```text
Create Groq LLM client.

Requirements:
- Send prompt
- Handle response
- Support streaming
- Error handling

File:
server/llm/groq_client.py
```

---

# 🧩 PART 7 — LLM CLIENT: DEEPSEEK

```text
Create DeepSeek client.

Requirements:
- Used only for complex queries
- Clean interface similar to Groq client

File:
server/llm/deepseek_client.py
```

---

# 🧩 PART 8 — DECISION ENGINE

```text
Build decision engine.

Input:
- user text
- memory
- context

Output:
- response OR action

Combine:
- router
- LLM clients

File:
server/decision_engine.py
```

---

# 🧩 PART 9 — MEMORY SYSTEM

```text
Implement memory system.

Requirements:
- Store conversation logs (SQLite)
- Store summaries (vector DB or placeholder)
- Retrieve relevant context

File:
server/memory/vector_db.py
server/memory/logs.db
```

---

# 🧩 PART 10 — MEMORY SUMMARIZATION

```text
Add summarization logic.

Requirements:
- Extract important info
- Compress conversation
- Store only useful data

Use LLM or simple rules.
```

---

# 🧩 PART 11 — API ENDPOINTS

```text
Expose endpoints:

POST /stt
POST /route
POST /chat
POST /tts
POST /memory/store
POST /memory/retrieve

Ensure clean request/response models.
```

---

# 🧩 PART 12 — EDGE AUDIO MODULE

```text
Build Raspberry Pi audio input module.

Requirements:
- Record audio
- Send to server STT endpoint
- Receive response

File:
edge/audio_input.py
```

---

# 🧩 PART 13 — EDGE CAMERA MODULE

```text
Implement camera module.

Requirements:
- Capture frames
- Send frames to processing functions

File:
edge/camera.py
```

---

# 🧩 PART 14 — FACE DETECTION & TRACKING

```text
Add face detection + tracking.

Requirements:
- Detect face
- Track position
- Output coordinates

Use OpenCV / MediaPipe.

File:
edge/tracking.py
```

---

# 🧩 PART 15 — POSTURE DETECTION

```text
Implement posture detection.

Requirements:
- Sitting / standing detection
- Basic rule logic

File:
edge/posture.py
```

---

# 🧩 PART 16 — SENSOR MODULE

```text
Implement sensor system.

Requirements:
- Temperature
- Humidity
- Return structured data

File:
edge/sensors.py
```

---

# 🧩 PART 17 — REMINDER ENGINE

```text
Build reminder system.

Requirements:
- Medicine reminder
- Water reminder
- Sitting detection alert

Rule-based system.

File:
server/reminders.py
```

---

# 🧩 PART 18 — ACTION SYSTEM

```text
Implement action execution.

Examples:
- mark_medicine_taken
- log_event
- trigger_alert

Should integrate with decision engine.
```

---

# 🧩 PART 19 — FULL PIPELINE INTEGRATION

```text
Connect all modules.

Flow:
Audio → STT → Router → LLM → Action → TTS

Ensure:
- async flow
- error handling
- logging
```

---

# 🧩 PART 20 — OPTIMIZATION & TESTING

```text
Optimize system.

Include:
- caching responses
- reducing latency
- fallback handling

Add:
- test scripts
- example inputs
```

---

# 🧠 FINAL INSTRUCTION TO AGENT

```text
Follow steps sequentially.
Do NOT skip steps.
Each step must be complete before moving to next.

Ensure:
- clean architecture
- modular design
- production-ready code
```

---
