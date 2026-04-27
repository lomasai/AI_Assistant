# 🤖 AI Robot System — Architecture.md

---

# 📌 1. Overview

This system is a **hybrid, multimodal AI robot architecture** designed for:

* 🎙️ Voice interaction (STT + TTS)
* 👁️ Vision (face tracking, posture detection)
* 🧠 Intelligent reasoning (multi-LLM system)
* 🧠 Memory & context awareness
* 🔔 Health & behavior monitoring
* ⚙️ Real-world actions (reminders, tracking, control)

---

## 🧠 Design Philosophy

> **“Fast by default, intelligent on demand.”**

* ⚡ Use fast models for most tasks
* 🧠 Escalate to powerful reasoning only when needed
* 🧩 Keep system modular and distributed

---

# 🧭 2. High-Level Architecture

```text
                    ┌────────────────────────────┐
                    │         👤 USER            │
                    └────────────┬───────────────┘
                                 │
                🎤 Voice / 🎥 Camera / 👆 Touch Input
                                 │
        ┌────────────────────────▼────────────────────────┐
        │              🤖 RASPBERRY PI (EDGE)             │
        │-------------------------------------------------│
        │ • Wake Word Detection                           │
        │ • Audio Capture                                 │
        │ • Camera Input                                  │
        │ • Face Detection & Tracking                     │
        │ • Posture Detection                             │
        │ • Sensor Data (Temp / Humidity)                 │
        │ • Touch Interface                               │
        │ • Motor Control                                 │
        └────────────────────┬────────────────────────────┘
                             │
                             │  (Local API / WebSocket)
                             ▼
        ┌────────────────────────────────────────────────┐
        │              🧠 AI SERVER (LAPTOP)              │
        │------------------------------------------------│
        │ 🎙️ STT Engine (Speech → Text)                 │
        │ 🧠 Intent Router (Core Logic)                 │
        │ 🧠 Memory System (FAISS + Logs)               │
        │ 🧠 Decision Engine                            │
        │ 🤖 LLM Layer (Groq + DeepSeek)               │
        │ 🔊 TTS Engine (Text → Speech)                │
        └────────────────────┬───────────────────────────┘
                             │
                             ▼
                      🔊 Speaker Output
```

---

# 🔁 3. End-to-End Interaction Flow

```text
1. User speaks
2. Raspberry Pi captures audio
3. Audio → STT → text
4. Text → Intent Router
5. Router decides:
      ├── Simple → Groq
      ├── Medium → Groq
      └── Complex → DeepSeek
6. Model returns:
      ├── Response text
      └── Tool/Action call
7. Execute action (if any)
8. Response → TTS
9. Robot speaks
10. Store memory (if important)
```

---

# 🧠 4. Intelligent Routing System (CORE)

## 🎯 Goal

* Minimize latency
* Minimize cost
* Maximize intelligence

---

## 🧩 Routing Diagram

```text
                ┌───────────────┐
                │   User Input  │
                └──────┬────────┘
                       ▼
                ┌───────────────┐
                │ Intent Router │
                └──────┬────────┘
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
   Simple          Medium          Complex
   (Fast)          (Normal)        (Heavy)
    │                │                │
    ▼                ▼                ▼
  Groq            Groq           DeepSeek
```

---

## 🧠 Routing Decision Pipeline

```text
User Input
   ↓
[1] Rule-Based Filter (fast)
   ↓
[2] Complexity Scoring Engine
   ↓
[3] Confidence Check
   ↓
[4] (Optional) LLM Classifier
   ↓
Final Decision:
   → Groq
   → DeepSeek
```

---

## 🧮 Complexity Scoring Logic

```python
score = 0

if long_sentence: score += 1
if reasoning_keywords: score += 1
if multi_part_query: score += 1

if score >= 3 → complex (DeepSeek)
elif score == 2 → medium (Groq)
else → simple (Groq)
```

---

## ⚠️ Important Rules

* ❌ Never call multiple LLMs sequentially
* ✅ Always route to ONE model
* ✅ Use fallback only if needed

---

# 🧠 5. Memory Architecture

## Types

### 🔹 Short-Term Memory

* Last few interactions
* Stored in RAM

### 🔹 Long-Term Memory

* Important summaries only
* Stored in FAISS

---

## Memory Flow

```text
Conversation
   ↓
Filter Important Info
   ↓
Summarize (LLM)
   ↓
Store (Vector DB)
   ↓
Retrieve when needed
```

---

# 👁️ 6. Vision System (Edge - Pi)

```text
Camera Input
   │
   ├── Face Detection
   │       → Identify user
   │
   ├── Face Tracking
   │       → Control robot movement
   │
   └── Posture Detection
           → Sitting / Standing / Walking / Falling
```

---

# 🔔 7. Reminder & Health Engine

## Rule-Based System (No LLM)

```python
if sitting_time > 60:
    alert("Take a walk")

if water_gap > 2 hours:
    alert("Drink water")

if medicine_time:
    ask("Did you take medicine?")
```

---

## Confirmation Loop

```text
Robot: "Did you take medicine?"
User: "Yes"
→ Update state
→ Log event
→ Confirm to user
```

---

# 🧠 8. Decision Engine

Combines:

* Vision data
* Sensor data
* Time
* Memory
* User behavior

---

## Example Context

```json
{
  "posture": "sitting",
  "duration": "2 hours",
  "time": "11 PM",
  "medicine_taken": false
}
```

---

## Output

```text
"You're sitting too long. Let's stretch and take your medicine."
```

---

# ⚙️ 9. System Modules

---

## 🤖 Edge (Raspberry Pi)

* Camera processing
* Sensors
* Wake word
* Audio capture
* Motor control

---

## 🧠 Server (Laptop)

* STT engine
* Router
* LLMs
* Memory
* Decision engine
* TTS

---

# 🔗 10. Communication Layer

* Protocol: REST / WebSocket
* Network: Local WiFi

---

## Example APIs

```text
POST /stt
POST /route
POST /llm/groq
POST /llm/deepseek
POST /tts
POST /memory/store
POST /memory/retrieve
```

---

# ⚡ 11. Latency Optimization

### Techniques

* Use single-model routing
* Cache repeated queries
* Stream responses
* Limit context size
* Preload models

---

# 🧱 12. Project Structure

```text
project/
│
├── edge/
│   ├── camera.py
│   ├── sensors.py
│   ├── tracking.py
│   ├── audio_input.py
│
├── server/
│   ├── stt.py
│   ├── tts.py
│   ├── router.py
│   ├── decision_engine.py
│   ├── memory/
│   │   ├── vector_db.py
│   │   ├── logs.db
│   ├── llm/
│       ├── groq_client.py
│       ├── deepseek_client.py
│
├── api/
│   ├── main.py
│
└── architecture.md
```

---

# ⚠️ 13. Design Principles

### 1. Use AI only when needed

→ Rules handle 70% of logic

### 2. Keep edge lightweight

→ Real-time tasks only

### 3. Separate brain & body

→ Pi = execution, Server = intelligence

### 4. Optimize routing

→ Intelligence on demand

---

# 🚀 14. Future Enhancements

* Personalized behavior learning
* Voice cloning
* Multi-user support
* Edge LLM optimization
* Reinforcement learning

---

# 🧠 Final Insight

> This is not a single AI model
> This is a **distributed cognitive system with layered intelligence**

---