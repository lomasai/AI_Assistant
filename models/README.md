# Models

What is committed here, and what you have to fetch yourself.

Small models live in git so that a clone runs. Large ones do not: a piper
voice is 60–100 MB and git keeps every version of a binary forever, which
turns a clone on a school's connection into a bad afternoon.

## In this repository

| file | size | what it does | licence |
|---|---|---|---|
| `face_detection_yunet_2023mar.onnx` | 227 KB | face detection, `face.detector: yunet` | Apache-2.0, from the [OpenCV Zoo](https://github.com/opencv/opencv_zoo) |

Nothing needs downloading for detection, tracking, pose or attention to work.

## Fetch these yourself

### A voice — `models/piper/`

Without one the robot is silent. Voices come from
[rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices); take both
the `.onnx` and its `.onnx.json`.

```
models/piper/en_US-lessac-medium.onnx
models/piper/hi_IN-pratham-medium.onnx
```

The names are config, under `speech.tts.voice`, one per language. Adding a
language is a voice file and a content pack, never a code change.

### Face recognition — `models/`

Only needed to put names on faces. Detection, tracking and attention all work
without it. Needs `pip install onnxruntime`, then a MobileFaceNet or ArcFace
model at the path in `face.embedder_model_path`.

Leave `face.embedder: mock` until you have one — enrolment still runs end to
end, it simply will not reliably tell two people apart.

### Wake word — `models/wake/`

Nothing drives the wake engine yet, so this is not needed. When it is, it is
an openwakeword `.onnx` at `speech.wake.model_path`.
