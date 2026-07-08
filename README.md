Here's a simpler, cleaner version:

````
# Emotionally Intelligent Social Robot Assistant

A multimodal AI assistant that understands how a person looks and sounds — not just what they say — and responds like an empathetic robot would.

It takes a video (or an image + audio), figures out:
- What the person said (speech-to-text)
- How their face looked over time (happy, sad, angry, etc.)
- How their voice sounded over time (separately from the face)

Then a large language model reasons over all of that and generates a caring, context-aware reply — plus a simulated robot action (speak, navigate, fetch an object, alert someone, etc.).

## Key Idea

Most emotion AI just gives one label like "Happy" for a whole video. This project does two things differently:

1. **Tracks emotion second-by-second**, not just once. A person can be calm for 5 seconds and upset for the next 5 — the app shows both.
2. **Keeps face and voice separate.** They're never merged into one blended score. If your face looks calm but your voice sounds upset, the app tells you that directly instead of picking one.

## How It Works

```
Video/Image + Audio
        |
        ├── Speech-to-text (Whisper)
        ├── Face emotion, tracked every second (FER)
        └── Voice emotion, tracked every second (YAMNet)
                |
        Groq LLM reads everything and responds
                |
        Robot speaks + takes a simulated action
                |
        Saved to memory (SQLite) for next time
```

## Project Files

| File | What it does |
|---|---|
| `app.py` | The Streamlit web app |
| `src/pipeline.py` | Runs the whole process end to end |
| `src/video_input.py` | Splits a video into frames + audio |
| `src/asr.py` | Speech-to-text (Whisper) |
| `src/emotion.py` | Face + voice emotion detection |
| `src/llm_reasoning.py` | Talks to the Groq LLM |
| `src/tts.py` | Makes the robot speak |
| `src/robot_control.py` | Simulated robot actions |
| `src/memory.py` | Remembers past conversations |
| `src/explain.py` | Logs why the robot responded the way it did |
| `src/vision.py` | Object/gesture detection (built but turned off for now) |

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Then add your free Groq API key (from console.groq.com) to the `.env` file.

## Run

```powershell
streamlit run app.py
```

Upload a video (or an image + audio file), click **Respond**, and see:
- The transcript
- Face emotion over time
- Voice emotion over time
- The robot's reply and action
- A full explanation of why it responded that way

## Good to Know

- This is a heavy setup — it uses TensorFlow, PyTorch, and a few large models. First run will download things and take a while.
- Object detection and gesture recognition are built but turned off by default — this version focuses on emotion, not full scene understanding.
- Robot actions are **simulated only** — no real robot is controlled. Swapping in real ROS2 code is straightforward (see `src/robot_control.py`).
- Runs best locally. Free-tier cloud hosting (like Hugging Face Spaces) may not have enough resources for this dependency stack.

## License

MIT
````

Want me to save this simplified version over the current `README.md`, or keep both (a simple one + the detailed technical one)?
