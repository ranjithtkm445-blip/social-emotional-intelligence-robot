# config.py — central configuration: paths, seeds, model names, feature flags
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

RANDOM_SEED = 42

# --- Paths -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MEMORY_DB_PATH = DATA_DIR / "memory" / "memory.db"
LOG_DIR = BASE_DIR / "logs"
DECISION_LOG_PATH = LOG_DIR / "decision_trace.jsonl"
ROBOT_ACTION_LOG_PATH = LOG_DIR / "robot_actions.jsonl"
AUDIO_TMP_DIR = BASE_DIR / "data" / "tmp_audio"
VIDEO_TMP_DIR = BASE_DIR / "data" / "tmp_video"

for p in (DATA_DIR, MEMORY_DB_PATH.parent, LOG_DIR, AUDIO_TMP_DIR, VIDEO_TMP_DIR):
    p.mkdir(parents=True, exist_ok=True)

# --- Speech recognition (Whisper) --------------------------------------
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")  # tiny/base/small/medium
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
SAMPLE_RATE = 16000

# --- Vision ---------------------------------------------------------------
# Face detection always runs (needed for facial emotion recognition from the photo).
# Object detection and gesture recognition are full environment/video perception —
# deferred until continuous video capture is added, so they default off for now.
ENABLE_OBJECT_DETECTION = os.getenv("ENABLE_OBJECT_DETECTION", "false").lower() == "true"
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolo11n.pt")
YOLO_CONF_THRESHOLD = 0.35
FACE_CASCADE_PATH = os.getenv(
    "FACE_CASCADE_PATH",
    str(Path(os.getenv("CV2_DATA_PATH", "")) / "haarcascade_frontalface_default.xml")
    if os.getenv("CV2_DATA_PATH")
    else "haarcascade_frontalface_default.xml",
)

# --- Gesture recognition (MediaPipe Hands) — deferred to the video phase, off for now
ENABLE_GESTURE_RECOGNITION = os.getenv("ENABLE_GESTURE_RECOGNITION", "false").lower() == "true"

# Facial emotion is sampled once per this many seconds across the clip's duration (not
# one static frame for the whole thing) — an expression can shift mid-clip just like a
# voice's tone can. See recognize_facial_emotion_timeline in src/emotion.py.
FACE_EMOTION_SAMPLE_INTERVAL_SECONDS = 1.0

# --- Emotion recognition -------------------------------------------------
# Facial and voice emotion are reported as two fully separate, independent readings —
# not blended into one fused score — so disagreement between them stays visible instead
# of being hidden behind fixed weights (the LLM decides how to reconcile them, if at all).
ENABLE_VOICE_EMOTION = os.getenv("ENABLE_VOICE_EMOTION", "true").lower() == "true"
# YAMNet (Google, AudioSet-trained) recognizes the sound *event* itself (e.g. "Crying,
# sobbing", "Screaming", "Laughter") rather than inferring an abstract arousal/valence
# position — direct event recognition sidesteps the failure mode of prior attempts
# (a dimensional wav2vec2 model, and before that a 4-class classifier), both of which
# conflated loud/high-arousal vocalization (screaming, crying) with "happy".
YAMNET_MODEL_URL = os.getenv("YAMNET_MODEL_URL", "https://tfhub.dev/google/yamnet/1")
# Voice emotion is classified in successive windows of this length across the whole
# clip's duration (not one score for the whole clip) — a speaker's tone can shift within
# a single utterance. See recognize_voice_emotion_timeline in src/emotion.py.
VOICE_EMOTION_WINDOW_SECONDS = 1.0
FACE_EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
FINAL_EMOTION_LABELS = [
    "Happy", "Sad", "Angry", "Fear", "Surprise", "Neutral", "Frustrated", "Confused",
]

# --- LLM reasoning (Groq-hosted Llama 3 / Qwen) --------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
LLM_TEMPERATURE = 0.6
LLM_MAX_TOKENS = 500
CONVERSATION_HISTORY_TURNS = 6  # how many past turns to feed back into the prompt

# --- Text-to-speech -------------------------------------------------------
TTS_ENGINE = os.getenv("TTS_ENGINE", "pyttsx3")  # "pyttsx3" or "piper"
PIPER_MODEL_PATH = os.getenv("PIPER_MODEL_PATH", str(BASE_DIR / "models" / "en_US-lessac-medium.onnx"))
TTS_RATE = 175

# --- Robot control (simulated) --------------------------------------------
ROBOT_MODE = os.getenv("ROBOT_MODE", "simulated")  # "simulated" or "ros2"

# --- Memory -----------------------------------------------------------------
DEFAULT_USER_ID = "default_user"
