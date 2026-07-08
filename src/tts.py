# src/tts.py — text-to-speech output (pyttsx3 offline engine, or Piper via config flag)
import subprocess
import time
from pathlib import Path

import config

def _speak_pyttsx3(text: str) -> None:
    # A fresh engine per call, rather than one cached across the process: pyttsx3's
    # sapi5 driver raises "run loop already started" if runAndWait() is invoked again
    # on an engine whose previous run loop didn't fully tear down (e.g. two overlapping
    # calls in a long-lived server process like Streamlit).
    import pyttsx3
    engine = pyttsx3.init()
    try:
        engine.setProperty("rate", config.TTS_RATE)
        engine.say(text)
        engine.runAndWait()
    finally:
        engine.stop()


def _speak_piper(text: str) -> Path:
    """Synthesize speech with a local Piper ONNX voice model and play it back.

    Requires `piper-tts` installed and a voice model downloaded to config.PIPER_MODEL_PATH.
    """
    out_path = config.AUDIO_TMP_DIR / f"tts_{int(time.time() * 1000)}.wav"
    subprocess.run(
        [
            "piper",
            "--model", config.PIPER_MODEL_PATH,
            "--output_file", str(out_path),
        ],
        input=text.encode("utf-8"),
        check=True,
    )
    import sounddevice as sd
    import soundfile as sf
    data, samplerate = sf.read(out_path)
    sd.play(data, samplerate)
    sd.wait()
    return out_path


def speak(text: str) -> None:
    """Speak `text` aloud using the configured TTS backend (config.TTS_ENGINE).

    Playback failures (e.g. no audio output device — the normal case on a headless
    cloud host like a Hugging Face Space) are logged and swallowed rather than raised,
    since the robot's spoken reply is a nice-to-have alongside the text response, not
    something that should crash the whole interaction turn.
    """
    if not text:
        return
    try:
        if config.TTS_ENGINE == "piper":
            _speak_piper(text)
        else:
            _speak_pyttsx3(text)
    except Exception as e:
        print(f"[tts] speech playback failed, continuing without audio: {e}")
