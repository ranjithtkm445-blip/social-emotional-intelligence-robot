# src/asr.py — speech capture and transcription via Whisper (faster-whisper)
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

import config

_model: Optional[WhisperModel] = None


def load_model() -> WhisperModel:
    """Lazily load the Whisper model so import time stays fast."""
    global _model
    if _model is None:
        _model = WhisperModel(
            config.WHISPER_MODEL_SIZE,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )
    return _model


def record_audio(duration_seconds: float = 5.0) -> Path:
    """Record `duration_seconds` of audio from the default microphone and save it to a wav file."""
    import sounddevice as sd  # lazy: querying audio hardware can hang on a headless host

    frames = sd.rec(
        int(duration_seconds * config.SAMPLE_RATE),
        samplerate=config.SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    out_path = config.AUDIO_TMP_DIR / f"utterance_{int(time.time() * 1000)}.wav"
    sf.write(out_path, frames, config.SAMPLE_RATE)
    return out_path


def transcribe_audio(audio_path: Path) -> dict:
    """Transcribe a wav file to text. Returns transcript, language, and a mean confidence score."""
    model = load_model()
    segments, info = model.transcribe(str(audio_path), beam_size=5)
    segments = list(segments)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    if segments:
        confidence = float(np.mean([np.exp(seg.avg_logprob) for seg in segments]))
    else:
        confidence = 0.0
    return {
        "transcript": text,
        "language": info.language,
        "confidence": confidence,
    }


def listen_and_transcribe(duration_seconds: float = 5.0) -> dict:
    """Convenience wrapper: record from the mic, then transcribe."""
    audio_path = record_audio(duration_seconds)
    result = transcribe_audio(audio_path)
    result["audio_path"] = str(audio_path)
    return result
