# src/emotion.py — facial + vocal emotion recognition, reported as two separate signals
import csv
from typing import Optional

import numpy as np
import soundfile as sf

import config

_face_detector = None
_yamnet_model = None
_yamnet_class_names: Optional[list[str]] = None

# Words that hint at the two "meta" emotional states not directly produced by the
# facial/voice recognizers but called for in the project spec (Frustrated, Confused).
_CONFUSION_MARKERS = {"what", "huh", "confused", "don't understand", "do not understand", "unclear", "sorry what"}
_FRUSTRATION_MARKERS = {"again", "still not", "not working", "ugh", "frustrated", "annoying", "come on", "seriously"}

_CORE_EMOTIONS = ["happy", "sad", "angry", "fear", "surprise", "neutral"]


def _load_face_detector():
    global _face_detector
    if _face_detector is None:
        from fer import FER
        _face_detector = FER(mtcnn=True)
    return _face_detector


def recognize_facial_emotion(frame: np.ndarray) -> dict[str, float]:
    """Return a probability distribution over the 7 FER emotion classes for the given BGR frame.

    Returns an empty dict if no face is detected.
    """
    detector = _load_face_detector()
    results = detector.detect_emotions(frame)
    if not results:
        return {}
    # Use the largest detected face (most likely the primary conversational partner).
    largest = max(results, key=lambda r: r["box"][2] * r["box"][3])
    return {k: float(v) for k, v in largest["emotions"].items()}


def recognize_facial_emotion_timeline(frames: list[tuple[float, np.ndarray]]) -> list[dict]:
    """Classify facial emotion in each sampled frame across the clip's duration —
    mirroring recognize_voice_emotion_timeline, since an expression can shift within a
    clip just as vocal tone can; one static frame isn't representative of the whole thing.
    Frames with no detected face are silently skipped. Returns a list of
    {"start": s, "end": s, "scores": {emotion: score}} windows."""
    interval = frames[1][0] - frames[0][0] if len(frames) > 1 else config.FACE_EMOTION_SAMPLE_INTERVAL_SECONDS

    timeline = []
    for t, frame in frames:
        probs = recognize_facial_emotion(frame)
        if not probs:
            continue
        core_probs = {k: v for k, v in probs.items() if k in _CORE_EMOTIONS or k == "disgust"}
        timeline.append({"start": round(t, 2), "end": round(t + interval, 2), "scores": _normalize(core_probs)})
    return timeline


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    total = sum(scores.values())
    if total <= 0:
        return {k: 0.0 for k in _CORE_EMOTIONS}
    return {k: scores.get(k, 0.0) / total for k in _CORE_EMOTIONS}


def _load_yamnet():
    global _yamnet_model, _yamnet_class_names
    if _yamnet_model is None:
        import tensorflow_hub as hub
        _yamnet_model = hub.load(config.YAMNET_MODEL_URL)
        class_map_path = _yamnet_model.class_map_path().numpy().decode("utf-8")
        with open(class_map_path, encoding="utf-8") as f:
            _yamnet_class_names = [row["display_name"] for row in csv.DictReader(f)]
    return _yamnet_model, _yamnet_class_names


# Map specific AudioSet event classes YAMNet recognizes directly onto our core emotions.
# This is direct event recognition ("this sound is crying") rather than inferring an
# abstract arousal/valence position and hoping it lands near the right prototype — which
# is exactly where prior approaches failed (a real scream and real baby-crying audio were
# both read as "happy" by two different arousal/valence-style models, since loud/high-
# energy vocalization was conflated with positive valence in both).
_YAMNET_EVENT_TO_EMOTION = {
    "Baby cry, infant cry": "sad",
    "Crying, sobbing": "sad",
    "Whimper": "sad",
    "Sigh": "sad",
    "Screaming": "angry",
    "Shout": "angry",
    "Yell": "angry",
    "Bellow": "angry",
    "Battle cry": "angry",
    "Children shouting": "angry",
    "Laughter": "happy",
    "Baby laughter": "happy",
    "Giggle": "happy",
    "Chuckle, chortle": "happy",
    "Belly laugh": "happy",
    "Snicker": "happy",
    "Gasp": "surprise",
    "Speech": "neutral",
    "Conversation": "neutral",
    "Narration, monologue": "neutral",
    "Babbling": "neutral",
    "Child speech, kid speaking": "neutral",
}


def _classify_audio_chunk(chunk: np.ndarray) -> dict[str, float]:
    """Run YAMNet on a chunk (expects float32 mono @ 16kHz) and map its recognized sound
    events onto our core emotion labels."""
    model, class_names = _load_yamnet()
    scores, _embeddings, _spectrogram = model(chunk)
    mean_scores = np.mean(scores.numpy(), axis=0)

    emotion_scores = {k: 0.0 for k in _CORE_EMOTIONS}
    for idx, score in enumerate(mean_scores):
        emotion = _YAMNET_EVENT_TO_EMOTION.get(class_names[idx])
        if emotion:
            emotion_scores[emotion] += float(score)
    return _normalize(emotion_scores)


def recognize_voice_emotion_timeline(audio_path: str) -> list[dict]:
    """Classify vocal emotion in successive windows across the whole clip's duration.

    A speaker's tone can shift within a single utterance (e.g. calm, then a sudden
    scream, then calm again) — collapsing the whole clip into one score averages that
    away. Returns a list of {"start": s, "end": s, "scores": {emotion: score}} windows.
    """
    if not config.ENABLE_VOICE_EMOTION:
        return []
    audio_array, native_rate = sf.read(audio_path, dtype="float32")
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    if native_rate != config.SAMPLE_RATE:
        import librosa
        audio_array = librosa.resample(audio_array, orig_sr=native_rate, target_sr=config.SAMPLE_RATE)

    window = int(config.VOICE_EMOTION_WINDOW_SECONDS * config.SAMPLE_RATE)
    if window <= 0 or len(audio_array) == 0:
        return []
    num_windows = max(1, len(audio_array) // window)

    timeline = []
    for i in range(num_windows):
        chunk = audio_array[i * window:(i + 1) * window]
        if len(chunk) < window * 0.5 and i > 0:
            continue  # trailing sliver too short to classify meaningfully
        timeline.append({
            "start": round(i * config.VOICE_EMOTION_WINDOW_SECONDS, 2),
            "end": round((i + 1) * config.VOICE_EMOTION_WINDOW_SECONDS, 2),
            "scores": _classify_audio_chunk(chunk),
        })
    return timeline


def summarize_emotion_timeline(timeline: list[dict], empty_label: str = "no timeline available") -> str:
    """Collapse a per-window emotion timeline (voice or face) into a short narrative, e.g.
    '0.0-9.0s neutral, 9.0-14.0s angry, 14.0-19.0s happy', so the LLM and the UI can see
    how tone/expression moved over the clip instead of one static label."""
    if not timeline:
        return empty_label
    segments = []
    current_label = max(timeline[0]["scores"], key=timeline[0]["scores"].get)
    seg_start = timeline[0]["start"]
    for window in timeline[1:]:
        label = max(window["scores"], key=window["scores"].get)
        if label != current_label:
            segments.append(f"{seg_start:.1f}-{window['start']:.1f}s {current_label}")
            current_label = label
            seg_start = window["start"]
    segments.append(f"{seg_start:.1f}-{timeline[-1]['end']:.1f}s {current_label}")
    return ", ".join(segments)


def _per_window_labels(timeline: list[dict]) -> list[str]:
    """Which emotion is the top prediction for each window — a simple per-window argmax,
    with no magnitude/confidence comparison between different emotions or across windows."""
    return [max(window["scores"], key=window["scores"].get) for window in timeline]


def _select_headline(timeline: list[dict]) -> tuple[str, float]:
    """Pick the headline label by which emotion wins the most seconds, categorically —
    never by comparing confidence magnitudes between different emotions.

    Comparing scores like neutral=0.998 vs. sad=0.97 doesn't tell you which one actually
    mattered in the video — it's an artifact of how confident the model happens to be in
    each moment, not of which emotion is more significant. Instead: each window already has
    a winner (its own top prediction); count how many seconds each emotion won, and prefer
    whichever non-neutral emotion won the most — neutral is the default/baseline state, so
    it doesn't get to win just because it edged out a real event by a fraction of a point.
    The reported "confidence" is purely informational (how many of the clip's seconds that
    label won) — it does not participate in picking the label.
    """
    if not timeline:
        return "Unknown", 0.0
    labels = _per_window_labels(timeline)
    counts = {label: labels.count(label) for label in set(labels)}
    non_neutral = {k: v for k, v in counts.items() if k != "neutral"}
    label_key = max(non_neutral, key=non_neutral.get) if non_neutral else "neutral"
    seconds_won = counts.get(label_key, 0)
    return label_key.capitalize(), round(seconds_won / len(timeline), 3)


def _detect_meta_states(
    face_timeline: list[dict], voice_timeline: list[dict], transcript: str, asr_confidence: float
) -> dict[str, bool]:
    """Frustrated/Confused aren't produced directly by either recognizer — flag them from
    transcript wording plus whether either modality's per-window winner (categorically, not
    by score magnitude) was ever angry/sad."""
    text = (transcript or "").lower()
    negative_present = any(label in ("angry", "sad") for label in _per_window_labels(face_timeline)) or any(
        label in ("angry", "sad") for label in _per_window_labels(voice_timeline)
    )

    frustrated = any(marker in text for marker in _FRUSTRATION_MARKERS) and negative_present
    confused = any(marker in text for marker in _CONFUSION_MARKERS) or asr_confidence < 0.4
    return {"frustrated": frustrated, "confused": confused}


def analyze(
    frames: list[tuple[float, np.ndarray]],
    audio_path: Optional[str],
    transcript: str = "",
    asr_confidence: float = 1.0,
) -> dict:
    """Facial recognition + voice recognition, each tracked as its own per-second timeline
    across the whole clip's duration and reported fully separately — no blended "detected
    emotion" score, so disagreement between the two stays visible instead of being hidden
    behind fixed fusion weights. Each modality's headline label/confidence reflect the most
    significant moment anywhere in the clip (not just the last instant); the full timeline
    is kept alongside it so the LLM can still reason about when things happened.

    frames: a list of (timestamp_seconds, BGR frame) samples; a single-element list
            (e.g. [(0.0, frame)]) is fine for a one-off photo with no duration.
    """
    face_timeline = recognize_facial_emotion_timeline(frames) if frames else []
    face_label, face_confidence = _select_headline(face_timeline)

    voice_timeline = recognize_voice_emotion_timeline(audio_path) if audio_path else []
    voice_label, voice_confidence = _select_headline(voice_timeline)

    meta_states = _detect_meta_states(face_timeline, voice_timeline, transcript, asr_confidence)

    return {
        "face": {
            "label": face_label,
            "confidence": face_confidence,
            "timeline": face_timeline,
            "timeline_summary": summarize_emotion_timeline(face_timeline, "no face timeline available"),
        },
        "voice": {
            "label": voice_label,
            "confidence": voice_confidence,
            "timeline": voice_timeline,
            "timeline_summary": summarize_emotion_timeline(voice_timeline, "no voice timeline available"),
        },
        "meta_states": meta_states,
    }
