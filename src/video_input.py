# src/video_input.py — split an uploaded video clip into a still frame (for facial emotion)
# and an audio track (for ASR + voice emotion); also standardizes standalone audio uploads
import time

import cv2
import numpy as np
from moviepy.editor import AudioFileClip, VideoFileClip

import config


def extract_frame_and_audio(video_path: str) -> dict:
    """Given a path to a video file, return the audio track plus frames sampled once every
    config.FACE_EMOTION_SAMPLE_INTERVAL_SECONDS across the whole duration (not just one
    static frame) — a facial expression can shift over a clip just like a voice's tone can.

    Returns {"frame": single representative BGR ndarray (for preview display),
             "frames": [(timestamp_seconds, BGR ndarray), ...] across the full duration,
             "audio_path": str or None}.
    """
    clip = VideoFileClip(str(video_path))
    try:
        interval = config.FACE_EMOTION_SAMPLE_INTERVAL_SECONDS
        num_samples = max(1, int(clip.duration // interval))
        sample_times = [min(i * interval, max(clip.duration - 0.01, 0)) for i in range(num_samples)]

        frames = []
        for t in sample_times:
            rgb_frame = clip.get_frame(t)
            bgr_frame = cv2.cvtColor(rgb_frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
            frames.append((round(t, 2), bgr_frame))

        audio_path = None
        if clip.audio is not None:
            audio_path = config.VIDEO_TMP_DIR / f"video_audio_{int(time.time() * 1000)}.wav"
            clip.audio.write_audiofile(str(audio_path), fps=config.SAMPLE_RATE, logger=None)
    finally:
        clip.close()

    representative_frame = frames[len(frames) // 2][1]
    return {
        "frame": representative_frame,
        "frames": frames,
        "audio_path": str(audio_path) if audio_path else None,
    }


def standardize_audio(audio_path: str) -> str:
    """Convert an arbitrary audio file (mp3/m4a/ogg/etc.) into the 16kHz wav format
    the ASR and voice-emotion modules expect. Returns the path to the converted wav."""
    clip = AudioFileClip(str(audio_path))
    try:
        out_path = config.AUDIO_TMP_DIR / f"audio_{int(time.time() * 1000)}.wav"
        clip.write_audiofile(str(out_path), fps=config.SAMPLE_RATE, logger=None)
    finally:
        clip.close()
    return str(out_path)
