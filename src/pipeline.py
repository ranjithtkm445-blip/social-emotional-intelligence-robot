# src/pipeline.py — orchestrates one full perceive -> reason -> act -> remember cycle
from typing import Optional

import numpy as np

from src import asr, emotion, explain, fusion, llm_reasoning, memory, robot_control, tts, vision

memory.init_db()


def run_once(
    user_id: str,
    frames: list[tuple[float, np.ndarray]],
    audio_path: Optional[str] = None,
    listen_duration: float = 5.0,
    speak_response: bool = True,
) -> dict:
    """Run one full interaction turn.

    frames: a list of (timestamp_seconds, BGR frame) samples across the interaction's
            duration, e.g. from video_input.extract_frame_and_audio. A single-element
            list (e.g. [(0.0, frame)]) is fine for a one-off photo.
    audio_path: path to a pre-recorded wav utterance. If omitted, records `listen_duration`
                seconds live from the microphone instead.
    """
    if audio_path is None:
        asr_result = asr.listen_and_transcribe(listen_duration)
    else:
        asr_result = asr.transcribe_audio(audio_path)
        asr_result["audio_path"] = audio_path

    representative_frame = frames[len(frames) // 2][1]
    vision_result = vision.analyze_frame(representative_frame)

    emotion_result = emotion.analyze(
        frames=frames,
        audio_path=asr_result.get("audio_path"),
        transcript=asr_result["transcript"],
        asr_confidence=asr_result["confidence"],
    )

    context = fusion.build_context(user_id, asr_result, vision_result, emotion_result)
    llm_output = llm_reasoning.generate_response(context)
    executed_action = robot_control.execute_action(llm_output["action"])

    if speak_response:
        tts.speak(llm_output["response_text"])

    memory.save_interaction(
        user_id=user_id,
        transcript=context["transcript"],
        face_emotion_label=emotion_result["face"]["label"],
        face_emotion_confidence=emotion_result["face"]["confidence"],
        voice_emotion_label=emotion_result["voice"]["label"],
        voice_emotion_confidence=emotion_result["voice"]["confidence"],
        response_text=llm_output["response_text"],
        action_type=llm_output["action"]["type"],
    )

    trace = explain.log_decision(context, llm_output, executed_action)

    return {
        "context": context,
        "llm_output": llm_output,
        "executed_action": executed_action,
        "trace": trace,
    }
