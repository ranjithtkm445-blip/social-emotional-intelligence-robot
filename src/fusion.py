# src/fusion.py — multimodal fusion: combine speech, vision, emotion, and memory into one context object
import time

from src import memory


def build_context(
    user_id: str,
    asr_result: dict,
    vision_result: dict,
    emotion_result: dict,
) -> dict:
    """Fuse all per-modality outputs plus retrieved memory into a single structured context.

    This is the object that gets serialized into the LLM prompt in llm_reasoning.py and
    persisted (minus large fields) by explain.py for auditability.
    """
    profile = memory.get_user_profile(user_id)

    return {
        "timestamp": time.time(),
        "user_id": user_id,
        "transcript": asr_result.get("transcript", ""),
        "asr_confidence": asr_result.get("confidence", 0.0),
        "language": asr_result.get("language", "en"),
        "emotion": emotion_result,
        "objects": [o["label"] for o in vision_result.get("objects", [])],
        "num_faces": len(vision_result.get("faces", [])),
        "gestures": vision_result.get("gestures", []),
        "scene_summary": vision_result.get("scene_summary", ""),
        "preferences": profile["preferences"],
        "recent_history": profile["recent_history"],
    }
