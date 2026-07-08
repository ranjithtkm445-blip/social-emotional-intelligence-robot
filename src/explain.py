# src/explain.py — explainable AI: records why the robot decided what it decided
#
# The reasoning engine here is a pretrained LLM rather than a model we train ourselves, so
# "explainability" means making the full decision trace auditable: exactly which perceptual
# signals (transcript, emotion scores, scene contents, memory) were fed in, and what the LLM
# and action dispatcher produced from them. This satisfies the "Explainable AI for robot
# decision-making" contribution without needing SHAP/GradCAM, which apply to models we'd train.
import json
import time

import config


def log_decision(context: dict, llm_output: dict, executed_action: dict) -> dict:
    """Persist one full decision trace to a JSONL log and return it for UI display."""
    trace = {
        "timestamp": time.time(),
        "inputs": {
            "transcript": context["transcript"],
            "asr_confidence": context["asr_confidence"],
            "emotion": context["emotion"],
            "scene_summary": context["scene_summary"],
            "objects": context["objects"],
            "gestures": context["gestures"],
            "preferences": context["preferences"],
        },
        "llm_reasoning": {
            "emotion_acknowledged": llm_output.get("emotion_acknowledged", ""),
            "response_text": llm_output.get("response_text", ""),
            "planned_action": llm_output.get("action", {}),
            "raw_llm_output": llm_output.get("raw_llm_output", ""),
        },
        "executed_action": executed_action,
    }
    with open(config.DECISION_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(trace) + "\n")
    return trace


def format_trace_summary(trace: dict) -> str:
    """Human-readable rationale for why the robot responded/acted the way it did."""
    inputs = trace["inputs"]
    reasoning = trace["llm_reasoning"]
    face = inputs["emotion"]["face"]
    voice = inputs["emotion"]["voice"]
    meta = inputs["emotion"]["meta_states"]
    return (
        f"Heard: \"{inputs['transcript']}\" (ASR confidence {inputs['asr_confidence']:.2f})\n"
        f"Facial expression over time: {face['timeline_summary']}\n"
        f"Voice tone over time: {voice['timeline_summary']}\n"
        f"Frustrated: {meta['frustrated']}, Confused: {meta['confused']}\n"
        f"Scene: {inputs['scene_summary']}\n"
        f"Because of this, the robot acknowledged: \"{reasoning['emotion_acknowledged']}\"\n"
        f"-> said: \"{reasoning['response_text']}\"\n"
        f"-> action taken: {reasoning['planned_action'].get('type')} "
        f"({reasoning['planned_action'].get('params')}) -> status: {trace['executed_action'].get('status')}"
    )


def read_recent_traces(n: int = 10) -> list[dict]:
    if not config.DECISION_LOG_PATH.exists():
        return []
    with open(config.DECISION_LOG_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()[-n:]
    return [json.loads(line) for line in lines]
