# src/llm_reasoning.py — contextual reasoning engine: Groq-hosted Llama 3 / Qwen
import json
from typing import Optional

from groq import Groq

import config

_client: Optional[Groq] = None

_VALID_ACTIONS = {"speak", "navigate", "fetch_object", "display_reminder", "call_for_assistance"}

_SYSTEM_PROMPT = """You are the reasoning engine of an emotionally intelligent social robot assistant.
You receive the user's spoken transcript, two SEPARATE and INDEPENDENT emotion readings
(facial expression and voice tone — deliberately not blended into one score, so you can
see when they disagree), what the robot currently sees (objects, people, gestures), the
user's stored preferences, and recent conversation history. Use all of this to produce an
empathetic, context-aware reply and, when useful, plan one concrete robot action.

Respond with ONLY a JSON object of this exact shape:
{
  "response_text": "<what the robot should say out loud, warm and emotion-aware>",
  "emotion_acknowledged": "<how you factored in the user's emotional state, one short phrase>",
  "action": {
    "type": "<one of: speak, navigate, fetch_object, display_reminder, call_for_assistance>",
    "params": {"<key>": "<value>"}
  }
}

Guidelines:
- Use "call_for_assistance" if the user expresses distress, an emergency, or a medical concern.
- Use "navigate" or "fetch_object" only when the user's request clearly implies physical movement or an object.
- Use "display_reminder" for reminders/schedules the user asks to be shown or repeated later.
- Otherwise use "speak" with empty params.
- Keep response_text concise (1-3 sentences) and natural to say aloud.

Facial expression and voice tone are each given as a timeline of plain segments, e.g.
"0.0-11.0s neutral, 12.0-19.0s angry" — that IS the emotional account, read it literally
and state it plainly:
- Do not talk about confidence, percentages, or scores — there aren't any to weigh, just
  segments and the emotion each one is.
- If a timeline has more than one segment, say what happened in order — e.g. "you were
  neutral, then you turned angry" — and react to the change itself, not just the last
  segment.
- If facial expression and voice tone disagree, say so plainly — e.g. "your face looks
  calm, but your voice sounds upset."
- Then decide what to do about it — e.g. if someone turned angry, your job is to help
  calm them down, not just narrate that they're angry.
"""


def _load_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client


def _format_history(history: list[dict]) -> str:
    if not history:
        return "(no prior conversation)"
    lines = []
    for turn in history:
        lines.append(
            f"- User said: \"{turn['transcript']}\" "
            f"(face: {turn['face_emotion_label']}, voice: {turn['voice_emotion_label']}) "
            f"-> Robot: \"{turn['response_text']}\""
        )
    return "\n".join(lines)


def _build_user_prompt(context: dict) -> str:
    face = context["emotion"]["face"]
    voice = context["emotion"]["voice"]
    return f"""Transcript: "{context['transcript']}"
Facial expression over the course of the clip: {face['timeline_summary']}
Voice tone over the course of speaking: {voice['timeline_summary']}
Frustrated: {context['emotion']['meta_states']['frustrated']}, Confused: {context['emotion']['meta_states']['confused']}
Scene: {context['scene_summary']}
Objects visible: {context['objects']}
Gestures observed: {context['gestures']}
People visible: {context['num_faces']}
Known user preferences: {context['preferences']}
Recent conversation history:
{_format_history(context['recent_history'])}

Generate the JSON response now."""


def _fallback_response(raw_text: str) -> dict:
    return {
        "response_text": raw_text.strip() or "I'm here with you, could you say that again?",
        "emotion_acknowledged": "unparsed_fallback",
        "action": {"type": "speak", "params": {}},
    }


def generate_response(context: dict) -> dict:
    """Call the Groq-hosted LLM with the fused multimodal context and return a structured decision."""
    client = _load_client()
    completion = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(context)},
        ],
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
        response_format={"type": "json_object"},
    )
    raw_text = completion.choices[0].message.content

    try:
        parsed = json.loads(raw_text)
        action = parsed.get("action") or {"type": "speak", "params": {}}
        if action.get("type") not in _VALID_ACTIONS:
            action["type"] = "speak"
        action.setdefault("params", {})
        parsed["action"] = action
        parsed.setdefault("response_text", "")
        parsed.setdefault("emotion_acknowledged", "")
    except (json.JSONDecodeError, AttributeError):
        parsed = _fallback_response(raw_text or "")

    parsed["raw_llm_output"] = raw_text
    return parsed
