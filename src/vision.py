# src/vision.py — environment perception: object detection, face detection, gesture recognition
import os
from typing import Optional

import cv2
import numpy as np

import config

_yolo_model = None
_face_cascade: Optional[cv2.CascadeClassifier] = None
_hands_detector = None  # mediapipe Hands instance, created lazily


def _load_yolo():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO  # lazy: heavy torch-backed import, unused unless enabled
        _yolo_model = YOLO(config.YOLO_MODEL_PATH)
    return _yolo_model


def _load_face_cascade() -> cv2.CascadeClassifier:
    global _face_cascade
    if _face_cascade is None:
        cascade_path = config.FACE_CASCADE_PATH
        if not os.path.isfile(cascade_path):
            # Fall back to the cascade bundled inside the installed opencv-python package
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade


def detect_objects(frame: np.ndarray) -> list[dict]:
    """Run YOLO object detection on a BGR frame. Returns a list of {label, confidence, bbox}."""
    if not config.ENABLE_OBJECT_DETECTION:
        return []
    model = _load_yolo()
    results = model.predict(frame, conf=config.YOLO_CONF_THRESHOLD, verbose=False)[0]
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        detections.append({
            "label": model.names[cls_id],
            "confidence": float(box.conf[0]),
            "bbox": [float(v) for v in box.xyxy[0].tolist()],
        })
    return detections


def detect_faces(frame: np.ndarray) -> list[dict]:
    """Detect faces in a BGR frame using a Haar cascade. Returns bounding boxes."""
    cascade = _load_face_cascade()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    return [{"bbox": [int(x), int(y), int(x + w), int(y + h)]} for (x, y, w, h) in faces]


def _load_hands_detector():
    global _hands_detector
    if _hands_detector is None:
        import mediapipe as mp
        _hands_detector = mp.solutions.hands.Hands(
            static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
        )
    return _hands_detector


def _classify_hand_pose(landmarks) -> str:
    """Simple heuristic gesture classifier from 21 MediaPipe hand landmarks.

    Compares fingertip y-coordinates to their corresponding knuckle joints to decide
    whether each finger is extended, then maps the extended-finger pattern to a gesture.
    """
    tips = [8, 12, 16, 20]
    pip_joints = [6, 10, 14, 18]
    extended = [landmarks[t].y < landmarks[p].y for t, p in zip(tips, pip_joints)]
    thumb_extended = landmarks[4].x < landmarks[3].x if landmarks[4].x < landmarks[0].x else landmarks[4].x > landmarks[3].x

    num_extended = sum(extended)
    if num_extended == 4 and thumb_extended:
        return "open_palm_wave"
    if num_extended == 0 and not thumb_extended:
        return "fist"
    if extended[0] and num_extended == 1:
        return "pointing"
    if extended[0] and extended[1] and num_extended == 2:
        return "peace_sign"
    return "unknown"


def detect_gestures(frame: np.ndarray) -> list[str]:
    """Detect hand gestures in a BGR frame using MediaPipe Hands. Disabled via config flag."""
    if not config.ENABLE_GESTURE_RECOGNITION:
        return []
    detector = _load_hands_detector()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = detector.process(rgb_frame)
    if not result.multi_hand_landmarks:
        return []
    return [_classify_hand_pose(hand.landmark) for hand in result.multi_hand_landmarks]


def summarize_scene(objects: list[dict], faces: list[dict], gestures: list[str]) -> str:
    """Build a short natural-language scene description for the LLM prompt."""
    parts = []
    if faces:
        parts.append(f"{len(faces)} person(s) visible")
    object_counts: dict[str, int] = {}
    for obj in objects:
        object_counts[obj["label"]] = object_counts.get(obj["label"], 0) + 1
    if object_counts:
        parts.append("objects seen: " + ", ".join(f"{v}x {k}" for k, v in object_counts.items()))
    if gestures:
        parts.append("gestures: " + ", ".join(gestures))
    return "; ".join(parts) if parts else "no significant objects or people detected"


def analyze_frame(frame: np.ndarray) -> dict:
    """Run the full vision stack on a single frame."""
    objects = detect_objects(frame)
    faces = detect_faces(frame)
    gestures = detect_gestures(frame)
    return {
        "objects": objects,
        "faces": faces,
        "gestures": gestures,
        "scene_summary": summarize_scene(objects, faces, gestures),
    }
