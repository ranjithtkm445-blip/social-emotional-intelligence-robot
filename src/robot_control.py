# src/robot_control.py — robot action execution (simulated by default; see config.ROBOT_MODE)
#
# Each function's signature and return shape mirror what a real ROS2 publisher/action-client
# would expose, so switching config.ROBOT_MODE to "ros2" later only requires swapping the
# function bodies below for rclpy publishers/action clients, not the calling code in pipeline.py.
#   navigate_to        -> nav2 "navigate_to_pose" action goal
#   fetch_object       -> MoveIt2 pick-and-place goal on the manipulator arm
#   display_reminder   -> a topic publish to the robot's on-board display/HRI panel
#   call_for_assistance -> a topic publish to a staff-alert / emergency-dispatch node
import json
import time

import config

_ACTION_TOPIC_MAP = {
    "navigate_to": "/navigate_to_pose",
    "fetch_object": "/arm_controller/pick_place",
    "display_reminder": "/hri/display_panel",
    "call_for_assistance": "/hri/emergency_alert",
}


def _log_action(action_type: str, params: dict, status: str) -> dict:
    entry = {
        "timestamp": time.time(),
        "mode": config.ROBOT_MODE,
        "action_type": action_type,
        "ros2_topic_equivalent": _ACTION_TOPIC_MAP.get(action_type),
        "params": params,
        "status": status,
    }
    with open(config.ROBOT_ACTION_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def navigate_to(location: str) -> dict:
    """Simulate sending a navigation goal to a location (e.g. 'kitchen', 'user's bedroom')."""
    return _log_action("navigate_to", {"location": location}, status="simulated_arrived")


def fetch_object(object_name: str) -> dict:
    """Simulate a pick-and-place goal for the given object."""
    return _log_action("fetch_object", {"object": object_name}, status="simulated_delivered")


def display_reminder(text: str) -> dict:
    """Simulate showing a reminder/message on the robot's display panel."""
    return _log_action("display_reminder", {"text": text}, status="simulated_displayed")


def call_for_assistance(reason: str) -> dict:
    """Simulate escalating to a human caregiver/staff member."""
    return _log_action("call_for_assistance", {"reason": reason}, status="simulated_alert_sent")


_DISPATCH = {
    "navigate": lambda params: navigate_to(params.get("location", "unknown")),
    "fetch_object": lambda params: fetch_object(params.get("object", "unknown")),
    "display_reminder": lambda params: display_reminder(params.get("text", "")),
    "call_for_assistance": lambda params: call_for_assistance(params.get("reason", "unspecified")),
    "speak": lambda params: _log_action("speak", params, status="handled_by_tts"),
}


def execute_action(action: dict) -> dict:
    """Dispatch an LLM-planned action ({"type": ..., "params": {...}}) to the matching handler."""
    handler = _DISPATCH.get(action.get("type", "speak"), _DISPATCH["speak"])
    return handler(action.get("params", {}))
