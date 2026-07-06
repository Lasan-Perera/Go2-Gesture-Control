#!/usr/bin/env python3
"""
gesture_control.py

Standalone, real-time DYNAMIC hand gesture detector.

No ROS, no simulator - just: webcam -> MediaPipe hand tracking ->
motion-based gesture classification -> on-screen + console command.

Gestures recognized:
    Swipe hand left   -> "TURN LEFT"
    Swipe hand right  -> "TURN RIGHT"
    Push hand forward -> "MOVE FORWARD"
    Pull hand back    -> "MOVE BACKWARD"
    Open palm, held   -> "STOP"

How it works (high level):
    1. Every frame, MediaPipe gives us 21 hand landmarks.
    2. We track the palm center (x, y) and a "palm size" proxy
       (distance from wrist to middle-finger base) over a short
       rolling window of recent frames.
    3. "Dynamic" gestures are detected by looking at how much that
       center/size CHANGED across the window, not a single frame.
    4. Once a gesture is confidently classified, we show/print it,
       then immediately clear the window and go back to watching -
       ready for the next gesture right away.

Run:
    pip install opencv-python mediapipe numpy
    python3 gesture_control.py

Press 'q' to quit.
"""

import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

# --------------------------------------------------------------------------
# Tunable parameters - adjust these if detection feels too sensitive/insensitive
# --------------------------------------------------------------------------

WINDOW_FRAMES = 12          # how many recent frames we look back over to judge motion
SWIPE_DX_THRESH = 0.16      # min horizontal movement (fraction of frame width) for a swipe
SIZE_CHANGE_THRESH = 0.045  # min change in palm-size proxy for push/pull
STILL_MOTION_THRESH = 0.015 # max movement allowed to count as "held still" (for STOP)
STOP_HOLD_FRAMES = 10       # how many consecutive still+open frames needed to trigger STOP
DISPLAY_HOLD_SECONDS = 0.9  # how long the detected command stays on screen before resetting
NO_HAND_RESET_FRAMES = 8    # if hand is lost this many frames, clear the buffer

# MediaPipe hand landmark indices we need
WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
MIDDLE_TIP = 12
INDEX_TIP = 8
INDEX_PIP = 6
MIDDLE_PIP = 10
RING_TIP = 16
RING_PIP = 14
PINKY_TIP = 20
PINKY_PIP = 18


def palm_center(landmarks):
    """Average of wrist + all MCP (knuckle) points -> stable center of the palm."""
    idxs = [0, 5, 9, 13, 17]
    xs = [landmarks[i].x for i in idxs]
    ys = [landmarks[i].y for i in idxs]
    return np.mean(xs), np.mean(ys)


def palm_size(landmarks):
    """Distance from wrist to middle-finger knuckle - grows as hand moves closer to camera."""
    wx, wy = landmarks[WRIST].x, landmarks[WRIST].y
    mx, my = landmarks[MIDDLE_MCP].x, landmarks[MIDDLE_MCP].y
    return float(np.hypot(mx - wx, my - wy))


def is_hand_open(landmarks):
    """Rough check: are the 4 fingers extended (tip above pip in y, i.e. higher on screen)?"""
    fingers = [
        (INDEX_TIP, INDEX_PIP),
        (MIDDLE_TIP, MIDDLE_PIP),
        (RING_TIP, RING_PIP),
        (PINKY_TIP, PINKY_PIP),
    ]
    extended = 0
    for tip, pip in fingers:
        if landmarks[tip].y < landmarks[pip].y:  # tip higher (smaller y) than pip = extended
            extended += 1
    return extended >= 3  # majority of fingers extended counts as "open"


def classify_gesture(buffer):
    """
    Look at the rolling buffer of (t, cx, cy, size, open) samples and decide
    if a dynamic gesture just happened. Returns a command string or None.
    """
    if len(buffer) < WINDOW_FRAMES:
        return None

    pts = list(buffer)[-WINDOW_FRAMES:]
    cx0, cy0, size0 = pts[0][1], pts[0][2], pts[0][3]
    cx1, cy1, size1 = pts[-1][1], pts[-1][2], pts[-1][3]

    dx = cx1 - cx0
    dy = cy1 - cy0
    dsize = size1 - size0

    horiz = abs(dx)
    vert = abs(dy)

    # --- Swipe left / right: big horizontal motion, dominant over vertical & size change ---
    if horiz > SWIPE_DX_THRESH and horiz > vert * 1.5 and abs(dsize) < SIZE_CHANGE_THRESH * 1.5:
        return "TURN RIGHT" if dx > 0 else "TURN LEFT"
        # Note: frame is mirrored (selfie view) before landmarks are drawn/read here,
        # so dx > 0 (moving toward the viewer's right) = robot's right.

    # --- Push forward / pull back: hand size changes a lot, without big lateral swipe ---
    if abs(dsize) > SIZE_CHANGE_THRESH and horiz < SWIPE_DX_THRESH * 0.7 and vert < SWIPE_DX_THRESH * 0.7:
        return "MOVE FORWARD" if dsize > 0 else "MOVE BACKWARD"

    # --- Stop: hand basically motionless AND open for a sustained number of frames ---
    still_and_open = all(
        abs(p[1] - pts[0][1]) < STILL_MOTION_THRESH and
        abs(p[2] - pts[0][2]) < STILL_MOTION_THRESH and
        p[4]
        for p in pts
    )
    if still_and_open and len(pts) >= STOP_HOLD_FRAMES:
        return "STOP"

    return None


def main():
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam (device 0). Check camera permissions/connection.")
        return

    buffer = deque(maxlen=WINDOW_FRAMES)
    no_hand_count = 0

    # Display-hold state: when a gesture fires, freeze on it briefly then reset
    display_command = None
    display_until = 0.0

    print("Gesture control running. Press 'q' in the video window to quit.")
    print("Show: swipe left/right, push toward camera, pull away, or hold an open palm still.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)  # mirror for natural "selfie" interaction
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        now = time.time()
        detected_this_frame = None

        if result.multi_hand_landmarks:
            no_hand_count = 0
            landmarks = result.multi_hand_landmarks[0].landmark
            mp_draw.draw_landmarks(frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

            cx, cy = palm_center(landmarks)
            size = palm_size(landmarks)
            open_hand = is_hand_open(landmarks)
            buffer.append((now, cx, cy, size, open_hand))

            # Only look for a new gesture if we're not currently freezing on a previous one
            if now >= display_until:
                cmd = classify_gesture(buffer)
                if cmd is not None:
                    detected_this_frame = cmd
        else:
            no_hand_count += 1
            if no_hand_count >= NO_HAND_RESET_FRAMES:
                buffer.clear()

        # If a new gesture was just recognized, lock it in for display + reset the buffer
        if detected_this_frame is not None:
            display_command = detected_this_frame
            display_until = now + DISPLAY_HOLD_SECONDS
            buffer.clear()  # instantly ready to start collecting the NEXT gesture
            print(f"[GESTURE DETECTED] -> {display_command}")

        # --- On-screen overlay ---
        h, w = frame.shape[:2]
        if now < display_until and display_command is not None:
            # Big, obvious command banner while "frozen" on the detected gesture
            cv2.rectangle(frame, (0, 0), (w, 70), (0, 130, 0), -1)
            cv2.putText(frame, display_command, (20, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
        else:
            display_command = None
            cv2.rectangle(frame, (0, 0), (w, 40), (60, 60, 60), -1)
            status = "watching..." if result.multi_hand_landmarks else "show your hand"
            cv2.putText(frame, status, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Dynamic Gesture Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()