#!/usr/bin/env python3
"""
gesture_control.py

Standalone, real-time DYNAMIC hand gesture detector, locked to ONE
enrolled target person's identity via face recognition.

No ROS, no simulator - just: webcam -> face ID -> hand tracking ->
gesture classification -> on-screen + console command.

Gestures recognized (only while ARMED - see below):
    Swipe hand left   -> "TURN LEFT"
    Swipe hand right  -> "TURN RIGHT"
    Push hand forward -> "MOVE FORWARD"
    Pull hand back    -> "MOVE BACKWARD"
    Open palm, held   -> "STOP"

Classification (trained model, with rule-based fallback):
    If gesture_model.pkl exists, gestures are classified by a RandomForest
    trained on your own recorded samples (see collect_gestures.py and
    train_gestures.py). The model has an explicit "NONE" class, so it can
    answer "nothing is happening" instead of being forced to pick a command
    for every window of hand motion.

    If no trained model is found, the script transparently falls back to the
    original hand-tuned threshold rules, so it still works end to end.

    Two guards reduce misfires:
      - CONFIDENCE_THRESHOLD: ignore predictions the model isn't sure about.
      - CONSECUTIVE_AGREE: require the same prediction on N consecutive
        frames before acting, so one-off jitter can't trigger a command.

Target-person identity lock (for crowded environments):
    On first run (or whenever target_face.npy is missing), you enroll:
    look at the camera and press 'e' to capture your face. That face
    embedding is saved to disk so you don't need to re-enroll next time.

    Every frame, ALL visible faces are compared against your enrolled
    embedding. Only YOUR face counts as "the target" - regardless of how
    many other people are in frame, and even after you fully leave and
    re-enter the frame.

    The control zone is NOT a fixed box - it dynamically follows your
    detected face. Only a hand inside that zone is ever considered.

Wake-gesture (arm / disarm) - stays rule-based, since "hold still" is
already reliable and doesn't benefit from being learned:
    - System starts LOCKED (idle). All motion is ignored.
    - Hold an OPEN palm still, inside your face-anchored zone, for about a
      second -> system ARMS. Only now are commands dispatched.
    - Hold a CLOSED FIST still, inside the zone, while armed -> DISARMS.

Install (Ubuntu - dlib needs build tools first):
    sudo apt install -y build-essential cmake libopenblas-dev liblapack-dev
    pip install -r requirements.txt

Run:
    python3 gesture_control.py
    (first run: look at the camera and press 'e' to enroll your face)

Press 'q' to quit, 'e' to (re-)enroll your face at any time.
"""

import os
import time
from collections import deque

import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", category=UserWarning, module="face_recognition_models")
import mediapipe as mp
import face_recognition

from gesture_common import (
    WINDOW_FRAMES,
    COL_OPEN,
    COL_CLOSED,
    palm_center,
    frame_row,
    held_still,
    classify_gesture_rules,
)
from gesture_model import load_model, predict_gesture, CONFIDENCE_THRESHOLD

# --------------------------------------------------------------------------
# Tunable parameters
# --------------------------------------------------------------------------

DISPLAY_HOLD_SECONDS = 0.9  # how long the detected command stays on screen before resetting
NO_HAND_RESET_FRAMES = 8    # if hand is lost this many frames, clear the buffer

# Require the model to predict the same gesture this many frames in a row
# before firing. 1 = fire immediately (twitchy). 2-3 = noticeably steadier.
CONSECUTIVE_AGREE = 2

# --- Arm / disarm (wake gesture) ---
ARM_HOLD_FRAMES = 15        # ~0.5-1s of held open palm (inside the zone) to ARM
DISARM_HOLD_FRAMES = 15     # ~0.5-1s of held closed fist (inside the zone) to DISARM
ARMED_TIMEOUT_SECONDS = 20  # auto-disarm if no command fires for this long (safety net)

# --- Face identity lock ---
ENROLLED_FACE_PATH = "target_face.npy"
FACE_MATCH_THRESHOLD = 0.55   # lower = stricter match (face_recognition "distance")
FACE_DOWNSCALE = 0.35         # shrink frame before face_recognition for speed
FACE_DETECT_EVERY_N_FRAMES = 2  # run face ID every Nth frame; reuse last zone in between
FACE_LOST_GRACE_FRAMES = 10   # keep using the last known zone this many frames after losing the face

# Dynamic control zone, expressed as multiples of the detected face box size.
ZONE_SIDE_MARGIN = 2.2   # how far left/right of the face the zone extends
ZONE_ABOVE_MARGIN = 0.3  # how far above the face the zone extends
ZONE_BELOW_MARGIN = 4.5  # how far below the face the zone extends


# --------------------------------------------------------------------------
# Face identity lock
# --------------------------------------------------------------------------

def load_enrolled_face():
    if os.path.exists(ENROLLED_FACE_PATH):
        return np.load(ENROLLED_FACE_PATH)
    return None


def save_enrolled_face(encoding):
    np.save(ENROLLED_FACE_PATH, encoding)


def enroll_from_frame(frame_bgr):
    """Find the largest face in this frame and return its encoding, or None."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb, model="hog")
    if not locations:
        return None
    # largest face box = closest to camera = most likely the operator
    locations.sort(key=lambda box: (box[2] - box[0]) * (box[1] - box[3]), reverse=True)
    encodings = face_recognition.face_encodings(rgb, known_face_locations=[locations[0]])
    if not encodings:
        return None
    return encodings[0]


def find_target_face(frame_bgr, enrolled_encoding):
    """
    Detect faces in a downscaled copy of the frame, compare each to the
    enrolled encoding, and return the best match's box in full-frame pixel
    coordinates: (top, right, bottom, left). None if no good match.
    """
    small = cv2.resize(frame_bgr, (0, 0), fx=FACE_DOWNSCALE, fy=FACE_DOWNSCALE)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb_small, model="hog")
    if not locations:
        return None
    encodings = face_recognition.face_encodings(rgb_small, known_face_locations=locations)
    if not encodings:
        return None

    distances = face_recognition.face_distance(encodings, enrolled_encoding)
    best_idx = int(np.argmin(distances))
    if distances[best_idx] > FACE_MATCH_THRESHOLD:
        return None

    top, right, bottom, left = locations[best_idx]
    scale = 1.0 / FACE_DOWNSCALE
    return (int(top * scale), int(right * scale), int(bottom * scale), int(left * scale))


def zone_from_face_box(face_box, frame_w, frame_h):
    """Build the dynamic control-zone rectangle (in pixels) around a face box."""
    top, right, bottom, left = face_box
    face_w = right - left
    face_h = bottom - top
    cx = (left + right) / 2.0

    x_min = max(0, int(cx - ZONE_SIDE_MARGIN * face_w))
    x_max = min(frame_w, int(cx + ZONE_SIDE_MARGIN * face_w))
    y_min = max(0, int(top - ZONE_ABOVE_MARGIN * face_h))
    y_max = min(frame_h, int(bottom + ZONE_BELOW_MARGIN * face_h))
    return x_min, y_min, x_max, y_max


def in_zone_px(cx_px, cy_px, zone):
    x_min, y_min, x_max, y_max = zone
    return x_min <= cx_px <= x_max and y_min <= cy_px <= y_max


def pick_zone_hand(multi_hand_landmarks, zone, frame_w, frame_h):
    """
    Among all detected hands, return the landmarks + index of whichever falls
    inside the dynamic zone. If several qualify, pick the one nearest the
    zone's horizontal center.
    """
    if not multi_hand_landmarks or zone is None:
        return None, None

    x_min, y_min, x_max, y_max = zone
    zone_cx = (x_min + x_max) / 2.0

    best_landmarks, best_idx, best_dist = None, None, None
    for idx, hand in enumerate(multi_hand_landmarks):
        landmarks = hand.landmark
        cx, cy = palm_center(landmarks)
        cx_px, cy_px = cx * frame_w, cy * frame_h
        if not in_zone_px(cx_px, cy_px, zone):
            continue
        d = abs(cx_px - zone_cx)
        if best_dist is None or d < best_dist:
            best_dist = d
            best_landmarks = landmarks
            best_idx = idx
    return best_landmarks, best_idx


def main():
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam (device 0). Check camera permissions/connection.")
        return

    # --- Classifier: trained model if present, else the original rules ---
    bundle = load_model()
    if bundle is not None:
        print(f"Loaded trained model ({bundle['n_samples']} training samples, "
              f"{len(bundle['classes'])} classes).")
        print(f"Confidence threshold {CONFIDENCE_THRESHOLD}, "
              f"requires {CONSECUTIVE_AGREE} agreeing frames.")
    else:
        print("No trained model found - using rule-based thresholds.")
        print("  To train one:  python3 collect_gestures.py   then   python3 train_gestures.py")

    enrolled_encoding = load_enrolled_face()
    if enrolled_encoding is None:
        print("No enrolled face found. Look at the camera and press 'e' to enroll.")
    else:
        print(f"Loaded enrolled face from {ENROLLED_FACE_PATH}. Press 'e' any time to re-enroll.")

    buffer = deque(maxlen=max(WINDOW_FRAMES, ARM_HOLD_FRAMES, DISARM_HOLD_FRAMES))
    pred_history = deque(maxlen=CONSECUTIVE_AGREE)
    no_hand_count = 0

    display_command = None
    display_conf = 0.0
    display_until = 0.0

    armed = False
    armed_since = 0.0

    frame_idx = 0
    last_face_box = None
    frames_since_face_seen = FACE_LOST_GRACE_FRAMES + 1  # start as "not seen"

    print("Press 'q' to quit.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)  # mirror for natural "selfie" interaction
        h, w = frame.shape[:2]
        frame_idx += 1
        now = time.time()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('e'):
            print("Enrolling... hold still and look at the camera.")
            enc = enroll_from_frame(frame)
            if enc is None:
                print("No face found - try again with better lighting/closer to camera.")
            else:
                enrolled_encoding = enc
                save_enrolled_face(enc)
                print(f"Enrolled and saved to {ENROLLED_FACE_PATH}.")

        # --- Face identity lock: find/update the target's face box ---
        if enrolled_encoding is not None and frame_idx % FACE_DETECT_EVERY_N_FRAMES == 0:
            face_box = find_target_face(frame, enrolled_encoding)
            if face_box is not None:
                last_face_box = face_box
                frames_since_face_seen = 0
            else:
                frames_since_face_seen += 1
        else:
            frames_since_face_seen += 1

        target_visible = enrolled_encoding is not None and frames_since_face_seen <= FACE_LOST_GRACE_FRAMES
        zone = zone_from_face_box(last_face_box, w, h) if (target_visible and last_face_box) else None

        # --- Hand detection ---
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_result = hands.process(rgb)

        detected_this_frame = None
        detected_conf = 0.0
        hand_in_zone = False

        landmarks, target_idx = pick_zone_hand(hand_result.multi_hand_landmarks, zone, w, h)

        if hand_result.multi_hand_landmarks:
            for idx, hand in enumerate(hand_result.multi_hand_landmarks):
                if idx == target_idx:
                    mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                else:
                    mp_draw.draw_landmarks(
                        frame, hand, mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(90, 90, 90), thickness=1, circle_radius=1),
                        mp_draw.DrawingSpec(color=(90, 90, 90), thickness=1),
                    )

        if landmarks is not None:
            no_hand_count = 0
            hand_in_zone = True
            buffer.append(frame_row(now, landmarks))

            if now >= display_until:
                if not armed:
                    # Idle: the only thing we look for is the arm (wake) gesture.
                    if held_still(buffer, ARM_HOLD_FRAMES, lambda p: p[COL_OPEN] > 0.5):
                        armed = True
                        armed_since = now
                        buffer.clear()
                        pred_history.clear()
                        print("[ARMED] Now listening for movement commands.")
                else:
                    # Armed: check disarm first, then classify.
                    if held_still(buffer, DISARM_HOLD_FRAMES, lambda p: p[COL_CLOSED] > 0.5):
                        armed = False
                        buffer.clear()
                        pred_history.clear()
                        print("[DISARMED] Back to locked/idle.")
                    else:
                        if bundle is not None:
                            cmd, conf = predict_gesture(buffer, bundle)
                        else:
                            cmd, conf = classify_gesture_rules(buffer), 1.0

                        # Smoothing: only act once the same command has been
                        # predicted on CONSECUTIVE_AGREE frames in a row.
                        if cmd is None:
                            pred_history.clear()
                        else:
                            pred_history.append((cmd, conf))
                            if (len(pred_history) == CONSECUTIVE_AGREE and
                                    len({c for c, _ in pred_history}) == 1):
                                detected_this_frame = cmd
                                detected_conf = float(np.mean([c for _, c in pred_history]))
        else:
            no_hand_count += 1
            if no_hand_count >= NO_HAND_RESET_FRAMES:
                buffer.clear()
                pred_history.clear()

        if armed and (now - armed_since) > ARMED_TIMEOUT_SECONDS and now >= display_until:
            armed = False
            buffer.clear()
            pred_history.clear()
            print("[DISARMED] Timed out with no commands.")

        if detected_this_frame is not None:
            display_command = detected_this_frame
            display_conf = detected_conf
            display_until = now + DISPLAY_HOLD_SECONDS
            armed_since = now
            buffer.clear()
            pred_history.clear()
            print(f"[GESTURE DETECTED] -> {display_command}  ({display_conf:.0%})")

        # --- Draw the dynamic, face-anchored control zone ---
        if zone is not None:
            zone_color = (0, 200, 0) if armed else (0, 165, 255)
            cv2.rectangle(frame, (zone[0], zone[1]), (zone[2], zone[3]), zone_color, 2)
        if last_face_box is not None and target_visible:
            top, right, bottom, left = last_face_box
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 200, 0), 2)

        # --- Status bar ---
        if now < display_until and display_command is not None:
            cv2.rectangle(frame, (0, 0), (w, 70), (0, 130, 0), -1)
            label = display_command
            if bundle is not None:
                label += f"  ({display_conf:.0%})"
            cv2.putText(frame, label, (20, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
        else:
            display_command = None
            bar_color = (0, 110, 0) if armed else (60, 60, 60)
            cv2.rectangle(frame, (0, 0), (w, 40), bar_color, -1)
            if enrolled_encoding is None:
                status = "NOT ENROLLED - press 'e' to enroll your face"
            elif not target_visible:
                status = "target face not visible..."
            elif armed:
                status = "ARMED - listening for commands"
            elif hand_result.multi_hand_landmarks and not hand_in_zone:
                status = "hand outside your control zone"
            elif hand_in_zone:
                status = "hold OPEN PALM still to ARM"
            else:
                status = "target found - show hand in your zone"
            cv2.putText(frame, status, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            mode = "model" if bundle is not None else "rules"
            cv2.putText(frame, mode, (w - 70, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow("Dynamic Gesture Control (face-locked)", frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
