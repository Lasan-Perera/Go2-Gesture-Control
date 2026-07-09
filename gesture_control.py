#!/usr/bin/env python3
"""
gesture_control.py

Standalone, real-time DYNAMIC hand gesture detector, locked to ONE
enrolled target person's identity via face recognition.

No ROS, no simulator - just: webcam -> face ID -> hand tracking ->
motion-based gesture classification -> on-screen + console command.

Gestures recognized (only while ARMED - see below):
    Swipe hand left   -> "TURN LEFT"
    Swipe hand right  -> "TURN RIGHT"
    Push hand forward -> "MOVE FORWARD"
    Pull hand back    -> "MOVE BACKWARD"
    Open palm, held   -> "STOP"

Target-person identity lock (for crowded environments):
    On first run (or whenever target_face.npy is missing), you enroll:
    look at the camera and press 'e' to capture your face. That face
    embedding is saved to disk so you don't need to re-enroll next time.

    Every frame, ALL visible faces are compared against your enrolled
    embedding. Only YOUR face counts as "the target" - regardless of
    how many other people are in frame, and even after you fully leave
    and re-enter the frame (no need to re-arm identity, just re-appear).

    The control zone is NOT a fixed box - it dynamically follows your
    detected face (a region below/around wherever your face currently
    is). Only a hand inside that zone is ever considered, and only
    while your face is the one being tracked. Press 'e' any time to
    re-enroll (e.g. if lighting changes badly or you want to swap who
    is being tracked).

Wake-gesture (arm / disarm) - this is what prevents random/incidental
hand movement from being treated as a command, even from you:
    - System starts LOCKED (idle). All motion is ignored.
    - Hold an OPEN palm still, inside your face-anchored zone, for
      about a second -> system ARMS. Only now do swipe/push/pull/
      stop commands get dispatched.
    - Hold a CLOSED FIST still, inside the zone, for about a second
      while armed -> system DISARMS back to idle/ignoring everything.

How it works (high level):
    1. Every frame, `face_recognition` finds all faces + their
       embeddings, and MediaPipe Hands finds up to 2 hands.
    2. We match the enrolled embedding against detected faces to find
       YOUR face box, then build a dynamic zone around/below it.
    3. Whichever detected hand falls inside that zone becomes "the"
       tracked hand for this frame. Others are drawn but ignored.
    4. We track that hand's palm center (x, y) and a "palm size" proxy
       over a short rolling window of recent frames.
    5. "Dynamic" gestures are detected by looking at how much that
       center/size CHANGED across the window, not a single frame.
    6. Once a gesture is confidently classified, we show/print it,
       then immediately clear the window and go back to watching.

Install (Ubuntu - dlib needs build tools first):
    sudo apt install -y build-essential cmake libopenblas-dev liblapack-dev
    pip install opencv-python mediapipe numpy face_recognition

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
import mediapipe as mp
import face_recognition

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

# --- Arm / disarm (wake gesture) ---
ARM_HOLD_FRAMES = 15        # ~0.5-1s of held open palm (inside the zone) to ARM
DISARM_HOLD_FRAMES = 15     # ~0.5-1s of held closed fist (inside the zone) to DISARM
ARMED_TIMEOUT_SECONDS = 20  # auto-disarm if no command fires for this long (safety net)

# --- Face identity lock ---
ENROLLED_FACE_PATH = "target_face.npy"
FACE_MATCH_THRESHOLD = 0.55   # lower = stricter match (face_recognition "distance"; 0.6 is their default cutoff)
FACE_DOWNSCALE = 0.35         # shrink frame before face_recognition for speed (it's slow at full res)
FACE_DETECT_EVERY_N_FRAMES = 2  # run face ID every Nth frame; reuse last zone in between for smoothness
FACE_LOST_GRACE_FRAMES = 10   # keep using the last known zone this many frames after losing the face

# Dynamic control zone, expressed as multiples of the detected face box size,
# anchored to the face position. This is what "follows you" around the frame.
ZONE_SIDE_MARGIN = 2.2   # how far left/right of the face the zone extends (x face width)
ZONE_ABOVE_MARGIN = 0.3  # how far above the face the zone extends
ZONE_BELOW_MARGIN = 4.5  # how far below the face the zone extends (down to roughly waist/hands)

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
    return float(np.mean(xs)), float(np.mean(ys))


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
        if landmarks[tip].y < landmarks[pip].y:
            extended += 1
    return extended >= 3


def is_hand_closed(landmarks):
    """Rough check: are all 4 fingers curled (tip below/near pip) -> a fist?"""
    fingers = [
        (INDEX_TIP, INDEX_PIP),
        (MIDDLE_TIP, MIDDLE_PIP),
        (RING_TIP, RING_PIP),
        (PINKY_TIP, PINKY_PIP),
    ]
    curled = 0
    for tip, pip in fingers:
        if landmarks[tip].y > landmarks[pip].y:
            curled += 1
    return curled >= 3


def _held_still(pts, n_frames, predicate):
    """True if the last n_frames samples are all within STILL_MOTION_THRESH
    of the first of those frames' position, AND all satisfy `predicate`."""
    if len(pts) < n_frames:
        return False
    window = pts[-n_frames:]
    x0, y0 = window[0][1], window[0][2]
    return all(
        abs(p[1] - x0) < STILL_MOTION_THRESH and
        abs(p[2] - y0) < STILL_MOTION_THRESH and
        predicate(p)
        for p in window
    )


def classify_gesture(buffer):
    """
    Look at the rolling buffer of (t, cx, cy, size, open, closed) samples
    and decide if a dynamic MOVEMENT gesture just happened. Returns a
    command string or None.
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

    if horiz > SWIPE_DX_THRESH and horiz > vert * 1.5 and abs(dsize) < SIZE_CHANGE_THRESH * 1.5:
        return "TURN RIGHT" if dx > 0 else "TURN LEFT"
        # frame is mirrored (selfie view), so dx > 0 = viewer's right = robot's right.

    if abs(dsize) > SIZE_CHANGE_THRESH and horiz < SWIPE_DX_THRESH * 0.7 and vert < SWIPE_DX_THRESH * 0.7:
        return "MOVE FORWARD" if dsize > 0 else "MOVE BACKWARD"

    if _held_still(pts, STOP_HOLD_FRAMES, lambda p: p[4]):
        return "STOP"

    return None


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
    """
    Find the largest face in this frame and return its encoding, or None
    if no face was found. Used for both first-time and re-enrollment.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb, model="hog")
    if not locations:
        return None
    # pick the largest face box (closest to camera = most likely the operator)
    locations.sort(key=lambda box: (box[2] - box[0]) * (box[1] - box[3]), reverse=True)
    encodings = face_recognition.face_encodings(rgb, known_face_locations=[locations[0]])
    if not encodings:
        return None
    return encodings[0]


def find_target_face(frame_bgr, enrolled_encoding):
    """
    Detect faces in a downscaled copy of the frame, compare each to the
    enrolled encoding, and return the best match's bounding box scaled
    back to full-frame pixel coordinates: (top, right, bottom, left).
    Returns None if no good match this frame.
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
    Among all detected hands, return the landmark list + index of whichever
    one falls inside the dynamic zone. If more than one qualifies, pick the
    one closest to the zone's horizontal center (most likely the real arm,
    not a stray hand at the zone's edge).
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

    enrolled_encoding = load_enrolled_face()
    if enrolled_encoding is None:
        print("No enrolled face found. Look at the camera and press 'e' to enroll.")
    else:
        print(f"Loaded enrolled face from {ENROLLED_FACE_PATH}. Press 'e' any time to re-enroll.")

    buffer = deque(maxlen=max(WINDOW_FRAMES, ARM_HOLD_FRAMES, DISARM_HOLD_FRAMES))
    no_hand_count = 0

    display_command = None
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

            cx, cy = palm_center(landmarks)
            size = palm_size(landmarks)
            open_hand = is_hand_open(landmarks)
            closed_hand = is_hand_closed(landmarks)
            buffer.append((now, cx, cy, size, open_hand, closed_hand))

            if now >= display_until:
                if not armed:
                    pts = list(buffer)
                    if _held_still(pts, ARM_HOLD_FRAMES, lambda p: p[4]):
                        armed = True
                        armed_since = now
                        buffer.clear()
                        print("[ARMED] Now listening for movement commands.")
                else:
                    pts = list(buffer)
                    if _held_still(pts, DISARM_HOLD_FRAMES, lambda p: p[5]):
                        armed = False
                        buffer.clear()
                        print("[DISARMED] Back to locked/idle.")
                    else:
                        cmd = classify_gesture(buffer)
                        if cmd is not None:
                            detected_this_frame = cmd
        else:
            no_hand_count += 1
            if no_hand_count >= NO_HAND_RESET_FRAMES:
                buffer.clear()

        if armed and (now - armed_since) > ARMED_TIMEOUT_SECONDS and now >= display_until:
            armed = False
            buffer.clear()
            print("[DISARMED] Timed out with no commands.")

        if detected_this_frame is not None:
            display_command = detected_this_frame
            display_until = now + DISPLAY_HOLD_SECONDS
            armed_since = now
            buffer.clear()
            print(f"[GESTURE DETECTED] -> {display_command}")

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
            cv2.putText(frame, display_command, (20, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
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

        cv2.imshow("Dynamic Gesture Control (face-locked)", frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()