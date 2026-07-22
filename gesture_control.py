#!/usr/bin/env python3
"""
gesture_control.py

Standalone, real-time DYNAMIC hand gesture detector, locked to ONE enrolled
target person's identity via face recognition.

webcam -> face ID -> hand tracking -> gesture classification -> dispatch

Gestures recognized (only while ARMED):
    Swipe hand left   -> "TURN LEFT"
    Swipe hand right  -> "TURN RIGHT"
    Push hand forward -> "MOVE FORWARD"
    Pull hand back    -> "MOVE BACKWARD"
    Open palm, held   -> "STOP"


DISPATCHING COMMANDS
--------------------
This module never assumes what a command *does*. run() takes a `dispatch`
callback with the signature:

    dispatch(command: str, confidence: float) -> None

The default just prints. To drive a robot instead, pass your own. A ROS 2
node becomes roughly:

    from gesture_control import run

    COMMAND_TO_TWIST = {...}

    def dispatch(command, confidence):
        publisher.publish(COMMAND_TO_TWIST[command])

    run(dispatch=dispatch)

That separation is the entire point: the vision pipeline below should not
need to change when the output changes.


CLASSIFICATION (trained model, with rule-based fallback)
--------------------------------------------------------
If gesture_model.pkl exists, gestures are classified by a RandomForest trained
on your own recorded samples (collect_gestures.py -> train_gestures.py). The
model has an explicit "NONE" class, so it can answer "nothing is happening"
rather than being forced to pick a command for every window of hand motion.

If no trained model is found, this falls back to the original hand-tuned
threshold rules, so the script always runs.

Two guards reduce misfires:
  - CONFIDENCE_THRESHOLD: ignore predictions the model isn't sure about.
  - CONSECUTIVE_AGREE: require the same prediction on N consecutive frames.


TARGET-PERSON IDENTITY LOCK
---------------------------
On first run, look at the camera and press 'e' to enroll your face. The
embedding is saved to disk. Every frame, all visible faces are compared
against it; only YOUR face counts as the target, however many other people
are in shot, and even after you leave and re-enter the frame.

The control zone is not a fixed box - it follows your detected face. Only a
hand inside that zone is ever considered.


ARM / DISARM (wake gesture)
---------------------------
Stays rule-based; "hold still" is already reliable and gains nothing from
being learned.

    - Starts LOCKED. All motion ignored.
    - Hold an OPEN palm still, inside your zone -> ARMS.
    - Hold a CLOSED FIST still, inside your zone -> DISARMS.
    - Auto-disarms after ARMED_TIMEOUT_SECONDS of no commands.
    - Auto-disarms once your face has been continuously missing for
      DISARM_GRACE_SECONDS (default 5s). Briefly occluding your own face with
      the gesturing hand is expected and does NOT disarm you - the control
      zone simply persists for FACE_ZONE_GRACE_SECONDS. Re-arming is always
      deliberate.


Install (Ubuntu - dlib needs build tools first):
    sudo apt install -y build-essential cmake libopenblas-dev liblapack-dev
    pip install -r requirements.txt

Run:
    python3 gesture_control.py
    (first run: look at the camera and press 'e' to enroll your face)

Press 'q' to quit, 'e' to (re-)enroll at any time.
"""

import os
import time
import warnings
from collections import deque

# MediaPipe and face_recognition_models emit a deprecation warning on every
# single frame, which buries the actual gesture output. Silence them here,
# before those libraries are imported.
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", category=UserWarning, module="face_recognition_models")

import cv2
import numpy as np
import face_recognition

import config
from gesture_common import (
    COL_OPEN,
    COL_CLOSED,
    COL_NFING,
    COL_THUMB,
    palm_center,
    frame_row,
    held_still,
    classify_gesture_rules,
)
from gesture_model import load_model, predict_gesture
from hand_landmarker import HandTracker


# --------------------------------------------------------------------------
# Default dispatcher
# --------------------------------------------------------------------------

def print_dispatch(command, confidence):
    """Default command sink: print to the console."""
    print(f"[GESTURE DETECTED] -> {command}  ({confidence:.0%})")


# --------------------------------------------------------------------------
# Frame-rate instrumentation
# --------------------------------------------------------------------------

class Perf:
    """
    Rolling frame-rate and face-detection latency.

    Worth having before optimizing anything: "it feels laggy" is not a number,
    and you cannot tell whether the cost is dlib, MediaPipe, or the display
    without measuring. face_ms is tracked separately because the HOG face
    detector is by far the most expensive step.
    """

    def __init__(self, window=30):
        self.frame_times = deque(maxlen=window)
        self.face_times = deque(maxlen=window)

    def tick(self, dt):
        self.frame_times.append(dt)

    def face(self, dt):
        self.face_times.append(dt)

    @property
    def fps(self):
        if not self.frame_times:
            return 0.0
        mean = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / mean if mean > 0 else 0.0

    @property
    def face_ms(self):
        if not self.face_times:
            return 0.0
        return 1000.0 * sum(self.face_times) / len(self.face_times)


# --------------------------------------------------------------------------
# Face identity lock
# --------------------------------------------------------------------------

def load_enrolled_face():
    if os.path.exists(config.ENROLLED_FACE_PATH):
        return np.load(config.ENROLLED_FACE_PATH)
    return None


def save_enrolled_face(encoding):
    np.save(config.ENROLLED_FACE_PATH, encoding)


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
    small = cv2.resize(frame_bgr, (0, 0), fx=config.FACE_DOWNSCALE, fy=config.FACE_DOWNSCALE)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb_small, model="hog")
    if not locations:
        return None
    encodings = face_recognition.face_encodings(rgb_small, known_face_locations=locations)
    if not encodings:
        return None

    distances = face_recognition.face_distance(encodings, enrolled_encoding)
    best_idx = int(np.argmin(distances))
    if distances[best_idx] > config.FACE_MATCH_THRESHOLD:
        return None

    top, right, bottom, left = locations[best_idx]
    scale = 1.0 / config.FACE_DOWNSCALE
    return (int(top * scale), int(right * scale), int(bottom * scale), int(left * scale))


def zone_from_face_box(face_box, frame_w, frame_h):
    """Build the dynamic control-zone rectangle (in pixels) around a face box."""
    top, right, bottom, left = face_box
    face_w = right - left
    face_h = bottom - top
    cx = (left + right) / 2.0

    x_min = max(0, int(cx - config.ZONE_SIDE_MARGIN * face_w))
    x_max = min(frame_w, int(cx + config.ZONE_SIDE_MARGIN * face_w))
    y_min = max(0, int(top - config.ZONE_ABOVE_MARGIN * face_h))
    y_max = min(frame_h, int(bottom + config.ZONE_BELOW_MARGIN * face_h))
    return x_min, y_min, x_max, y_max


def in_zone_px(cx_px, cy_px, zone):
    x_min, y_min, x_max, y_max = zone
    return x_min <= cx_px <= x_max and y_min <= cy_px <= y_max


def pick_zone_hand(hands_found, zone, frame_w, frame_h):
    """
    Among all detected hands, return the landmarks + index of whichever falls
    inside the dynamic zone. If several qualify, pick the one nearest the
    zone's horizontal center.

    `hands_found` is the Tasks API shape: a list where each element is itself
    a list of 21 landmark objects (landmarks[i].x/.y). This replaces the old
    `.multi_hand_landmarks[k].landmark` access - each element here already IS
    the landmark list, so there is no `.landmark` to unwrap.
    """
    if not hands_found or zone is None:
        return None, None

    x_min, y_min, x_max, y_max = zone
    zone_cx = (x_min + x_max) / 2.0

    best_landmarks, best_idx, best_dist = None, None, None
    for idx, landmarks in enumerate(hands_found):
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


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

def run(dispatch=print_dispatch, camera_index=0):
    """
    Open the webcam and run the gesture pipeline until the user quits.

    dispatch(command: str, confidence: float) is called once per recognized
    gesture. Everything else - drawing, arming, identity - is internal.
    """
    tracker = HandTracker(
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: could not open webcam (device {camera_index}). "
              f"Check camera permissions/connection.")
        return

    bundle = load_model()
    if bundle is not None:
        print(f"Loaded trained model ({bundle['n_samples']} training samples, "
              f"{len(bundle['classes'])} classes).")
        cv_f1 = bundle.get("cv_macro_f1")
        if cv_f1 is not None:
            print(f"  cross-validated macro F1: {cv_f1:.3f}")
        print(f"  confidence threshold {config.CONFIDENCE_THRESHOLD}, "
              f"requires {config.CONSECUTIVE_AGREE} agreeing frames.")
    else:
        print("No trained model found - using rule-based thresholds.")
        print("  To train one:  python3 collect_gestures.py   then   python3 train_gestures.py")

    enrolled_encoding = load_enrolled_face()
    if enrolled_encoding is None:
        print("No enrolled face found. Look at the camera and press 'e' to enroll.")
    else:
        print(f"Loaded enrolled face from {config.ENROLLED_FACE_PATH}. "
              f"Press 'e' any time to re-enroll.")

    buffer = deque(maxlen=max(config.WINDOW_FRAMES,
                             config.ARM_HOLD_FRAMES,
                             config.DISARM_HOLD_FRAMES))
    pred_history = deque(maxlen=config.CONSECUTIVE_AGREE)
    no_hand_count = 0

    display_command = None
    display_conf = 0.0
    display_until = 0.0

    armed = False
    armed_since = 0.0

    # Frames remaining before arm/disarm may fire again. Stops a single gesture
    # from toggling the mode twice as it passes through fist and open poses.
    mode_cooldown = 0

    frame_idx = 0
    last_face_box = None
    # Timestamp of the last SUCCESSFUL face match. Using a timestamp rather
    # than a frame counter fixes two bugs at once: the old counter ticked up on
    # frames where face detection never even ran (we only detect every Nth
    # frame), and a frame count means something different at 30 fps than at 10.
    last_face_seen_t = 0.0  # 0.0 = never seen

    perf = Perf()
    last_frame_t = time.time()

    print("Press 'q' to quit.\n")

    def disarm(reason):
        nonlocal armed
        armed = False
        buffer.clear()
        pred_history.clear()
        print(f"[DISARMED] {reason}")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)  # mirror for natural "selfie" interaction
        h, w = frame.shape[:2]
        frame_idx += 1
        now = time.time()

        perf.tick(now - last_frame_t)
        last_frame_t = now

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
                if armed:
                    disarm("re-enrolled, arm state reset")
                print(f"Enrolled and saved to {config.ENROLLED_FACE_PATH}.")

        # --- Face identity lock: find/update the target's face box ---
        if (enrolled_encoding is not None
                and frame_idx % config.FACE_DETECT_EVERY_N_FRAMES == 0):
            t0 = time.time()
            face_box = find_target_face(frame, enrolled_encoding)
            perf.face(time.time() - t0)
            if face_box is not None:
                last_face_box = face_box
                last_face_seen_t = now
        # NOTE: no `else` branch. Frames where we simply didn't look must not
        # count as evidence that the face is gone.

        face_gone_for = (now - last_face_seen_t) if last_face_seen_t > 0 else float("inf")

        # Keep using the last known zone through brief losses - your face has
        # not teleported, and you may well be occluding it with the very hand
        # you are gesturing with.
        target_visible = (enrolled_encoding is not None
                          and face_gone_for <= config.FACE_ZONE_GRACE_SECONDS)
        zone = (zone_from_face_box(last_face_box, w, h)
                if (target_visible and last_face_box) else None)

        # Tick the arm/disarm cooldown. Decremented here, once per processed
        # frame, rather than inside the landmark branch - a cooldown measured in
        # frames must not stall just because the hand briefly left the picture.
        if mode_cooldown > 0:
            mode_cooldown -= 1

        # --- SAFETY: never stay armed once the operator has really gone ---
        # Deliberately a much longer grace than the zone one above: a hand
        # sweeping across your face is not you walking away, and disarming on
        # that would make the system unusable.
        if (armed and config.DISARM_ON_FACE_LOST
                and face_gone_for > config.DISARM_GRACE_SECONDS):
            disarm(f"target face gone for {config.DISARM_GRACE_SECONDS:.0f}s")

        # --- Hand detection (Tasks API) ---
        # detect() returns a list of hands, each a list of 21 landmarks with
        # .x/.y in [0,1]. Same shape the old .multi_hand_landmarks[k].landmark
        # gave, so pick_zone_hand and frame_row are unchanged.
        hands_found = tracker.detect(frame, now * 1000.0)

        detected_this_frame = None
        detected_conf = 0.0
        hand_in_zone = False

        landmarks, target_idx = pick_zone_hand(hands_found, zone, w, h)

        # Draw the detected hands: the in-zone (target) hand in green, any
        # others in gray. The legacy mp_draw.draw_landmarks + HAND_CONNECTIONS
        # is gone with mp.solutions, so we draw the keypoints ourselves - a
        # light touch that keeps the "which hand is selected" feedback.
        for idx, hand_lms in enumerate(hands_found):
            color = (0, 255, 0) if idx == target_idx else (90, 90, 90)
            for lm in hand_lms:
                px, py = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (px, py), 3, color, -1)

        if landmarks is not None:
            no_hand_count = 0
            hand_in_zone = True
            buffer.append(frame_row(now, landmarks))

            if now >= display_until:
                if not armed:
                    # Idle: the only thing we look for is the arm (wake) gesture.
                    # A FULL open palm - all four fingers extended - not merely
                    # "open-ish", so a hand drifting through a half-open pose on
                    # its way somewhere else does not arm the system.
                    if (mode_cooldown <= 0 and held_still(
                            buffer, config.ARM_HOLD_FRAMES,
                            lambda p: p[COL_NFING] >= 4)):
                        armed = True
                        armed_since = now
                        mode_cooldown = config.ARM_DISARM_COOLDOWN_FRAMES
                        buffer.clear()
                        pred_history.clear()
                        print("[ARMED] Now listening for movement commands.")
                else:
                    # Armed: check disarm first, then classify.
                    #
                    # A TIGHT fist: every finger curled AND the thumb wrapped in.
                    # A resting hand is loosely curled with the thumb loose, and
                    # no longer disarms. STOP's opening fist is also brief and
                    # moving, so it fails the held-still test.
                    if (mode_cooldown <= 0 and held_still(
                            buffer, config.DISARM_HOLD_FRAMES,
                            lambda p: p[COL_NFING] <= 0 and p[COL_THUMB] < 0.5)):
                        disarm("tight fist held")
                        mode_cooldown = config.ARM_DISARM_COOLDOWN_FRAMES
                    else:
                        if bundle is not None:
                            cmd, conf = predict_gesture(buffer, bundle)
                        else:
                            cmd, conf = classify_gesture_rules(buffer), 1.0

                        # Smoothing: act only once the same command has been
                        # predicted on CONSECUTIVE_AGREE frames in a row.
                        if cmd is None:
                            pred_history.clear()
                        else:
                            pred_history.append((cmd, conf))
                            if (len(pred_history) == config.CONSECUTIVE_AGREE and
                                    len({c for c, _ in pred_history}) == 1):
                                detected_this_frame = cmd
                                detected_conf = float(np.mean([c for _, c in pred_history]))
        else:
            no_hand_count += 1
            if no_hand_count >= config.NO_HAND_RESET_FRAMES:
                buffer.clear()
                pred_history.clear()

        if (armed and (now - armed_since) > config.ARMED_TIMEOUT_SECONDS
                and now >= display_until):
            disarm("timed out with no commands")

        if detected_this_frame is not None:
            display_command = detected_this_frame
            display_conf = detected_conf
            display_until = now + config.DISPLAY_HOLD_SECONDS
            armed_since = now
            buffer.clear()
            pred_history.clear()
            dispatch(display_command, display_conf)

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
            elif armed and not target_visible:
                # Occluded but still armed - show the countdown to auto-disarm.
                remaining = max(0.0, config.DISARM_GRACE_SECONDS - face_gone_for)
                status = f"ARMED - face hidden, disarming in {remaining:.1f}s"
            elif not target_visible:
                status = "target face not visible..."
            elif armed:
                status = "ARMED - listening for commands"
            elif hands_found and not hand_in_zone:
                status = "hand outside your control zone"
            elif hand_in_zone:
                status = "hold OPEN PALM still to ARM"
            else:
                status = "target found - show hand in your zone"
            cv2.putText(frame, status, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # --- Footer: mode + performance ---
        mode = "model" if bundle is not None else "rules"
        footer = mode
        if config.SHOW_FPS:
            footer = f"{perf.fps:4.1f} fps | face {perf.face_ms:3.0f} ms | {mode}"
        cv2.putText(frame, footer, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow("Dynamic Gesture Control (face-locked)", frame)

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()


def main():
    run(dispatch=print_dispatch)


if __name__ == "__main__":
    main()