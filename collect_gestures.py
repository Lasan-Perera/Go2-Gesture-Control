#!/usr/bin/env python3
"""
collect_gestures.py

Record labeled training samples for the gesture classifier.

Each sample is a window of WINDOW_FRAMES consecutive frames of your hand,
saved as a .npy file under data/<LABEL>/. Train on them with:

    python3 train_gestures.py

Usage:
    python3 collect_gestures.py

Keys:
    1  TURN LEFT        (swipe hand left)
    2  TURN RIGHT       (swipe hand right)
    3  MOVE FORWARD     (push hand toward camera)
    4  MOVE BACKWARD    (pull hand away from camera)
    5  STOP             (open palm, held still)
    6  NONE             (anything that is NOT a command - see below)

    SPACE  record ONE sample of the selected label (3..2..1 countdown)
    c      toggle CONTINUOUS recording of the selected label
    u      undo / delete the most recently recorded sample
    q      quit

Two recording modes:

  SINGLE (SPACE) - use for the 5 command gestures.
    A 3..2..1 countdown appears, then the script captures exactly
    WINDOW_FRAMES frames. Perform the gesture during that capture. If your
    hand leaves the frame mid-capture, the sample is discarded.

  CONTINUOUS ('c') - use for NONE.
    Records a fresh overlapping window every CONTINUOUS_STRIDE frames for as
    long as it's on. Turn it on, then just behave normally in front of the
    camera for a minute - rest your hand, drift, fidget, scratch your face,
    move between positions - and it will harvest dozens of NONE samples
    automatically. This is by far the fastest way to build the NONE class.

    Do NOT use continuous mode for swipes: it would also record your hand
    travelling BACK to its start position and label that motion a swipe too.

On the NONE class (the most important one):
    Record plenty of NONE: hand drifting slowly, hand moving between
    positions, scratching your face, small fidgets, hand half out of frame,
    gesturing while talking. Without a rich NONE class the model has no way
    to say "nothing is happening" and will misfire constantly.

    CRITICAL - do not record STOP as NONE. Your STOP gesture is "open palm,
    held still". If you rest your open hand in front of the camera during a
    NONE session, you are recording STOP and labelling it NONE. The model
    then receives directly contradictory labels and both classes degrade.

    During NONE sessions: keep your hand MOVING (aimlessly), or keep it
    CLOSED/relaxed if it is still. Never open-palm-and-freeze.

    Aim for roughly:
        ~40 samples for each of the 5 command gestures  (SPACE, one at a time)
        ~80-120 samples for NONE                        (one 'c' session)

    NONE should be the largest class, but not by more than about 2-3x. If
    NONE dwarfs everything (say 300 vs 20), overall accuracy stops meaning
    anything - a model that always answers NONE would look "accurate".

    Vary distance from the camera, speed, and which hand you use. Variety
    matters much more than raw sample count.

Note: this script does NOT use face recognition. Features are computed from
hand landmarks alone, so the face-identity lock in gesture_control.py has no
effect on them - skipping it here keeps collection fast and simple.
"""

import os
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

from gesture_common import (
    WINDOW_FRAMES,
    GESTURE_CLASSES,
    NONE_CLASS,
    class_to_slug,
    frame_row,
)

DATA_DIR = "data"
COUNTDOWN_SECONDS = 3

# In continuous mode, save a new window every N frames. Set to WINDOW_FRAMES so
# consecutive samples share NO frames.
#
# This matters more than it looks. With a stride of 4 and a 12-frame window,
# consecutive samples overlap by 8 frames - they are near-duplicates. Those
# near-duplicates then land on both sides of the train/test split during
# training, so the model is scored on windows it has effectively already seen.
# The reported accuracy goes up; live performance does not. Non-overlapping
# windows keep each saved sample genuinely independent.
CONTINUOUS_STRIDE = WINDOW_FRAMES

# key -> class label
KEY_TO_CLASS = {ord(str(i + 1)): label for i, label in enumerate(GESTURE_CLASSES)}


def sample_dir(label):
    return os.path.join(DATA_DIR, class_to_slug(label))


def count_samples(label):
    d = sample_dir(label)
    if not os.path.isdir(d):
        return 0
    return len([f for f in os.listdir(d) if f.endswith(".npy")])


def save_sample(label, window):
    d = sample_dir(label)
    os.makedirs(d, exist_ok=True)
    idx = 0
    while True:
        path = os.path.join(d, f"sample_{idx:04d}.npy")
        if not os.path.exists(path):
            break
        idx += 1
    np.save(path, np.asarray(window, dtype=np.float64))
    return path


def draw_text(frame, text, y, scale=0.7, color=(255, 255, 255), thickness=2):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


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
        print("ERROR: could not open webcam (device 0).")
        return

    # Recording state machine: idle -> countdown -> capturing -> idle
    # Continuous mode runs alongside "idle" and harvests sliding windows.
    state = "idle"
    active_label = GESTURE_CLASSES[0]
    countdown_end = 0.0
    capture_buf = []
    last_saved_path = None
    flash_msg = ""
    flash_until = 0.0

    continuous = False
    cont_buf = deque(maxlen=WINDOW_FRAMES)   # rolling window for continuous mode
    frames_since_cont_save = 0

    print("Collecting gesture samples.")
    print("  1-6    select the gesture to record")
    print("  SPACE  record ONE sample (countdown)")
    print("  c      toggle CONTINUOUS recording (best for NONE)")
    print("  u      undo last sample     q  quit\n")
    for i, label in enumerate(GESTURE_CLASSES):
        print(f"  {i + 1}  {label}")
    print()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)  # mirror, same as gesture_control.py
        h, w = frame.shape[:2]
        now = time.time()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        landmarks = None
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            landmarks = hand.landmark
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        # --- select which label we're recording ---
        if key in KEY_TO_CLASS and state == "idle":
            active_label = KEY_TO_CLASS[key]
            if continuous:
                # changing label mid-session would mislabel the rolling window
                continuous = False
                cont_buf.clear()
                flash_msg = f"selected {active_label} (continuous OFF)"
            else:
                flash_msg = f"selected {active_label}"
            flash_until = now + 1.0

        # --- toggle continuous mode ---
        if key == ord('c') and state == "idle":
            continuous = not continuous
            cont_buf.clear()
            frames_since_cont_save = 0
            if continuous and active_label != NONE_CLASS:
                print(f"[WARN] continuous mode on '{active_label}'. This also captures "
                      f"your hand returning to its start position, which is NOT the "
                      f"gesture. Recommended for NONE only.")
            flash_msg = f"continuous {'ON' if continuous else 'OFF'}: {active_label}"
            flash_until = now + 1.2
            print(f"[CONTINUOUS] {'ON' if continuous else 'OFF'} for {active_label}")

        # --- start a single-shot recording ---
        if key == ord(' ') and state == "idle" and not continuous:
            countdown_end = now + COUNTDOWN_SECONDS
            capture_buf = []
            state = "countdown"

        if key == ord('u') and state == "idle":
            if last_saved_path and os.path.exists(last_saved_path):
                os.remove(last_saved_path)
                flash_msg = f"deleted {os.path.basename(last_saved_path)}"
                print(f"[UNDO] {flash_msg}")
                last_saved_path = None
            else:
                flash_msg = "nothing to undo"
            flash_until = now + 1.2

        # --- continuous harvesting ---
        if continuous and state == "idle":
            if landmarks is None:
                cont_buf.clear()  # a window spanning a hand-loss gap is meaningless
            else:
                cont_buf.append(frame_row(now, landmarks))
                frames_since_cont_save += 1
                if len(cont_buf) == WINDOW_FRAMES and frames_since_cont_save >= CONTINUOUS_STRIDE:
                    last_saved_path = save_sample(active_label, list(cont_buf))
                    frames_since_cont_save = 0

            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
            draw_text(frame, f"CONTINUOUS: {active_label}   ({count_samples(active_label)})",
                      h - 20, 0.8, (0, 0, 255), 2)

        # --- countdown ---
        if state == "countdown":
            remaining = countdown_end - now
            if remaining <= 0:
                state = "capturing"
            else:
                n = int(np.ceil(remaining))
                cv2.putText(frame, str(n), (w // 2 - 40, h // 2 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 200, 255), 8)
                draw_text(frame, f"get ready: {active_label}", h - 20, 0.9, (0, 200, 255), 2)

        # --- capturing (single shot) ---
        elif state == "capturing":
            if landmarks is None:
                flash_msg = "hand lost - sample discarded"
                print(f"[DISCARD] {flash_msg}")
                flash_until = now + 1.5
                state = "idle"
                capture_buf = []
            else:
                capture_buf.append(frame_row(now, landmarks))
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
                draw_text(frame, f"RECORDING {active_label}  [{len(capture_buf)}/{WINDOW_FRAMES}]",
                          h - 20, 0.9, (0, 0, 255), 2)

                if len(capture_buf) >= WINDOW_FRAMES:
                    last_saved_path = save_sample(active_label, capture_buf)
                    total = count_samples(active_label)
                    flash_msg = f"saved {active_label}  (total: {total})"
                    print(f"[SAVED] {active_label} -> {last_saved_path}  (total: {total})")
                    flash_until = now + 1.2
                    state = "idle"
                    capture_buf = []

        # --- overlay: status + per-class counts ---
        if state == "idle" and not continuous:
            if now < flash_until and flash_msg:
                cv2.rectangle(frame, (0, 0), (w, 40), (0, 120, 0), -1)
                draw_text(frame, flash_msg, 28, 0.7)
            else:
                cv2.rectangle(frame, (0, 0), (w, 40), (60, 60, 60), -1)
                status = (f"[{active_label}]  SPACE=record  c=continuous"
                          if landmarks is not None else "show your hand")
                draw_text(frame, status, 28, 0.7)

        if state == "idle":
            y = 70
            for i, label in enumerate(GESTURE_CLASSES):
                colour = (0, 255, 255) if label == active_label else (200, 200, 200)
                draw_text(frame, f"{i + 1}: {label:<14} {count_samples(label):>3}", y, 0.55,
                          colour, 1)
                y += 22

        cv2.imshow("Collect Gestures", frame)

    cap.release()
    cv2.destroyAllWindows()

    print("\nSample counts:")
    for label in GESTURE_CLASSES:
        print(f"  {label:<15} {count_samples(label)}")
    print("\nNext: python3 train_gestures.py")


if __name__ == "__main__":
    main()