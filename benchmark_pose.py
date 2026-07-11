#!/usr/bin/env python3
"""
benchmark_pose.py

P3 benchmark: is MediaPipe Pose Landmarker good enough to give us the
operator's WRISTS, at usable framerate, at the robot's low tilted-up angle?

What this decides
-----------------
The operator lock (P8) matches each detected hand to the nearest body wrist,
then throws away hands that don't belong to the operator. That needs wrist
positions. This script answers two questions before we commit to that design:

  1. SPEED  - does adding pose keep us above a usable framerate?
              Your hand-only baseline is ~22-25 fps; we compare against that.
  2. WRISTS - does Pose actually lock onto your wrists reliably, especially
              from a low camera tilted UP at you? You can only judge this by
              WATCHING, so the skeleton is drawn on screen with the wrists
              highlighted big and bright.

If both are good -> MediaPipe Pose is enough, no need for RTMPose.
If speed is fine but wrists are flaky at the low angle, or speed tanks when
Pose + Hands run together -> that's the signal to try RTMPose (one wholebody
model in a single pass) instead.

Three modes
-----------
    python3 benchmark_pose.py            # Pose ALONE  (speed + wrist quality)
    python3 benchmark_pose.py --combined # Pose + Hands together (the real cost)
    python3 benchmark_pose.py --tilt     # tips for testing the low/up angle

Keys while running:  q = quit.  The on-screen readout shows live fps and
whether each wrist is currently being found.

Model bundle
------------
Needs the pose model, downloaded once (like the hand one):

    curl -L -o models/pose_landmarker_lite.task \\
      https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task

Lite is the fastest variant and plenty for wrist positions. If wrists look
jittery you can try pose_landmarker_full (swap 'lite' for 'full' in the URL
and the path below), which is more accurate but slower.
"""

import os
import sys
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from hand_landmarker import HandTracker


# Pose landmark indices (MediaPipe Pose, 33 points). These are stable.
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16

# Upper-body connections we care about for the operator lock: shoulder-elbow-wrist
# on each arm, plus the shoulder line. We don't draw legs - irrelevant here.
UPPER_BODY_CONNECTIONS = [
    (L_SHOULDER, R_SHOULDER),
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
]

POSE_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "pose_landmarker_full.task"
)


def make_pose(model_path=POSE_MODEL_PATH):
    if not os.path.exists(model_path):
        sys.exit(
            f"Pose model not found at:\n    {model_path}\n\n"
            f"Download it once:\n"
            f"    curl -L -o {model_path} \\\n"
            f"      https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            f"pose_landmarker_lite/float16/1/pose_landmarker_lite.task\n"
        )
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def draw_pose(frame, landmarks, w, h):
    """Draw upper-body skeleton; make the wrists impossible to miss."""
    def px(i):
        return int(landmarks[i].x * w), int(landmarks[i].y * h)

    # bones
    for a, b in UPPER_BODY_CONNECTIONS:
        cv2.line(frame, px(a), px(b), (0, 200, 200), 2)

    # joints (small)
    for i in (L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW):
        cv2.circle(frame, px(i), 4, (255, 200, 0), -1)

    # WRISTS - big and bright, because this is the whole point
    for i, name in ((L_WRIST, "L"), (R_WRIST, "R")):
        p = px(i)
        cv2.circle(frame, p, 12, (0, 0, 255), 2)
        cv2.circle(frame, p, 4, (0, 0, 255), -1)
        cv2.putText(frame, name, (p[0] + 14, p[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def wrist_visible(landmarks, i):
    """
    A wrist counts as 'found' if MediaPipe reports it with reasonable
    visibility. .visibility is [0,1]; below ~0.5 the point is a guess.
    """
    try:
        return landmarks[i].visibility >= 0.5
    except AttributeError:
        return True  # some builds omit visibility; treat as present


def run_pose_only():
    pose = make_pose()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("ERROR: could not open webcam (device 0).")

    fps_win = deque(maxlen=30)
    last = time.time()
    l_seen = r_seen = total = 0

    print("POSE ALONE. Show your upper body; move your hands around.")
    print("Watch the red wrist circles - do they track your actual wrists?")
    print("Press 'q' to quit.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.time()
        fps_win.append(now - last); last = now

        result = pose.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
            int(now * 1000),
        )

        total += 1
        lw = rw = False
        if result.pose_landmarks:
            lms = result.pose_landmarks[0]
            draw_pose(frame, lms, w, h)
            lw = wrist_visible(lms, L_WRIST)
            rw = wrist_visible(lms, R_WRIST)
            l_seen += int(lw); r_seen += int(rw)

        fps = 1.0 / (sum(fps_win) / len(fps_win)) if fps_win else 0.0
        cv2.rectangle(frame, (0, 0), (w, 40), (40, 40, 40), -1)
        cv2.putText(frame, f"POSE ONLY  {fps:4.1f} fps   "
                           f"L wrist:{'Y' if lw else '-'}  R wrist:{'Y' if rw else '-'}",
                    (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("P3 benchmark: Pose only", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release(); cv2.destroyAllWindows(); pose.close()
    report("POSE ALONE", fps_win, l_seen, r_seen, total)


def run_combined():
    pose = make_pose()
    hands = HandTracker(max_num_hands=2)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("ERROR: could not open webcam (device 0).")

    fps_win = deque(maxlen=30)
    last = time.time()
    l_seen = r_seen = total = 0

    print("POSE + HANDS together - this is the REAL cost of the two-model path.")
    print("Compare this fps against your ~22-25 hand-only baseline.")
    print("Press 'q' to quit.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.time()
        fps_win.append(now - last); last = now
        ts = int(now * 1000)

        # both models, same frame
        result = pose.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
            ts,
        )
        hands_found = hands.detect(frame, now * 1000.0)

        total += 1
        lw = rw = False
        if result.pose_landmarks:
            lms = result.pose_landmarks[0]
            draw_pose(frame, lms, w, h)
            lw = wrist_visible(lms, L_WRIST); rw = wrist_visible(lms, R_WRIST)
            l_seen += int(lw); r_seen += int(rw)

        # draw hands as small green dots so you can see both models cooperating
        for hand in hands_found:
            for lm in hand:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, (0, 255, 0), -1)

        fps = 1.0 / (sum(fps_win) / len(fps_win)) if fps_win else 0.0
        cv2.rectangle(frame, (0, 0), (w, 40), (40, 40, 40), -1)
        cv2.putText(frame, f"POSE+HANDS  {fps:4.1f} fps   "
                           f"hands:{len(hands_found)}  "
                           f"L:{'Y' if lw else '-'} R:{'Y' if rw else '-'}",
                    (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("P3 benchmark: Pose + Hands", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release(); cv2.destroyAllWindows(); pose.close(); hands.close()
    report("POSE + HANDS", fps_win, l_seen, r_seen, total)


def report(label, fps_win, l_seen, r_seen, total):
    fps = 1.0 / (sum(fps_win) / len(fps_win)) if fps_win else 0.0
    print("\n" + "=" * 52)
    print(f"{label} - result")
    print(f"  framerate (last 30 frames): {fps:.1f} fps")
    if total:
        print(f"  left wrist found : {l_seen}/{total} ({100*l_seen/total:.0f}%)")
        print(f"  right wrist found: {r_seen}/{total} ({100*r_seen/total:.0f}%)")
    print("=" * 52)
    print("How to read this:")
    print("  - fps comfortably above ~15 and near your hand-only baseline -> speed OK")
    print("  - wrist-found > ~90% while your arms were in view -> reliable enough for P8")
    print("  - if either fails, especially at the low/tilted angle, that's the")
    print("    signal to benchmark RTMPose next.")


def tilt_tips():
    print("""
Testing the LOW, TILTED-UP angle (the robot's real viewpoint)
-------------------------------------------------------------
Your webcam normally sits at eye level and looks straight at you. The robot's
D435i sits ~45-50 cm off the floor and looks UP at ~20 degrees. Pose behaves
differently there, so test it properly:

  1. Put the laptop/webcam low - on the floor, or a low stool - and prop the
     lid so the camera tilts UP toward you.
  2. Stand up, a couple of metres back, as a person would when commanding the
     robot.
  3. Run:  python3 benchmark_pose.py
  4. Watch the red wrist circles as you raise a hand to gesture. Do they stay
     on your wrist? Or jump/vanish when your arm is up?

Why it matters: the operator lock pairs hands to wrists. If the wrist is
unreliable at this angle, that pairing is unreliable, and that's better to
find out now than after recording a dataset.

Then run the real-cost version:  python3 benchmark_pose.py --combined
""")


if __name__ == "__main__":
    if "--tilt" in sys.argv:
        tilt_tips()
    elif "--combined" in sys.argv:
        run_combined()
    else:
        run_pose_only()
