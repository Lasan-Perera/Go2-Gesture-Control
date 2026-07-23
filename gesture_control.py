#!/usr/bin/env python3
"""
gesture_control.py

Standalone, real-time DYNAMIC hand gesture detector, locked to ONE operator by
BODY POSE rather than by face.

    webcam -> pose + hands -> operator lock -> gesture classification -> dispatch

Gestures recognized (only while ARMED):
    Two fingers, arm toward camera   -> "COME"
    Thumb out, arm moved sideways    -> "FOLLOW"
    Fist springing open              -> "STOP"
    Open palm lowered to the floor   -> "STAY"
    Open palm pushed forward         -> "BACK OFF"
    Hand waved side to side          -> "RELEASE"


DISPATCHING COMMANDS
--------------------
This module never assumes what a command *does*. run() takes a `dispatch`
callback with the signature:

    dispatch(command: str, confidence: float) -> None

The default just prints. To drive a robot instead, pass your own. Note that in
the wider system this node must publish INTENT, never velocity - the IT2-FLS
owns /cmd_vel, because the whole point is that the same gesture produces
different motion in different contexts. A ROS 2 node becomes roughly:

    from gesture_control import run

    def dispatch(command, confidence):
        msg = GestureIntent()
        msg.label = command
        msg.gesture_confidence = confidence      # NOT thresholded downstream
        publisher.publish(msg)

    run(dispatch=dispatch)

That separation is the entire point: the vision pipeline below should not need
to change when the output changes.


CLASSIFICATION (trained model, with rule-based fallback)
--------------------------------------------------------
If gesture_model.pkl exists, gestures are classified by a RandomForest over
windowed motion AND hand-shape features. The model has an explicit "NONE"
class, so it can answer "nothing is happening" rather than being forced to pick
a command for every window of hand motion.

Two guards reduce misfires:
  - CONFIDENCE_THRESHOLD: ignore predictions the model isn't sure about.
  - CONSECUTIVE_AGREE: require the same prediction on N consecutive frames.

STOP is exempt from the first and handled asymmetrically in gesture_model.py:
a false STOP is inconvenient, a missed STOP on a walking robot is not.


OPERATOR LOCK (pose-based)
--------------------------
The robot must obey ONE person and ignore everyone else.

This used to be done with face recognition: find the enrolled face, draw a
control zone around it, accept hands inside that zone. That had a hole in it -
the zone knew WHERE to look, not WHOSE hand it was looking at, so a bystander
reaching into the zone was obeyed. It also could not work at all for this
robot: during a FOLLOW the robot sees the operator's back, the camera sits
~50 cm off the floor and mostly sees chins, and dlib's HOG detector was the
pipeline's single biggest cost on CPU - and will not build on the Jetson.

The replacement uses body pose:

  1. Pose estimation gives every visible person's WRIST positions.
  2. Whoever holds a full open palm still becomes the operator. No face
     database, no enrolment step, no biometric data stored - which also makes
     the ethics application for the human study considerably simpler.
  3. Each detected hand is matched to the nearest OPERATOR wrist, and hands
     that belong to nobody's operator body are discarded. THIS is the actual
     exclusivity mechanism; the face lock never provided it.
  4. The operator is followed frame to frame by shoulder-midpoint continuity,
     which works from behind and through partial occlusion.

The old face-lock implementation is preserved on the `face-identity-lock`
branch. It was a deliberate design, not a mistake - it just answered the wrong
question.


TWO-LEVEL STATE
---------------
    ENROLLED : the system knows who you are and is tracking you
    ARMED    : the system is listening for commands

They are separate on purpose. Disarming should not make the system forget who
you are, and losing you should not leave it armed for whoever walks in next.

    no operator  + open palm held  -> enrolled AND armed
    enrolled     + open palm held  -> armed
    armed        + tight fist held -> disarmed (still enrolled)
    operator pose lost > grace     -> released and disarmed
    armed, nothing for N seconds   -> disarmed (still enrolled)
"""

import os
import time
import warnings
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import config
from gesture_common import (
    COL_NFING,
    COL_THUMB,
    palm_center,
    frame_row,
    held_still,
    classify_gesture_rules,
)
from gesture_model import load_model, predict_gesture
from hand_landmarker import HandTracker

warnings.filterwarnings("ignore", category=UserWarning)


# Pose landmark indices (MediaPipe Pose, 33 points). These are stable.
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16

POSE_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "pose_landmarker_full.task"
)


# --------------------------------------------------------------------------
# Default dispatch
# --------------------------------------------------------------------------

def print_dispatch(command, confidence):
    print(f"[COMMAND] {command}   ({confidence:.0%})")


# --------------------------------------------------------------------------
# Frame-rate instrumentation
# --------------------------------------------------------------------------

class Perf:
    """
    Rolling frame rate and pose-estimation latency.

    Worth having before optimizing anything: "it feels laggy" is not a number,
    and you cannot tell whether the cost is pose, hands, or the display without
    measuring. pose_ms is tracked separately because pose is now the most
    expensive step - and because the Jetson port may have to fall back from the
    `full` model to `lite` plus smoothing, which is a decision that needs a
    measurement behind it.
    """

    def __init__(self, window=30):
        self.frame_times = deque(maxlen=window)
        self.pose_times = deque(maxlen=window)

    def tick(self, dt):
        self.frame_times.append(dt)

    def pose(self, dt):
        self.pose_times.append(dt)

    @property
    def fps(self):
        if not self.frame_times:
            return 0.0
        mean = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / mean if mean > 0 else 0.0

    @property
    def pose_ms(self):
        if not self.pose_times:
            return 0.0
        return 1000.0 * sum(self.pose_times) / len(self.pose_times)


# --------------------------------------------------------------------------
# Pose helpers
# --------------------------------------------------------------------------

def make_pose(model_path=POSE_MODEL_PATH, num_poses=2):
    """
    num_poses=2 so a bystander is SEEN rather than ignored - we need their
    skeleton in order to prove their hand is not the operator's.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Pose model not found at:\n    {model_path}\n\n"
            f"Download it once:\n"
            f"    curl -L -o {model_path} \\\n"
            f"      https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            f"pose_landmarker_full/float16/1/pose_landmarker_full.task\n"
        )
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def pose_anchor(landmarks):
    """
    A stable per-person point for identity continuity: the midpoint between the
    shoulders. Steadier than any single joint, and present whenever the upper
    body is visible - including from behind, which FOLLOW mode requires.
    """
    return np.array([
        (landmarks[L_SHOULDER].x + landmarks[R_SHOULDER].x) / 2.0,
        (landmarks[L_SHOULDER].y + landmarks[R_SHOULDER].y) / 2.0,
    ])


def wrists_of(landmarks):
    """The operator's two wrist points, x/y in [0,1]."""
    return {
        "L": np.array([landmarks[L_WRIST].x, landmarks[L_WRIST].y]),
        "R": np.array([landmarks[R_WRIST].x, landmarks[R_WRIST].y]),
    }


def operator_hand(hands_found, operator_wrists):
    """
    Of all detected hands, return the one belonging to the OPERATOR - the hand
    closest to one of their wrists, provided it is close enough to plausibly be
    attached to that arm.

    This is the whole operator lock in four lines. A bystander's hand sits near
    the bystander's own wrists, far from the operator's, so it fails the
    distance test and is dropped even when it is well inside the frame and
    waving enthusiastically.

    Returns (index, landmarks) or (None, None).
    """
    if not hands_found or operator_wrists is None:
        return None, None

    best_i, best_d = None, None
    for i, hand in enumerate(hands_found):
        cx, cy = palm_center(hand)
        p = np.array([cx, cy])
        d = min(np.linalg.norm(p - wp) for wp in operator_wrists.values())
        if best_d is None or d < best_d:
            best_d, best_i = d, i

    if best_d is not None and best_d <= config.HAND_WRIST_MAX_DIST:
        return best_i, hands_found[best_i]
    return None, None


def draw_pose(frame, landmarks, w, h, color):
    """Upper body only - legs are irrelevant here and just add clutter."""
    def px(i):
        return int(landmarks[i].x * w), int(landmarks[i].y * h)

    for a, b in ((L_SHOULDER, R_SHOULDER),
                 (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
                 (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST)):
        cv2.line(frame, px(a), px(b), color, 2)
    for i in (L_WRIST, R_WRIST):
        cv2.circle(frame, px(i), 8, color, 2)


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

def run(dispatch=print_dispatch, camera_index=0):
    """
    Open the webcam and run the gesture pipeline until the user quits.

    dispatch(command: str, confidence: float) is called once per recognized
    gesture. Everything else - drawing, arming, operator identity - is internal.
    """
    tracker = HandTracker(
        max_num_hands=3,          # operator + a bystander hand, no more
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    pose = make_pose(num_poses=2)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: could not open webcam (device {camera_index}). "
              f"Check camera permissions/connection.")
        return

    # Ask for a smaller frame. Cameras are free to ignore this and hand back
    # the nearest mode they support, so we read the result rather than assume
    # it - a silent 1280x720 would quietly cost ~3x the inference time.
    if config.CAMERA_WIDTH and config.CAMERA_HEIGHT:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {cap_w}x{cap_h}"
          + ("" if (cap_w, cap_h) == (config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
             else "  (camera refused the requested size)"))

    bundle = load_model()
    if bundle is not None:
        print(f"Loaded trained model ({bundle['n_samples']} training samples, "
              f"{len(bundle['classes'])} classes).")
        cv_f1 = bundle.get("cv_macro_f1")
        if cv_f1 is not None:
            print(f"  cross-validated macro F1: {cv_f1:.3f}")
        print(f"  confidence threshold {config.CONFIDENCE_THRESHOLD}, "
              f"requires {config.CONSECUTIVE_AGREE} agreeing frames.")
        print(f"  STOP override at {config.STOP_CONFIDENCE_THRESHOLD} "
              f"(safety asymmetry).")
    else:
        print("No trained model found - using rule-based thresholds.")

    buffer = deque(maxlen=max(config.WINDOW_FRAMES,
                              config.ARM_HOLD_FRAMES,
                              config.DISARM_HOLD_FRAMES))
    pred_history = deque(maxlen=config.CONSECUTIVE_AGREE)
    no_hand_count = 0

    display_command = None
    display_conf = 0.0
    display_until = 0.0

    # --- operator lock state ---
    operator_anchor = None        # shoulder midpoint of the locked operator
    operator_lost_since = None    # when their pose went missing, or None

    # Last pose result. These persist across frames where pose is skipped -
    # that persistence IS the optimisation, so they must live out here rather
    # than being rebuilt each iteration.
    people = []
    operator_lms = None
    frame_idx = 0

    # --- arm state ---
    armed = False
    armed_since = 0.0
    mode_cooldown = 0             # frames before arm/disarm may fire again

    perf = Perf()
    last_frame_t = time.time()

    print("\nHold a FULL OPEN PALM still to become the operator and arm.")
    print("Hold a TIGHT FIST still to disarm.   'r' release lock   'q' quit\n")

    def disarm(reason):
        nonlocal armed
        armed = False
        buffer.clear()
        pred_history.clear()
        print(f"[DISARMED] {reason}")

    def release(reason):
        nonlocal operator_anchor, operator_lost_since, operator_lms
        operator_anchor = None
        operator_lost_since = None
        # Also drop the cached skeleton: on a pose-skipped frame the stale one
        # would otherwise still be treated as the live operator.
        operator_lms = None
        if armed:
            disarm(reason)
        else:
            print(f"[RELEASED] {reason}")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)  # mirror for natural "selfie" interaction
        h, w = frame.shape[:2]
        now = time.time()

        perf.tick(now - last_frame_t)
        last_frame_t = now

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            release("released by user")

        # --- Perception ---
        # Hands run EVERY frame: they are what the gesture features are made
        # of, and a skipped frame is a hole in the motion window.
        #
        # Pose runs every Nth frame and its result is reused in between. It is
        # used for one thing - deciding which hand is the operator's - and a
        # wrist does not travel far in 60 ms. This is what buys back the frame
        # rate; running both on every frame costs roughly half of it.
        frame_idx += 1
        run_pose = (frame_idx % config.POSE_EVERY_N_FRAMES == 0)

        if run_pose:
            t0 = time.time()
            pose_res = pose.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                int(now * 1000),
            )
            perf.pose(time.time() - t0)
            people = pose_res.pose_landmarks if pose_res.pose_landmarks else []

            # --- Re-find the locked operator among the visible people ---
            operator_lms = None
            if operator_anchor is not None and people:
                best_i, best_d = None, None
                for i, lms in enumerate(people):
                    d = np.linalg.norm(pose_anchor(lms) - operator_anchor)
                    if best_d is None or d < best_d:
                        best_d, best_i = d, i
                # Accept only if they did not teleport. A large jump means this
                # is somebody else, not the operator having moved.
                if best_d is not None and best_d < config.OPERATOR_MATCH_MAX_DIST:
                    operator_lms = people[best_i]
                    operator_anchor = pose_anchor(operator_lms)
                    operator_lost_since = None

            # --- SAFETY: never stay locked onto someone who has gone ---
            # The pose equivalent of the old "disarm when the face disappears",
            # but it survives the operator turning their back, which the face
            # never did.
            #
            # Evaluated INSIDE the run_pose branch on purpose. On a skipped
            # frame we did not look, and "did not look" is not evidence of
            # absence - treating it as such is exactly the bug the old
            # frame-counted face grace period had.
            if operator_anchor is not None and operator_lms is None:
                if operator_lost_since is None:
                    operator_lost_since = now
                elif now - operator_lost_since > config.OPERATOR_LOST_SECONDS:
                    release(f"operator gone for {config.OPERATOR_LOST_SECONDS:.0f}s")

        hands_found = tracker.detect(frame, now * 1000.0)

        if mode_cooldown > 0:
            mode_cooldown -= 1

        # --- Which hand, if any, is the operator's? ---
        operator_wrists = wrists_of(operator_lms) if operator_lms is not None else None
        target_idx, landmarks = operator_hand(hands_found, operator_wrists)

        # Before anyone is enrolled there is no operator wrist to match against,
        # so the enrolling palm is simply the first hand we can see.
        if operator_anchor is None and hands_found:
            target_idx, landmarks = 0, hands_found[0]

        detected_this_frame = None
        detected_conf = 0.0

        if landmarks is not None:
            no_hand_count = 0
            buffer.append(frame_row(now, landmarks))

            if now >= display_until:
                if operator_anchor is None:
                    # --- Enrolment: a full open palm held still claims the system ---
                    if (mode_cooldown <= 0 and people and held_still(
                            buffer, config.ARM_HOLD_FRAMES,
                            lambda p: p[COL_NFING] >= 4)):
                        # Whoever's wrist is nearest that palm becomes operator.
                        cx, cy = palm_center(landmarks)
                        p = np.array([cx, cy])
                        best_i, best_d = None, None
                        for i, lms in enumerate(people):
                            d = min(np.linalg.norm(p - wp)
                                    for wp in wrists_of(lms).values())
                            if best_d is None or d < best_d:
                                best_d, best_i = d, i
                        if best_i is not None:
                            operator_anchor = pose_anchor(people[best_i])
                            armed = True
                            armed_since = now
                            mode_cooldown = config.ARM_DISARM_COOLDOWN_FRAMES
                            buffer.clear()
                            pred_history.clear()
                            print("[LOCKED + ARMED] operator enrolled.")

                elif not armed:
                    # Known operator, just not listening. Same palm re-arms.
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
                    # A TIGHT fist - every finger curled AND the thumb wrapped
                    # in. A resting hand is loosely curled with the thumb loose,
                    # and no longer disarms; STOP's opening fist is brief and
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
                                detected_conf = float(
                                    np.mean([c for _, c in pred_history]))
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

        # --- Draw: operator green, everyone else grey ---
        for lms in people:
            is_op = operator_lms is not None and lms is operator_lms
            draw_pose(frame, lms, w, h, (0, 220, 0) if is_op else (110, 110, 110))

        for idx, hand in enumerate(hands_found):
            cx, cy = palm_center(hand)
            px, py = int(cx * w), int(cy * h)
            if idx == target_idx and operator_anchor is not None:
                cv2.circle(frame, (px, py), 16, (0, 255, 0), 3)
            else:
                cv2.circle(frame, (px, py), 12, (90, 90, 90), 2)

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

            if not people:
                status = "no person detected - step into frame"
            elif operator_anchor is None:
                status = "hold FULL OPEN PALM still to become operator"
            elif operator_lms is None:
                remaining = max(0.0, config.OPERATOR_LOST_SECONDS -
                                (now - (operator_lost_since or now)))
                status = f"operator lost - releasing in {remaining:.1f}s"
            elif armed:
                status = "ARMED - listening for commands"
            elif landmarks is None:
                status = "LOCKED - show your hand"
            else:
                status = "LOCKED - hold OPEN PALM still to ARM"

            cv2.putText(frame, status, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # --- Footer: mode + performance ---
        mode = "model" if bundle is not None else "rules"
        footer = mode
        if config.SHOW_FPS:
            footer = (f"{perf.fps:4.1f} fps | pose {perf.pose_ms:3.0f} ms | "
                      f"{w}x{h} | {len(people)} people | {mode}")
        cv2.putText(frame, footer, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow("Gesture Control (pose-locked operator)", frame)

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()
    pose.close()


def main():
    run(dispatch=print_dispatch)


if __name__ == "__main__":
    main()