#!/usr/bin/env python3
"""
verify_migration.py

Prove that the Tasks API produces the same features as the old mp.solutions
API, BEFORE you record a dataset on it.

Why this matters
----------------
The whole project depends on training data and live inference computing the
exact same features (this is why gesture_common.py exists). Swapping the hand
tracker underneath is only safe if the new tracker produces landmark positions
close enough that extract_features() gives effectively the same vector. If the
two APIs disagreed meaningfully, a model trained on Tasks-API data would
misfire on... Tasks-API inference too, but more importantly, any comparison to
your old frozen model would be meaningless, and you would not know which half
was wrong.

This script runs BOTH trackers on the same webcam frames and reports, per
frame, how far apart their palm-center, palm-size, and open/closed readings
are. Small, stable differences = safe to proceed. Large or erratic = stop and
investigate before recording.

Requirements
------------
This is the ONE place both APIs must coexist. Run it in an environment that
still has the legacy API working - i.e. your current pinned setup with
mediapipe==0.10.14 - PLUS the models/hand_landmarker.task bundle for the new
API. If your environment has already moved past 0.10.14 and mp.solutions is
broken, you cannot run the legacy half; in that case skip straight to
verify_features_only() below, which just sanity-checks the new path.

Usage
-----
    python3 verify_migration.py            # side-by-side, needs both APIs
    python3 verify_migration.py --newonly  # only exercise the Tasks path

Press 'q' to quit. Watch the printed deltas; they should be small (< ~0.02 in
normalized units) and steady.
"""

import sys
import time

import cv2
import numpy as np

from gesture_common import palm_center, palm_size, is_hand_open, is_hand_closed
from hand_landmarker import HandTracker


def read_new(tracker, frame, ts_ms):
    """Get (cx, cy, size, open, closed) from the Tasks API, or None."""
    hands = tracker.detect(frame, ts_ms)
    if not hands:
        return None
    lm = hands[0]
    cx, cy = palm_center(lm)
    return (cx, cy, palm_size(lm), is_hand_open(lm), is_hand_closed(lm))


def read_legacy(hands_legacy, frame):
    """Get (cx, cy, size, open, closed) from mp.solutions, or None."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands_legacy.process(rgb)
    if not result.multi_hand_landmarks:
        return None
    lm = result.multi_hand_landmarks[0].landmark
    cx, cy = palm_center(lm)
    return (cx, cy, palm_size(lm), is_hand_open(lm), is_hand_closed(lm))


def verify_features_only():
    """Just run the new path and confirm it yields sane, stable readings."""
    tracker = HandTracker(max_num_hands=1)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: no webcam.")
        return
    print("New-path-only check. Show your hand; press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        vals = read_new(tracker, frame, time.time() * 1000.0)
        if vals:
            cx, cy, size, op, cl = vals
            txt = f"cx={cx:.3f} cy={cy:.3f} size={size:.3f} open={int(op)} closed={int(cl)}"
        else:
            txt = "no hand"
        cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("verify (new only)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    tracker.close()


def verify_side_by_side():
    """Run both APIs on the same frames and report per-frame deltas."""
    try:
        import mediapipe as mp
        hands_legacy = mp.solutions.hands.Hands(
            model_complexity=0, max_num_hands=1,
            min_detection_confidence=0.6, min_tracking_confidence=0.6,
        )
    except Exception as e:
        print(f"Legacy mp.solutions unavailable ({e}).")
        print("Falling back to --newonly. To run the real comparison, use an")
        print("environment with mediapipe==0.10.14 where mp.solutions still works.")
        verify_features_only()
        return

    tracker = HandTracker(max_num_hands=1)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: no webcam.")
        return

    print("Side-by-side check. Show your hand and move it around.")
    print("Watch the deltas - they should stay small and steady.\n")

    d_center, d_size = [], []
    open_agree = closed_agree = both_seen = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        ts = time.time() * 1000.0

        new = read_new(tracker, frame, ts)
        old = read_legacy(hands_legacy, frame)

        if new and old:
            both_seen += 1
            dc = np.hypot(new[0] - old[0], new[1] - old[1])
            ds = abs(new[2] - old[2])
            d_center.append(dc)
            d_size.append(ds)
            open_agree += int(new[3] == old[3])
            closed_agree += int(new[4] == old[4])
            txt = f"dCenter={dc:.4f}  dSize={ds:.4f}"
            color = (0, 255, 0) if dc < 0.02 else (0, 165, 255)
        else:
            txt = "need both APIs to see the hand"
            color = (150, 150, 150)

        cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow("verify (legacy vs tasks)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()

    if both_seen:
        print("\n" + "=" * 52)
        print(f"Frames where both APIs saw the hand: {both_seen}")
        print(f"  palm-center delta : mean {np.mean(d_center):.4f}  "
              f"max {np.max(d_center):.4f}   (normalized 0-1)")
        print(f"  palm-size delta   : mean {np.mean(d_size):.4f}  "
              f"max {np.max(d_size):.4f}")
        print(f"  open flag agree   : {open_agree}/{both_seen} "
              f"({100*open_agree/both_seen:.0f}%)")
        print(f"  closed flag agree : {closed_agree}/{both_seen} "
              f"({100*closed_agree/both_seen:.0f}%)")
        print("=" * 52)
        if np.mean(d_center) < 0.02 and open_agree > 0.9 * both_seen:
            print("VERDICT: APIs agree closely. Safe to record on the Tasks API.")
        else:
            print("VERDICT: notable disagreement. Investigate before recording -")
            print("  check lighting, and that both are set to max_num_hands=1.")
    else:
        print("\nNever saw the hand in both APIs at once - inconclusive.")


if __name__ == "__main__":
    if "--newonly" in sys.argv:
        verify_features_only()
    else:
        verify_side_by_side()
