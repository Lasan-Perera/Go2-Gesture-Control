#!/usr/bin/env python3
"""
gesture_common.py

Shared building blocks for the gesture-control project. Imported by:
    - collect_gestures.py   (recording training samples)
    - train_gestures.py     (training the classifier)
    - gesture_control.py    (live inference)

Deliberately depends on numpy and config ONLY (no mediapipe, no cv2, no
sklearn), so that all three scripts compute features through the exact same
code path. If feature extraction lived separately in the collector and the
runner, they would eventually drift apart and the model would silently
degrade - this is the classic "train/serve skew" bug.
"""

import os
import re

import numpy as np

from config import (
    WINDOW_FRAMES,
    SWIPE_DX_THRESH,
    SIZE_CHANGE_THRESH,
    STILL_MOTION_THRESH,
    STOP_HOLD_FRAMES,
)

# --------------------------------------------------------------------------
# Sample format
# --------------------------------------------------------------------------
# A "sample" is a window of consecutive frames, stored as an array of shape
# (WINDOW_FRAMES, 8). Each row is one frame:
#
#     [ t, cx, cy, size, open, closed, nfing, thumb ]
#
#   t      : timestamp (seconds). Recorded for future use; NOT used as a
#            feature, because tying features to frame timing would make the
#            model sensitive to your machine's current FPS.
#   cx, cy : palm center, as a fraction of frame width/height (0..1)
#   size   : palm-size proxy (wrist -> middle knuckle distance). Grows as
#            the hand moves closer to the camera.
#   open   : 1.0 if the hand is open (fingers extended), else 0.0
#   closed : 1.0 if the hand is a fist (fingers curled), else 0.0
#   nfing  : how many of the 4 fingers are extended (0..4)
#   thumb  : 1.0 if the thumb is extended away from the palm, else 0.0
#
# Why nfing and thumb exist
# -------------------------
# The first six columns describe MOTION and nothing else. That is enough to
# separate gestures that move differently (COME scored 0.95, STAY 0.90), and
# useless for gestures that differ only in HAND SHAPE.
#
# It cost us STOP. STOP (fist -> open palm) and BACK OFF (open palm pushed
# forward) are both "open hand moving toward the camera" in motion space, and
# the model confused them in both directions - STOP scored 0.59, the worst in
# the vocabulary and the one gesture that must never fail.
#
# The open/closed flags were already there, but they are BINARY and each needs
# 3 of 4 fingers to agree, so a hand mid-open registers as neither. A graded
# count exposes the whole opening trajectory:
#
#     STOP     : nfing goes 0 -> 4          (delta +4)
#     BACK OFF : nfing stays 4              (delta  0)
#     COME     : nfing stays 2              (two fingers throughout)
#
# Nothing else in the vocabulary changes finger count, so the delta alone
# separates STOP. `thumb` is tracked separately because the thumb abducts
# sideways rather than curling, so the tip/pip test used for the other four
# does not apply to it - and because FOLLOW is defined by an extended thumb.

COL_T, COL_CX, COL_CY, COL_SIZE, COL_OPEN, COL_CLOSED, COL_NFING, COL_THUMB = range(8)

# Columns in one stored frame row. Bumped 6 -> 8 when finger shape was added.
# load_windows() checks this and skips older samples, so mixing old and new
# recordings cannot silently corrupt training.
ROW_WIDTH = 8

# --------------------------------------------------------------------------
# Gesture classes
# --------------------------------------------------------------------------
# NONE is not optional. A classifier trained on only the 5 real gestures is
# forced to assign every window it ever sees to one of them - including your
# hand just resting, or drifting between gestures. NONE gives the model an
# explicit "nothing is happening" answer, and is where most of your recorded
# samples should go.

# The six commands + the wake gesture. Sourced from the Zenodo-27 dataset
# (see extract_dataset.py for the class mapping). Each was chosen so that it
# differs from the others on a feature this project ACTUALLY measures - palm
# centre, palm size, open/closed. Palm ORIENTATION is invisible to these
# features, which is why several "open palm raised" source classes were
# rejected: they look different to a human and identical to the model.

MOVEMENT_CLASSES = [
    "COME",       # 2 fingers + arm toward camera  -> size grows, open=0
    "FOLLOW",     # thumb out + lateral translate  -> high path, high net
    "STOP",       # fist -> open palm              -> open flag flips
    "STAY",       # open palm lowers to floor      -> ndy positive
    "BACK OFF",   # open palm pushed forward       -> size grows, open=1
    "RELEASE",    # lateral wave                   -> high path, ~0 net
]
NONE_CLASS = "NONE"
GESTURE_CLASSES = MOVEMENT_CLASSES + [NONE_CLASS]

# Filesystem-safe folder name <-> class label
def class_to_slug(label):
    return label.replace(" ", "_")


def slug_to_class(slug):
    return slug.replace("_", " ")


# --------------------------------------------------------------------------
# Subject grouping
# --------------------------------------------------------------------------
# Sample files are named so the SUBJECT who performed the gesture can be
# recovered from the filename alone:
#
#     z27_u07_c22_r1_00042.npy   -> subject "z27_u07"   (Zenodo-27, user 7)
#     ipn_u013_00007.npy         -> subject "ipn_u013"  (IPN-Hand, subject 13)
#     sample_0003.npy            -> subject "sample_0003" (own group)
#
# train_gestures.py splits BY SUBJECT, not at random, for two reasons:
#
#   1. It answers the question that actually matters: does this work on a
#      person the model has never seen? A random split lets the SAME person
#      appear in train and test, which flatters the score and tells you
#      nothing about a new operator walking up to the robot.
#
#   2. It makes overlapping windows safe. Near-duplicate windows cut from one
#      clip all carry that clip's subject id, so they cannot straddle the
#      split - which is what lets extract_dataset.py use EXTRACT_STRIDE < a
#      full window and rescue the short classes.
#
# A file with no parseable subject becomes its own group: it can never leak
# against anything else, which is the safe default.

_SUBJECT_RE = re.compile(r"^([a-zA-Z0-9]+_u\d+)")


def subject_from_filename(fname):
    """Recover the subject id from a sample filename. Falls back to the stem."""
    base = os.path.basename(fname)
    m = _SUBJECT_RE.match(base)
    if m:
        return m.group(1)
    return os.path.splitext(base)[0]


# --------------------------------------------------------------------------
# MediaPipe hand landmark indices
# --------------------------------------------------------------------------

WRIST = 0
THUMB_MCP = 2
THUMB_TIP = 4
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

_FINGER_TIP_PIP_PAIRS = [
    (INDEX_TIP, INDEX_PIP),
    (MIDDLE_TIP, MIDDLE_PIP),
    (RING_TIP, RING_PIP),
    (PINKY_TIP, PINKY_PIP),
]

# Thumb tip -> index knuckle distance, in palm-size units, above which the
# thumb counts as extended. Tucked across the palm that ratio sits near 0.4;
# swung clear it passes 1.0. 0.65 is the gap between those two regimes.
#
# This is a structural fact about hand geometry, not a behaviour to tune, so it
# lives here rather than in config.py.
THUMB_EXTENDED_RATIO = 0.65


def palm_center(landmarks):
    """Average of wrist + all MCP (knuckle) points -> stable center of the palm."""
    idxs = [0, 5, 9, 13, 17]
    xs = [landmarks[i].x for i in idxs]
    ys = [landmarks[i].y for i in idxs]
    return float(np.mean(xs)), float(np.mean(ys))


def palm_size(landmarks):
    """Distance from wrist to middle-finger knuckle - grows as hand nears the camera."""
    wx, wy = landmarks[WRIST].x, landmarks[WRIST].y
    mx, my = landmarks[MIDDLE_MCP].x, landmarks[MIDDLE_MCP].y
    return float(np.hypot(mx - wx, my - wy))


def is_hand_open(landmarks):
    """Are at least 3 of the 4 fingers extended (tip higher on screen than pip)?"""
    extended = sum(1 for tip, pip in _FINGER_TIP_PIP_PAIRS
                   if landmarks[tip].y < landmarks[pip].y)
    return extended >= 3


def is_hand_closed(landmarks):
    """Are at least 3 of the 4 fingers curled (tip lower on screen than pip)?"""
    curled = sum(1 for tip, pip in _FINGER_TIP_PIP_PAIRS
                 if landmarks[tip].y > landmarks[pip].y)
    return curled >= 3


def count_extended_fingers(landmarks):
    """
    How many of the four fingers are extended (0..4).

    Same tip-above-pip test the open/closed flags use, but GRADED rather than
    thresholded. The binary flags need 3 of 4 to agree, so a half-open hand
    registers as neither open nor closed and the transition is invisible. The
    count exposes it, which is what separates STOP (0 -> 4) from BACK OFF
    (4 -> 4).
    """
    return sum(1 for tip, pip in _FINGER_TIP_PIP_PAIRS
               if landmarks[tip].y < landmarks[pip].y)


def is_thumb_extended(landmarks):
    """
    Is the thumb held away from the palm?

    The thumb needs its own test: it abducts sideways rather than curling
    toward the wrist, so the tip-above-pip comparison used for the other four
    fingers does not apply to it.

    Instead we measure how far the thumb tip sits from the index knuckle, in
    units of palm size. Tucked across the palm the tip lands close to that
    knuckle; extended, it swings well clear. Dividing by palm size keeps the
    test independent of how near the hand is to the camera, the same trick the
    motion features use.
    """
    ps = palm_size(landmarks)
    if ps <= 1e-6:
        return False
    tx, ty = landmarks[THUMB_TIP].x, landmarks[THUMB_TIP].y
    ix, iy = landmarks[INDEX_MCP].x, landmarks[INDEX_MCP].y
    return (np.hypot(tx - ix, ty - iy) / ps) > THUMB_EXTENDED_RATIO


def frame_row(t, landmarks):
    """Build one (t, cx, cy, size, open, closed, nfing, thumb) row."""
    cx, cy = palm_center(landmarks)
    return (
        float(t),
        cx,
        cy,
        palm_size(landmarks),
        1.0 if is_hand_open(landmarks) else 0.0,
        1.0 if is_hand_closed(landmarks) else 0.0,
        float(count_extended_fingers(landmarks)),
        1.0 if is_thumb_extended(landmarks) else 0.0,
    )


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------

# Feature layout (per sample), all derived from the window:
#   per-frame (7 x WINDOW_FRAMES):
#       ndx[i]  : horizontal displacement from frame 0, normalized by palm size
#       ndy[i]  : vertical displacement from frame 0, normalized by palm size
#       nsz[i]  : palm size relative to frame 0 (1.0 = unchanged, >1 = closer)
#       open[i] : open flag
#       cls[i]  : closed flag
#       nfg[i]  : extended-finger count, 0..4
#       thb[i]  : thumb-extended flag
#   aggregate (17):
#       MOTION (11): net dx / dy / size-change, path length, peak step speed,
#           spread of dx/dy/size, fraction of frames open/closed, and
#           horizontal-vs-vertical dominance (which separates swipes from
#           incidental vertical drift).
#       SHAPE (6): first and last finger count, the CHANGE between them, mean
#           and spread of the count, and fraction of frames with the thumb out.
#
# The shape aggregates exist for STOP. `nfg[-1] - nfg[0]` is +4 for STOP (fist
# opening) and ~0 for BACK OFF (already open), which is the one thing that
# separates two gestures identical in motion space. Nothing else in the
# vocabulary changes finger count, so this feature is close to a STOP detector
# on its own.
#
# Why normalize by palm size? Because a swipe performed close to the camera
# covers far more of the frame than the same swipe performed further away.
# Dividing by palm size (a proxy for distance) makes the features roughly
# scale-invariant - something the old fixed pixel-fraction thresholds could
# never do. Finger count needs no such treatment: it is already a count.

FEATURE_DIM = 7 * WINDOW_FRAMES + 17


def extract_features(window):
    """
    Turn a (WINDOW_FRAMES, ROW_WIDTH) window into a 1-D feature vector.
    Returns None if the window is the wrong shape or degenerate.
    """
    w = np.asarray(window, dtype=np.float64)
    if w.ndim != 2 or w.shape[0] != WINDOW_FRAMES or w.shape[1] != ROW_WIDTH:
        return None

    cx = w[:, COL_CX]
    cy = w[:, COL_CY]
    size = w[:, COL_SIZE]
    op = w[:, COL_OPEN]
    cl = w[:, COL_CLOSED]
    nfg = w[:, COL_NFING]
    thb = w[:, COL_THUMB]

    s0 = float(size[0])
    if not np.isfinite(s0) or s0 <= 1e-6:
        return None  # can't normalize against a zero-size palm

    ndx = (cx - cx[0]) / s0
    ndy = (cy - cy[0]) / s0
    nsz = size / s0

    if not np.all(np.isfinite(ndx)) or not np.all(np.isfinite(ndy)) or not np.all(np.isfinite(nsz)):
        return None

    per_frame = np.concatenate([ndx, ndy, nsz, op, cl, nfg, thb])

    step = np.hypot(np.diff(ndx), np.diff(ndy))
    agg = np.array([
        # --- motion ---
        ndx[-1],                                  # net horizontal travel
        ndy[-1],                                  # net vertical travel
        nsz[-1] - 1.0,                            # net size change (push/pull)
        float(step.sum()),                        # total path length
        float(step.max()) if step.size else 0.0,  # peak per-step speed
        float(ndx.std()),
        float(ndy.std()),
        float(nsz.std()),
        float(op.mean()),                         # fraction of frames hand was open
        float(cl.mean()),                         # fraction of frames hand was a fist
        abs(ndx[-1]) - abs(ndy[-1]),              # horizontal dominance (swipe signature)

        # --- shape ---
        float(nfg[0]),                            # fingers at the start
        float(nfg[-1]),                           # fingers at the end
        float(nfg[-1] - nfg[0]),                  # THE STOP feature: +4 opening, ~0 already open
        float(nfg.mean()),                        # typical finger count (COME sits near 2)
        float(nfg.std()),                         # did the shape change at all?
        float(thb.mean()),                        # fraction of frames thumb was out (FOLLOW)
    ], dtype=np.float64)

    feats = np.concatenate([per_frame, agg])
    if feats.shape[0] != FEATURE_DIM or not np.all(np.isfinite(feats)):
        return None
    return feats


# --------------------------------------------------------------------------
# Rule-based fallback classifier
# --------------------------------------------------------------------------
# This is the original hand-tuned logic. gesture_control.py falls back to it
# if no trained model file is present, so the project still runs end-to-end
# before you've collected any data. Thresholds live in config.py.


def held_still(pts, n_frames, predicate):
    """
    True if the last n_frames samples all sit within STILL_MOTION_THRESH of
    where that stretch started, AND every one satisfies `predicate`.
    `pts` is a sequence of (t, cx, cy, size, open, closed) rows.
    Used for the STOP rule and for the arm/disarm wake gestures.
    """
    if len(pts) < n_frames:
        return False
    window = list(pts)[-n_frames:]
    x0, y0 = window[0][COL_CX], window[0][COL_CY]
    return all(
        abs(p[COL_CX] - x0) < STILL_MOTION_THRESH and
        abs(p[COL_CY] - y0) < STILL_MOTION_THRESH and
        predicate(p)
        for p in window
    )


def classify_gesture_rules(buffer):
    """Original threshold-based classifier. Returns a command string or None."""
    if len(buffer) < WINDOW_FRAMES:
        return None

    pts = list(buffer)[-WINDOW_FRAMES:]
    cx0, cy0, size0 = pts[0][COL_CX], pts[0][COL_CY], pts[0][COL_SIZE]
    cx1, cy1, size1 = pts[-1][COL_CX], pts[-1][COL_CY], pts[-1][COL_SIZE]

    dx = cx1 - cx0
    dy = cy1 - cy0
    dsize = size1 - size0
    horiz, vert = abs(dx), abs(dy)

    if horiz > SWIPE_DX_THRESH and horiz > vert * 1.5 and abs(dsize) < SIZE_CHANGE_THRESH * 1.5:
        # frame is mirrored (selfie view), so dx > 0 = viewer's right
        return "TURN RIGHT" if dx > 0 else "TURN LEFT"

    if abs(dsize) > SIZE_CHANGE_THRESH and horiz < SWIPE_DX_THRESH * 0.7 and vert < SWIPE_DX_THRESH * 0.7:
        return "MOVE FORWARD" if dsize > 0 else "MOVE BACKWARD"

    if held_still(pts, STOP_HOLD_FRAMES, lambda p: p[COL_OPEN] > 0.5):
        return "STOP"

    return None