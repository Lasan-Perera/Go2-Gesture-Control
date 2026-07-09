#!/usr/bin/env python3
"""
gesture_common.py

Shared building blocks for the gesture-control project. Imported by:
    - collect_gestures.py   (recording training samples)
    - train_gestures.py     (training the classifier)
    - gesture_control.py    (live inference)

Deliberately depends on numpy ONLY (no mediapipe, no cv2, no sklearn), so
that all three scripts compute features through the exact same code path.
If feature extraction lived separately in the collector and the runner,
they would eventually drift apart and the model would silently degrade -
this is the classic "train/serve skew" bug.
"""

import numpy as np

# --------------------------------------------------------------------------
# Sample format
# --------------------------------------------------------------------------
# A "sample" is a window of consecutive frames, stored as an array of shape
# (WINDOW_FRAMES, 6). Each row is one frame:
#
#     [ t, cx, cy, size, open, closed ]
#
#   t      : timestamp (seconds). Recorded for future use; NOT used as a
#            feature, because tying features to frame timing would make the
#            model sensitive to your machine's current FPS.
#   cx, cy : palm center, as a fraction of frame width/height (0..1)
#   size   : palm-size proxy (wrist -> middle knuckle distance). Grows as
#            the hand moves closer to the camera.
#   open   : 1.0 if the hand is open (fingers extended), else 0.0
#   closed : 1.0 if the hand is a fist (fingers curled), else 0.0

WINDOW_FRAMES = 12   # frames per sample. Must match between collection and inference.

COL_T, COL_CX, COL_CY, COL_SIZE, COL_OPEN, COL_CLOSED = range(6)

# --------------------------------------------------------------------------
# Gesture classes
# --------------------------------------------------------------------------
# NONE is not optional. A classifier trained on only the 5 real gestures is
# forced to assign every window it ever sees to one of them - including your
# hand just resting, or drifting between gestures. NONE gives the model an
# explicit "nothing is happening" answer, and is where most of your recorded
# samples should go.

MOVEMENT_CLASSES = [
    "TURN LEFT",
    "TURN RIGHT",
    "MOVE FORWARD",
    "MOVE BACKWARD",
    "STOP",
]
NONE_CLASS = "NONE"
GESTURE_CLASSES = MOVEMENT_CLASSES + [NONE_CLASS]

# Filesystem-safe folder name <-> class label
def class_to_slug(label):
    return label.replace(" ", "_")


def slug_to_class(slug):
    return slug.replace("_", " ")


# --------------------------------------------------------------------------
# MediaPipe hand landmark indices
# --------------------------------------------------------------------------

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

_FINGER_TIP_PIP_PAIRS = [
    (INDEX_TIP, INDEX_PIP),
    (MIDDLE_TIP, MIDDLE_PIP),
    (RING_TIP, RING_PIP),
    (PINKY_TIP, PINKY_PIP),
]


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


def frame_row(t, landmarks):
    """Build one (t, cx, cy, size, open, closed) row from raw hand landmarks."""
    cx, cy = palm_center(landmarks)
    return (
        float(t),
        cx,
        cy,
        palm_size(landmarks),
        1.0 if is_hand_open(landmarks) else 0.0,
        1.0 if is_hand_closed(landmarks) else 0.0,
    )


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------

# Feature layout (per sample), all derived from the window:
#   per-frame (5 x WINDOW_FRAMES):
#       ndx[i]  : horizontal displacement from frame 0, normalized by palm size
#       ndy[i]  : vertical displacement from frame 0, normalized by palm size
#       nsz[i]  : palm size relative to frame 0 (1.0 = unchanged, >1 = closer)
#       open[i] : open flag
#       cls[i]  : closed flag
#   aggregate (11):
#       net dx / dy / size-change, path length, peak step speed, spread of
#       dx/dy/size, fraction of frames open/closed, and horizontal-vs-vertical
#       dominance (which separates swipes from incidental vertical drift).
#
# Why normalize by palm size? Because a swipe performed close to the camera
# covers far more of the frame than the same swipe performed further away.
# Dividing by palm size (a proxy for distance) makes the features roughly
# scale-invariant - something the old fixed pixel-fraction thresholds could
# never do.

FEATURE_DIM = 5 * WINDOW_FRAMES + 11


def extract_features(window):
    """
    Turn a (WINDOW_FRAMES, 6) window into a 1-D feature vector.
    Returns None if the window is the wrong length or degenerate.
    """
    w = np.asarray(window, dtype=np.float64)
    if w.ndim != 2 or w.shape[0] != WINDOW_FRAMES or w.shape[1] != 6:
        return None

    cx = w[:, COL_CX]
    cy = w[:, COL_CY]
    size = w[:, COL_SIZE]
    op = w[:, COL_OPEN]
    cl = w[:, COL_CLOSED]

    s0 = float(size[0])
    if not np.isfinite(s0) or s0 <= 1e-6:
        return None  # can't normalize against a zero-size palm

    ndx = (cx - cx[0]) / s0
    ndy = (cy - cy[0]) / s0
    nsz = size / s0

    if not np.all(np.isfinite(ndx)) or not np.all(np.isfinite(ndy)) or not np.all(np.isfinite(nsz)):
        return None

    per_frame = np.concatenate([ndx, ndy, nsz, op, cl])

    step = np.hypot(np.diff(ndx), np.diff(ndy))
    agg = np.array([
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
# before you've collected any data.

SWIPE_DX_THRESH = 0.16
SIZE_CHANGE_THRESH = 0.045
STILL_MOTION_THRESH = 0.015
STOP_HOLD_FRAMES = 10


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
