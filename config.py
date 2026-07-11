#!/usr/bin/env python3
"""
config.py

Every tunable constant for the project, in one place.

This module deliberately imports nothing. It sits at the bottom of the
dependency graph so that gesture_common, gesture_model, collect_gestures,
train_gestures and gesture_control can all import from it without any risk
of a circular import.

Structural facts that are NOT tunable (the gesture class list, the column
layout of a sample, MediaPipe landmark indices) live in gesture_common.py
instead - changing those changes the meaning of the data, not its behaviour.
"""

# --------------------------------------------------------------------------
# Sample window
# --------------------------------------------------------------------------
# Frames per sample. Changing this invalidates every recorded sample AND any
# trained model - both encode this length. gesture_model.load_model() checks
# it and refuses a mismatched model rather than silently predicting garbage.
WINDOW_FRAMES = 12


# --------------------------------------------------------------------------
# Live inference
# --------------------------------------------------------------------------

# Minimum probability the winning class needs before we act on it.
# Raise if you get false positives; lower if real gestures are ignored.
CONFIDENCE_THRESHOLD = 0.60

# Require the model to predict the same gesture this many frames in a row
# before firing. 1 = fire immediately (twitchy). 2-3 = noticeably steadier.
CONSECUTIVE_AGREE = 2

DISPLAY_HOLD_SECONDS = 0.9   # how long a detected command stays on screen
NO_HAND_RESET_FRAMES = 8     # clear the motion buffer after this many hand-less frames

SHOW_FPS = True              # draw a frame-rate / latency readout


# --------------------------------------------------------------------------
# Arm / disarm (wake gesture)
# --------------------------------------------------------------------------

ARM_HOLD_FRAMES = 15         # frames of held open palm (inside the zone) to ARM
DISARM_HOLD_FRAMES = 15      # frames of held closed fist (inside the zone) to DISARM
ARMED_TIMEOUT_SECONDS = 20   # auto-disarm if no command fires for this long

# SAFETY: disarm once the enrolled operator's face has been gone for a while.
#
# Without this, walking out of shot while ARMED leaves the system armed. You
# return some time later and the very first hand movement you make - reaching
# for the keyboard, scratching your nose - is dispatched as a command. Harmless
# when the output is a print(); not harmless when the output is a robot.
#
# Re-arming should always be a deliberate act.
DISARM_ON_FACE_LOST = True

# How long the face must be CONTINUOUSLY missing before we disarm.
#
# This is deliberately much longer than FACE_ZONE_GRACE_SECONDS below. They
# answer different questions:
#
#   "keep using the last known control zone?"  -> your face has not teleported,
#                                                 a short grace is plenty
#   "has the operator actually left?"          -> needs a long grace; raising
#                                                 your hand in front of your own
#                                                 face while gesturing is normal,
#                                                 and must not disarm you
#
# Set generously. Occlusion by your own hand, turning your head, or a person
# walking past should all ride through this untouched.
DISARM_GRACE_SECONDS = 5.0


# --------------------------------------------------------------------------
# Face identity lock
# --------------------------------------------------------------------------

ENROLLED_FACE_PATH = "target_face.npy"

FACE_MATCH_THRESHOLD = 0.55     # lower = stricter (face_recognition "distance")
FACE_DOWNSCALE = 0.35           # shrink the frame before face detection, for speed
FACE_DETECT_EVERY_N_FRAMES = 2  # run face ID every Nth frame; reuse the zone between

# How long to keep using the last known control zone after losing the face.
#
# Time-based, not frame-count-based, on purpose. A frame count silently means
# something different at 30 fps than at 10 fps - and this pipeline's frame rate
# swings with dlib's mood. Seconds mean seconds.
FACE_ZONE_GRACE_SECONDS = 1.5

# Control zone, as multiples of the detected face box. This is what follows
# you around the frame.
ZONE_SIDE_MARGIN = 2.2   # left/right extent, in face widths
ZONE_ABOVE_MARGIN = 0.3  # extent above the face, in face heights
ZONE_BELOW_MARGIN = 4.5  # extent below the face (down to roughly waist/hands)


# --------------------------------------------------------------------------
# Rule-based fallback classifier
# --------------------------------------------------------------------------
# Used only when no trained model is present, so the project still runs
# end-to-end before any data has been recorded.

SWIPE_DX_THRESH = 0.16      # min horizontal travel (frame fractions) for a swipe
SIZE_CHANGE_THRESH = 0.045  # min palm-size change for push/pull
STILL_MOTION_THRESH = 0.015 # max movement that still counts as "held still"
STOP_HOLD_FRAMES = 10       # consecutive still+open frames needed for STOP


# --------------------------------------------------------------------------
# Data collection
# --------------------------------------------------------------------------

DATA_DIR = "data"
COUNTDOWN_SECONDS = 3

# In continuous mode, save a new window every N frames. Set to WINDOW_FRAMES so
# consecutive samples share NO frames.
#
# With a shorter stride, consecutive samples overlap heavily - they are
# near-duplicates. Those near-duplicates then land on both sides of the
# train/test split, so the model is scored on windows it has effectively
# already seen. Reported accuracy rises; live performance does not.
CONTINUOUS_STRIDE = WINDOW_FRAMES


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

MODEL_PATH = "gesture_model.pkl"

MIN_SAMPLES_PER_CLASS = 10   # hard floor - below this, training is meaningless
RECOMMENDED_PER_CLASS = 25   # soft warning threshold

AUGMENT = True
N_NOISE_COPIES = 2           # noisy variants per (original, mirrored) window
NOISE_POS_STD = 0.004        # landmark jitter on palm center, in frame fractions
NOISE_SIZE_STD = 0.0015      # landmark jitter on the palm-size proxy