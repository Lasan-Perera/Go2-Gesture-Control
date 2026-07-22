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


# --------------------------------------------------------------------------
# STOP safety asymmetry
# --------------------------------------------------------------------------
# STOP is not just another class. The two ways of getting it wrong are not
# equally bad:
#
#   missed STOP  - the operator says stop, the robot keeps going, or (worse)
#                  reads it as BACK OFF and moves. On 15 kg of walking robot
#                  that is the failure that matters.
#   false STOP   - the robot halts when nobody asked. Annoying. Harmless.
#
# So STOP gets a lower bar than every other command: whenever the classifier
# gives STOP at least this much probability, STOP is emitted even if another
# label scored higher. Everything else still has to clear CONFIDENCE_THRESHOLD.
#
# This is deliberately NOT symmetric and NOT a tie-break - it is an override.
#
# Tune it from data, not by feel: train_gestures.py prints a sweep of this
# value against the held-out set showing exactly how many missed STOPs each
# threshold recovers and how many false STOPs it costs. Lower = safer and
# twitchier. Set to 1.01 to disable the override entirely.
STOP_CLASS = "STOP"
STOP_CONFIDENCE_THRESHOLD = 0.35

DISPLAY_HOLD_SECONDS = 0.9   # how long a detected command stays on screen
NO_HAND_RESET_FRAMES = 8     # clear the motion buffer after this many hand-less frames

SHOW_FPS = True              # draw a frame-rate / latency readout


# --------------------------------------------------------------------------
# Arm / disarm (wake gesture)
# --------------------------------------------------------------------------
# These used to test is_hand_open / is_hand_closed, which need only 3 of the 4
# fingers to agree. That was fine for the old swipe vocabulary, where no real
# gesture ever held still. It is not fine now:
#
#   - a RELAXED hand is a loosely curled hand held still, so simply pausing
#     disarmed the system
#   - STOP begins as a fist, so starting a STOP disarmed the system
#   - STAY and BACK OFF are open palms, so slowing down mid-gesture armed it
#
# The finger count fixes this. A deliberate fist has ALL four fingers curled
# and the thumb wrapped in; a relaxed hand almost never does. A deliberate open
# palm has all four extended. Requiring the extremes leaves the ambiguous
# middle - which is where resting hands and gesture transitions live - doing
# nothing at all.

ARM_HOLD_FRAMES = 20         # frames of held FULL open palm (all 4 fingers) to ARM
DISARM_HOLD_FRAMES = 40      # frames of held TIGHT fist (0 fingers, thumb in) to DISARM
ARMED_TIMEOUT_SECONDS = 20   # auto-disarm if no command fires for this long

# Frames to ignore arm/disarm for after either one fires.
#
# Without this, STOP (fist -> open palm) can disarm on its opening frames and
# re-arm on its closing frames, all inside one gesture. The cooldown makes each
# state change cost a deliberate pause, which is what a mode switch should cost.
ARM_DISARM_COOLDOWN_FRAMES = 30

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
# Dataset extraction (extract_dataset.py)
# --------------------------------------------------------------------------

# Stride between windows cut from a dataset clip. DELIBERATELY smaller than
# WINDOW_FRAMES - i.e. overlapping windows - which is normally the leakage bug
# described above.
#
# It is safe HERE, and only here, because the dataset has 21 named subjects and
# train_gestures.py splits BY SUBJECT (GroupKFold). Every window cut from one
# person's clip carries that person's id, so all of them land on the same side
# of the split. A near-duplicate cannot leak across a boundary it can't cross.
#
# This matters most for STOP: the source gesture (fist -> open palm) averages
# only ~16 frames, so a non-overlapping stride yields barely one window per
# clip and starves the single most safety-critical class.
EXTRACT_STRIDE = 4

# Temporal augmentation: sample every Nth frame to synthesise the same gesture
# performed at different speeds.
#
# The source videos are 30 fps. The live pipeline runs ~16-20 fps once pose
# estimation is alongside hand tracking. A 12-frame window therefore spans
# ~0.4 s in the dataset but ~0.6-0.75 s live, so an identical gesture presents
# with different per-frame displacement. Feeding the model both stride-1 and
# stride-2 versions teaches it the gesture SHAPE rather than one frame rate's
# idea of its speed.
#
# Stride 2 needs 24 source frames for a 12-frame window, so it simply doesn't
# fire on the shortest clips - that's expected, not an error.
TEMPORAL_STRIDES = (1, 2)

# Frames to add either side of the annotated gesture segment.
#
# The annotation marks the gesture proper, but STOP (fist -> open palm) is only
# ~16 frames, barely longer than one window. The frames immediately either side
# are not noise: they are the hand already in fist, or already open, which is
# exactly the state the open/closed flags encode. A small pad recovers usable
# windows for the short classes without dragging in the long drift into and out
# of position.
TIMING_PAD = 4

# Ceiling on windows taken from a single clip.
#
# Overlapping windows scale with clip length, so without a cap the long classes
# run away: ATTENTION (~86 frames) would yield ~35 windows per clip while STOP
# (~16 frames) yields 2 - a 17x imbalance that is an artefact of gesture
# duration, not of how much there is to learn. Sampling evenly across the clip
# up to this many windows keeps every class in the same order of magnitude, and
# class_weight="balanced" mops up the rest.
MAX_WINDOWS_PER_CLIP = 8


# --------------------------------------------------------------------------
# IPN-Hand NONE extraction (extract_ipn_none.py)
# --------------------------------------------------------------------------

# Only the D0X (non-gesture) segments become NONE.
#
# NOT B0A/B0B ("pointing with one/two fingers"): B0B is two fingers moving
# around, and COME is two fingers moved toward the camera. Labelling B0B as
# NONE would hand the model contradictory labels for near-identical motion and
# degrade both classes - the same trap as recording STOP during a NONE session.
# The G0x classes are worse still (G05/G06 throw left/right resemble RELEASE;
# G10/G11 zoom in/out resemble COME/BACK OFF).
IPN_NONE_LABEL = "D0X"

# Windows per D0X segment. Deliberately far below MAX_WINDOWS_PER_CLIP.
#
# There are 1431 D0X segments averaging 147 frames. At 8 windows each that is
# ~11,000 NONE windows against a largest command class of 986 - NONE would stop
# being a class and start being the answer. At 2 each it lands near 2,800,
# roughly 3x the biggest command class, which is the ratio that suppresses
# false positives without swamping training.
#
# Two windows from 1431 different segments also beats eight from fewer: NONE
# needs BREADTH ("all the ways nothing is happening"), not depth.
IPN_MAX_WINDOWS_PER_SEGMENT = 2

# Frames sampled from the middle of each D0X segment.
#
# We only need enough to build the capped number of windows, so there is no
# reason to run the landmarker over an entire 147-frame segment and throw 90%
# of it away. Sampling the middle avoids the edges, where the hand is still
# arriving from (or leaving toward) the neighbouring real gesture.
IPN_FRAMES_PER_SEGMENT = 40


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