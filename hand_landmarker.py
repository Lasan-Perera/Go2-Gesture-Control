#!/usr/bin/env python3
"""
hand_landmarker.py

A thin wrapper around MediaPipe's *Tasks* HandLandmarker that presents its
output in the SAME shape the old mp.solutions.Hands API produced.

Why this file exists
--------------------
The rest of the project (frame_row, palm_center, palm_size, pick_zone_hand,
the finger open/closed tests) reads hand landmarks as:

    landmarks[i].x        # i in 0..20, x/y in [0,1], z relative depth

The legacy `mp.solutions.hands` API is deprecated and already broken on
current mediapipe (0.10.31+ on Python 3.12 throws "module 'mediapipe' has no
attribute 'solutions'"). The Tasks API is the supported replacement and the
only path onto the Jetson (it uses a portable .task model bundle).

But the Tasks API hands you results in a slightly different shape:

    result.hand_landmarks[hand_index][i].x     # a plain list, no .landmark

This wrapper hides that difference. It returns a list of "hands", where each
hand is directly indexable as landmarks[i].x/.y/.z - exactly what the old
`.multi_hand_landmarks[k].landmark` gave you. So NOTHING downstream changes:
gesture_common.py, feature extraction, the trained model, all untouched.

This is the whole point of keeping gesture_common.py free of mediapipe: the
API swap is isolated to the two files that actually capture frames.

Usage
-----
    from hand_landmarker import HandTracker

    tracker = HandTracker(max_num_hands=2)
    ...
    hands = tracker.detect(frame_bgr, timestamp_ms)   # list of hand-landmark lists
    if hands:
        landmarks = hands[0]          # first hand, indexable landmarks[i].x
        cx, cy = palm_center(landmarks)

Handedness (which was awkward on the legacy API) is available too:
    tracker.last_handedness  ->  ["Right", ...] parallel to the hands list.
This is not used yet, but the operator-lock work (hand-to-wrist association)
will want it, and it now comes for free.
"""

import os

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


# Default location of the model bundle. Download once with:
#   mkdir -p models
#   curl -L -o models/hand_landmarker.task \
#     https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "hand_landmarker.task"
)


class HandTracker:
    """
    Wraps a Tasks HandLandmarker in VIDEO running mode and returns landmarks
    in the legacy `.landmark`-style shape.

    VIDEO mode (not IMAGE) is deliberate: it lets the landmarker use the
    previous frame's hand box to track across frames instead of re-running
    palm detection every frame, which is both faster and steadier - and
    steadiness across a 12-frame window is exactly what the gesture features
    depend on. VIDEO mode requires a monotonically increasing timestamp on
    every call, which detect() enforces.
    """

    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        min_presence_confidence=0.6,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Hand landmarker model not found at:\n    {model_path}\n\n"
                f"Download it once with:\n"
                f"    mkdir -p {os.path.dirname(model_path)}\n"
                f"    curl -L -o {model_path} \\\n"
                f"      https://storage.googleapis.com/mediapipe-models/"
                f"hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task\n"
            )

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

        # VIDEO mode demands strictly increasing timestamps. If two frames ever
        # arrive with the same millisecond, we nudge forward by 1 ms so the
        # call doesn't raise.
        self._last_ts_ms = -1

        # Populated on every detect(): handedness label per returned hand,
        # parallel to the returned list. e.g. ["Right", "Left"].
        self.last_handedness = []

    def detect(self, frame_bgr, timestamp_ms):
        """
        Run hand landmarking on one BGR frame (the format OpenCV gives you).

        Returns a list of hands. Each hand is a list of 21 landmark objects,
        indexable as landmarks[i].x / .y / .z - the same shape the old
        `.multi_hand_landmarks[k].landmark` produced.

        Empty list => no hands found this frame.
        """
        ts = int(timestamp_ms)
        if ts <= self._last_ts_ms:
            ts = self._last_ts_ms + 1
        self._last_ts_ms = ts

        # OpenCV is BGR; MediaPipe wants RGB.
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._landmarker.detect_for_video(mp_image, ts)

        # result.hand_landmarks is already a list of lists of landmark objects
        # with .x/.y/.z. That is exactly the shape we want to hand back, so no
        # per-landmark conversion is needed - just surface handedness alongside.
        self.last_handedness = []
        if result.handedness:
            for hand_cats in result.handedness:
                # hand_cats is a list of Category; index 0 is the top label.
                self.last_handedness.append(
                    hand_cats[0].category_name if hand_cats else "Unknown"
                )

        return result.hand_landmarks if result.hand_landmarks else []

    def close(self):
        """Release the underlying landmarker. Safe to call more than once."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    # Context-manager sugar, so callers can `with HandTracker() as t:` if they
    # want deterministic cleanup.
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
