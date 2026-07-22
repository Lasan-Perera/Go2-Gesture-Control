#!/usr/bin/env python3
"""
gesture_model.py

Loads the trained classifier and runs inference on a live motion buffer.
Kept separate from gesture_control.py (and free of mediapipe/cv2 imports)
so it can be tested without a webcam.
"""

import os

import numpy as np

from config import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    WINDOW_FRAMES,
    STOP_CLASS,
    STOP_CONFIDENCE_THRESHOLD,
)
from gesture_common import (
    FEATURE_DIM,
    NONE_CLASS,
    extract_features,
)


def load_model(path=MODEL_PATH):
    """
    Return the model bundle, or None if there's no usable model on disk.
    Never raises - a missing or stale model must not crash the live loop,
    it should just fall back to the rule-based classifier.
    """
    if not os.path.exists(path):
        return None
    try:
        import joblib
        bundle = joblib.load(path)
    except Exception as e:  # corrupt file, sklearn version mismatch, etc.
        print(f"WARNING: could not load {path} ({e}). Falling back to rules.")
        return None

    # A model trained with a different window length or feature layout would
    # silently produce garbage predictions. Refuse it loudly instead.
    if bundle.get("window_frames") != WINDOW_FRAMES or bundle.get("feature_dim") != FEATURE_DIM:
        print(f"WARNING: {path} was trained with a different feature layout "
              f"(window={bundle.get('window_frames')}, dim={bundle.get('feature_dim')}; "
              f"expected window={WINDOW_FRAMES}, dim={FEATURE_DIM}). "
              f"Re-run train_gestures.py. Falling back to rules.")
        return None

    return bundle


def predict_gesture(buffer, bundle):
    """
    Classify the most recent WINDOW_FRAMES of `buffer`.

    Returns (label, confidence). `label` is None when the model is not
    confident enough, or when it believes nothing is happening (NONE).

    STOP is handled asymmetrically - see apply_stop_override().
    """
    if bundle is None or len(buffer) < WINDOW_FRAMES:
        return None, 0.0

    window = list(buffer)[-WINDOW_FRAMES:]
    feats = extract_features(window)
    if feats is None:
        return None, 0.0

    model = bundle["model"]
    probs = model.predict_proba(feats.reshape(1, -1))[0]
    return decide(probs, list(model.classes_))


def apply_stop_override(probs, classes):
    """
    Emit STOP whenever it has at least STOP_CONFIDENCE_THRESHOLD probability,
    even if another label scored higher.

    Why an override and not a tie-break: the classifier confuses STOP with
    BACK OFF in both directions (they are both "open hand moving toward the
    camera" once the fingers have finished opening), and the two errors are not
    equally costly. Reading BACK OFF as STOP makes the robot halt when it was
    asked to retreat - inconvenient. Reading STOP as BACK OFF makes the robot
    move when it was asked to halt - the failure this whole system exists to
    avoid.

    Returns (index, label, confidence) of the winning class.
    """
    best = int(np.argmax(probs))
    label = str(classes[best])

    if STOP_CLASS in classes and label != STOP_CLASS:
        i_stop = classes.index(STOP_CLASS)
        if float(probs[i_stop]) >= STOP_CONFIDENCE_THRESHOLD:
            return i_stop, STOP_CLASS, float(probs[i_stop])

    return best, label, float(probs[best])


def decide(probs, classes):
    """
    Turn class probabilities into a dispatched command, or None.

    Split out from predict_gesture so the exact same decision rule can be
    replayed over the held-out set in train_gestures.py - if the sweep there
    used different logic from the live loop, its numbers would be fiction.
    """
    _, label, conf = apply_stop_override(probs, classes)

    if label == NONE_CLASS:
        return None, conf

    # STOP already cleared its own (lower) bar inside the override.
    if label == STOP_CLASS:
        return label, conf

    if conf < CONFIDENCE_THRESHOLD:
        return None, conf
    return label, conf