#!/usr/bin/env python3
"""
gesture_model.py

Loads the trained classifier and runs inference on a live motion buffer.
Kept separate from gesture_control.py (and free of mediapipe/cv2 imports)
so it can be tested without a webcam.
"""

import os

import numpy as np

from config import MODEL_PATH, CONFIDENCE_THRESHOLD, WINDOW_FRAMES
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
    """
    if bundle is None or len(buffer) < WINDOW_FRAMES:
        return None, 0.0

    window = list(buffer)[-WINDOW_FRAMES:]
    feats = extract_features(window)
    if feats is None:
        return None, 0.0

    model = bundle["model"]
    probs = model.predict_proba(feats.reshape(1, -1))[0]
    best = int(np.argmax(probs))
    label = model.classes_[best]
    conf = float(probs[best])

    if label == NONE_CLASS or conf < CONFIDENCE_THRESHOLD:
        return None, conf
    return str(label), conf