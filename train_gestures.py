#!/usr/bin/env python3
"""
train_gestures.py

Train a gesture classifier on the samples recorded by collect_gestures.py,
evaluate it honestly, and save it to gesture_model.pkl for gesture_control.py
to load.

Usage:
    python3 train_gestures.py

Why RandomForest and not an LSTM:
    With ~30-80 samples per class on a CPU-only laptop, a RandomForest over
    engineered motion features is the right tool. It trains in seconds,
    predicts in microseconds (no added lag in the live loop), needs no
    feature scaling, gives usable class probabilities for a confidence
    threshold, and - unlike a neural net - tells you which features actually
    mattered. An LSTM would need TensorFlow/PyTorch, far more data, and would
    add real latency to a pipeline already fighting for frames.

Data augmentation:
    Each recorded window is expanded into several variants (see augment()).
    Augmentation is applied to TRAINING data only, never to the data a score
    is computed on. Mirroring a test sample into the training set would leak
    that sample across the split and inflate the reported accuracy - the
    model would look great on paper and misfire on your webcam.
"""

import os
import sys
from collections import Counter

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
)

from config import (
    WINDOW_FRAMES,
    DATA_DIR,
    MODEL_PATH,
    MIN_SAMPLES_PER_CLASS,
    RECOMMENDED_PER_CLASS,
    AUGMENT,
    N_NOISE_COPIES,
    NOISE_POS_STD,
    NOISE_SIZE_STD,
)
from gesture_common import (
    FEATURE_DIM,
    GESTURE_CLASSES,
    NONE_CLASS,
    COL_CX,
    COL_CY,
    COL_SIZE,
    class_to_slug,
    extract_features,
)

# Mirroring a window flips it left<->right, so swipe labels must swap too.
# Every other class is mirror-symmetric (a fist is a fist either way).
MIRROR_LABEL = {
    "TURN LEFT": "TURN RIGHT",
    "TURN RIGHT": "TURN LEFT",
}


# --------------------------------------------------------------------------
# Augmentation
# --------------------------------------------------------------------------

def mirror_window(w):
    """Reflect the window horizontally. cx lives in [0,1], so cx -> 1 - cx."""
    w2 = w.copy()
    w2[:, COL_CX] = 1.0 - w2[:, COL_CX]
    return w2


def jitter_window(w, rng):
    """Add small Gaussian noise, imitating MediaPipe's frame-to-frame landmark wobble."""
    w2 = w.copy()
    n = w2.shape[0]
    w2[:, COL_CX] += rng.normal(0.0, NOISE_POS_STD, n)
    w2[:, COL_CY] += rng.normal(0.0, NOISE_POS_STD, n)
    w2[:, COL_SIZE] = np.maximum(w2[:, COL_SIZE] + rng.normal(0.0, NOISE_SIZE_STD, n), 1e-4)
    return w2


def augment(windows, labels, rng):
    """
    Expand each window into 2 * (1 + N_NOISE_COPIES) variants: the original and
    its mirror, plus noisy copies of each.

    Note what is deliberately NOT here: translation and scale augmentation.
    extract_features() measures displacement relative to the window's first
    frame and divides by palm size, so shifting or rescaling a whole window
    leaves its feature vector unchanged. Those augmentations would add
    duplicate rows and nothing else.
    """
    out_w, out_y = [], []
    for w, lab in zip(windows, labels):
        mw = mirror_window(w)
        ml = MIRROR_LABEL.get(lab, lab)

        out_w.append(w);  out_y.append(lab)
        out_w.append(mw); out_y.append(ml)
        for _ in range(N_NOISE_COPIES):
            out_w.append(jitter_window(w, rng));  out_y.append(lab)
            out_w.append(jitter_window(mw, rng)); out_y.append(ml)
    return out_w, out_y


def to_features(windows, labels):
    """Vectorize a list of windows, dropping any that fail feature extraction."""
    X, y = [], []
    for w, lab in zip(windows, labels):
        f = extract_features(w)
        if f is not None:
            X.append(f)
            y.append(lab)
    if not X:
        return np.empty((0, FEATURE_DIM)), np.array([])
    return np.vstack(X), np.array(y)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_windows():
    """Load every saved window as a raw array (features come later, post-split)."""
    windows, labels, skipped = [], [], 0
    for label in GESTURE_CLASSES:
        d = os.path.join(DATA_DIR, class_to_slug(label))
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if not fname.endswith(".npy"):
                continue
            w = np.load(os.path.join(d, fname))
            if w.shape != (WINDOW_FRAMES, 6) or extract_features(w) is None:
                skipped += 1
                continue
            windows.append(w)
            labels.append(label)
    return windows, np.array(labels), skipped


def make_clf():
    return RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=1,
        class_weight="balanced",  # keeps a large NONE class from swamping the rest
        random_state=42,
        n_jobs=-1,
    )


def main():
    if not os.path.isdir(DATA_DIR):
        print(f"No '{DATA_DIR}/' directory found. Record samples first:\n"
              f"    python3 collect_gestures.py")
        sys.exit(1)

    windows, y_all, skipped = load_windows()
    if len(windows) == 0:
        print("No usable samples found. Record some with collect_gestures.py first.")
        sys.exit(1)

    rng = np.random.default_rng(42)
    counts = Counter(y_all)

    print(f"Loaded {len(windows)} recorded samples "
          f"({FEATURE_DIM} features each, window = {WINDOW_FRAMES} frames)")
    if skipped:
        print(f"  ({skipped} malformed sample(s) skipped)")
    print("\nSamples per class:")
    for label in GESTURE_CLASSES:
        n = counts.get(label, 0)
        flag = ""
        if n == 0:
            flag = "  <- MISSING"
        elif n < MIN_SAMPLES_PER_CLASS:
            flag = "  <- TOO FEW"
        elif n < RECOMMENDED_PER_CLASS:
            flag = "  <- thin, consider more"
        print(f"  {label:<15} {n:>4}{flag}")

    missing = [c for c in GESTURE_CLASSES if counts.get(c, 0) < MIN_SAMPLES_PER_CLASS]
    if missing:
        print(f"\nCannot train: these classes have fewer than {MIN_SAMPLES_PER_CLASS} "
              f"samples:\n  {', '.join(missing)}")
        print("Record more with collect_gestures.py, then re-run this script.")
        sys.exit(1)

    # Class-balance diagnosis is deferred until after cross-validation, where
    # it can be reported alongside the always-predict-NONE baseline.
    command_counts = [counts.get(c, 0) for c in GESTURE_CLASSES if c != NONE_CLASS]
    mean_command = sum(command_counts) / max(len(command_counts), 1)
    none_count = counts.get(NONE_CLASS, 0)

    if AUGMENT:
        factor = 2 * (1 + N_NOISE_COPIES)
        print(f"\nAugmentation ON: mirror + {N_NOISE_COPIES} noise copies "
              f"-> {factor}x training data (test data never augmented).")

    # ---- Honest held-out evaluation --------------------------------------
    idx = np.arange(len(windows))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.25, random_state=42, stratify=y_all
    )

    train_w = [windows[i] for i in idx_train]
    train_y = y_all[idx_train]
    if AUGMENT:
        train_w, train_y = augment(train_w, train_y, rng)

    X_train, y_train = to_features(train_w, train_y)
    X_test, y_test = to_features([windows[i] for i in idx_test], y_all[idx_test])

    clf = make_clf()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\n" + "=" * 62)
    print("HELD-OUT TEST PERFORMANCE (25% of data, never seen during training)")
    print("=" * 62)
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"balanced accuracy: {balanced_accuracy_score(y_test, y_pred):.3f}   "
          f"macro F1: {f1_score(y_test, y_pred, average='macro', zero_division=0):.3f}")
    print("(ignore the 'accuracy' line above if NONE dominates - see baseline below)")

    labels_present = sorted(set(y_all))
    cm = confusion_matrix(y_test, y_pred, labels=labels_present)
    print("Confusion matrix (rows = true, cols = predicted):")
    header = "".join(f"{l[:8]:>10}" for l in labels_present)
    print(f"{'':<15}{header}")
    for i, label in enumerate(labels_present):
        row = "".join(f"{v:>10}" for v in cm[i])
        print(f"{label:<15}{row}")
    print("\nRead this: off-diagonal cells are your real problem gestures.")
    print("If TURN LEFT is being confused with NONE, record more of both.")

    # ---- Cross-validation, augmenting each training fold only -------------
    #
    # We score with MACRO F1, not accuracy. With a dominant NONE class, plain
    # accuracy is close to worthless: a model that always answers NONE scores
    # whatever fraction of the data is NONE. Macro F1 weights every class
    # equally, so a model that ignores your swipes cannot hide behind NONE.
    n_splits = min(5, min(counts.values()))
    cv_f1 = None
    if n_splits >= 2:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_f1, fold_acc = [], []
        for tr, te in skf.split(idx, y_all):
            fw = [windows[i] for i in tr]
            fy = y_all[tr]
            if AUGMENT:
                fw, fy = augment(fw, fy, rng)
            Xtr, ytr = to_features(fw, fy)
            Xte, yte = to_features([windows[i] for i in te], y_all[te])
            m = make_clf()
            m.fit(Xtr, ytr)
            pred = m.predict(Xte)
            fold_f1.append(f1_score(yte, pred, average="macro", zero_division=0))
            fold_acc.append((pred == yte).mean())
        fold_f1, fold_acc = np.array(fold_f1), np.array(fold_acc)
        cv_f1 = fold_f1.mean()

        baseline = max(counts.values()) / sum(counts.values())
        print(f"\n{n_splits}-fold cross-validation:")
        print(f"  macro F1  : {cv_f1:.3f} +/- {fold_f1.std():.3f}   <- the number that matters")
        print(f"  accuracy  : {fold_acc.mean():.3f} +/- {fold_acc.std():.3f}")
        print(f"  (a model that ALWAYS answers '{max(counts, key=counts.get)}' "
              f"would score {baseline:.3f} accuracy)")

        if len(y_test) < 40:
            print(f"\n  NOTE: the held-out set is only {len(y_test)} samples, so the "
                  f"report above is noisy. Trust the cross-validation numbers.")

    # ---- Diagnose the class balance, and advise accordingly ---------------
    print()
    thin_commands = [c for c in GESTURE_CLASSES
                     if c != NONE_CLASS and counts.get(c, 0) < RECOMMENDED_PER_CLASS]
    if none_count > 3 * mean_command:
        print(f"NOTE: NONE ({none_count}) is much larger than the command classes "
              f"({mean_command:.0f} avg).")
        print("  That is FINE, and good for live use - more idle examples means fewer")
        print("  false-positive gestures. class_weight='balanced' stops it dominating")
        print("  training. Just never judge this model by the 'accuracy' line: read")
        print("  macro F1 above, and compare it against the baseline.")
        if thin_commands:
            print(f"  Do make sure the command classes are not starved. Thin: "
                  f"{', '.join(thin_commands)}")
    elif none_count < 1.5 * mean_command:
        print(f"WARNING: NONE has {none_count} samples vs. {mean_command:.0f} average per "
              f"command gesture.")
        print(f"  NONE should be your largest class - aim for at least "
              f"{int(1.5 * mean_command)}.")
        print("  A thin NONE class is a major cause of false-positive gestures.")
        print("  Fastest fix: collect_gestures.py -> press 6, then 'c', then move")
        print("  your hand around aimlessly for a minute (do NOT hold an open palm still).")

    if cv_f1 is not None:
        if cv_f1 < 0.80:
            print(f"\nMacro F1 of {cv_f1:.2f} is not good enough for comfortable live use.")
            print("  Look at the confusion matrix above and fix the specific classes")
            print("  that are bleeding into each other - more data alone may not help.")
        elif cv_f1 < 0.90:
            print(f"\nMacro F1 of {cv_f1:.2f} is usable but will misfire sometimes.")
        else:
            print(f"\nMacro F1 of {cv_f1:.2f} - good. Try it live.")

    # ---- Refit on ALL data (augmented), then save -------------------------
    all_w, all_y = list(windows), y_all
    if AUGMENT:
        all_w, all_y = augment(all_w, all_y, rng)
    X_full, y_full = to_features(all_w, all_y)

    clf_final = make_clf()
    clf_final.fit(X_full, y_full)

    bundle = {
        "model": clf_final,
        "classes": list(clf_final.classes_),
        "window_frames": WINDOW_FRAMES,
        "feature_dim": FEATURE_DIM,
        "n_samples": int(len(windows)),
        "n_training_rows": int(len(X_full)),
        "cv_macro_f1": float(cv_f1) if cv_f1 is not None else None,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"  {len(windows)} recorded samples -> {len(X_full)} training rows after augmentation")
    print("Now run:  python3 gesture_control.py")


if __name__ == "__main__":
    main()