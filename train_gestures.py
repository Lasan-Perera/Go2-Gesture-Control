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
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
)

from config import (
    WINDOW_FRAMES,
    STOP_CLASS,
    STOP_CONFIDENCE_THRESHOLD,
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
    subject_from_filename,
    ROW_WIDTH,
)

# Mirroring a window flips it left<->right, so any label whose meaning depends
# on direction must swap. The old TURN LEFT/TURN RIGHT pair needed that.
#
# The current vocabulary has NO direction-dependent labels:
#   - COME / BACK OFF are depth motions (toward/away), unchanged by mirroring
#   - STOP / STAY / ATTENTION are orientation or vertical, unchanged
#   - RELEASE is a symmetric lateral wave
#   - FOLLOW deliberately merges source classes 26 (thumb-left) and 27
#     (thumb-right), so its mirror is still FOLLOW
#
# So the map is empty, and mirroring is pure free augmentation: every class
# mirrors to itself.
MIRROR_LABEL = {}


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
    """
    Load every saved window, plus the SUBJECT each one came from.

    The subject is what makes the evaluation honest. Windows are grouped by the
    person who performed them so that a person never appears in both train and
    test - see gesture_common.subject_from_filename().
    """
    windows, labels, groups, skipped = [], [], [], 0
    for label in GESTURE_CLASSES:
        d = os.path.join(DATA_DIR, class_to_slug(label))
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if not fname.endswith(".npy"):
                continue
            w = np.load(os.path.join(d, fname))
            # Samples recorded before finger shape was added are 6 columns wide
            # and are skipped here rather than silently mixed in.
            if w.shape != (WINDOW_FRAMES, ROW_WIDTH) or extract_features(w) is None:
                skipped += 1
                continue
            windows.append(w)
            labels.append(label)
            groups.append(subject_from_filename(fname))
    return windows, np.array(labels), np.array(groups), skipped


def make_clf():
    return RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=1,
        class_weight="balanced",  # keeps a large NONE class from swamping the rest
        random_state=42,
        n_jobs=-1,
    )


def report_stop_sweep(clf, X_test, y_test):
    """
    Show what each STOP_CONFIDENCE_THRESHOLD actually costs and buys.

    STOP gets an override in gesture_model.py: if the classifier gives STOP at
    least this much probability, STOP is emitted even when another label scored
    higher. That is a deliberate safety asymmetry, and the right value for it is
    a judgement about consequences, not accuracy - so this prints the exact
    trade so the number can be chosen from evidence rather than feel.

    MISSED  = a real STOP that does not come out as STOP. The robot keeps
              moving, or backs off, when told to halt. The failure that matters.
    FALSE   = something else that comes out as STOP. The robot halts when it
              was not asked. Inconvenient, harmless.
    """
    if STOP_CLASS not in set(y_test):
        return

    classes = list(clf.classes_)
    if STOP_CLASS not in classes:
        return

    probs = clf.predict_proba(X_test)
    i_stop = classes.index(STOP_CLASS)
    is_stop = (y_test == STOP_CLASS)
    n_stop = int(is_stop.sum())

    print("\n" + "=" * 62)
    print("STOP SAFETY ASYMMETRY - threshold sweep on the held-out set")
    print("=" * 62)
    print(f"{'thresh':>7}{'STOP recall':>13}{'missed':>9}{'false':>8}   note")

    for t in (1.01, 0.60, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20):
        # replay the live rule: STOP wins outright if it clears t
        pred = []
        for row in probs:
            best = int(np.argmax(row))
            if classes[best] != STOP_CLASS and float(row[i_stop]) >= t:
                pred.append(STOP_CLASS)
            else:
                pred.append(classes[best])
        pred = np.array(pred)

        caught = int(((pred == STOP_CLASS) & is_stop).sum())
        missed = n_stop - caught
        false = int(((pred == STOP_CLASS) & ~is_stop).sum())

        note = ""
        if t > 1.0:
            note = "override off (baseline)"
        print(f"{t:>7.2f}{caught / n_stop:>13.2f}{missed:>9}{false:>8}   {note}")

    print(f"\n({n_stop} real STOP windows in the test set, "
          f"{len(y_test) - n_stop} non-STOP)")
    print(f"Currently set: STOP_CONFIDENCE_THRESHOLD = {STOP_CONFIDENCE_THRESHOLD}")
    print("Pick the row where 'missed' is acceptably low and 'false' is still")
    print("tolerable, then set that value in config.py.")


def main():
    if not os.path.isdir(DATA_DIR):
        print(f"No '{DATA_DIR}/' directory found. Record samples first:\n"
              f"    python3 collect_gestures.py")
        sys.exit(1)

    windows, y_all, groups, skipped = load_windows()
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

    # Split BY SUBJECT, not at random. The question that matters is "does this
    # work on a person it has never seen", and a random split cannot answer it:
    # the same person would appear on both sides. This is also what makes the
    # overlapping windows from extract_dataset.py safe.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    idx_train, idx_test = next(gss.split(idx, y_all, groups=groups))

    n_train_subj = len(set(groups[idx_train]))
    n_test_subj = len(set(groups[idx_test]))
    print(f"\nSubject-wise split: {n_train_subj} subjects train / "
          f"{n_test_subj} subjects test (no person appears in both)")

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

    # ---- STOP safety asymmetry: what does each threshold actually buy? ----
    report_stop_sweep(clf, X_test, y_test)

    # ---- Cross-validation, augmenting each training fold only -------------
    #
    # We score with MACRO F1, not accuracy. With a dominant NONE class, plain
    # accuracy is close to worthless: a model that always answers NONE scores
    # whatever fraction of the data is NONE. Macro F1 weights every class
    # equally, so a model that ignores your swipes cannot hide behind NONE.
    n_splits = min(5, len(set(groups)), min(counts.values()))
    cv_f1 = None
    if n_splits >= 2:
        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_f1, fold_acc = [], []
        for tr, te in skf.split(idx, y_all, groups=groups):
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
        print(f"\n{n_splits}-fold cross-validation (grouped by subject):")
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