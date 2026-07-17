#!/usr/bin/env python3
"""
extract_dataset.py

Turn the Zenodo-27 dynamic hand gesture videos into training windows in THIS
project's format, using THIS project's pipeline.

Why it works this way
---------------------
The .avi files go through the exact same code path as the live webcam:

    frame -> HandTracker (Tasks API) -> frame_row() -> (t,cx,cy,size,open,closed)

So the training data and the live data are computed by identical code. That is
the whole reason gesture_common.py has no mediapipe import. Extracting features
any other way would reintroduce train/serve skew.

Timing CSV
----------
Each .avi is 120 frames @ 30fps, but the gesture only occupies part of it - the
rest is the hand drifting into and out of position. hand_gesture_timing_stats.csv
gives start_frame/end_frame per gesture, so we cut ONLY the real motion. Feeding
the dead frames in would teach the model that "drifting into position" is part
of the gesture.

Windows
-------
Windows are sliced with stride = WINDOW_FRAMES, so consecutive samples share no
frames. Overlapping windows are near-duplicates that land on both sides of the
train/test split and inflate the score without improving live behaviour.

Usage
-----
    python3 extract_dataset.py --dry-run     # check paths + counts, no work
    python3 extract_dataset.py               # do the extraction
    python3 extract_dataset.py --classes 22  # one class only (quick test)
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

from config import (
    WINDOW_FRAMES,
    DATA_DIR,
    EXTRACT_STRIDE,
    TEMPORAL_STRIDES,
    TIMING_PAD,
    MAX_WINDOWS_PER_CLIP,
)
from gesture_common import class_to_slug, frame_row
from hand_landmarker import HandTracker


# --------------------------------------------------------------------------
# The mapping: Zenodo-27 class -> our command
# --------------------------------------------------------------------------
# Chosen so every command differs on a feature we ACTUALLY measure
# (palm centre, palm size, open/closed). Palm ORIENTATION is invisible to our
# features, which is why classes 3/10/16/19 (all "open palm raised") were
# dropped - they would be near-identical to the model no matter how different
# they look to a human.
#
# FOLLOW takes two source classes (26 + 27, thumb-left and thumb-right); both
# are the same command performed in mirror, so they merge into one label.

CLASS_MAP = {
    22: "COME",       # 2 fingers + arm toward camera -> size grows, open=0
    11: "STOP",       # fist -> open palm            -> open flag flips
    17: "STAY",       # open palm lowers to floor    -> ndy positive
    18: "BACK OFF",   # open palm pushed forward     -> size grows, open=1
    12: "RELEASE",    # lateral wave                 -> high path, ~0 net
    26: "FOLLOW",     # thumb out + move left        -> high path, high net
    27: "FOLLOW",     # thumb out + move right       -> mirror of 26
    24: "ATTENTION",  # thumbs up                    -> static, low motion
}

VIDEO_ROOT = os.path.join(DATA_DIR, "hand_gesture_dataset_videos")
TIMING_CSV = os.path.join(DATA_DIR, "hand_gesture_timing_stats.csv")

FPS = 30.0
N_USERS = 21
REPS_PER_USER = 3          # UserY_1.avi, UserY_2.avi, UserY_3.avi


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------

def load_timing(path=TIMING_CSV):
    """
    (class, user, rep) -> (start_frame, end_frame)

    Missing file is not fatal: we fall back to using the whole clip, just with
    a warning. Better to extract something than to refuse to run.
    """
    if not os.path.exists(path):
        print(f"WARNING: {path} not found - using full clips (noisier windows).")
        return None

    timing = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                cls = int(row["class"])
                usr = int(row["user"])
            except (KeyError, ValueError):
                continue
            for rep in (1, 2, 3):
                s, e = row.get(f"start_frame_{rep}"), row.get(f"end_frame_{rep}")
                if s in (None, "") or e in (None, ""):
                    continue
                try:
                    timing[(cls, usr, rep)] = (int(float(s)), int(float(e)))
                except ValueError:
                    continue
    print(f"Loaded timing for {len(timing)} clips.")
    return timing


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

def rows_from_video(tracker, path, start_f, end_f):
    """
    Run the hand tracker over frames [start_f, end_f] of one .avi and return a
    list of frame_row tuples.

    A frame with no hand BREAKS the run: a window spanning a hand-loss gap is
    physically meaningless (the hand teleports across the gap), so we return
    segments and let the caller window each segment separately.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []

    segments, current = [], []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx > end_f:
            break
        if idx >= start_f:
            hands = tracker.detect(frame, (idx / FPS) * 1000.0)
            if hands:
                current.append(frame_row(idx / FPS, hands[0]))
            elif current:
                segments.append(current)
                current = []
        idx += 1

    cap.release()
    if current:
        segments.append(current)
    return segments


def windows_from_segments(segments):
    """
    Slice each segment into WINDOW_FRAMES windows.

    Two things happen here that look like bugs and are not:

    OVERLAP (EXTRACT_STRIDE < WINDOW_FRAMES): consecutive windows share frames.
    That is only safe because train_gestures.py splits by SUBJECT, so every
    near-duplicate from one clip stays on one side of the split. Without it,
    STOP (~16 frames) would yield ~1 window per clip.

    TEMPORAL STRIDE (TEMPORAL_STRIDES): a window is also cut by taking every
    2nd frame, which is the same gesture rendered slower. The source is 30 fps
    and the live pipeline is ~16-20 fps, so this is what stops the model from
    learning one frame rate's notion of speed. Stride 2 needs twice the frames,
    so it silently doesn't fire on short clips.
    """
    out = []
    for seg in segments:
        for tstride in TEMPORAL_STRIDES:
            span = WINDOW_FRAMES * tstride      # source frames a window needs
            if len(seg) < span:
                continue
            for i in range(0, len(seg) - span + 1, EXTRACT_STRIDE):
                w = seg[i:i + span:tstride]
                if len(w) == WINDOW_FRAMES:
                    out.append(w)

    # Cap per clip, sampling EVENLY across it rather than truncating. Truncating
    # would keep only the start of every gesture and throw the ending away.
    if len(out) > MAX_WINDOWS_PER_CLIP:
        keep = np.linspace(0, len(out) - 1, MAX_WINDOWS_PER_CLIP).astype(int)
        out = [out[i] for i in sorted(set(keep))]
    return out


def save_windows(label, windows, counter, user, cls, rep):
    """
    Write windows with the SUBJECT encoded in the filename:

        z27_u07_c22_r1_00042.npy

    gesture_common.subject_from_filename() reads "z27_u07" back out, which is
    what lets training group by person.
    """
    d = os.path.join(DATA_DIR, class_to_slug(label))
    os.makedirs(d, exist_ok=True)
    for w in windows:
        path = os.path.join(
            d, f"z27_u{user:02d}_c{cls:02d}_r{rep}_{counter[label]:05d}.npy"
        )
        np.save(path, np.asarray(w, dtype=np.float64))
        counter[label] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="check paths and counts, extract nothing")
    ap.add_argument("--classes", type=int, nargs="+", default=None,
                    help="only these source class numbers (e.g. --classes 22 18)")
    args = ap.parse_args()

    if not os.path.isdir(VIDEO_ROOT):
        sys.exit(f"Videos not found at: {VIDEO_ROOT}")

    wanted = {c: lab for c, lab in CLASS_MAP.items()
              if args.classes is None or c in args.classes}
    if not wanted:
        sys.exit("No classes selected.")

    print("Source class -> our command:")
    for c, lab in sorted(wanted.items()):
        print(f"  class_{c:02d} -> {lab}")
    print()

    # --- dry run: just verify the tree is where we think it is -------------
    if args.dry_run:
        total = 0
        for c in sorted(wanted):
            cdir = os.path.join(VIDEO_ROOT, f"class_{c:02d}")
            if not os.path.isdir(cdir):
                print(f"  MISSING {cdir}")
                continue
            avis = sum(len([f for f in files if f.endswith(".avi")])
                       for _, _, files in os.walk(cdir))
            total += avis
            print(f"  class_{c:02d}: {avis} videos")
        print(f"\nTotal videos to process: {total}")
        print("Looks right? Re-run without --dry-run.")
        return

    timing = load_timing()
    tracker = HandTracker(max_num_hands=1)
    counter = defaultdict(int)
    no_hand_clips = []

    for c in sorted(wanted):
        label = wanted[c]
        cdir = os.path.join(VIDEO_ROOT, f"class_{c:02d}")
        if not os.path.isdir(cdir):
            print(f"SKIP class_{c:02d} (not found)")
            continue

        print(f"class_{c:02d} -> {label}")
        for u in range(1, N_USERS + 1):
            udir = os.path.join(cdir, f"User{u}_")
            if not os.path.isdir(udir):
                continue
            for rep in range(1, REPS_PER_USER + 1):
                avi = os.path.join(udir, f"User{u}_{rep}.avi")
                if not os.path.exists(avi):
                    continue

                if timing and (c, u, rep) in timing:
                    s, e = timing[(c, u, rep)]
                    # pad, clamped to the clip (120 frames @ 30fps)
                    s = max(0, s - TIMING_PAD)
                    e = min(119, e + TIMING_PAD)
                else:
                    s, e = 0, 119

                segs = rows_from_video(tracker, avi, s, e)
                wins = windows_from_segments(segs)
                if not wins:
                    no_hand_clips.append(f"class_{c:02d}/User{u}_/{rep}")
                save_windows(label, wins, counter, u, c, rep)

            print(f"    User{u}_ done   ({counter[label]} windows so far)", end="\r")
        print(f"    -> {counter[label]} windows{' ' * 20}")

    tracker.close()

    print("\n" + "=" * 46)
    print("Windows written per class:")
    for lab in sorted(counter):
        print(f"  {lab:<12} {counter[lab]:>5}")
    print("=" * 46)
    if no_hand_clips:
        print(f"{len(no_hand_clips)} clip(s) produced no windows "
              f"(hand not found / too short).")
        for x in no_hand_clips[:5]:
            print(f"    {x}")
        if len(no_hand_clips) > 5:
            print(f"    ... and {len(no_hand_clips) - 5} more")
    print("\nNONE class still needed - that comes from IPN-Hand.")
    print("Then: python3 train_gestures.py")


if __name__ == "__main__":
    main()