#!/usr/bin/env python3
"""
extract_ipn_none.py

Build the NONE class from IPN-Hand's non-gesture (D0X) segments.

Why NONE needs its own dataset
------------------------------
NONE is "my hand is visible but I am not commanding anything". Without it the
classifier is forced to label EVERY window as some command - including a hand
resting or drifting between positions - and misfires constantly. It should end
up the largest class, roughly 2-3x the biggest command class.

Zenodo-27 cannot supply it: every clip there is a deliberate gesture. IPN-Hand
can, because IPN was recorded CONTINUOUSLY and annotates the gaps between
gestures as D0X. Those gaps are exactly "hand visible, nothing intended",
recorded across 50 subjects in cluttered homes and offices with real lighting -
realism we could not stage.

Why ONLY D0X
------------
IPN's other labels are traps for THIS vocabulary:

    B0B  pointing with two fingers  ~  COME (two fingers toward camera)
    B0A  pointing with one finger   ~  ambiguous
    G05/G06  throw left/right       ~  RELEASE (lateral wave)
    G10/G11  zoom in/out            ~  COME / BACK OFF (depth motion)

Labelling any of those NONE would hand the model directly contradictory labels
for near-identical motion, and both classes would degrade. It is the same
mistake as resting an open palm during a NONE session and thereby labelling
STOP as NONE. B0A/B0B are also enormous (~225k frames each) and would swamp
everything.

Subjects
--------
IPN video names look like `1CM1_4_R_#229`. The first two tokens (`1CM1_4`)
identify the subject: there are exactly 50, matching IPN's documented 50
participants, and the metadata's Sex field is consistent within every group,
which confirms it. Each subject maps to a stable index so filenames read
`ipn_u07_00042.npy` and gesture_common.subject_from_filename() recovers
"ipn_u07" for the subject-wise split.

Layout expected
---------------
    data/ipn/Annot_List.txt
    data/ipn/videos/1CM1_4_R_#229.avi

Usage
-----
    python3 extract_ipn_none.py --dry-run
    python3 extract_ipn_none.py --limit 5      # quick test
    python3 extract_ipn_none.py
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

from config import (
    DATA_DIR,
    WINDOW_FRAMES,
    IPN_NONE_LABEL,
    IPN_MAX_WINDOWS_PER_SEGMENT,
    IPN_FRAMES_PER_SEGMENT,
)
from gesture_common import NONE_CLASS, class_to_slug, frame_row
from hand_landmarker import HandTracker
from extract_dataset import windows_from_segments   # same windowing as Zenodo-27

IPN_ROOT = os.path.join(DATA_DIR, "ipn")
ANNOT = os.path.join(IPN_ROOT, "Annot_List.txt")
VIDEO_DIR = os.path.join(IPN_ROOT, "videos")

FPS = 30.0


def subject_of(video_name):
    """`1CM1_4_R_#229` -> `1CM1_4`. Verified: 50 unique, Sex-consistent."""
    return "_".join(video_name.split("_")[:2])


def load_d0x():
    """
    video -> [(start0, end0), ...] for D0X segments only.
    Annotation frames are 1-indexed inclusive; we return 0-indexed.
    """
    if not os.path.exists(ANNOT):
        sys.exit(
            f"Annotation not found: {ANNOT}\n"
            f"Put IPN's Annot_List.txt there, and the videos under {VIDEO_DIR}/"
        )
    per_video = defaultdict(list)
    with open(ANNOT, newline="") as f:
        for row in csv.DictReader(f):
            if row["label"] != IPN_NONE_LABEL:
                continue
            try:
                s = int(row["t_start"]) - 1
                e = int(row["t_end"]) - 1
            except (KeyError, ValueError):
                continue
            if e - s + 1 >= WINDOW_FRAMES:      # too short for even one window
                per_video[row["video"]].append((s, e))
    return per_video


def middle_slice(s, e, n):
    """
    Centre `n` frames inside [s, e].

    The edges of a D0X gap are where the neighbouring REAL gesture is still
    finishing or already starting. Taking the middle keeps NONE clean.
    """
    length = e - s + 1
    if length <= n:
        return s, e
    pad = (length - n) // 2
    return s + pad, s + pad + n - 1


def runs_for_segments(tracker, path, segments):
    """
    One pass over the video, running the landmarker ONLY on the frames needed.

    grab() decodes without converting and is far cheaper than read(); we only
    pay retrieve() on the ~40 frames per segment we actually want. Across 200
    videos of ~4000 frames that is the difference between minutes and hours.

    A frame with no hand BREAKS the run: a window spanning a hand-loss gap is
    meaningless, so each unbroken stretch becomes its own run.
    """
    wanted = {}
    for (s, e) in segments:
        ms, me = middle_slice(s, e, IPN_FRAMES_PER_SEGMENT)
        for i in range(ms, me + 1):
            wanted[i] = (s, e)
    if not wanted:
        return []

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []

    last = max(wanted)
    runs = defaultdict(list)
    current = defaultdict(list)

    idx = 0
    while idx <= last:
        if idx in wanted:
            ok, frame = cap.read()
            if not ok:
                break
            key = wanted[idx]
            hands = tracker.detect(frame, (idx / FPS) * 1000.0)
            if hands:
                current[key].append(frame_row(idx / FPS, hands[0]))
            elif current[key]:
                runs[key].append(current[key])
                current[key] = []
        else:
            if not cap.grab():
                break
        idx += 1
    cap.release()

    for key, rows in current.items():
        if rows:
            runs[key].append(rows)
    return list(runs.values())      # per segment: list of unbroken runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N videos (quick test)")
    args = ap.parse_args()

    per_video = load_d0x()
    n_seg = sum(len(v) for v in per_video.values())
    subjects = sorted({subject_of(v) for v in per_video})
    subj_idx = {s: i + 1 for i, s in enumerate(subjects)}

    print(f"D0X segments >= {WINDOW_FRAMES} frames: {n_seg}")
    print(f"  across {len(per_video)} videos, {len(subjects)} subjects")
    print(f"  cap {IPN_MAX_WINDOWS_PER_SEGMENT}/segment "
          f"-> at most ~{n_seg * IPN_MAX_WINDOWS_PER_SEGMENT} NONE windows")

    if args.dry_run:
        missing = [v for v in per_video
                   if not os.path.exists(os.path.join(VIDEO_DIR, v + ".avi"))]
        found = len(per_video) - len(missing)
        print(f"\nvideos present: {found}/{len(per_video)}")
        if missing:
            print(f"MISSING {len(missing)}, e.g. {missing[:3]}")
            print(f"Expected under: {VIDEO_DIR}/")
        else:
            print("All videos found. Re-run without --dry-run.")
        return

    tracker = HandTracker(max_num_hands=1)
    out_dir = os.path.join(DATA_DIR, class_to_slug(NONE_CLASS))
    os.makedirs(out_dir, exist_ok=True)

    videos = sorted(per_video)
    if args.limit:
        videos = videos[:args.limit]

    total = 0
    for n, video in enumerate(videos, 1):
        path = os.path.join(VIDEO_DIR, video + ".avi")
        if not os.path.exists(path):
            continue
        u = subj_idx[subject_of(video)]

        for seg_runs in runs_for_segments(tracker, path, per_video[video]):
            for w in windows_from_segments(
                seg_runs, max_windows=IPN_MAX_WINDOWS_PER_SEGMENT
            ):
                np.save(os.path.join(out_dir, f"ipn_u{u:02d}_{total:05d}.npy"),
                        np.asarray(w, dtype=np.float64))
                total += 1

        print(f"  [{n}/{len(videos)}] {video} -> {total} windows      ", end="\r")

    tracker.close()
    print(f"\n\n{'=' * 46}")
    print(f"NONE windows written: {total}")
    print(f"{'=' * 46}")
    print("\nNow: python3 train_gestures.py")


if __name__ == "__main__":
    main()
