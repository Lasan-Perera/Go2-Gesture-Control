# Go2 Gesture Control

Real-time hand-gesture recognition for the **Unitree Go2**, built as Layer 2
(Contextual Gesture Response) of a socially-aware navigation system driven by
an Interval Type-2 Fuzzy Logic System (IT2-FLS).

Part of the WSO2-sponsored research project at the University of Moratuwa.
Companion repos: the simulation
([`go2-social-nav`](https://github.com/Lasan-Perera/go2-social-nav)) and the
LiDAR-camera fusion pipeline
([`lidar-cam-fusion-lab`](https://github.com/KaveeshwaraBandara/lidar-cam-fusion-lab)).

---

## What this component does

The robot must respond to human gestures — but not with a fixed reaction every
time. A "come here" in an open corridor should trigger a direct approach; the
same gesture in a crowd should trigger a slower, replanned path. That
context-sensitivity is the job of the fuzzy engine, not this node.

**This node's job is narrower and deliberate:** recognise *which* gesture an
*enrolled operator* is performing, with a confidence score, and publish that as
*intent* — never as a velocity command. The fuzzy engine is the only thing that
touches `/cmd_vel`. Keeping gesture recognition free of motor commands is what
lets the same gesture mean different things in different contexts.

```
camera  ->  hand + pose landmarks  ->  operator lock  ->  gesture classifier
                                                                  |
                                                                  v
                                                     GestureIntent  (label,
                                                     confidence, operator
                                                     distance/bearing)
                                                                  |
                                                                  v
                                                          IT2-FLS  ->  /cmd_vel
```

---

## Current status

Active development. The pipeline is being built and validated standalone
(webcam in, intent out) before ROS 2 integration, following a phase-gated
roadmap.

| Phase | Work | Status |
|---|---|---|
| P0 | Gesture vocabulary + interface spec | done |
| P2 | Migrate to MediaPipe **Tasks API** | done |
| P3 | Pose estimation benchmark (wrists for operator lock) | done |
| P5 | Record dataset (robot viewpoint, multiple people) | next |
| P8 | Operator lock (hand-to-wrist association) | planned |
| P9 | ROS 2 node publishing `GestureIntent` | planned |

The earlier face-recognition operator lock is **frozen** on branch
`face-identity-lock` (tag `v0.1-face-lock`) and replaced by a pose-based
approach — see [Operator lock](#operator-lock-in-progress).

---

## Gesture vocabulary

Six commands plus a wake gesture, chosen so that most have a genuinely
context-dependent response (the point of the whole system). One-handed, either
hand.

| Gesture | Type | Static/Dynamic | Context-sensitive |
|---|---|---|---|
| **Attention** (wake) | precondition | static | — |
| **Come** | goal | dynamic | high |
| **Follow** | mode | static | high |
| **Stop** | safety | static | no (by design) |
| **Stay** | mode | static | medium |
| **Back off** | constraint | dynamic | high |
| **Release** | mode exit | dynamic | no |

`Stop` is intentionally context-free: a safety command whose meaning varies is
not a safety command. `Come`, `Follow`, and `Back off` are where the fuzzy
engine earns its keep — `Back off` especially, because whether the robot *can*
comply depends on the free space behind it.

A `NONE` class ("hand visible, no command") is the most important class for
avoiding false triggers. See [Why NONE matters](#why-none-matters).

---

## How recognition works

**Windowed motion, not single frames.** A gesture is classified from a window
of `WINDOW_FRAMES` consecutive frames (palm centre, palm size, open/closed
flags), so dynamic gestures like a beckon or a push are representable — a
per-frame pose classifier could not represent them.

**Scale-normalized features.** Displacement is measured relative to the
window's first frame and divided by palm size (a distance proxy), so a gesture
performed 1 m from the camera looks the same to the model as one at 2 m. Fixed
pixel thresholds cannot do this.

**RandomForest, not an LSTM.** With the sample counts realistic for a
hand-recorded dataset, a forest over engineered motion features trains in
seconds, predicts in microseconds (no added latency in the live loop), needs no
feature scaling, and gives usable class probabilities for a confidence
threshold. That confidence is passed onward as a *fuzzy antecedent*, never
thresholded to a hard yes/no — thresholding it would discard the uncertainty
the IT2-FLS exists to absorb.

**Shared feature code.** `gesture_common.py` is imported by collection,
training, and inference so all three compute features through the exact same
path. Splitting that logic is the classic route to train/serve skew.

---

## Hand tracking: MediaPipe Tasks API

The pipeline uses the **MediaPipe Tasks API** (`HandLandmarker`), not the
deprecated `mp.solutions`. The swap is isolated in `hand_landmarker.py`, a thin
`HandTracker` wrapper that returns landmarks in the same shape the old API
produced — so nothing downstream changed.

Why the Tasks API:

- The legacy `mp.solutions` API is deprecated and already breaks on current
  MediaPipe (the reason the old code was pinned to `mediapipe==0.10.14`). That
  pin is gone.
- The Tasks API uses a portable `.task` model bundle — the only realistic path
  onto the Jetson Orin NX, which is where this must ultimately run.
- VIDEO running mode tracks the hand across frames using the previous frame's
  box, giving the steadiness the windowed features depend on.
- Handedness comes for free, which the operator lock will use.

The migration was verified with `verify_migration.py`, which runs both APIs on
the same frames: palm-centre agreement was within ~0.6% of frame width, and the
open-palm flag agreed on 99% of frames — confirming a model trained on either
API behaves the same.

---

## Operator lock (in progress)

The robot must obey **one** enrolled operator, not any hand in view. The
original design used face recognition; it is frozen and being replaced, because:

- During `Follow`, the robot sees the operator's back — no face to recognise.
- From a camera ~45 cm off the floor tilted up, a standing person's face is
  often just the chin.
- `dlib`'s HOG detector is CPU-only and the pipeline's bottleneck; it would be
  worse on ARM.
- Storing facial embeddings of study participants complicates ethics approval.

The replacement uses **body pose** instead of face identity:

1. **Pose estimation** (MediaPipe Pose Landmarker) gives body keypoints,
   including wrists. Benchmarked in P3: reliable wrists at the robot's low,
   tilted-up angle, ~16–17 fps running alongside hand tracking.
2. **Hand-to-wrist association** matches each detected hand to the nearest body
   wrist and discards hands that aren't the operator's — so a bystander's wave
   is ignored. This is the actual exclusivity mechanism (face never provided
   it).
3. **Enrol by action:** whoever performs the wake gesture becomes the operator,
   identified by track ID — no face database.
4. **Re-acquisition** after occlusion via a lightweight appearance embedding
   (clothing/build, not face) is planned for later.

Operator identity confidence is passed to the fuzzy engine as an input, not a
gate: low confidence makes the robot cautious rather than inert.

---

## Repository layout

| File | Role |
|---|---|
| `hand_landmarker.py` | MediaPipe Tasks API wrapper (`HandTracker`) |
| `gesture_common.py` | Landmark helpers, feature extraction, class list |
| `gesture_model.py` | Model loading + inference (headless-testable) |
| `collect_gestures.py` | Record labeled training samples |
| `train_gestures.py` | Train, evaluate, save the classifier |
| `gesture_control.py` | Live application (webcam in, intent out) |
| `config.py` | All tunable constants |
| `verify_migration.py` | Confirms Tasks API matches the legacy API |
| `benchmark_pose.py` | P3 pose benchmark (fps + wrist reliability) |
| `models/` | `.task` model bundles (committed) |

---

## Setup

Requires Python 3.12, Ubuntu, a webcam, and even lighting (detection degrades
in the dark).

```bash
git clone https://github.com/Lasan-Perera/Go2-Gesture-Control
cd Go2-Gesture-Control
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download the model bundles once (not pip dependencies):

```bash
mkdir -p models
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
curl -L -o models/pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task
```

---

## Usage

Three scripts, run in order. The first two are one-time setup per dataset.

```bash
# 1. Record samples (see the script header for per-class guidance)
python3 collect_gestures.py

# 2. Train and evaluate
python3 train_gestures.py

# 3. Run live
python3 gesture_control.py
```

`train_gestures.py` reports **macro F1** as the headline metric (not accuracy,
which a large NONE class makes meaningless) and prints a confusion matrix.
Off-diagonal cells are the gestures bleeding into each other. Target macro F1
≥ 0.90.

---

## Why NONE matters

`NONE` is "my hand is visible but I'm not commanding anything." It is the single
biggest determinant of whether this works live. A classifier trained on only the
command gestures is forced to label *every* window as some command — including a
hand at rest — and misfires constantly.

Two rules, both learned the hard way:

- **Don't record `Stop` as `NONE`.** `Stop` is an open palm held still. Resting
  an open hand during a NONE session labels `Stop` as `NONE`; both classes
  degrade. During NONE, keep the hand moving, or closed if still.
- **A large NONE class is good** — biggest class by 2–3×. `class_weight="balanced"`
  stops it dominating training.

---

## Notes for deployment

Target hardware is the **Jetson Orin NX** (aarch64) on the Go2. Two things to
validate before deployment:

- MediaPipe has no official aarch64 wheel; the Jetson build is a separate,
  from-source effort. The Tasks API `.task` bundles are the portable path.
- Pose Landmarker `full` was chosen for steady wrists on the laptop GPU; if it's
  too slow on the Orin NX, the fallback is `lite` plus a smoothing filter
  (higher fps, jitter removed in software).

The gesture node **never** publishes to `/cmd_vel`. It publishes `GestureIntent`;
the IT2-FLS owns all motion.
