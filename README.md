# Go2 Gesture Control

Real-time **dynamic hand gesture recognition**, locked to a single enrolled
operator via face recognition, driven by a **classifier trained on your own
recordings** rather than hand-tuned thresholds.

Built as a standalone control interface for navigating a simulated Unitree Go2
robot dog. It's a companion to
[`go2-social-nav`](https://github.com/KaveeshwaraBandara/go2-social-nav) — a
ROS 2 + Gazebo simulation where a Go2 navigates among simulated pedestrians —
with the goal of replacing keyboard teleop with hand gestures.

Right now it runs **standalone**: webcam in, gesture command out. The detection
pipeline was deliberately built and validated on its own before being wired
into ROS 2's `/cmd_vel`.

## Gestures

| Gesture | Motion | Command |
|---|---|---|
| Swipe hand left | fast horizontal motion, left | `TURN LEFT` |
| Swipe hand right | fast horizontal motion, right | `TURN RIGHT` |
| Push hand toward camera | palm grows larger | `MOVE FORWARD` |
| Pull hand away | palm shrinks | `MOVE BACKWARD` |
| Open palm, held still | no significant motion, palm open | `STOP` |

A sixth class, `NONE`, is the most important one — see
[Why NONE matters](#why-none-matters).

## Why this is more than hand tracking

A naive gesture demo reacts to *any* hand movement from *anyone* in view.
Three layers sit on top of raw MediaPipe hand tracking to make this usable in a
real environment:

**1. Trained classifier, not thresholds.**
Gestures are recognized by a RandomForest trained on windows of your own
recorded hand motion. Features are normalized by palm size, so a swipe
performed 1 m from the camera looks the same to the model as one at 2 m —
something fixed pixel thresholds fundamentally cannot do. The model has an
explicit `NONE` class, so it can answer *"nothing is happening"* rather than
being forced to label every window as some command.

**2. Arm / disarm wake gesture.**
The system starts **locked**, ignoring all motion. Hold an open palm still for
about a second to **arm** it; only then are commands dispatched. Hold a closed
fist still to **disarm**. This stops incidental movement — talking with your
hands, scratching your face, adjusting your glasses — from being read as
commands. It stays rule-based, because "hold still" is already reliable and
gains nothing from being learned.

**3. Face-recognition identity lock.**
Enroll your face once (press `e`). Every frame, the system identifies *your*
face among everyone visible and builds a control zone that follows you around
the frame. Other people's hands are drawn in gray and never treated as
commands. Because it's identity-based rather than positional, you can leave the
frame entirely, or be blocked by someone walking past, and control resumes the
moment your face is visible again.

## Requirements

- Ubuntu (tested on 22.04 / 24.04) with a working webcam
- Python 3.12
- Decent, even lighting — face and hand detection both degrade in the dark

## Installation

**1. Install system build tools.** `face_recognition` depends on `dlib`, which
compiles from source and has no prebuilt wheel:

```bash
sudo apt install -y build-essential cmake libopenblas-dev liblapack-dev
```

**2. Clone and create a virtual environment:**

```bash
git clone https://github.com/Lasan-Perera/Go2-Gesture-Control
cd Go2-Gesture-Control
python3 -m venv venv
source venv/bin/activate
```

**3. Install Python dependencies:**

```bash
pip install -r requirements.txt
```

This compiles `dlib` and can take several minutes. That's expected, not a hang.

## Usage

There are three scripts, run in order. The first two are one-time setup.

### Step 1 — Record training samples

```bash
python3 collect_gestures.py
```

| Key | Action |
|---|---|
| `1`–`6` | select which gesture to record |
| `SPACE` | record ONE sample (3-2-1 countdown) |
| `c` | toggle continuous recording |
| `u` | undo the last sample |
| `q` | quit |

**For the 5 command gestures:** select with `1`–`5`, then `SPACE`, one at a
time. Aim for **~40 samples each**. Vary distance from the camera, speed, and
which hand you use — *variety matters far more than raw count*. A model can't
learn a gesture you perform four different ways from a handful of examples.

**For NONE:** press `6`, then `c`, then move your hand around aimlessly for
about a minute. Drift, fidget, scratch your face, reach in and out of frame,
gesture while talking. Aim for **~100 samples**.

Samples are written to `data/<CLASS>/sample_NNNN.npy`.

### Step 2 — Train

```bash
python3 train_gestures.py
```

Reads `data/`, trains a RandomForest, evaluates it honestly, and writes
`gesture_model.pkl`.

**Read the output carefully:**

- **Macro F1** is the number that matters. It weights every class equally.
- **Ignore the `accuracy` line.** With a large NONE class, accuracy is close to
  worthless — the script prints the score a model would get by *always*
  answering NONE, so you can see for yourself.
- **The confusion matrix** shows which gestures bleed into each other.
  Off-diagonal cells are your real problems.

Target **macro F1 ≥ 0.90**. Below ~0.80, more data alone may not help — look at
the matrix and fix the specific classes that are colliding.

Expect to iterate: record → train → read the matrix → record more. That
iteration *is* the work.

### Step 3 — Run

```bash
python3 gesture_control.py
```

**First run:** look at the camera and press `e` to enroll your face. This saves
`target_face.npy`; you won't need to do it again. Press `e` any time to
re-enroll (different lighting, or to hand control to someone else).

**Then:**
1. Status bar shows `LOCKED` until your face is recognized.
2. Hold an **open palm still** inside the control zone → `ARMED`.
3. Perform any gesture. It flashes on screen with a confidence score and prints
   to the console.
4. Hold a **closed fist still** to `DISARM`. It also auto-disarms after ~20 s
   of inactivity.
5. Press `q` (with the video window focused) to quit.

If no trained model is found, the script transparently falls back to the
original threshold rules and shows `rules` in the corner, so it always runs.

## Why NONE matters

`NONE` is the class for *"my hand is visible but I am not commanding
anything."* It is the single biggest determinant of whether this works.

A classifier trained on only the 5 real gestures is mathematically forced to
assign every window it sees to one of them — including your hand simply
resting, or drifting between positions. The result is constant misfiring.

Two mistakes to avoid, both learned the hard way:

**Don't record STOP as NONE.** `STOP` is *"open palm, held still."* If you rest
your open hand in front of the camera during a NONE session, you are recording
STOP and labelling it NONE. The model receives directly contradictory labels
and both classes degrade. During NONE sessions, keep your hand **moving**, or
keep it **closed** if it's still.

**A large NONE class is good.** It should be your biggest class by 2–3x or
more. It teaches the model what idle looks like, across variety, and
`class_weight="balanced"` stops it dominating training. Just never judge the
model by raw accuracy once it's large.

## Architecture

```
webcam frame
    │
    ├─→ face_recognition (dlib) ──→ is this MY face?  ──→ control zone (follows the face)
    │                                                            │
    └─→ MediaPipe Hands ──→ which hand is inside the zone? ──────┘
                                    │
                                    ▼
                        rolling buffer of 12 frames
                     (palm center, palm size, open, closed)
                                    │
                     ┌──────────────┴──────────────┐
                     ▼                             ▼
            arm/disarm (rules)          extract_features() → RandomForest
            "held still + open"                    │
            "held still + fist"          confidence ≥ 0.60?
                                         2 consecutive frames agree?
                                                   │
                                                   ▼
                                            command dispatched
```

### Files

| File | Role |
|---|---|
| `gesture_common.py` | Landmark helpers, feature extraction, class list, rule-based fallback |
| `gesture_model.py` | Model loading + inference (no webcam needed — testable headless) |
| `collect_gestures.py` | Record labeled training samples |
| `train_gestures.py` | Train, evaluate, save the classifier |
| `gesture_control.py` | The live application |

`gesture_common.py` is imported by all three entry points so that collection,
training, and inference compute features through **the exact same code**.
Splitting that logic is the classic way to get train/serve skew — a model that
scores 97% in testing and behaves badly on your webcam.

### Design decisions

**RandomForest, not an LSTM.** With ~40–100 samples per class on a CPU-only
laptop, a forest over engineered motion features is the right tool: trains in
seconds, predicts in microseconds (no added latency in the live loop), needs no
feature scaling, and gives usable class probabilities for a confidence
threshold. An LSTM would need TensorFlow/PyTorch, far more data, and would add
real lag to a pipeline already fighting for frames.

**Features are scale-normalized, not translation-augmented.** Displacement is
measured relative to the window's first frame and divided by palm size.
A consequence worth knowing: shifting or uniformly rescaling a window leaves
its feature vector *bit-identical*, so translation and scale augmentation would
add nothing but duplicate rows. Mirroring and landmark noise are the
augmentations that actually carry signal.

**Augmentation never touches test data.** Mirroring a test sample into the
training set would leak it across the split and inflate the reported score.
`train_gestures.py` augments each training fold only.

**Continuous recording windows don't overlap.** With a stride shorter than the
window, consecutive samples share most of their frames — they're
near-duplicates, and they land on both sides of the train/test split, inflating
scores without improving live behavior.

## Tuning

Live behavior is the only test that counts. Tune from what you observe:

| Symptom | Fix |
|---|---|
| Fires while your hand idles | `CONFIDENCE_THRESHOLD` 0.60 → 0.75 in `gesture_model.py` |
| Still twitchy / double-fires | `CONSECUTIVE_AGREE` 2 → 3 in `gesture_control.py` |
| Real gestures ignored | `CONFIDENCE_THRESHOLD` down to 0.50 |
| Face lock too strict / too loose | `FACE_MATCH_THRESHOLD` in `gesture_control.py` (lower = stricter) |
| Control zone wrong size | `ZONE_*_MARGIN` constants in `gesture_control.py` |
| Laggy | raise `FACE_DETECT_EVERY_N_FRAMES`, lower `FACE_DOWNSCALE` |

## Known issues

**`AttributeError: module 'mediapipe' has no attribute 'solutions'`**
A packaging regression in mediapipe 0.10.31–0.10.35 on Python 3.12 breaks the
legacy `mp.solutions` API. `requirements.txt` pins `mediapipe==0.10.14`, the
last known-good version.

**`ModuleNotFoundError: No module named 'pkg_resources'`**
`face_recognition_models` still imports the deprecated `pkg_resources`, which
`setuptools >= 82.0.0` (Feb 2026) removed. `requirements.txt` pins
`setuptools<82`. You'll still see a *deprecation warning* — that's harmless.

**It's laggy.** `dlib`'s HOG face detector is CPU-only and by far the heaviest
part of the pipeline. Accepted for now. A background-threaded face detector,
decoupling video display from detection latency, is the next real fix.

**`MOVE BACKWARD` sometimes reads as `STOP`.** A genuine ambiguity in the
feature space: a palm shrinking slowly is hard to distinguish from a palm held
still while drifting away from the camera. Pull back sharply and decisively.
More data does not fix this one.

**A venv broke after moving its folder.** Python venvs bake in absolute paths at
creation. Delete and recreate it in the new location.

## What is not committed

`.gitignore` excludes, deliberately:

- `venv/` — regenerate from `requirements.txt`
- `target_face.npy` — a face embedding. Personal biometric data, and useless to
  anyone else since their face won't match it. Regenerate by pressing `e`.
- `data/` and `gesture_model.pkl` — recordings of you, and a model fitted to
  your camera, lighting, and personal swipe style. It would generalize poorly
  to anyone else. **Clone this repo and record your own.**

## Roadmap

- [ ] Publish commands to ROS 2 `/cmd_vel` (`geometry_msgs/Twist`) to drive the
      simulated Go2 in `go2-social-nav`
- [ ] Decide command semantics: discrete nudge per gesture vs. continuous
      velocity with timeout
- [ ] `twist_mux` so gesture control and keyboard teleop can coexist safely
- [ ] Background-threaded face detection
- [ ] Unit tests for the pure functions in `gesture_common.py`
- [ ] Configurable gesture set / bindings
