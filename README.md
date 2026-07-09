# Go2 Gesture Control

Real-time **dynamic hand gesture recognition**, locked to a single enrolled
operator via face recognition, built as a standalone control interface for
navigating a simulated Unitree Go2 robot dog.

This started as a companion tool for [`go2-social-nav`](https://github.com/KaveeshwaraBandara/go2-social-nav)
(a ROS 2 + Gazebo simulation of a Go2 navigating among simulated pedestrians)
— the idea being to replace keyboard teleop with hand gestures. Right now it
runs standalone (webcam in, gesture out) so the detection pipeline can be
built and tuned independently before wiring it into `/cmd_vel`.

## What it does

- Opens your webcam and tracks your hand in real time using MediaPipe.
- Recognizes 5 **dynamic** gestures — motion over time, not static poses:

| Gesture | Motion | Command |
|---|---|---|
| Swipe hand left | fast horizontal motion, left | `TURN LEFT` |
| Swipe hand right | fast horizontal motion, right | `TURN RIGHT` |
| Push hand toward camera | palm moves closer/larger | `MOVE FORWARD` |
| Pull hand away | palm moves farther/smaller | `MOVE BACKWARD` |
| Open palm, held still | no significant motion, palm open | `STOP` |

- The moment a gesture is recognized, it's shown on screen and printed to
  the console, then detection immediately resets — ready for the next
  gesture with no cooldown beyond a brief visual confirmation.

## Why it's more than "just MediaPipe hand tracking"

A naive version of this reacts to *any* hand movement from *anyone* in the
camera's view — unusable in a real environment with other people around.
This version adds two layers on top of raw gesture detection:

**1. Arm / disarm (wake gesture).** The system starts **locked**, ignoring
all motion. You hold an open palm still for about a second to **arm** it;
only then do movement gestures get dispatched. Holding a closed fist still
**disarms** it again. This stops incidental movement (talking with your
hands, adjusting your glasses, walking past the camera) from being
misread as commands.

**2. Face-recognition identity lock.** On first run you enroll your face
(press `e`). From then on, every frame, the system identifies *your* face
specifically among everyone visible, and builds a control zone that
dynamically follows you around the frame. Hands belonging to anyone else —
even if their hand crosses through where yours was — are tracked visually
(drawn in gray) but never treated as a command. Because it's identity-based
rather than just positional tracking, you can fully leave the frame and
come back, or be temporarily blocked by someone walking past, and control
resumes automatically once your face is visible again.

## Demo

*(add a screenshot or short GIF here showing the control zone, arm/disarm
state, and a recognized gesture banner — this section is a placeholder)*

## Requirements

- Ubuntu (tested on 22.04/24.04) with a working webcam
- Python 3.12
- A well-lit environment for reliable face/hand detection

## Installation

**1. Install system build tools** (needed to compile `dlib`, which
`face_recognition` depends on — there's no prebuilt wheel for it):

```bash
sudo apt install -y build-essential cmake libopenblas-dev liblapack-dev
```

**2. Clone the repo and set up a virtual environment:**

```bash
git clone <this-repo-url>
cd <this-repo>
python3 -m venv venv
source venv/bin/activate
```

**3. Install Python dependencies:**

```bash
pip install -r requirements.txt
```

This step compiles `dlib` from source and can take several minutes —
that's expected, not a hang.

## Usage

**Run it:**

```bash
python3 gesture_control.py
```

**First run — enroll your face:**
Look at the camera and press `e`. This captures your face, saves an
embedding to `target_face.npy` next to the script, and enrolls you as the
tracked target. You won't need to do this again on future runs (the file
persists). Press `e` at any time to re-enroll — useful if lighting has
changed a lot, or to hand control to someone else.

**Controlling it:**
1. Status bar shows `LOCKED` until your face is recognized.
2. Hold an **open palm still**, inside the control zone (the box that
   follows your face), for about a second → status changes to `ARMED`.
3. Perform any of the 5 gestures from the table above — each one flashes
   on screen and prints to the console.
4. Hold a **closed fist still** in the zone to `DISARM`, or just stop
   giving commands — it auto-disarms after ~20 seconds of inactivity.
5. Press `q` to quit.

## How it works

1. **Face ID** (`face_recognition` / `dlib`): every few frames, detects
   all faces in a downscaled copy of the frame and compares each against
   your enrolled embedding. The best match (if under a distance threshold)
   becomes "the target," and a control zone is built as a region
   below/around that face box.
2. **Hand tracking** (MediaPipe Hands): detects up to 2 hands per frame.
   Whichever hand falls inside the current control zone is treated as the
   active hand; others are drawn but ignored.
3. **Motion buffer**: the active hand's palm center and a palm-size proxy
   (used to detect push/pull) are tracked over a short rolling window of
   recent frames.
4. **Gesture classification**: swipe/push/pull are detected by how much
   the palm center/size *changed* across that window (not a single-frame
   pose). Stop is detected by the hand staying still and open for a
   sustained number of frames.
5. **Arm/disarm state machine**: gates whether movement gestures are
   dispatched at all, using the same still-hand detection logic applied to
   open-palm (arm) vs. closed-fist (disarm).

## Tuning

All thresholds live as named constants near the top of `gesture_control.py`
— worth adjusting if detection feels too sensitive, too sluggish, or the
control zone doesn't fit your setup:

| Constant | Controls |
|---|---|
| `SWIPE_DX_THRESH` | how far a swipe must travel to register |
| `SIZE_CHANGE_THRESH` | how much palm size must change for push/pull |
| `STILL_MOTION_THRESH` | how "still" counts as still (for stop/arm/disarm) |
| `ARM_HOLD_FRAMES` / `DISARM_HOLD_FRAMES` | how long you must hold the wake gesture |
| `FACE_MATCH_THRESHOLD` | how strict the face match must be (lower = stricter) |
| `ZONE_SIDE_MARGIN` / `ZONE_ABOVE_MARGIN` / `ZONE_BELOW_MARGIN` | size of the control zone relative to your detected face |
| `FACE_DETECT_EVERY_N_FRAMES` / `FACE_DOWNSCALE` | face-detection performance vs. responsiveness trade-off |

## Known issues / troubleshooting

**`AttributeError: module 'mediapipe' has no attribute 'solutions'`**
A packaging regression in mediapipe 0.10.31–0.10.35 on Python 3.12 breaks
the legacy `mp.solutions` API this project uses. `requirements.txt` pins
`mediapipe==0.10.14`, the last known-good version, to avoid this.

**`ModuleNotFoundError: No module named 'pkg_resources'`**
`face_recognition_models` still imports the deprecated `pkg_resources`
module, which `setuptools >= 82.0.0` (Feb 2026) removed entirely.
`requirements.txt` pins `setuptools<82` to avoid this.

**It's laggy.** Face recognition (`dlib`'s HOG detector) is CPU-only and
the heaviest part of this pipeline by far. Levers, in order of impact:
lower `CAPTURE_WIDTH`/`CAPTURE_HEIGHT`, increase `FACE_DETECT_EVERY_N_FRAMES`,
decrease `FACE_DOWNSCALE`, or set `number_of_times_to_upsample=0` in the
`face_recognition.face_locations()` calls. A background-thread face
detector (decoupling video display from detection latency) would be the
next real step if these aren't enough.

**A venv stopped working after moving its folder.** Python venvs bake in
absolute paths at creation time; moving the folder can break the
`activate` script. Simplest fix is deleting and recreating the venv in its
new location.

## Roadmap

- [ ] Publish gesture commands to ROS 2 `/cmd_vel` (`geometry_msgs/Twist`)
      to drive the simulated Go2 in `go2-social-nav` directly
- [ ] Background-threaded face detection for smoother video while ID runs
- [ ] Unit tests for the pure gesture-classification functions
- [ ] Configurable gesture set / bindings

## Project structure

```
.
├── gesture_control.py     # main script
├── requirements.txt       # pinned Python dependencies
├── target_face.npy        # created on first enrollment (gitignored - do not commit)
└── README.md
```

## License

MIT (or update to whatever license you've chosen for the repo).
