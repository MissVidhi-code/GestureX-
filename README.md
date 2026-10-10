# GestureX: A Touchless Screen Control System

Control your laptop with hand gestures: move the cursor, click, scroll, and change volume or brightness, using only a webcam. No touching the trackpad or screen.

Built for people who find trackpads and touchscreens hard to use, and for anyone who'd rather not reach over just to pause a video or change the volume.

## Features

- **Cursor control** with your index finger, smoothed so it doesn't shake
- **Left click and right click**
- **Scroll up/down** while a gesture is held
- **Volume and brightness sliders** controlled by finger spread
- A live HUD showing the current gesture, FPS, latency, and volume/brightness bars
- A **hard-pause key** (SPACE) for safe demos

## Gestures

Finger state is `[thumb, index, middle, ring, pinky]`.

| Gesture | Fingers | Action |
|---|---|---|
| Index only | `[0,1,0,0,0]` | Move cursor |
| Index + middle | `[0,1,1,0,0]` | Left click (bring the two fingertips together) |
| Pinky only | `[0,0,0,0,1]` | Right click |
| Three fingers | `[0,1,1,1,0]` | Scroll down (while held) |
| Four fingers | `[0,1,1,1,1]` | Scroll up (while held) |
| Thumb + index | `[1,1,0,0,0]` | Volume (finger gap = level) |
| Thumb + index + pinky | `[1,1,0,0,1]` | Brightness (finger gap = level) |
| Open palm | `[1,1,1,1,1]` | Rest. Nothing triggers. |

## How it works

```
webcam frame -> mirror -> MediaPipe hand landmarks (21 points)
   -> which fingers are up? -> stabilise (must agree for N frames)
   -> map gesture to an OS action -> draw HUD
```

The hand tracking comes from MediaPipe. The work in this project is making it *usable*:

- **One Euro filter** ([Casiez, Roussel, Vogel, CHI 2012](https://gery.casiez.net/1euro/)). Raw landmarks jitter, so a fixed smoothing strength is either shaky or laggy. This filter smooths heavily when the hand is still and lightly when it moves fast.
- **Gesture stabiliser.** A gesture only counts after several consecutive frames agree, so one misread frame can't fire a click or jump to another mode.
- **Shape-based scrolling.** Reading scroll direction from hand motion was fragile, since it depends on the hand travelling far enough, fast enough, without leaving the tracked region. Two finger shapes, one per direction, only need the system to recognise a shape.
- **Native mouse events.** On macOS the cursor is moved by posting events directly through Quartz instead of calling a general-purpose library on every frame, which cuts the per-frame cost.
- **Throttled system calls.** Setting volume launches a process, so writes only happen when the value changes enough and enough time has passed.
- **Honest brightness HUD.** Brightness uses Apple's DisplayServices framework, because the usual `brightness` command fails on Apple Silicon built-in displays. If a write can't be confirmed, the HUD bar turns grey instead of pretending it worked.

## Project structure

```
GestureX-/
├── air_control.py            # main app: gestures, filters, OS control, HUD
├── Hand_Tracking_Module.py   # MediaPipe wrapper (handdetector class)
├── requirements.txt
└── README.md
```

`air_control.py` imports `handdetector` from `Hand_Tracking_Module.py`, so both files must sit in the same folder.

## Setup

**Platform:** macOS. The mouse, volume and brightness code uses macOS APIs (Quartz, AppleScript, DisplayServices).

```bash
pip install opencv-python mediapipe numpy pyautogui
```

Grant your terminal (or VS Code) permission under **System Settings > Privacy & Security > Accessibility**, and allow camera access. Without Accessibility permission nothing will move.

```bash
python air_control.py
```

Press `q` to quit and `SPACE` to pause.

## Tuning

Everything is a constant at the top of the file. The ones you'll most likely change:

- `PINCH_CLICK`: how close (in pixels) the two fingertips must be to count as a click
- `SLIDER_MIN`, `SLIDER_MAX`: thumb-to-index distance (in pixels) that maps to 0% and 100%. The HUD prints the live distance while you're in volume or brightness mode, so you can measure your own.
- `EURO_MIN_CUTOFF`: lower for less jitter when holding still
- `EURO_BETA`: higher for less lag on fast movement
- `STABLE_FRAMES`: higher is more reliable but adds latency
- `SCROLL_TICK_SECONDS`: lower for faster scrolling
- `MARGIN_X`, `MARGIN_TOP`, `MARGIN_BOT`: the size of the active box your hand moves in

## Limitations

- macOS only, for now.
- Pinch thresholds are in camera pixels, so they depend on how far you sit from the camera and may need recalibrating.
- The thumb is the least reliable finger for the tracker, so thumb-based gestures (volume, brightness) can occasionally misfire.
- Gestures take about 100 to 130 ms to register, which is the cost of the stabiliser.
- Lighting and background affect tracking quality.
- Designed with accessibility in mind, but **not yet tested with users who have motor impairments**. That user testing is the most important next step.

## Roadmap

- [ ] Hand-size-relative thresholds, so calibration doesn't depend on camera distance
- [ ] Zoom gesture
- [ ] Air writing, with text recognition from a CNN trained on my own handwriting
- [ ] Windows/Linux backend
- [ ] Testing with the users this is designed for

## Acknowledgements

- [MediaPipe Hands](https://developers.google.com/mediapipe) for hand landmark detection
- Casiez, Roussel and Vogel, *1 Euro Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems*, CHI 2012

## License

MIT
