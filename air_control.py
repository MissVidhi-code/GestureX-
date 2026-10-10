"""
AirControl -- touchless computer control via hand gestures.

Control the cursor, click, scroll, system volume and screen brightness
without touching the laptop. Built for accessibility: users with motor
impairments who cannot reliably operate a trackpad or touchscreen.

GESTURE MAP (finger state = [thumb, index, middle, ring, pinky])
    [0,1,0,0,0]  index only            ->  MOVE cursor
    [0,1,1,0,0]  index + middle        ->  LEFT CLICK  (pinch 8-12 together)
    [0,0,0,0,1]  pinky only            ->  RIGHT CLICK
    [0,1,1,1,0]  three fingers         ->  SCROLL DOWN (held, fixed rate)
    [0,1,1,1,1]  four fingers          ->  SCROLL UP   (held, fixed rate)
    [1,1,0,0,0]  thumb + index         ->  VOLUME      (pinch width = level)
    [1,1,0,0,1]  thumb + index + pinky ->  BRIGHTNESS  (pinch width = level)
    [1,1,1,1,1]  open palm             ->  NEUTRAL, everything frozen

FIVE THINGS THAT MAKE THIS USABLE
  1. One Euro filter -- adaptive smoothing. Heavy when the hand is still (kills
     landmark jitter), light when it moves fast (no lag). A fixed-strength
     filter cannot do both, which is why naive gesture cursors feel either
     shaky or sluggish.
  2. Gesture stabiliser -- a mode change requires N consecutive agreeing frames.
     Landmark noise causes single-frame misreads; without debouncing, one bad
     frame fires a click or jumps modes mid-gesture.
  3. Direct Quartz mouse events -- pyautogui.moveTo does validation, tweening
     setup and a screen-size lookup on every call (~15 ms). Posting the CGEvent
     directly is ~1 ms: the difference between 25 and 55 fps.
  4. Discrete-direction, fixed-rate scrolling -- up and down are two separate
     gestures, each scrolling a fixed number of units per tick while held.
     Earlier versions inferred direction from hand motion (move up = scroll
     up), which is fragile live: it depends on the hand travelling far enough,
     fast enough, without leaving the tracked region. A dedicated gesture per
     direction removes that failure mode entirely -- the system only has to
     recognise a finger shape, not read motion.
  5. DisplayServices brightness -- the `brightness` CLI fails on Apple Silicon
     built-in panels (IOKit error -536870201), so brightness goes through the
     private DisplayServices framework via ctypes, with a key-code fallback.

SETUP
    pip install pyautogui
    macOS: System Settings > Privacy & Security > Accessibility
           -> enable your terminal or VS Code, or nothing will move.

Run:  python air_control.py     ('q' quits, SPACE hard-pauses)
"""

import ctypes
import math
import subprocess
import time
from collections import deque

import cv2
import numpy as np
import pyautogui

from Hand_Tracking_Module import handdetector

# --- fast native mouse movement ---------------------------------------------
# Quartz ships as a pyautogui dependency on macOS. If unavailable we fall back
# to pyautogui, which works but costs ~15 ms per call.
try:
    import Quartz
    HAS_QUARTZ = True
except ImportError:
    HAS_QUARTZ = False

pyautogui.FAILSAFE = False   # corner-of-screen abort would fire constantly here
pyautogui.PAUSE = 0          # its default 0.1s sleep per call would cap us at 10 fps

# --- camera / mapping --------------------------------------------------------
# 640x480, not 720p: MediaPipe downsamples internally, so extra pixels cost
# frame time and buy nothing.
CAM_W, CAM_H = 640, 480
MARGIN_X, MARGIN_TOP, MARGIN_BOT = 80, 90, 90    # dead border: a smaller box
                                                   # maps to the whole screen so
                                                   # you never reach for edges

# --- gesture thresholds ------------------------------------------------------
PINCH_CLICK = 28             # px below which two fingertips count as pinched
CLICK_COOLDOWN = 0.6         # seconds between discrete clicks
STABLE_FRAMES = 4            # consecutive agreeing frames to commit a mode change

# --- scrolling ---------------------------------------------------------------
# Held-gesture scroll: fires a fixed number of units on a timer for as long as
# the gesture is held. No hand-motion tracking involved, so there is nothing
# to mis-detect except the finger shape itself.
SCROLL_UNITS_PER_TICK = 1    # units emitted per tick
SCROLL_TICK_SECONDS = 0.09   # seconds between emissions -- lower = faster scroll

# --- slider calibration ------------------------------------------------------
# Thumb-index pixel distance at minimum / maximum, at 640x480. Measure your own:
# the HUD prints the live distance while in VOLUME or BRIGHTNESS mode.
SLIDER_MIN, SLIDER_MAX = 20.0, 120.0

# --- system-call throttling --------------------------------------------------
SYS_STEP = 2                 # ignore changes smaller than this (0-100 scale)
SYS_INTERVAL = 0.1           # seconds between writes to the OS

# --- One Euro tuning ---------------------------------------------------------
# beta up          -> less lag on fast movement
# min_cutoff down  -> less jitter when holding still
EURO_MIN_CUTOFF = 0.8
EURO_BETA = 0.02


# =============================================================================
#  FAST MOUSE MOVE
# =============================================================================
def move_mouse(x, y):
    """Post a mouse-move event straight to the HID tap (~1 ms)."""
    if HAS_QUARTZ:
        evt = Quartz.CGEventCreateMouseEvent(
            None, Quartz.kCGEventMouseMoved, (float(x), float(y)), 0)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, evt)
    else:
        pyautogui.moveTo(x, y, duration=0)


# =============================================================================
#  BRIGHTNESS BACKEND
# =============================================================================
class BrightnessBackend:
    """Set built-in display brightness on Apple Silicon.

    The Homebrew `brightness` CLI uses an IOKit path that M-series internal
    panels do not expose, so it fails with error -536870201. DisplayServices is
    a private framework that does work. Every step is guarded: if the framework
    or symbol is missing we degrade to stepped key codes and report that we
    could not confirm the write.
    """

    def __init__(self):
        self.set_fn = None
        self.get_fn = None
        self.display_id = None

        try:
            ds = ctypes.CDLL(
                "/System/Library/PrivateFrameworks/"
                "DisplayServices.framework/DisplayServices"
            )
            ds.DisplayServicesSetBrightness.argtypes = [ctypes.c_uint32, ctypes.c_float]
            ds.DisplayServicesSetBrightness.restype = ctypes.c_int
            ds.DisplayServicesGetBrightness.argtypes = [
                ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)]
            ds.DisplayServicesGetBrightness.restype = ctypes.c_int

            if not HAS_QUARTZ:
                raise ImportError("Quartz needed for CGMainDisplayID")

            self.display_id = Quartz.CGMainDisplayID()
            self.set_fn = ds.DisplayServicesSetBrightness
            self.get_fn = ds.DisplayServicesGetBrightness
        except (OSError, AttributeError, ImportError):
            pass                                  # falls back to key codes

    @property
    def available(self):
        return self.set_fn is not None

    def get(self):
        """Current brightness as 0-100, or 50 if unreadable."""
        if not self.available:
            return 50
        level = ctypes.c_float(0.0)
        if self.get_fn(self.display_id, ctypes.byref(level)) == 0:
            return int(level.value * 100)
        return 50

    def set(self, percent, going_up=True):
        """Set 0-100. Returns True only if the write actually landed."""
        if self.available:
            rc = self.set_fn(self.display_id, ctypes.c_float(percent / 100.0))
            if rc == 0:
                return True

        # Fallback: key code 144 = brighter, 145 = dimmer. Stepped, and there is
        # no way to verify it worked, so the caller is told False.
        code = 144 if going_up else 145
        subprocess.run(
            ["osascript", "-e",
             f'tell application "System Events" to key code {code}'],
            check=False, capture_output=True)
        return False


# =============================================================================
#  ONE EURO FILTER
# =============================================================================
class OneEuroFilter:
    """Adaptive low-pass filter (Casiez et al., CHI 2012).

    Cutoff frequency rises with signal speed:
        cutoff = min_cutoff + beta * |velocity|

    Slow movement -> low cutoff  -> heavy smoothing -> no jitter.
    Fast movement -> high cutoff -> light smoothing -> no lag.
    """

    def __init__(self, freq=30.0, min_cutoff=EURO_MIN_CUTOFF,
                 beta=EURO_BETA, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None       # previous filtered value
        self.dx_prev = 0.0       # previous filtered derivative
        self.t_prev = None       # previous timestamp

    @staticmethod
    def _alpha(cutoff, freq):
        """Smoothing factor for a given cutoff frequency and sample rate."""
        tau = 1.0 / (2.0 * math.pi * cutoff)    # time constant
        te = 1.0 / freq                         # sampling period
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x, t=None):
        t = time.time() if t is None else t

        if self.x_prev is None:                 # first sample, nothing to filter
            self.x_prev, self.t_prev = x, t
            return x

        # Derive the real sample rate from timestamps rather than assuming it:
        # frame rate varies, and a wrong freq detunes the entire filter.
        dt = t - self.t_prev
        if dt > 0:
            self.freq = 1.0 / dt
        self.t_prev = t

        # Filter the derivative first, then use it to choose the cutoff.
        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff, self.freq)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        self.dx_prev = dx_hat

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, self.freq)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        return x_hat

    def reset(self):
        """Clear history, so re-entering a mode does not drag the old value in."""
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None


# =============================================================================
#  GESTURE STABILISER
# =============================================================================
class GestureStabiliser:
    """Commit a mode change only after N consecutive agreeing frames.

    MediaPipe occasionally misreads a finger for a single frame -- the thumb
    especially, since its extension is judged sideways. Without debouncing, one
    bad frame fires a click or drops the cursor into volume mode mid-sentence.
    """

    def __init__(self, n=STABLE_FRAMES):
        self.buf = deque(maxlen=n)
        self.committed = None

    def update(self, state):
        """Feed the current finger state; return the committed (stable) state."""
        self.buf.append(tuple(state) if state else None)
        if len(self.buf) == self.buf.maxlen and len(set(self.buf)) == 1:
            self.committed = self.buf[0]        # whole buffer agrees -> commit
        return self.committed


# =============================================================================
#  HOLD SCROLLER
# =============================================================================
class HoldScroller:
    """Fixed-direction, fixed-rate scrolling while a gesture is held.

    Direction comes from WHICH gesture is active (three fingers vs four),
    not from how the hand moves -- so there is nothing to mis-detect except
    the finger shape itself. Call .tick(direction) every frame the relevant
    gesture is held; it self-throttles via SCROLL_TICK_SECONDS so holding the
    gesture doesn't flood the OS with scroll events every frame.
    """

    def __init__(self, units_per_tick=SCROLL_UNITS_PER_TICK,
                 tick_seconds=SCROLL_TICK_SECONDS):
        self.units_per_tick = units_per_tick
        self.tick_seconds = tick_seconds
        self.last_tick = 0.0

    def reset(self):
        """Called on entering a scroll mode, so switching directions or
        re-entering after a pause doesn't fire an event instantly."""
        self.last_tick = 0.0

    def tick(self, direction: int) -> int:
        """direction: +1 scrolls up, -1 scrolls down. Returns units emitted
        this frame (0 most frames, since emission is rate-limited)."""
        now = time.time()
        if now - self.last_tick < self.tick_seconds:
            return 0
        self.last_tick = now
        units = self.units_per_tick * direction
        pyautogui.scroll(units)
        return units


# =============================================================================
#  SYSTEM CONTROL  (volume + brightness, both throttled)
# =============================================================================
class SystemControl:
    """Set macOS volume and brightness, with change- and rate-gating.

    Volume forks an osascript process (~15 ms) per write. Calling it per frame
    would halve the frame rate, so a write happens only when the value moved by
    at least SYS_STEP and SYS_INTERVAL seconds have passed.
    """

    def __init__(self):
        self._bright_backend = BrightnessBackend()
        self.vol = self._read_volume()
        self.bright = self._bright_backend.get()
        self.brightness_ok = self._bright_backend.available

        self._last_sent = {"vol": -1, "bright": -1}
        self._last_write = {"vol": 0.0, "bright": 0.0}

    @property
    def has_brightness(self):
        return self._bright_backend.available

    @staticmethod
    def _read_volume():
        """Read the current system volume, so the HUD starts truthful."""
        try:
            out = subprocess.run(
                ["osascript", "-e", "output volume of (get volume settings)"],
                capture_output=True, text=True, timeout=2,
            )
            return int(out.stdout.strip())
        except (ValueError, subprocess.SubprocessError):
            return 50

    def _should_write(self, key, value):
        """True when the value moved enough AND enough time has passed."""
        now = time.time()
        if (abs(value - self._last_sent[key]) >= SYS_STEP
                and now - self._last_write[key] >= SYS_INTERVAL):
            self._last_sent[key] = value
            self._last_write[key] = now
            return True
        return False

    def set_volume(self, percent):
        percent = int(max(0, min(100, percent)))
        self.vol = percent
        if not self._should_write("vol", percent):
            return
        subprocess.run(
            ["osascript", "-e", f"set volume output volume {percent}"],
            check=False, capture_output=True,
        )

    def set_brightness(self, percent):
        percent = int(max(0, min(100, percent)))
        going_up = percent > self.bright
        self.bright = percent
        if not self._should_write("bright", percent):
            return
        # The backend reports whether the write actually landed, so the HUD
        # cannot animate while the screen does nothing.
        self.brightness_ok = self._bright_backend.set(percent, going_up)


# =============================================================================
#  HUD
# =============================================================================
def draw_bar(img, x, y, value, label, colour, h=150):
    """Vertical 0-100 bar. Judges need to see state, not infer it."""
    # Reversed output range: in images y=0 is the TOP.
    fill_y = int(np.interp(value, [0, 100], [y + h, y]))
    cv2.rectangle(img, (x, y), (x + 28, y + h), colour, 2)
    cv2.rectangle(img, (x, fill_y), (x + 28, y + h), colour, cv2.FILLED)
    cv2.putText(img, f"{value}%", (x - 8, y + h + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
    cv2.putText(img, label, (x - 4, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1)


def draw_hud(img, mode, fps, latency_ms, paused, sysctl, extra=""):
    h, w = img.shape[:2]

    # Solid panel, not addWeighted: blending the full frame every iteration is
    # measurable overhead for a cosmetic effect.
    cv2.rectangle(img, (0, 0), (w, 70), (18, 18, 18), cv2.FILLED)

    colour = (0, 165, 255) if paused else (0, 255, 120)
    cv2.putText(img, f"{'PAUSED' if paused else mode}", (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2)
    cv2.putText(img, f"{fps:3.0f} FPS | {latency_ms:4.1f} ms  {extra}",
                (14, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

    # Grey bar = the backend could not confirm the write. An honest HUD beats
    # one that animates while nothing happens on screen.
    bri_colour = (0, 200, 255) if sysctl.brightness_ok else (90, 90, 90)
    draw_bar(img, w - 55, 100, sysctl.vol, "VOL", (255, 0, 255))
    draw_bar(img, w - 110, 100, sysctl.bright, "BRI", bri_colour)

    # The active region: only movement inside this box maps to the screen.
    cv2.rectangle(img, (MARGIN_X, MARGIN_TOP),
                  (CAM_W - MARGIN_X, CAM_H - MARGIN_BOT), (110, 110, 110), 1)
    return img


# =============================================================================
#  MAIN
# =============================================================================
def main() -> int:
    screen_w, screen_h = pyautogui.size()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open camera. Check camera permissions.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    # modelComplexity=0 is the lite model: ~2x faster, and the accuracy loss is
    # invisible for fingertip tracking.
    detector = handdetector(maxHands=1, modelComplexity=0,
                            detectionCon=0.7, trackCon=0.7)

    sysctl = SystemControl()
    stabiliser = GestureStabiliser()
    scroller = HoldScroller()

    # One filter per axis; they are stateful, so they live outside the loop.
    fx = OneEuroFilter()
    fy = OneEuroFilter()

    last_click = 0.0
    paused = False
    p_time = 0.0
    mode = "IDLE"
    prev_mode = None

    if not HAS_QUARTZ:
        print("[warn] Quartz unavailable -- falling back to pyautogui.moveTo, "
              "which is ~15x slower per call.")
    if not sysctl.has_brightness:
        print("[warn] DisplayServices unavailable -- brightness will be stepped "
              "and unverifiable. The BRI bar will show grey.")

    try:
        while True:
            t_start = time.time()          # start of end-to-end latency window

            success, img = cap.read()
            if not success:
                continue

            img = cv2.flip(img, 1)         # mirror, so movement matches screen

            # draw=False: rendering 21 circles and 21 connections per frame in
            # Python is expensive, and we draw our own cursor indicator anyway.
            img = detector.findHands(img, draw=False)
            lm_list, bbox = detector.findPosition(img, draw=False)

            latency_ms = 0.0
            extra = ""

            # Debounce the raw finger reading into a committed mode.
            raw = detector.fingersUp() if lm_list else None
            fingers = stabiliser.update(raw)

            if lm_list and fingers and not paused:
                x_idx, y_idx = lm_list[8][1:]         # index fingertip

                # --- MOVE -----------------------------------------------------
                if fingers == (0, 1, 0, 0, 0):
                    mode = "MOVE"
                    if prev_mode != "MOVE":               # fresh entry
                        fx.reset()
                        fy.reset()

                    # np.interp clamps, so a hand outside the box parks the
                    # cursor at the edge instead of going out of range.
                    sx = np.interp(x_idx, (MARGIN_X, CAM_W - MARGIN_X),
                                   (0, screen_w))
                    sy = np.interp(y_idx, (MARGIN_TOP, CAM_H - MARGIN_BOT),
                                   (0, screen_h))
                    move_mouse(fx(sx), fy(sy))
                    cv2.circle(img, (x_idx, y_idx), 10, (0, 255, 120), cv2.FILLED)

                # --- LEFT CLICK -----------------------------------------------
                elif fingers == (0, 1, 1, 0, 0):
                    mode = "CLICK"
                    length, info, img = detector.findDistance(8, 12, img)
                    extra = f"pinch={int(length)}"
                    if length < PINCH_CLICK and time.time() - last_click > CLICK_COOLDOWN:
                        pyautogui.click()
                        last_click = time.time()
                        if info:
                            cv2.circle(img, (info[4], info[5]), 18,
                                       (0, 255, 0), cv2.FILLED)

                # --- RIGHT CLICK ----------------------------------------------
                elif fingers == (0, 0, 0, 0, 1):
                    mode = "RIGHT CLICK"
                    if time.time() - last_click > CLICK_COOLDOWN:
                        pyautogui.rightClick()
                        last_click = time.time()

                # --- SCROLL DOWN ------------------------------------------------
                elif fingers == (0, 1, 1, 1, 0):
                    mode = "SCROLL DOWN"
                    if prev_mode != "SCROLL DOWN":
                        scroller.reset()                  # no instant fire on entry
                    units = scroller.tick(direction=-1)
                    extra = f"units={units:+d}"
                    dot = (255, 130, 0) if units else (90, 90, 90)
                    cv2.circle(img, (x_idx, y_idx), 10, dot, cv2.FILLED)

                # --- SCROLL UP --------------------------------------------------
                elif fingers == (0, 1, 1, 1, 1):
                    mode = "SCROLL UP"
                    if prev_mode != "SCROLL UP":
                        scroller.reset()
                    units = scroller.tick(direction=1)
                    extra = f"units={units:+d}"
                    dot = (0, 200, 255) if units else (90, 90, 90)
                    cv2.circle(img, (x_idx, y_idx), 10, dot, cv2.FILLED)

                # --- VOLUME ---------------------------------------------------
                elif fingers == (1, 1, 0, 0, 0):
                    mode = "VOLUME"
                    length, info, img = detector.findDistance(4, 8, img)
                    value = int(np.interp(length, [SLIDER_MIN, SLIDER_MAX], [0, 100]))
                    sysctl.set_volume(value)
                    extra = f"pinch={int(length)}px"

                # --- BRIGHTNESS -----------------------------------------------
                elif fingers == (1, 1, 0, 0, 1):
                    mode = "BRIGHTNESS"
                    length, info, img = detector.findDistance(4, 8, img)
                    value = int(np.interp(length, [SLIDER_MIN, SLIDER_MAX], [0, 100]))
                    sysctl.set_brightness(value)
                    extra = f"pinch={int(length)}px"

                # --- NEUTRAL --------------------------------------------------
                elif fingers == (1, 1, 1, 1, 1):
                    mode = "NEUTRAL"
                    fx.reset()
                    fy.reset()
                    scroller.reset()

                else:
                    mode = "IDLE"

                prev_mode = mode
                latency_ms = (time.time() - t_start) * 1000

            elif not lm_list:
                mode = "NO HAND"
                prev_mode = None
                scroller.reset()

            c_time = time.time()
            fps = 1 / max(c_time - p_time, 1e-6)
            p_time = c_time

            draw_hud(img, mode, fps, latency_ms, paused, sysctl, extra)
            cv2.imshow("AirControl", img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused             # hard stop, for demo safety
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
