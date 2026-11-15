"""Simple cartoon eyes for the ST7789 (or laptop simulator).

Idle blink + look-left / look-right — enough to prove the display path.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

from robot.device.base import Display, DeviceError

BG = (8, 12, 28)
SCLERA = (240, 244, 255)
IRIS = (70, 140, 220)
PUPIL = (12, 16, 28)
LID = (8, 12, 28)
HIGHLIGHT = (255, 255, 255)


@dataclass(frozen=True)
class EyePose:
    """Gaze offset in pixels (positive x = right) and lid close amount 0–1."""

    gaze_x: float = 0.0
    gaze_y: float = 0.0
    blink: float = 0.0  # 0 open … 1 fully closed


class EyeDemo:
    """Draw a pair of eyes and step through idle / look animations."""

    def __init__(self, display: Display) -> None:
        self.display = display
        w, h = display.width, display.height
        self.eye_r = max(18, min(w, h) // 5)
        gap = self.eye_r // 2
        cx = w // 2
        cy = h // 2
        self.left = (cx - self.eye_r - gap // 2, cy)
        self.right = (cx + self.eye_r + gap // 2, cy)

    def draw(self, pose: EyePose) -> None:
        d = self.display
        d.fill(BG)
        for cx, cy in (self.left, self.right):
            self._draw_eye(cx, cy, pose)
        d.show()

    def _draw_eye(self, cx: int, cy: int, pose: EyePose) -> None:
        d = self.display
        r = self.eye_r
        # Sclera
        self._fill_circle(cx, cy, r, SCLERA)
        # Iris + pupil follow gaze, clamped inside the sclera
        max_off = r // 3
        gx = int(max(-max_off, min(max_off, pose.gaze_x)))
        gy = int(max(-max_off, min(max_off, pose.gaze_y)))
        iris_r = r // 2
        self._fill_circle(cx + gx, cy + gy, iris_r, IRIS)
        self._fill_circle(cx + gx, cy + gy, iris_r // 2, PUPIL)
        # Specular highlight
        self._fill_circle(cx + gx - iris_r // 3, cy + gy - iris_r // 3, max(2, r // 10), HIGHLIGHT)
        # Eyelids (top + bottom grow toward center as blink → 1)
        if pose.blink > 0:
            lid_h = int(r * 2 * min(1.0, max(0.0, pose.blink)))
            half = lid_h // 2
            d.draw_rect(cx - r, cy - r, r * 2 + 1, half + 1, LID, filled=True)
            d.draw_rect(cx - r, cy + r - half, r * 2 + 1, half + 1, LID, filled=True)

    def _fill_circle(self, cx: int, cy: int, radius: int, color: tuple[int, int, int]) -> None:
        if radius <= 0:
            return
        d = self.display
        r2 = radius * radius
        for y in range(-radius, radius + 1):
            # span of x for this scanline
            dx = int(math.sqrt(r2 - y * y))
            d.draw_rect(cx - dx, cy + y, dx * 2 + 1, 1, color, filled=True)

    def animate_sequence(self, duration_sec: float = 12.0) -> None:
        """Idle blinks and look left/right until duration elapses (or window closes)."""
        t0 = time.monotonic()
        phase = 0.0
        while True:
            elapsed = time.monotonic() - t0
            if elapsed >= duration_sec:
                break
            # Cycle: center → blink → look left → center → blink → look right → …
            cycle = elapsed % 6.0
            if cycle < 0.15:
                pose = EyePose(blink=cycle / 0.15)
            elif cycle < 0.3:
                pose = EyePose(blink=1.0 - (cycle - 0.15) / 0.15)
            elif cycle < 1.5:
                pose = EyePose()
            elif cycle < 2.5:
                pose = EyePose(gaze_x=-self.eye_r * 0.45)
            elif cycle < 2.65:
                pose = EyePose(blink=(cycle - 2.5) / 0.15)
            elif cycle < 2.8:
                pose = EyePose(blink=1.0 - (cycle - 2.65) / 0.15)
            elif cycle < 4.0:
                pose = EyePose()
            elif cycle < 5.0:
                pose = EyePose(gaze_x=self.eye_r * 0.45)
            else:
                pose = EyePose()
            try:
                self.draw(pose)
            except DeviceError:
                break
            # Tiny sleep so headless / test paths don't spin a core.
            time.sleep(1 / 30)
            phase = elapsed


def run_eye_demo(display: Display, duration_sec: float = 12.0) -> None:
    EyeDemo(display).animate_sequence(duration_sec=duration_sec)
