"""Display adapter + eye demo smoke tests (no hardware window required)."""

from __future__ import annotations

import numpy as np
import pytest

from robot.device.base import DeviceError
from robot.device.esp32 import Esp32Display
from robot.device.laptop import LaptopDisplay
from robot.device.pins import DEFAULT_ST7789_PINS, ST7789_HEIGHT, ST7789_WIDTH
from robot.display.eyes import EyeDemo, EyePose


def test_st7789_default_resolution() -> None:
    assert ST7789_WIDTH == 240
    assert ST7789_HEIGHT == 240


def test_pin_table_covers_module() -> None:
    rows = {r[0]: r for r in DEFAULT_ST7789_PINS.as_table_rows()}
    assert rows["SCL"][2] == "18"
    assert rows["SDA"][2] == "23"
    assert rows["CS"][2] == "5"
    assert rows["DC"][2] == "2"
    assert rows["RES"][2] == "4"
    assert rows["BLK"][2] == "32"


def test_laptop_display_draw_and_blit() -> None:
    d = LaptopDisplay(headless=True, scale=1)
    try:
        d.fill((10, 20, 30))
        assert tuple(d._fb[0, 0]) == (10, 20, 30)
        d.draw_pixel(1, 2, (255, 0, 0))
        assert tuple(d._fb[2, 1]) == (255, 0, 0)
        d.draw_rect(10, 10, 4, 3, (0, 255, 0), filled=True)
        assert tuple(d._fb[11, 11]) == (0, 255, 0)
        img = np.zeros((2, 3, 3), dtype=np.uint8)
        img[:] = (0, 0, 255)
        d.blit_rgb(0, 0, img)
        assert tuple(d._fb[0, 0]) == (0, 0, 255)
        # one white RGB565 pixel: 0xFFFF little-endian
        d.blit_rgb565(5, 5, 1, 1, b"\xff\xff")
        assert tuple(d._fb[5, 5]) == (255, 255, 255)
        d.set_backlight(False)
        d.show()
    finally:
        d.close()


def test_eye_demo_draws_without_crash() -> None:
    d = LaptopDisplay(headless=True, scale=1)
    try:
        eyes = EyeDemo(d)
        eyes.draw(EyePose())
        eyes.draw(EyePose(blink=1.0))
        eyes.draw(EyePose(gaze_x=-10))
        eyes.animate_sequence(duration_sec=0.05)
        # Sclera should have painted something non-background near center
        assert d._fb.sum() > 0
    finally:
        d.close()


def test_esp32_display_stub_raises() -> None:
    d = Esp32Display()
    assert d.width == 240
    with pytest.raises(DeviceError, match="SCK=18"):
        d.clear()
