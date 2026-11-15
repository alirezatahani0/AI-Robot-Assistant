"""Default ESP32 DevKit GPIO map for the 1.3\" ST7789 module.

Module silk labels SCL/SDA are SPI SCK/MOSI on these boards — not I2C.
"""

from __future__ import annotations

from dataclasses import dataclass


# Typical 1.3" ST7789 resolution (confirm on your panel's datasheet).
ST7789_WIDTH = 240
ST7789_HEIGHT = 240


@dataclass(frozen=True)
class St7789Esp32Pins:
    """VSPI-friendly defaults for a 38-pin ESP32-WROOM DevKit."""

    sck: int = 18  # module SCL
    mosi: int = 23  # module SDA
    cs: int = 5
    dc: int = 2
    res: int = 4
    blk: int = 32
    # No MISO — panel is write-only.

    def as_table_rows(self) -> list[tuple[str, str, str]]:
        """(module_pin, signal, gpio_label) rows for docs / CLI."""
        return [
            ("GND", "Ground", "GND"),
            ("VCC", "3.3 V power", "3V3"),
            ("SCL", "SPI SCK", str(self.sck)),
            ("SDA", "SPI MOSI", str(self.mosi)),
            ("RES", "Reset", str(self.res)),
            ("DC", "Data/Command", str(self.dc)),
            ("CS", "Chip Select", str(self.cs)),
            ("BLK", "Backlight", str(self.blk)),
        ]


DEFAULT_ST7789_PINS = St7789Esp32Pins()
