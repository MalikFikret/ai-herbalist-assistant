#!/usr/bin/env python3
"""Install chat shell background plate (native copy, no blur or resize)."""

from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src/herbalist_assistant/ui/static"
REFERENCE_PLATE = ROOT / "assets/reference/chat_herbal_background.png"
OUT_PLATE = STATIC / "chat_shell_background.png"
# Very subtle softness on leaves (native resolution, no resize).
LIGHT_BLUR_RADIUS = 1.2


def build() -> None:
    if not REFERENCE_PLATE.is_file():
        raise SystemExit(f"Missing reference plate: {REFERENCE_PLATE}")

    STATIC.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REFERENCE_PLATE, OUT_PLATE)
    img = Image.open(OUT_PLATE).convert("RGBA")
    w, h = img.size
    if LIGHT_BLUR_RADIUS > 0:
        img = img.filter(ImageFilter.GaussianBlur(LIGHT_BLUR_RADIUS))
        img.save(OUT_PLATE, format="PNG", optimize=True)
    print(f"Wrote {OUT_PLATE} ({w}x{h}, light blur r={LIGHT_BLUR_RADIUS})")


if __name__ == "__main__":
    build()
