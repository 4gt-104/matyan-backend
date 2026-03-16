#!/usr/bin/env python3
"""Generate favicon.ico from matyan green logo SVG."""

from __future__ import annotations

import os
import sys

# Paths relative to repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SVG_PATH = os.path.join(REPO_ROOT, "extra", "matyan-ui", "web", "src", "assets", "logo.svg")
ICO_PATH = os.path.join(REPO_ROOT, "extra", "matyan-ui", "web", "public", "favicon.ico")


def main() -> None:
    import cairosvg
    from PIL import Image
    import io

    if not os.path.isfile(SVG_PATH):
        print(f"SVG not found: {SVG_PATH}", file=sys.stderr)
        sys.exit(1)

    sizes = [16, 32, 48]
    pil_images = []

    for size in sizes:
        png_data = cairosvg.svg2png(
            url=SVG_PATH,
            output_width=size,
            output_height=size,
        )
        img = Image.open(io.BytesIO(png_data))
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        pil_images.append(img)

    pil_images[0].save(
        ICO_PATH,
        format="ICO",
        sizes=[(im.width, im.height) for im in pil_images],
        append_images=pil_images[1:] if len(pil_images) > 1 else [],
    )
    print(f"Wrote {ICO_PATH}")


if __name__ == "__main__":
    main()
