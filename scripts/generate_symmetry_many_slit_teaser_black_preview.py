from __future__ import annotations

from pathlib import Path

import generate_symmetry_many_slit_paths_phasors_interference_teaser as teaser


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "content"
    / "drafts"
    / "animations"
    / "symmetry-many-slit-paths-phasors-interference-teaser-palette-black-preview.png"
)

# Neutral near-black gradient; all foreground colors remain those of the
# current Wine / Powder Blue master.
BG_TOP = (2, 2, 3)
BG_BOTTOM = (10, 8, 10)


def main() -> None:
    teaser.BG_TOP = BG_TOP
    teaser.BG_BOTTOM = BG_BOTTOM
    teaser.BACKGROUND = teaser.make_background()
    frame_index = round(teaser.DURATION * teaser.FPS) - 1
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    teaser.draw_frame(frame_index).save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
